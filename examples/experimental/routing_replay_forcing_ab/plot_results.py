"""Figure and summary of the routing replay forcing A/B (run_ab.py), OFF vs ON per step. Test-only.

    python plot_results.py --source local --off off.log --on on.log [--out-dir DIR]
    python plot_results.py --source wandb --wandb-project PROJECT [--wandb-entity ENTITY] --off RUN --on RUN

--source local reads a saved launcher log (`run_ab.py train ... 2>&1 | tee off.log`); --source wandb reads the W&B
run with that name (run_ab.py names runs after --run-id). An arm may list several comma-separated logs or runs, for a
run resumed from a checkpoint; later ones win on shared steps. Writes fig-3-off-vs-on.png and summary.json: the mean
ON - OFF log-prob gap with a 95% interval, overall and per 25 steps, and which README.md stop rules fired. Miles logs
rollout/r3/forcing_refused_turns only when nonzero, so an ON step without it counts as 0.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIGURE_NAME = "fig-3-off-vs-on.png"

LOGPROB_GAP = "train/train_rollout_logprob_abs_diff"
KL = "train/train_rollout_kl"
TIS_CLIPFRAC = "train/tis_clipfrac"
RETRACTED = "rollout/r3/retracted_turn_frac"
FORCED = "rollout/r3/forced_turn_frac"
REFUSED = "rollout/r3/forcing_refused_turns"
ABORTED_GROUPS = "rollout/fully_async/aborted_groups_filtered"
WEIGHT_SYNC = "perf/update_weights_time"
REWARD = "rollout/episode_raw_reward"
# Metrics Miles logs only when nonzero, and the arms that can log them: there a step that has rollout metrics but
# not the key counts 0. Forcing is off in the OFF arm, so its refused turns are absent, not zero.
_ZERO_WHEN_MISSING = {REFUSED: ("on",)}
_DIVERGENCE_KEYS = ("train/loss", "train/grad_norm", LOGPROB_GAP)

# Stop rules; README.md states the same thresholds.
SESSIONS_PER_STEP = 256  # 32 prompts x 8 samples
REFUSED_SHARE_LIMIT = 0.02
REFUSED_CONSECUTIVE_STEPS = 3
GAP_BLOWUP_FACTOR = 3.0
GAP_BASELINE_STEPS = 10
GAP_BLOWUP_CONSECUTIVE_STEPS = 3
WINDOW_STEPS = 25  # one checkpoint interval
# Samples up to 4 versions stale correlate nearby steps, so the interval resamples blocks of 5 consecutive steps.
BOOTSTRAP_BLOCK = 5

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_METRIC_LINE = re.compile(r" - (?P<kind>step|perf|rollout|multi_turn) (?P<index>\d+): (?P<body>\{.*)")
_SESSION_REFUSAL_LINE = re.compile(r"Session \S+: (?P<reason>.+); its sample will be aborted")
_NON_FINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_NUMPY_SCALARS = {"float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint8", "bool_"}


@dataclass
class RunMetrics:
    """Per-rollout-step scalars of one arm, plus what loading them found."""

    steps: dict[int, dict[str, float]] = field(default_factory=dict)
    rollout_steps: set[int] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)
    parse_errors: int = 0
    refusal_log_lines: dict[str, int] = field(default_factory=dict)

    def series(self, key: str, *, missing_as_zero: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Steps and values of ``key``, sorted by step; NaN and inf are kept so a plot shows the gap.

        ``missing_as_zero`` counts a step that has rollout metrics but not ``key`` as 0 (see ``_ZERO_WHEN_MISSING``).
        """
        points = []
        for step in sorted(self.steps):
            value = self.steps[step].get(key)
            if value is None and missing_as_zero and step in self.rollout_steps:
                value = 0.0
            if value is not None:
                points.append((step, value))
        if not points:
            return np.array([], dtype=int), np.array([], dtype=float)
        steps, values = zip(*points, strict=True)
        return np.asarray(steps, dtype=int), np.asarray(values, dtype=float)

    def merge(self, other: RunMetrics) -> None:
        for step, values in other.steps.items():
            self.steps.setdefault(step, {}).update(values)
        self.rollout_steps |= other.rollout_steps
        self.sources += other.sources
        self.parse_errors += other.parse_errors
        for reason, count in other.refusal_log_lines.items():
            self.refusal_log_lines[reason] = self.refusal_log_lines.get(reason, 0) + count


# ---------------------------------------------------------------------------------------------------------------
# Parsing


def parse_log_lines(lines: Iterable[str], *, steps_per_rollout: int = 1, namespace: str | None = None) -> RunMetrics:
    """Per-step metrics from Miles log lines.

    A ``step N`` line is a train step, mapped to rollout step ``N // steps_per_rollout`` (several train steps of one
    rollout are averaged); ``perf``, ``rollout`` and ``multi_turn`` lines carry the rollout id. ``namespace`` strips a
    ``<trainer model id>/`` key prefix. Lines that do not parse are counted, never guessed at.
    """
    run = RunMetrics()
    train_sums: dict[int, dict[str, list[float]]] = {}
    for raw in lines:
        line = _ANSI.sub("", raw)
        if (refusal := _SESSION_REFUSAL_LINE.search(line)) is not None:
            reason = _refusal_reason(refusal["reason"])
            run.refusal_log_lines[reason] = run.refusal_log_lines.get(reason, 0) + 1
            continue
        match = _METRIC_LINE.search(line)
        if match is None:
            continue
        body = match["body"]
        try:
            # Ray may append " [repeated Nx across cluster]" after the dict.
            values = _scalars(_literal(ast.parse(body[: body.rindex("}") + 1], mode="eval").body))
        except (SyntaxError, ValueError):  # rindex raises ValueError on a line cut before its closing brace
            run.parse_errors += 1
            continue
        values = _strip_namespace(values, namespace)
        index = int(match["index"])
        if match["kind"] == "step":
            if not any(key.startswith("train/") for key in values):
                continue
            rollout_step = index // steps_per_rollout
            sums = train_sums.setdefault(rollout_step, {})
            for key, value in values.items():
                if key.startswith("train/"):
                    sums.setdefault(key, []).append(value)
        else:
            run.steps.setdefault(index, {}).update(values)
            if match["kind"] == "perf" and any(key.startswith("rollout/") for key in values):
                run.rollout_steps.add(index)
    for rollout_step, sums in train_sums.items():
        run.steps.setdefault(rollout_step, {}).update({key: float(np.mean(v)) for key, v in sums.items()})
    return run


def records_from_rows(rows: Iterable[dict], *, steps_per_rollout: int = 1, namespace: str | None = None) -> RunMetrics:
    """Per-step metrics from W&B history rows: ``train/*`` keys by ``train/step``, the rest by ``rollout/step``."""
    run = RunMetrics()
    train_sums: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        values = _strip_namespace(_scalars({k: v for k, v in row.items() if not k.startswith("_")}), namespace)
        train_step, rollout_step = values.pop("train/step", None), values.pop("rollout/step", None)
        if train_step is not None:
            sums = train_sums.setdefault(int(train_step) // steps_per_rollout, {})
            for key, value in values.items():
                if key.startswith("train/"):
                    sums.setdefault(key, []).append(value)
        if rollout_step is not None:
            rest = {key: value for key, value in values.items() if not key.startswith("train/")}
            run.steps.setdefault(int(rollout_step), {}).update(rest)
            if any(key.startswith("rollout/") for key in rest):
                run.rollout_steps.add(int(rollout_step))
    for rollout_step, sums in train_sums.items():
        run.steps.setdefault(rollout_step, {}).update({key: float(np.mean(v)) for key, v in sums.items()})
    return run


def _literal(node: ast.AST):
    """``ast.literal_eval`` plus the reprs Miles' dicts can contain: nan, inf, np.float64(x), tensor(x)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values, strict=True)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Name) and node.id in ("nan", "inf"):
        return float(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and len(node.args) == 1 and _is_scalar_wrapper(node.func):
        return _literal(node.args[0])
    raise ValueError(f"not a literal: {ast.dump(node)[:80]}")


def _is_scalar_wrapper(func: ast.AST) -> bool:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("np", "numpy"):
        return func.attr in _NUMPY_SCALARS
    return isinstance(func, ast.Name) and func.id in ("tensor", "float", "int")


def _scalars(values) -> dict[str, float]:
    """Numeric entries only; W&B returns non-finite values as the strings "NaN", "Infinity" and "-Infinity"."""
    if not isinstance(values, dict):
        raise ValueError("the logged metrics are not a dict")
    out = {}
    for key, value in values.items():
        if isinstance(value, str) and value in _NON_FINITE:
            value = _NON_FINITE[value]
        if isinstance(key, str) and isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _strip_namespace(values: dict[str, float], namespace: str | None) -> dict[str, float]:
    if namespace is None:
        return values
    prefix = f"{namespace}/"
    return {key.removeprefix(prefix): value for key, value in values.items()}


def _refusal_reason(message: str) -> str:
    """Bucket a session-server refusal line by cause: SGLang's code, or which forcing check failed."""
    if "without forcing their routes" in message:
        return "unforced_tokens"
    if "reported no unforced_tokens" in message:
        return "no_forcing_counters"
    if (code := re.search(r"refused the turn: (\w+)", message)) is not None:
        return code[1]
    return "other"


# ---------------------------------------------------------------------------------------------------------------
# Loading


def load_local(log: Path, **parse_kwargs) -> RunMetrics:
    with log.open(encoding="utf-8", errors="replace") as lines:
        run = parse_log_lines(lines, **parse_kwargs)
    run.sources.append(f"local:{log}")
    return run


def load_wandb(run_name: str, *, project: str, entity: str | None, **parse_kwargs) -> RunMetrics:
    import wandb  # optional: only this source needs it

    api = wandb.Api(timeout=60)
    path = f"{entity or api.default_entity}/{project}"
    runs = list(api.runs(path, filters={"display_name": run_name}))
    if len(runs) != 1:
        raise LookupError(f"{len(runs)} W&B runs named {run_name!r} in {path}")
    run = records_from_rows(runs[0].scan_history(page_size=1000), **parse_kwargs)
    run.sources.append(f"wandb:{path}/{runs[0].id}")
    return run


def load_arm(names: list[str], *, source: str, project: str | None, entity: str | None, **kw) -> RunMetrics:
    arm = RunMetrics()
    for name in names:
        if source == "wandb":
            arm.merge(load_wandb(name, project=project, entity=entity, **kw))
        else:
            arm.merge(load_local(Path(name), **kw))
    return arm


# ---------------------------------------------------------------------------------------------------------------
# Analysis


def paired_difference(
    off: RunMetrics,
    on: RunMetrics,
    key: str = LOGPROB_GAP,
    *,
    first: int = 0,
    last: int | None = None,
    block: int = BOOTSTRAP_BLOCK,
    resamples: int = 4000,
    seed: int = 0,
) -> dict:
    """Mean ON - OFF over the steps both arms logged, with a 95% interval from resampling blocks of steps."""
    off_steps, off_values = off.series(key)
    on_steps, on_values = on.series(key)
    off_by_step = dict(zip(off_steps.tolist(), off_values.tolist(), strict=True))
    pairs = [
        (step, value - off_by_step[step])
        for step, value in zip(on_steps.tolist(), on_values.tolist(), strict=True)
        if step in off_by_step and step >= first and (last is None or step <= last)
    ]
    diffs = np.asarray([d for _, d in pairs if math.isfinite(d)], dtype=float)
    return {
        "key": key,
        "first_step": first,
        "last_step": last,
        "paired_steps": int(diffs.size),
        "mean": float(diffs.mean()) if diffs.size else None,
        "ci95": _block_bootstrap_ci(diffs, block=block, resamples=resamples, seed=seed),
    }


def _block_bootstrap_ci(
    diffs: np.ndarray, *, block: int = BOOTSTRAP_BLOCK, resamples: int = 4000, seed: int = 0
) -> list[float] | None:
    """95% interval of the mean from resampling blocks of consecutive steps; None below two blocks."""
    if diffs.size < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, diffs.size - block + 1, size=(resamples, math.ceil(diffs.size / block)))
    samples = diffs[(starts[..., None] + np.arange(block)).reshape(resamples, -1)[:, : diffs.size]]
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return [float(low), float(high)]


def window_differences(off: RunMetrics, on: RunMetrics, *, window: int = WINDOW_STEPS) -> list[dict]:
    last = max(set(off.steps) & set(on.steps), default=-1)
    return [paired_difference(off, on, first=start, last=start + window - 1) for start in range(0, last + 1, window)]


def check_stop_rules(off: RunMetrics, on: RunMetrics) -> list[str]:
    """The README's stop rules that fired, each with the step it fired at; empty when none did."""
    fired = []
    if count := on.refusal_log_lines.get("unforced_tokens", 0):
        fired.append(f"ON: {count} turn(s) reported unforced tokens (session-server log)")
    steps, refused = on.series(REFUSED, missing_as_zero=True)
    limit = REFUSED_SHARE_LIMIT * SESSIONS_PER_STEP
    if (step := _first_run_above(steps, refused, limit, REFUSED_CONSECUTIVE_STEPS)) is not None:
        fired.append(
            f"ON: refused turns above {limit:g} per step for {REFUSED_CONSECUTIVE_STEPS} steps, from step {step}"
        )
    for name, arm in (("OFF", off), ("ON", on)):
        for key in _DIVERGENCE_KEYS:
            steps, values = arm.series(key)
            if (bad := steps[~np.isfinite(values)]).size:
                fired.append(f"{name}: {key} is NaN or inf at step {int(bad[0])}")
        steps, gap = arm.series(LOGPROB_GAP)
        baseline = gap[:GAP_BASELINE_STEPS][np.isfinite(gap[:GAP_BASELINE_STEPS])]
        if baseline.size == GAP_BASELINE_STEPS:
            limit = GAP_BLOWUP_FACTOR * float(np.median(baseline))
            if (step := _first_run_above(steps, gap, limit, GAP_BLOWUP_CONSECUTIVE_STEPS)) is not None:
                fired.append(
                    f"{name}: log-prob gap above {GAP_BLOWUP_FACTOR:g}x its first-10-step median from step {step}"
                )
    return fired


def _first_run_above(steps: np.ndarray, values: np.ndarray, limit: float, length: int) -> int | None:
    run_start, run_length, previous = None, 0, None
    for step, value in zip(steps.tolist(), values.tolist(), strict=True):
        consecutive = previous is not None and step == previous + 1
        if value > limit:
            run_start, run_length = (run_start, run_length + 1) if consecutive and run_length else (step, 1)
            if run_length >= length:
                return run_start
        else:
            run_length = 0
        previous = step
    return None


def summarize(off: RunMetrics, on: RunMetrics) -> dict:
    return {
        "sources": {"off": off.sources, "on": on.sources},
        "steps": {"off": len(off.steps), "on": len(on.steps)},
        "parse_errors": {"off": off.parse_errors, "on": on.parse_errors},
        "refusal_log_lines": {"off": off.refusal_log_lines, "on": on.refusal_log_lines},
        "logprob_gap_on_minus_off": paired_difference(off, on),
        "logprob_gap_on_minus_off_by_window": window_differences(off, on),
        "stop_rules_fired": check_stop_rules(off, on),
    }


# ---------------------------------------------------------------------------------------------------------------
# Plot

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
OFF_COLOR = "#eb6834"  # orange: routes recomputed under new weights
ON_COLOR = "#2a78d6"  # blue: first-computed routes
# ON also keeps additional R3 on under retract (radixark/miles#3307), so the arms differ in more than forcing.
ARMS = (("off", "OFF (default)", OFF_COLOR), ("on", "ON (forcing)", ON_COLOR))
SMOOTH_STEPS = 10
_CUMULATIVE = (ABORTED_GROUPS,)  # rare counts read better as a running total than as a 10-step mean
_COUNTS = (REFUSED, ABORTED_GROUPS)  # whole numbers: integer ticks, and a y range that shows 0 as a count
_SCALE = {KL: 1e3, TIS_CLIPFRAC: 1e4}  # plotted in units of 1e-3 and 1e-4
_PERCENT = (RETRACTED, FORCED)  # shares of turns, shown in %

PANELS = (
    (LOGPROB_GAP, "Train/rollout log-prob gap", "mean |log p_train − log p_rollout|", ("off", "on")),
    ("paired", "Log-prob gap, ON − OFF per step", "difference (below 0: ON closer)", ()),
    (KL, "Train/rollout KL", "KL, ×10⁻³", ("off", "on")),
    (TIS_CLIPFRAC, "TIS clip fraction", "share of tokens, ×10⁻⁴", ("off", "on")),
    (RETRACTED, "Retracted turns", "% of turns", ("off", "on")),
    (FORCED, "Forced turns (ON)", "% of turns", ("on",)),
    (REFUSED, "Refused turns (ON)", "turns per step", ("on",)),
    (ABORTED_GROUPS, "Dropped (aborted) groups, cumulative", "groups since step 0", ("off", "on")),
    (WEIGHT_SYNC, "Weight sync before the step", "seconds", ("off", "on")),
    (REWARD, "Reward", "mean episode reward", ("off", "on")),
)


def plot(off: RunMetrics, on: RunMetrics, out_path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    arms = {"off": off, "on": on}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    figure, axes = plt.subplots(5, 2, figsize=(13, 15), facecolor=SURFACE, sharex=True)
    for ax, (key, name, unit, arm_names) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_ylabel(unit)
        if key == "paired":
            _plot_paired(ax, off, on)
            continue
        if key in _PERCENT:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        drawn, all_zero, peak = False, True, 0.0
        for arm_name, label, color in ARMS:
            if arm_name not in arm_names:
                continue
            steps, values = arms[arm_name].series(key, missing_as_zero=arm_name in _ZERO_WHEN_MISSING.get(key, ()))
            if not steps.size:
                continue
            drawn = True
            values = values * _SCALE.get(key, 1)
            all_zero = all_zero and bool(np.all(np.nan_to_num(values) == 0))
            if key == REFUSED:
                _plot_refused(ax, steps, values, color=color, label=label)
                peak = max(peak, float(np.nanmax(values)))
                continue
            if key in _CUMULATIVE:
                total = np.cumsum(np.nan_to_num(values))
                ax.step(steps, total, where="post", color=color, linewidth=2, label=label)
                peak = max(peak, float(total[-1]))
                continue
            ax.plot(steps, values, color=color, linewidth=1, alpha=0.35)
            ax.plot(steps, _rolling_mean(values), color=color, linewidth=2, label=label)
        if not drawn:
            ax.text(0.5, 0.5, "not logged", transform=ax.transAxes, ha="center", va="center", color=INK2)
        elif key in _COUNTS:
            ax.set_ylim(-0.5, max(5.0, 1.15 * peak))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if key in _CUMULATIVE and all_zero and len(arm_names) > 1:
                # The arms' lines coincide at 0, so only the last one drawn is visible.
                ax.text(0.5, 0.5, "0 in both arms", transform=ax.transAxes, ha="center", va="center", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("step (one policy update and one weight sync per step)")
    handles = [plt.Line2D([], [], color=color, linewidth=2, label=label) for _, label, color in ARMS]
    figure.legend(handles=handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.995))
    figure.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=13, fontweight="bold")
    figure.text(
        0.01,
        0.968,
        f"thin lines: per step; thick lines: mean of the last {SMOOTH_STEPS} steps",
        fontsize=9,
        color=INK2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return out_path


def _plot_refused(ax, steps: np.ndarray, values: np.ndarray, *, color: str, label: str) -> None:
    # A line, not bars: a zero-height bar draws nothing, so a run without refusals would look unlogged.
    ax.step(steps, values, where="mid", color=color, linewidth=2, label=label)
    if np.all(values == 0):
        ax.text(
            0.5,
            0.5,
            f"0 on every step ({values.size} steps)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
        )


def _plot_paired(ax, off: RunMetrics, on: RunMetrics) -> None:
    off_steps, off_values = off.series(LOGPROB_GAP)
    on_steps, on_values = on.series(LOGPROB_GAP)
    off_by_step = dict(zip(off_steps.tolist(), off_values.tolist(), strict=True))
    pairs = [
        (s, v - off_by_step[s]) for s, v in zip(on_steps.tolist(), on_values.tolist(), strict=True) if s in off_by_step
    ]
    ax.axhline(0, color=INK2, linewidth=1)
    if not pairs:
        ax.text(0.5, 0.5, "no paired steps", transform=ax.transAxes, ha="center", va="center", color=INK2)
        return
    steps, diffs = (np.asarray(v) for v in zip(*pairs, strict=True))
    overall = paired_difference(off, on)
    if overall["ci95"] is not None:
        # One mean over all paired steps, not a per-step band: drawn as a dashed line with its interval.
        mean, (low, high) = overall["mean"], overall["ci95"]
        ax.axhspan(low, high, color=GRID, zorder=0)
        ax.axhline(mean, color=INK, linewidth=1.2, linestyle="--", zorder=1)
        ax.text(
            0.01,
            0.97,
            _minus(f"dashed: mean {mean:.4f} [{low:.4f}, {high:.4f}] (95%), all {overall['paired_steps']} steps"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=INK2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
    ax.plot(steps, diffs, color=INK2, linewidth=1, alpha=0.35)
    ax.plot(steps, _rolling_mean(diffs), color=INK, linewidth=2)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.15 * (high - low))  # room for the note above the zero line


def _minus(text: str) -> str:
    """Typographic minus signs, matching matplotlib's tick labels."""
    return text.replace("-", "−")


def _rolling_mean(values: np.ndarray, window: int = SMOOTH_STEPS) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for i in range(values.size):
        chunk = values[max(0, i - window + 1) : i + 1]
        chunk = chunk[np.isfinite(chunk)]
        out[i] = chunk.mean() if chunk.size else np.nan
    return out


# ---------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off", required=True, help="the OFF arm's W&B run names or log files, comma-separated")
    parser.add_argument("--on", required=True, help="the ON arm's W&B run names or log files, comma-separated")
    parser.add_argument("--source", choices=("wandb", "local"), required=True)
    parser.add_argument("--wandb-project", help="--source wandb: the project the runs logged to")
    parser.add_argument(
        "--wandb-entity", help="--source wandb: the W&B entity (default: the API key's default entity)"
    )
    parser.add_argument(
        "--steps-per-rollout",
        type=int,
        default=1,
        help="train steps per rollout (1 when the global batch equals the rollout batch)",
    )
    parser.add_argument("--namespace", default=None, help="strip this <trainer model id>/ prefix from keys")
    parser.add_argument("--title", default="Routing replay forcing, OFF vs ON")
    parser.add_argument("--out-dir", type=Path, default=Path("ab-results"))
    args = parser.parse_args(argv)
    if args.source == "wandb" and not args.wandb_project:
        parser.error("--source wandb needs --wandb-project")

    common = {
        "source": args.source,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "steps_per_rollout": args.steps_per_rollout,
        "namespace": args.namespace,
    }
    off = load_arm(args.off.split(","), **common)
    on = load_arm(args.on.split(","), **common)
    summary = {"off": args.off.split(","), "on": args.on.split(","), **summarize(off, on)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure = plot(off, on, args.out_dir / FIGURE_NAME, title=args.title)
    print(json.dumps(summary, indent=2))
    print(f"wrote {figure} and {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
