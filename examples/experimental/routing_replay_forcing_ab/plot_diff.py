"""Figure of the routing replay forcing A/B (run_ab.py): the paired ON - OFF difference per step. Test-only.

    python plot_diff.py --off off.log --on on.log [--out-dir DIR] [--stats FILE] [--title T] [--subtitle S]

Reads the same saved launcher logs as plot_results.py, which must sit next to this file. An arm may list several
comma-separated logs, oldest first, for a run resumed from a checkpoint; a later log wins on a step both logged. Only
steps both arms logged are paired; the subtitle counts them. The 95% interval resamples blocks of 5 steps, as
plot_results.py does.

Writes fig-4-on-minus-off.png to --out-dir, and the numbers the panels state to --stats when given.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from plot_results import (
    BOOTSTRAP_BLOCK,
    GRID,
    INK,
    INK2,
    KL,
    LOGPROB_GAP,
    OFF_COLOR,
    ON_COLOR,
    REWARD,
    SMOOTH_STEPS,
    SURFACE,
    TIS_CLIPFRAC,
    _block_bootstrap_ci,
    _minus,
    _rolling_mean,
    load_arm,
)

FIGURE_NAME = "fig-4-on-minus-off.png"
TITLE = "ON (forcing) − OFF (default), per step"

RESPONSE_LEN = "rollout/response_len/mean"

LATE = 20  # every panel's "last steps" window: the last LATE paired steps
SIMILAR_LENGTH = 0.10  # panel (f): steps whose mean response lengths differ by at most this fraction

BAND = "#cfcdc4"
TIS_SCALE = 1e4  # plotted in units of 1e-4


def paired(off: dict[int, dict], on: dict[int, dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps both arms logged ``key`` at (finite in both), with the OFF and ON values."""
    rows = [
        (step, off[step][key], on[step][key])
        for step in sorted(set(off) & set(on))
        if _finite(off[step].get(key)) and _finite(on[step].get(key))
    ]
    if not rows:
        return np.array([], dtype=int), np.array([]), np.array([])
    steps, off_values, on_values = (np.asarray(column) for column in zip(*rows, strict=True))
    return steps.astype(int), off_values.astype(float), on_values.astype(float)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def running_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running mean over steps 0..i and its 95% interval (NaN until two blocks exist)."""
    mean = np.cumsum(values) / np.arange(1, values.size + 1)
    low, high = np.full(values.size, np.nan), np.full(values.size, np.nan)
    for i in range(values.size):
        if (ci := _block_bootstrap_ci(values[: i + 1])) is not None:
            low[i], high[i] = ci
    return mean, low, high


def analyze(off: dict[int, dict], on: dict[int, dict]) -> dict:
    out: dict = {}
    steps, off_gap, on_gap = paired(off, on, LOGPROB_GAP)
    diff = on_gap - off_gap
    gap_diff_by_step = dict(zip(steps.tolist(), diff.tolist(), strict=True))
    out["gap"] = {
        "steps": int(steps.size),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "on_lower_steps": int((diff < 0).sum()),
        "mean_on_minus_off": float(diff.mean()),
        "ci95": _block_bootstrap_ci(diff),
        "mean_off": float(off_gap.mean()),
        "mean_on": float(on_gap.mean()),
        "relative_pct": float(100 * (on_gap.mean() / off_gap.mean() - 1)),
        f"relative_last{LATE}_pct": float(100 * (on_gap[-LATE:].mean() / off_gap[-LATE:].mean() - 1)),
    }
    steps, off_kl, on_kl = paired(off, on, KL)
    ratio = on_kl / off_kl
    out["kl"] = {
        "steps": int(steps.size),
        "on_lower_steps": int((ratio < 1).sum()),
        "geometric_mean_ratio": float(np.exp(np.log(ratio).mean())),
        f"geometric_mean_ratio_last{LATE}": float(np.exp(np.log(ratio[-LATE:]).mean())),
    }
    steps, off_tis, on_tis = paired(off, on, TIS_CLIPFRAC)
    out["tis_clipfrac"] = {
        "steps": int(steps.size),
        "on_lower_steps": int((on_tis < off_tis).sum()),
        "mean_on_minus_off": float((on_tis - off_tis).mean()),
        "mean_off": float(off_tis.mean()),
        "mean_on": float(on_tis.mean()),
    }
    steps, off_reward, on_reward = paired(off, on, REWARD)
    diff = on_reward - off_reward
    out["reward"] = {
        "steps": int(steps.size),
        "mean_on_minus_off": float(diff.mean()),
        f"mean_on_minus_off_last{LATE}": float(diff[-LATE:].mean()),
    }
    steps, off_len, on_len = paired(off, on, RESPONSE_LEN)
    diff = on_len - off_len
    # Does ON's lower gap come from its shorter responses? Compare the gap only on steps whose lengths are close.
    similar = np.abs(on_len / off_len - 1) <= SIMILAR_LENGTH
    similar_gap_diffs = [gap_diff_by_step[s] for s in steps[similar].tolist() if s in gap_diff_by_step]
    out["response_len"] = {
        "steps": int(steps.size),
        "mean_on_minus_off": float(diff.mean()),
        "ci95": _block_bootstrap_ci(diff),
        "late_window": [int(steps[-LATE]), int(steps[-1])],
        "late_mean_off": float(off_len[-LATE:].mean()),
        "late_mean_on": float(on_len[-LATE:].mean()),
        "similar_length_steps": len(similar_gap_diffs),
        "similar_length_gap_on_lower_steps": sum(d < 0 for d in similar_gap_diffs),
    }
    return out


# ---------------------------------------------------------------------------------------------------------------
# Plot


def plot(off: dict[int, dict], on: dict[int, dict], stats: dict, out_path: Path, *, title: str, subtitle: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.labelsize": 10,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "text.color": INK,
        }
    )
    figure, grid = plt.subplots(3, 2, figsize=(10, 10.6), facecolor=SURFACE, sharex=True)
    axes = grid.ravel()
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    gap = stats["gap"]
    steps, off_v, on_v = paired(off, on, LOGPROB_GAP)
    _difference(ax := axes[0], steps, on_v - off_v, band=True)
    _title(ax, "(a) Log-prob gap, ON − OFF", "Δ mean |log p_train − log p_rollout|")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _signed(v, "{:.4f}")))
    _note(ax, f"ON lower on {gap['on_lower_steps']}/{gap['steps']} steps\n" + _mean_ci(gap, "{:+.4f}"))

    _difference(ax := axes[1], steps, 100 * (on_v / off_v - 1))
    _title(ax, "(b) Log-prob gap, ON / OFF − 1", "relative difference, %")
    # The note compares the arms' mean gaps, not the mean of the per-step ratios the panel plots.
    _note(
        ax,
        f"mean gap: ON {_lower(gap['relative_pct'])} ({gap['mean_on']:.4f} vs {gap['mean_off']:.4f})\n"
        f"last {LATE} steps: {_lower(gap[f'relative_last{LATE}_pct'])}",
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _signed(v, "{:.0f}") + "%"))

    kl = stats["kl"]
    steps, off_v, on_v = paired(off, on, KL)
    ax = axes[2]
    ratio = on_v / off_v
    ax.axhline(1, color=INK2, linewidth=1)
    ax.plot(steps, ratio, color=INK2, linewidth=0.9, alpha=0.55)
    ax.plot(steps, np.exp(_rolling_mean(np.log(ratio))), color=INK, linewidth=2.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    _title(ax, "(c) Train/rollout KL, ON / OFF", "ratio (log scale)")
    _note(
        ax,
        f"ON lower on {kl['on_lower_steps']}/{kl['steps']} steps\ngeometric mean {kl['geometric_mean_ratio']:.2f}×, "
        f"last {LATE} steps {kl[f'geometric_mean_ratio_last{LATE}']:.2f}×",
    )
    _headroom(ax)
    _tint(ax, zero=1.0)

    tis = stats["tis_clipfrac"]
    steps, off_v, on_v = paired(off, on, TIS_CLIPFRAC)
    _difference(ax := axes[3], steps, TIS_SCALE * (on_v - off_v), note_top=False)
    _title(ax, "(d) TIS clip fraction, ON − OFF", "Δ share of tokens clipped, ×10⁻⁴")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _signed(v, "{:.0f}")))
    _note(
        ax,
        f"ON lower on {tis['on_lower_steps']}/{tis['steps']} steps\n"
        f"mean OFF {TIS_SCALE * tis['mean_off']:.2f}, ON {TIS_SCALE * tis['mean_on']:.2f} (×10⁻⁴)",
        top=False,
    )

    reward = stats["reward"]
    steps, off_v, on_v = paired(off, on, REWARD)
    _difference(ax := axes[4], steps, on_v - off_v)
    _title(ax, "(e) Reward, ON − OFF", "Δ mean episode reward")
    late = on_v[-LATE:] - off_v[-LATE:]
    _note(
        ax,
        _minus(f"mean {reward['mean_on_minus_off']:+.3f}; last {LATE} steps {late.mean():+.3f}")
        + "\none seed, not a claim",
    )

    length = stats["response_len"]
    steps, off_v, on_v = paired(off, on, RESPONSE_LEN)
    _difference(ax := axes[5], steps, on_v - off_v, headroom=0.5)  # room for a four-line note
    _title(ax, "(f) Response length, ON − OFF", "Δ mean response tokens")
    late_pct = 100 * (length["late_mean_on"] / length["late_mean_off"] - 1)
    _note(
        ax,
        _minus(
            f"mean {length['mean_on_minus_off']:+.0f} tokens\nlast {LATE} steps: ON {length['late_mean_on']:.0f} "
            f"vs OFF {length['late_mean_off']:.0f} ({late_pct:+.0f}%)"
        )
        + f"\nsteps where both arms' mean lengths are within {SIMILAR_LENGTH:.0%}:\nlog-prob gap ON lower on "
        f"{length['similar_length_gap_on_lower_steps']}/{length['similar_length_steps']}",
    )

    for ax in grid[-1]:
        ax.set_xlabel("step (one policy update each)")
    last_step = max(gap["last_step"], length["late_window"][1])
    for ax in axes:
        ax.set_xlim(gap["first_step"] - 1, last_step + 1)

    figure.suptitle(title, x=0.012, y=0.988, ha="left", fontsize=12.5, fontweight="bold")
    lead = f"{subtitle}, " if subtitle else ""
    figure.text(
        0.012,
        0.952,
        f"{lead}one seed, {gap['steps']} steps; steps are matched by training step, not by prompt. "
        "Below 0 (below 1× in c): ON lower.\nLower is better in a–d (trainer and rollout agree more); "
        "higher is better in e. Blue tint: ON lower; orange: OFF lower.\n"
        "ON = forcing plus additional R3 under retract (radixark/miles#3307).",
        ha="left",
        va="top",
        fontsize=10,
        color=INK2,
    )
    handles = [
        Line2D([], [], color=INK2, linewidth=0.9, alpha=0.55, label="per step"),
        Line2D([], [], color=INK, linewidth=2.2, label=f"{SMOOTH_STEPS}-step trailing mean"),
        Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 2)), label="(a) running mean over steps 0 to x"),
        Patch(facecolor=BAND, edgecolor="none", label="(a) 95% interval of the running mean"),
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.006, 0.899),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
        columnspacing=1.4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.909), h_pad=1.4, w_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(figure)
    return out_path


def _difference(
    ax, steps: np.ndarray, diff: np.ndarray, *, band: bool = False, note_top: bool = True, headroom: float = 0.2
) -> None:
    ax.axhline(0, color=INK2, linewidth=1)
    if band:
        mean, low, high = running_ci(diff)
        ax.fill_between(steps, low, high, color=BAND, linewidth=0, zorder=1)
        ax.plot(steps, mean, color=INK, linewidth=1.2, linestyle=(0, (4, 2)), zorder=3)
    ax.plot(steps, diff, color=INK2, linewidth=0.9, alpha=0.55, zorder=2)
    ax.plot(steps, _rolling_mean(diff), color=INK, linewidth=2.2, zorder=4)
    if note_top:
        _headroom(ax, headroom)
    _tint(ax, zero=0.0)


def _headroom(ax, fraction: float = 0.2) -> None:
    """Room above the data for the panel's top-left note, so it hides no point."""
    low, high = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(low, high * (high / low) ** fraction)
    else:
        ax.set_ylim(low, high + (high - low) * fraction)


def _tint(ax, *, zero: float) -> None:
    """Very light half-plane tints in the arm colors: below the no-difference line ON is lower, above it OFF is."""
    low, high = ax.get_ylim()
    ax.axhspan(low, zero, color=ON_COLOR, alpha=0.05, linewidth=0, zorder=0)
    ax.axhspan(zero, high, color=OFF_COLOR, alpha=0.05, linewidth=0, zorder=0)
    ax.set_ylim(low, high)


def _title(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel)


def _note(ax, text: str, *, top: bool = True) -> None:
    ax.text(
        0.015,
        0.97 if top else 0.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top" if top else "bottom",
        fontsize=9.3,
        color=INK,
        zorder=6,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )


def _mean_ci(stats: dict, fmt: str) -> str:
    text = "mean " + fmt.format(stats["mean_on_minus_off"])
    if stats["ci95"] is not None:
        low, high = stats["ci95"]
        text += f" [{fmt.format(low)}, {fmt.format(high)}] (95%)"
    return _minus(text)


def _lower(pct: float) -> str:
    return f"{abs(pct):.0f}% {'lower' if pct < 0 else 'higher'}"


def _signed(value: float, fmt: str) -> str:
    return "0" if abs(value) < 1e-12 else _minus(fmt.format(value))


# ---------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--off", required=True, help="the OFF arm's log files, comma-separated, oldest first")
    parser.add_argument("--on", required=True, help="the ON arm's log files, comma-separated, oldest first")
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--subtitle", default="", help="leads the subtitle, e.g. the model name")
    parser.add_argument("--out-dir", type=Path, default=Path("ab-results"))
    parser.add_argument("--stats", type=Path, help="also write the panels' numbers to this JSON file")
    args = parser.parse_args(argv)

    off_logs, on_logs = args.off.split(","), args.on.split(",")
    off = load_arm(off_logs, source="local", project=None, entity=None).steps
    on = load_arm(on_logs, source="local", project=None, entity=None).steps
    need = max(2 * BOOTSTRAP_BLOCK, LATE)
    counts = {key: paired(off, on, key)[0].size for key in (LOGPROB_GAP, KL, TIS_CLIPFRAC, REWARD, RESPONSE_LEN)}
    if min(counts.values()) < need:
        print(f"need {need} paired steps of each metric; have {counts}", file=sys.stderr)
        return 1
    stats = {"off": off_logs, "on": on_logs, **analyze(off, on)}
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n")
        paths = [args.stats]
    else:
        paths = []
    paths = [plot(off, on, stats, args.out_dir / FIGURE_NAME, title=args.title, subtitle=args.subtitle)] + paths
    print(json.dumps(stats, indent=2))
    print("wrote " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
