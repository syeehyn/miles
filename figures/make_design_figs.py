"""Design figures for routing replay forcing (MoE expert routes only) in Miles and SGLang.

Usage (any Python with matplotlib):
    python make_design_figs.py [--out-dir DIR]

Writes a 2x PNG on a white background for each figure (default: the directory of this script):
  fig-1-what-happens   one session across a weight sync, without and with forcing
  fig-2-how-it-works   the system: Miles session server v1, router, SGLang engine, held-routes store

Drawn in pixel coordinates (1 data unit = 1 px at 100 dpi); the PNG is saved at 200 dpi. Multi-line text is drawn
one line at a time at a fixed pitch, because matplotlib sizes each line by its own glyphs. Every box records any
text that leaves it, and the script prints those as OVERFLOW lines. The figures contain no experiment data.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent

WHITE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_FILL = "#f0efec"
PANEL_FILL = "#f7f6f3"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_WASH = "#e3eefb"
GOOD = "#0c8a0c"
CRITICAL = "#d03b3b"
CRITICAL_WASH = "#fbe4e4"

# DejaVu Sans metrics, in ems: the ascent of "l", the descent of "p", and the line pitch used for multi-line text.
ASCENT = 0.76
DESCENT = 0.21
PITCH = 1.36
_OVERFLOWS: list[str] = []


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "hatch.linewidth": 1.4,
        }
    )


def canvas(w: int, h: int):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def crop(fig, ax, w: int, top: float, bottom: float) -> None:
    """Crop the canvas to [top, bottom], keeping 1 data unit = 1 px."""
    fig.set_size_inches(w / 100, (bottom - top) / 100)
    ax.set_ylim(bottom, top)


def save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def em(size: float) -> float:
    """One em of a font size in points, in px (data units)."""
    return size * 100 / 72


def txt(ax, x, y, s, size=10.5, color=INK, weight="normal", ha="left", va="center", **kw):
    """Draw text. A multi-line string is drawn line by line at a fixed pitch and returns a list of Text objects."""
    lines = s.split("\n")
    if len(lines) == 1:
        return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)
    e, pitch = em(size), PITCH * em(size)
    if va == "top":
        base0 = y + ASCENT * e
    elif va == "center":
        # A line's visual center sits (ASCENT - DESCENT) / 2 ems above its baseline.
        base0 = y + (ASCENT - DESCENT) / 2 * e - (len(lines) - 1) * pitch / 2
    else:
        raise ValueError(f"va={va!r}")
    return [
        ax.text(x, base0 + i * pitch, line, fontsize=size, color=color, fontweight=weight, ha=ha, va="baseline", **kw)
        for i, line in enumerate(lines)
        if line
    ]


def block_height(s: str, size: float) -> float:
    """Height in px of the ink block that ``txt(..., va="top")`` draws for ``s``."""
    return (ASCENT + DESCENT) * em(size) + (s.count("\n")) * PITCH * em(size)


def text_width(fig, t) -> float:
    return t.get_window_extent(renderer=fig.canvas.get_renderer()).width


def _flat(texts) -> list:
    out = []
    for t in texts:
        out.extend(t if isinstance(t, list) else [t])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Figure 1: what happens to one session across a weight sync


SEGS = [
    ("prompt", 118),
    ("out₁", 98),
    ("tool₁", 98),
    ("out₂\nbefore sync", 124),
    ("out₂\nafter sync", 124),
    ("tool₂", 98),
    ("out₃", 98),
]
SEG_X0 = 246
FLOW_W = 1020
LABEL = 12.5  # row labels
BAR = 11.0  # labels inside bars
NOTE = 11.0  # notes under a row


def seg_edges() -> list[float]:
    xs = [SEG_X0]
    for _, w in SEGS:
        xs.append(xs[-1] + w)
    return xs


def span_bar(ax, a: int, b: int, y: float, color: str | None, *, hatch=False, outline=False, label=None, h=28):
    e = seg_edges()
    x, w = e[a] + 2, e[b + 1] - e[a] - 4
    if outline:
        ax.add_patch(
            Rectangle(
                (x, y - h / 2), w, h, facecolor="none", edgecolor=MUTED, linewidth=1.2, linestyle=(0, (3, 2)), zorder=3
            )
        )
    else:
        ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=color, edgecolor="none", zorder=3))
        if hatch:
            ax.add_patch(
                Rectangle((x, y - h / 2), w, h, facecolor="none", edgecolor=WHITE, linewidth=0, hatch="///", zorder=4)
            )
    if label:
        lc = INK2 if outline else WHITE
        bbox = dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor="none") if hatch else None
        txt(ax, x + w / 2, y, label, size=BAR, color=lc, ha="center", zorder=5, bbox=bbox)


def stripes(ax, a: int, b: int, y: float, colors: list[str], h=28):
    e = seg_edges()
    x, w = e[a] + 2, e[b + 1] - e[a] - 4
    sh = (h - 3 * (len(colors) - 1)) / len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((x, y - h / 2 + i * (sh + 3)), w, sh, facecolor=c, edgecolor="none", zorder=3))


def verdict(ax, seg: int, y: float, ok: bool):
    e = seg_edges()
    cx = (e[seg] + e[seg + 1]) / 2
    txt(ax, cx, y, "✓" if ok else "✗", size=17, color=GOOD if ok else CRITICAL, weight="bold", ha="center")


def sync_line(ax, y: float):
    e = seg_edges()
    ax.plot([e[0] - 6, e[-1] + 6], [y, y], color=INK2, linewidth=1.3, linestyle=(0, (5, 3)), zorder=2)
    txt(ax, 20, y, "Weight sync w0 → w1", size=LABEL - 0.5, color=INK2, weight="bold")
    txt(
        ax,
        (e[0] + e[-1]) / 2,
        y,
        "retract turn 2 · flush the cache",
        size=NOTE,
        color=INK2,
        ha="center",
        zorder=4,
        bbox=dict(boxstyle="square,pad=0.3", facecolor=WHITE, edgecolor="none"),
    )


def note(ax, y_first: float, lines: list[str], color: str) -> None:
    """A note under a row: an arrow, then lines at a fixed pitch. White behind the text hides the column grid."""
    bbox = dict(boxstyle="square,pad=0.15", facecolor=WHITE, edgecolor="none")
    txt(ax, SEG_X0 + 4, y_first, "↳", size=NOTE, color=color, zorder=4, bbox=bbox)
    for i, s in enumerate(lines):
        txt(ax, SEG_X0 + 22, y_first + i * PITCH * em(NOTE), s, size=NOTE, color=color, zorder=4, bbox=bbox)


def draw_what_happens(out_dir: Path) -> Path:
    """One session across one weight sync. Turn 2 is in flight when the sync arrives: the retract drops its KV, the
    flush empties the cache, and its re-prefill recomputes every earlier token under w1. Turn 3 re-prefills its
    history (radix cache off, or a cache miss); with a cache hit it would reuse that re-prefill's KV, which carries
    the same routes, so the outcome is the same."""
    W = FLOW_W
    fig, ax = canvas(W, 1600)
    e = seg_edges()
    y_head = 34
    txt(ax, 20, y_head, "Tokens of one session →", size=LABEL - 1, color=INK2, weight="bold")
    for (name, _), x0, x1 in zip(SEGS, e[:-1], e[1:], strict=True):
        txt(ax, (x0 + x1) / 2, y_head, name, size=11.5, color=INK, ha="center")

    note_pitch = PITCH * em(NOTE)

    def lane(y0: float, forcing: bool) -> float:
        ax.plot([10, W - 10], [y0 - 26, y0 - 26], color=AXIS, linewidth=0.9)
        if forcing:
            txt(ax, 20, y0 + 4, "With routing replay forcing", size=15, weight="bold", color=GOOD)
        else:
            txt(ax, 20, y0 + 4, "Without forcing (today)", size=15, weight="bold", color=CRITICAL)
        y = y0 + 54
        txt(ax, 20, y, "Turn 1 · w0", size=LABEL)
        span_bar(ax, 0, 0, y, BLUE, label="prefill")
        span_bar(ax, 1, 1, y, BLUE, label="decode")
        y += 50
        txt(ax, 20, y, "Turn 2 · w0", size=LABEL)
        span_bar(ax, 0, 1, y, None, outline=True, label="cached")
        span_bar(ax, 2, 2, y, BLUE, label="prefill")
        span_bar(ax, 3, 3, y, BLUE, label="decode")
        y += 48
        sync_line(ax, y)
        y += 48
        txt(ax, 20, y, "Turn 2 resumes · w1", size=LABEL)
        if forcing:
            span_bar(ax, 0, 3, y, BLUE, hatch=True, label="re-prefill: held routes r0")
        else:
            span_bar(ax, 0, 3, y, ORANGE, label="re-prefill: new routes r1")
        span_bar(ax, 4, 4, y, ORANGE, label="decode")
        y += 32
        if forcing:
            lines = [
                "the retract keeps turn 2's held and computed routes; the re-prefill reuses them;",
                "only gate weights and KV values are recomputed under w1",
            ]
        else:
            lines = [
                "turn 2's reply reports r1 for every token, even out₂ before the sync,",
                "which was sampled with r0 (radixark/miles#3277); under retract Miles",
                "keeps the last turn's routes for all tokens",
            ]
        note(ax, y, lines, INK2 if forcing else CRITICAL)
        y += (len(lines) - 1) * note_pitch + 46
        txt(ax, 20, y - 9, "Turn 3 · w1", size=LABEL)
        txt(ax, 20, y + 12, "(radix cache off or a miss)", size=10.5, color=INK2)
        if forcing:
            span_bar(ax, 0, 3, y, BLUE, hatch=True, label="re-prefill: held routes r0")
            span_bar(ax, 4, 4, y, ORANGE, hatch=True, label="held r1")
        else:
            span_bar(ax, 0, 4, y, ORANGE, label="re-prefill: new routes r1")
        span_bar(ax, 5, 5, y, ORANGE, label="prefill")
        span_bar(ax, 6, 6, y, ORANGE, label="decode")
        y += 40
        ax.plot([20, W - 20], [y, y], color=GRID, linewidth=1)
        y += 38
        txt(ax, 20, y, "Trainer replays (R3)", size=LABEL, weight="bold")
        if forcing:
            span_bar(ax, 0, 3, y, BLUE)
            span_bar(ax, 4, 6, y, ORANGE)
        else:
            span_bar(ax, 0, 6, y, ORANGE)
        y += 50
        txt(ax, 20, y, "Sampling used", size=LABEL, weight="bold")
        if forcing:
            span_bar(ax, 0, 3, y, BLUE)
        else:
            stripes(ax, 0, 3, y, [BLUE, ORANGE])
        span_bar(ax, 4, 6, y, ORANGE)
        y += 50
        txt(ax, 20, y, "Same routes?", size=LABEL, weight="bold")
        for s in range(7):
            verdict(ax, s, y, ok=forcing or s >= 4)
        return y + 42

    y_end = lane(104, forcing=False)
    y_end = lane(y_end + 34, forcing=True)
    for x in e:
        ax.plot([x, x], [y_head + 24, y_end - 18], color=GRID, linewidth=0.8, zorder=1)

    # Legend, placed by measured text widths and wrapped at the canvas edge.
    ax.plot([10, W - 10], [y_end - 6, y_end - 6], color=AXIS, linewidth=0.9)
    items = [
        ("swatch", BLUE, "routes r0, computed under weights w0"),
        ("swatch", ORANGE, "routes r1, computed under weights w1"),
        ("hatch", BLUE, "held routes reused (only the expert ids)"),
        ("outline", None, "cached KV: nothing recomputed"),
        ("stripes", None, "sampling used both: tokens sampled under w0 and under w1 depend on these"),
        ("ok", None, "the trainer replays the routes the sampling used"),
        ("bad", None, "the trainer replays r1 here, but tokens sampled under w0 used r0"),
        ("text", None, "outₖ: tokens sampled in turn k · toolₖ: tool-result tokens"),
        (
            "text",
            None,
            "a re-prefill under the same weights can still flip near-tied experts; forcing removes that too",
        ),
    ]
    x, ly = 20, y_end + 28
    for kind, color, label in items:
        dx = 0 if kind == "text" else 40
        t = txt(ax, x + dx, ly, label, size=11.5, color=INK2 if kind == "text" else INK)
        width = dx + text_width(fig, t) + 30
        if x + width > W - 10 and x > 20:
            t.remove()
            x, ly = 20, ly + 32
            t = txt(ax, x + dx, ly, label, size=11.5, color=INK2 if kind == "text" else INK)
        if x + width - 30 > W - 10:
            _OVERFLOWS.append(f"figure 1 legend item {label[:30]!r} runs past the canvas")
        if kind == "outline":
            ax.add_patch(
                Rectangle(
                    (x, ly - 10), 30, 20, facecolor="none", edgecolor=MUTED, linewidth=1.2, linestyle=(0, (3, 2))
                )
            )
        elif kind == "stripes":
            ax.add_patch(Rectangle((x, ly - 10), 30, 8.5, facecolor=BLUE, edgecolor="none"))
            ax.add_patch(Rectangle((x, ly + 1.5), 30, 8.5, facecolor=ORANGE, edgecolor="none"))
        elif kind in ("swatch", "hatch"):
            ax.add_patch(Rectangle((x, ly - 10), 30, 20, facecolor=color, edgecolor="none"))
            if kind == "hatch":
                ax.add_patch(
                    Rectangle((x, ly - 10), 30, 20, facecolor="none", edgecolor=WHITE, linewidth=0, hatch="///")
                )
        elif kind in ("ok", "bad"):
            txt(
                ax,
                x + 15,
                ly,
                "✓" if kind == "ok" else "✗",
                size=16,
                weight="bold",
                ha="center",
                color=GOOD if kind == "ok" else CRITICAL,
            )
        x += width
    _check_inside(fig, ax, "figure 1", 0, 0, W, ly + 26)
    crop(fig, ax, W, 0, ly + 26)
    return save(fig, out_dir, "fig-1-what-happens")


def _check_inside(fig, ax, name: str, x0: float, y0: float, x1: float, y1: float) -> None:
    """Record any text that leaves the canvas."""
    r = fig.canvas.get_renderer()
    for t in ax.texts:
        bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        bx0, bx1 = sorted((bb.x0, bb.x1))
        by0, by1 = sorted((bb.y0, bb.y1))
        if bx0 < x0 + 2 or bx1 > x1 - 2 or by0 < y0 or by1 > y1:
            _OVERFLOWS.append(f"{name}: text {t.get_text()[:30]!r} leaves the canvas")


# ----------------------------------------------------------------------------------------------------------------
# Figure 2: how it works


FIG2_W = 1120
BODY = 10.0  # body text in boxes
SMALL = 10.0  # arrow labels and tags
TITLE_ROW = 33  # px from a box's top to its body's first line


def box_height(body: str, body_size: float = BODY) -> float:
    return TITLE_ROW + block_height(body, body_size) + 11


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body=None,
    *,
    fill=NEUTRAL_FILL,
    edge=AXIS,
    title_size=11,
    body_size=BODY,
    lw=1.2,
    title_color=INK,
    z=2.0,
    tag=None,
    inline=None,
):
    """A rounded box with a bold title and a top-aligned body (or an ``inline`` body on the title row). ``h=None``
    sizes the box to its body. Records any text that leaves the box; returns the box's bottom."""
    if h is None:
        h = box_height(body, body_size)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=7",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=z,
        )
    )
    texts = [txt(ax, x + 12, y + 17, title, size=title_size, weight="bold", color=title_color, zorder=z + 3)]
    if body:
        texts.append(txt(ax, x + 12, y + TITLE_ROW, body, size=body_size, color=INK, va="top", zorder=z + 3))
    if inline:
        tx = x + 12 + text_width(ax.figure, texts[0]) + 14
        texts.append(txt(ax, tx, y + 17.5, inline, size=body_size, color=INK, zorder=z + 3))
    if tag:
        # A small "fail closed" pill at the title row's right end.
        texts.append(
            txt(
                ax,
                x + w - 12,
                y + 17,
                tag,
                size=SMALL,
                color=CRITICAL,
                weight="bold",
                ha="right",
                zorder=z + 4,
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.6",
                    facecolor=CRITICAL_WASH,
                    edgecolor=CRITICAL,
                    linewidth=0.9,
                ),
            )
        )
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t in _flat(texts):
        bb = t.get_window_extent(renderer=r).transformed(inv)
        x0, x1 = sorted((bb.x0, bb.x1))
        y0, y1 = sorted((bb.y0, bb.y1))
        if x0 < x + 4 or x1 > x + w - 4 or y0 < y + 2 or y1 > y + h - 2:
            _OVERFLOWS.append(
                f"{title!r}: text {t.get_text()[:30]!r} spans x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}; "
                f"box x {x}-{x + w}, y {y:.0f}-{y + h:.0f}"
            )
    if tag:
        tb = texts[-1].get_window_extent(renderer=r).transformed(inv)
        hb = texts[0].get_window_extent(renderer=r).transformed(inv)
        if min(tb.x0, tb.x1) < max(hb.x0, hb.x1) + 6:
            _OVERFLOWS.append(f"{title!r}: tag overlaps the title")
    return y + h


def arrow(ax, p0, p1, *, color=INK2, lw=1.5, rad=0.0, z=8, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=f"{style},head_length=8,head_width=4.5",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=lw,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def polyline_arrow(ax, pts, *, color=INK2, lw=1.5):
    xs, ys = zip(*pts[:-1], strict=True)
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=8, solid_joinstyle="miter", solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def label(ax, x, y, s, color=INK2, ha="left", size=SMALL, **kw):
    return txt(ax, x, y, s, size=size, color=color, ha=ha, zorder=9, **kw)


EACH_TURN = (
    "routed_experts_start_len = F:\n"
    "   rows of routes the session holds\n"
    "routed_experts_session_key\n"
    "   = the session id  (new)\n"
    "n = session length after this turn"
)
CHECK_REPLY = (
    "refused, or unforced_tokens\n"
    "missing or > 0?\n"
    "no: keep only this turn's new rows\n"
    "[F, n-1), also under retract (today\n"
    "Miles keeps the last turn's rows)"
)
REFUSED = (
    "HTTP 503 for this and later turns;\n"
    "the session yields no sample; its\n"
    "group is dropped like any aborted\n"
    "group. Counted in\n"
    "rollout/r3/forcing_refused_turns"
)
ROUTER = (
    "keeps a session\n"
    "on one engine\n"
    "(policy manual or\n"
    "consistent_hashing);\n"
    "forwards the\n"
    "session key (new)"
)
ADMIT = (
    "before any compute: are rows [0, F)\n"
    "held? No: refuse with\n"
    "routed_experts_missing. Yes: the turn\n"
    "gets a per-session radix namespace"
)
FORWARD = (
    "held positions get the held expert\n"
    "ids; gate weights use the current\n"
    "router logits"
)
RETRACT = (
    "at a weight sync or KV shortage,\n"
    "before KV is freed, the turn keeps\n"
    "its routes; the re-prefill reuses them"
)
COMMIT = (
    "unless the turn aborted: store\n"
    "[0, F) + this turn's routes; an\n"
    "untrusted commit drops the key:\n"
    "its next turn is refused"
)
REPLY = (
    "routes [F, n-1), as R3 today,\n"
    "plus counters in\n"
    "meta_info.routed_experts_forcing"
)
STORE = (
    "one per TP rank,\n"
    "host memory\n"
    "\n"
    "per session key:\n"
    "the expert ids first\n"
    "computed for each\n"
    "token (MoE layers)\n"
    "\n"
    "LRU, token budget"
)


def draw_how_it_works(out_dir: Path) -> Path:
    """The system. Columns: Miles (client) | router | SGLang engine (server); the trainer spans the bottom.

    Engine steps follow the code's order: admission in Scheduler.handle_generate_request before the request is
    queued; forcing in select_experts (after any routing override, before capture and the EPLB remap); the retract
    snapshot in release_req before its KV is freed; the commit in _maybe_collect_routed_experts for a request that
    did not abort; the counters in meta_info["routed_experts_forcing"].
    """
    W = FIG2_W
    fig, ax = canvas(W, 1200)
    FC = "fail closed"
    GAP = 12
    top = 136  # first row of boxes inside the panels

    # Columns.
    mx, mw = 16, 316  # Miles panel
    ix, iw = mx + 14, mw - 28  # boxes inside it
    rx, rw = 346, 164  # router
    ex0 = 528  # engine panel
    ew0 = W - 16 - ex0
    sx, sw = ex0 + 34, 300  # engine steps; the left margin holds the retract arrow
    hx = sx + sw + 44  # held-routes store
    hw = ex0 + ew0 - 28 - hx

    # Rows, from the text each box holds.
    eng = [
        ("1  Admit", ADMIT, FC),
        ("2  Forward: select_experts", FORWARD, None),
        ("3  Retract", RETRACT, None),
        ("4  Commit at finish", COMMIT, FC),
        ("5  Reply", REPLY, None),
    ]
    ey, y = [], top
    for _, body, _ in eng:
        ey.append((y, box_height(body)))
        y += box_height(body) + GAP
    engine_bottom = ey[-1][0] + ey[-1][1] + 16
    et_bottom = top + box_height(EACH_TURN)
    cr_y = et_bottom + 34
    cr_bottom = cr_y + box_height(CHECK_REPLY)
    col_y = cr_bottom + 32
    miles_bottom = col_y + 36 + 18
    ry = engine_bottom - box_height(REFUSED)  # the refused box ends level with the engine panel
    ty = engine_bottom + 66  # trainer

    # Miles column ------------------------------------------------------------------------------------------
    box(ax, mx, 24, mw, 36, "Agent", None, fill=WHITE, inline="multi-turn tool loop, chat API")
    box(ax, mx, 100, mw, miles_bottom - 100, "Miles session server v1", None, fill=PANEL_FILL, title_size=12)
    box(ax, ix, top, iw, None, "Each turn", EACH_TURN, fill=BLUE_WASH, edge=BLUE)
    box(ax, ix, cr_y, iw, None, "Check the reply", CHECK_REPLY, fill=BLUE_WASH, edge=BLUE, tag=FC)
    cx0 = ix + 64
    box(ax, cx0, col_y, ix + iw - cx0, 36, "Collect", None, fill=NEUTRAL_FILL, inline="merges the turns")
    arrow(ax, (mx + mw - 56, 60), (mx + mw - 56, top), style="<|-|>")
    box(ax, mx, ry, mw, None, "Refused turn", REFUSED, fill=CRITICAL_WASH, edge=CRITICAL, title_color=CRITICAL)
    yes_x = ix + 26
    arrow(ax, (yes_x, cr_bottom), (yes_x, ry), color=CRITICAL)
    label(ax, yes_x + 8, cr_bottom + 16, "yes", color=CRITICAL)
    no_x = cx0 + 130
    arrow(ax, (no_x, cr_bottom), (no_x, col_y), color=GOOD)
    label(ax, no_x + 8, cr_bottom + 16, "no", color=GOOD)

    # Router ------------------------------------------------------------------------------------------------
    box(ax, rx, top, rw, None, "SGLang router", ROUTER, fill=BLUE_WASH, edge=BLUE)

    # Engine column -----------------------------------------------------------------------------------------
    box(ax, ex0, 100, ew0, engine_bottom - 100, "SGLang engine  (sglang-miles)", None, fill=PANEL_FILL, title_size=12)
    for (title, body, tag), (y, _) in zip(eng, ey, strict=True):
        box(
            ax,
            sx,
            y,
            sw,
            None,
            title,
            body,
            fill=NEUTRAL_FILL if title.startswith("5") else BLUE_WASH,
            edge=AXIS if title.startswith("5") else BLUE,
            tag=tag,
        )
    # Retract -> forward: the re-prefill is forced to the snapshot. The bulge (rad * span / 2) stays in the margin.
    (y2, h2), (y3, h3) = ey[1], ey[2]
    arrow(ax, (sx, y3 + h3 / 2), (sx, y2 + h2 / 2), color=BLUE, lw=1.4, rad=-0.28)

    # Held-routes store, one per TP rank (stacked copies behind).
    hy = ey[0][0] + 50
    write_y = ey[3][0] + 44  # the commit box's title row
    hh = max(box_height(STORE), write_y + 34 - hy)
    for k in (2, 1):
        ax.add_patch(
            FancyBboxPatch(
                (hx + 7 * k, hy - 7 * k),
                hw,
                hh,
                boxstyle="round,pad=0,rounding_size=7",
                facecolor=BLUE_WASH,
                edgecolor=BLUE,
                linewidth=1.0,
                zorder=2.1 + 0.1 * (2 - k),
            )
        )
    box(ax, hx, hy, hw, hh, "Held-routes store", STORE, fill=BLUE_WASH, edge=BLUE, z=2.5)
    read_y = hy + 30
    arrow(ax, (hx, read_y), (sx + sw, read_y), color=BLUE)
    label(ax, (sx + sw + hx) / 2, read_y - 13, "read", color=BLUE, ha="center")
    arrow(ax, (sx + sw, write_y), (hx, write_y), color=BLUE)
    label(ax, (sx + sw + hx) / 2, write_y - 13, "write", color=BLUE, ha="center")

    # Request: each turn -> router -> admit.
    req_y = top + 60
    arrow(ax, (ix + iw, req_y), (rx, req_y))
    arrow(ax, (rx + rw, req_y), (sx, req_y))
    # Reply: engine -> check the reply.
    gx = rx + rw / 2
    reply_y = ey[4][0] + 50
    check_y = cr_y + box_height(CHECK_REPLY) / 2
    polyline_arrow(ax, [(sx, reply_y), (gx, reply_y), (gx, check_y), (ix + iw, check_y)])
    label(ax, gx + 8, (reply_y + check_y) / 2, "reply")

    # Trainer and weight sync --------------------------------------------------------------------------------
    box(
        ax,
        mx,
        ty,
        W - 32,
        36,
        "Trainer  (unchanged)",
        None,
        fill=NEUTRAL_FILL,
        inline="R3 replays the sample's routes: every token is trained with the routes that sampled it",
    )
    ax_x = mx + mw + 10
    polyline_arrow(ax, [(ix + iw, col_y + 18), (ax_x, col_y + 18), (ax_x, ty)], color=GOOD)
    label(ax, ax_x - 8, (engine_bottom + ty) / 2, "accepted: one sample with exact routes", color=GOOD, ha="right")
    wx = ex0 + ew0 - 90
    arrow(ax, (wx, ty), (wx, engine_bottom), color=INK)
    label(
        ax,
        wx - 10,
        (engine_bottom + ty) / 2,
        "weight sync: pause (retract) → flush cache → update → resume\n"
        "an engine's first sync: no routed_experts_forcing in /server_info → the run stops",
        color=INK,
        ha="right",
    )

    # Legend ------------------------------------------------------------------------------------------------
    ly = ty + 68
    items = [
        (BLUE_WASH, BLUE, "new (off by default)"),
        (NEUTRAL_FILL, AXIS, "existing R3 path"),
        (CRITICAL_WASH, CRITICAL, "fail closed: refuse the turn instead of guessing routes"),
    ]
    x = 20
    for fill, edge, text in items:
        ax.add_patch(
            FancyBboxPatch(
                (x, ly - 9),
                28,
                18,
                boxstyle="round,pad=0,rounding_size=3",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.1,
            )
        )
        t = txt(ax, x + 36, ly, text, size=10.5)
        x += 36 + text_width(fig, t) + 28
    if x - 28 > W:
        _OVERFLOWS.append(f"legend runs to x={x:.0f} > {W}")
    _check_inside(fig, ax, "figure 2", 0, 6, W, ly + 22)
    crop(fig, ax, W, 6, ly + 22)
    return save(fig, out_dir, "fig-2-how-it-works")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    paths = [draw_what_happens(args.out_dir), draw_how_it_works(args.out_dir)]
    for o in _OVERFLOWS:
        print("OVERFLOW", o)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
