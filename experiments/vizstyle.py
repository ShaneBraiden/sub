"""Shared look of the result figures (summary lengths, sentiment curves): the validated categorical
palette in its fixed order, a blue/red diverging pair with a gray midpoint, recessive hairline axes."""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# categorical slots, assigned in this order and never cycled
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# diverging poles (sentiment: positive / negative) and the neutral midpoint
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL = "#c3c2b7"


def style_axes(ax) -> None:
    """Hairline solid grid on the value axis, no top/right spines, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
