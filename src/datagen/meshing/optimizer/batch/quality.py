"""
The delivered meshes' quality against the study's gates: a table for the log, and a histogram per gate of each
mesh's worst value. Read from the run store: a re-mesh is delivered only once it hashes to its run, so the
run's measured quality is the delivered mesh's.

    Optim_remesh_quality   both, for a re-mesh's delivered meshes
    Optim_quality_table    the table
    Optim_quality_figure   the histograms, as a PNG
"""

import statistics
from pathlib import Path
from typing import NamedTuple, Sequence

from src.datagen.meshing.optimizer.manifest.store import RunStore
from src.datagen.meshing.optimizer.search.events import Optim_gate_value

# The figure's colours: one series on the light chart surface, the gate in secondary ink, recessive grid
_SURFACE, _BAR, _INK, _INK2, _GRID = "#fcfcfb", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3de"

FIGURE_NAME = "quality.png"


class _Gate(NamedTuple):
    key: str      # its name in failed_gates
    limit: str    # its limit's settings column
    above: bool   # whether above the limit fails
    title: str
    spec: str     # how its values print


GATES = (
    _Gate("skew", "gate_skew_max", True, "Worst skewness", ".4f"),
    _Gate("ortho", "gate_ortho_min", False, "Worst orthogonal quality", ".4f"),
    _Gate("jac", "gate_jac_min", False, "Worst scaled Jacobian", ".4f"),
    _Gate("aspect_ratio", "gate_ar_max", True, "Worst aspect ratio", ".3g"),
    _Gate("size_ratio", "gate_size_ratio_max", True, "Worst size ratio", ".4f"),
    _Gate("geo_dev", "gate_geo_dev_max", True, "Worst boundary deviation [c]", ".2e"),
)


def Optim_remesh_quality(store: RunStore, outcomes: Sequence, out_root) -> list[str]:
    """
    The quality of a re-mesh's delivered meshes, judged by the settings their runs ran under: the table's lines,
    then where the histograms went (`<out_root>/quality.png`).

    Args:
        store (RunStore): The run store the re-mesh read
        outcomes (Sequence[RemeshOutcome]): The re-mesh's outcomes; those not delivered are left out
        out_root (str | Path): The re-mesh's folder

    Returns:
        list[str]: The lines for the log; none if nothing was delivered
    """
    runs = [(o.name, store.run(o.remeshed.run_id)) for o in outcomes if o.remeshed is not None]
    if not runs:
        return []
    settings = store.settings(runs[0][1]["settings_id"])
    path = Optim_quality_figure(runs, settings, Path(out_root) / FIGURE_NAME)
    return [*Optim_quality_table(runs, settings), f"Histograms: {path}"]


def _values(runs: Sequence[tuple[str, dict]], gate: _Gate) -> list[tuple[float, str]]:
    return [(v, name) for name, row in runs if (v := Optim_gate_value(row, gate.key)) is not None]


def Optim_quality_table(runs: Sequence[tuple[str, dict]], settings: dict) -> list[str]:
    """
    Each gate's limit, and the median and worst of the meshes' worst values, with the worst's airfoil.

    Args:
        runs (Sequence[tuple[str, dict]]): (airfoil, run row) per mesh
        settings (dict): The settings row the runs ran under, for the limits

    Returns:
        list[str]: A header, then a line per gate
    """
    out = [f"Quality of the {len(runs)} delivered meshes, each mesh's worst cell:",
           f"{'gate':<13} {'limit':>10} {'median':>10} {'worst':>10}  airfoil"]
    for g in GATES:
        values = _values(runs, g)
        limit = f"{'<=' if g.above else '>='} {settings[g.limit]:g}"
        if not values:
            out.append(f"{g.key:<13} {limit:>10} {'-':>10} {'-':>10}")
            continue
        worst = (max if g.above else min)(values)
        median = statistics.median(v for v, _ in values)
        out.append(f"{g.key:<13} {limit:>10} {median:>10{g.spec}} {worst[0]:>10{g.spec}}  {worst[1]}")
    return out


def Optim_quality_figure(runs: Sequence[tuple[str, dict]], settings: dict, path) -> Path:
    """
    One histogram per gate of the meshes' worst values, the gate drawn where it falls inside the data's reach.

    Args:
        runs (Sequence[tuple[str, dict]]): (airfoil, run row) per mesh
        settings (dict): The settings row the runs ran under, for the limits
        path (str | Path): The PNG; its folder must exist

    Returns:
        Path: `path`
    """
    from matplotlib.figure import Figure  # here alone: nothing else in a re-mesh loads matplotlib
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(14, 8), dpi=110, facecolor=_SURFACE, layout="constrained")
    fig.suptitle(f"Quality of the {len(runs)} delivered meshes: each mesh's worst cell against the study's gate",
                 x=0.01, ha="left", color=_INK, fontsize=13)
    for ax, g in zip(fig.subplots(2, 3).flat, GATES):
        _panel(ax, g, _values(runs, g), settings[g.limit], MaxNLocator)
    path = Path(path)
    fig.savefig(path, facecolor=_SURFACE)
    return path


def _panel(ax, gate: _Gate, values: list[tuple[float, str]], limit: float, locator) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(gate.title, loc="left", color=_INK, fontsize=11, pad=32)
    sign = "≤" if gate.above else "≥"
    if not values:
        ax.text(0.5, 0.5, f"not measured; gate {sign} {limit:g}", transform=ax.transAxes, ha="center", color=_INK2)
        return
    data = [v for v, _ in values]
    ax.hist(data, bins=20, color=_BAR, edgecolor=_SURFACE, linewidth=1.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # headroom for the gate's label
    ax.yaxis.set_major_locator(locator(integer=True))
    ax.xaxis.set_major_locator(locator(5))
    ax.set_ylabel("meshes", color=_INK2, fontsize=9)
    lo, hi = min(data), max(data)
    span = (hi - lo) or abs(hi) * 0.01 or 1.0
    shown = max(lo - limit, limit - hi, 0.0) <= 3 * span  # near enough to draw without squashing the bars
    if shown:
        ax.axvline(limit, color=_INK2, linewidth=1.5)
        ax.annotate(f"gate {limit:g}", xy=(limit, 1), xycoords=("data", "axes fraction"), xytext=(-4 if gate.above
                    else 4, -4), textcoords="offset points", ha="right" if gate.above else "left", va="top",
                    color=_INK2, fontsize=9)
    worst = (max if gate.above else min)(values)
    median = statistics.median(data)
    ax.text(0, 1.02, f"median {median:{gate.spec}}, worst {worst[0]:{gate.spec}} ({worst[1]})\n"
            f"gate {sign} {limit:g}" + ("" if shown else ", off the axis"), transform=ax.transAxes, color=_INK2,
            fontsize=9, linespacing=1.4)
