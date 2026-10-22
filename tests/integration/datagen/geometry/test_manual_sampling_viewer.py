"""
Manual check of Sequence 3: one matplotlib window showing, on one x/c, y/c plot, an airfoil's points as
read from its .dat, its processed shape (the stored CST: normalized, closed, fitted), that shape with the
TE its mesher needs (handled), and the points sampled from it. Same convention as the geometry viewer:
opt in with `pytest -m interactive`.

Each `_CASES` entry is a stored airfoil's name with its TE handling ("te", and "gap" for a blunt or
round one) and any sampling knobs (`SamplingFields`' names); what is left out takes the studies'
defaults (the knobs' defaults, the gap TE_GAP_DEFAULT).

Keys:
    1-9   case 1-9 of `_CASES`
    n     .dat points as read / moved into the processed frame (normalized, not closed)
    z     whole airfoil / LE close-up / TE close-up / TE cap close-up
The toolbar zoom and pan work as usual.

Reads the repo's own manifest under `data/`; `test_viewer_cases_sample` keeps `_CASES` in step with it,
and with the sampler, in every batch run.
"""

from typing import NamedTuple

import numpy as np
import pytest

from src.datagen.geometry.build.normalize import Geom_normalize
from src.datagen.geometry.build.parametrization import Geom_cst_contour
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.io import Geom_read_dat
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.geometry.sample.contour import Geom_build_contour
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.optimizer.sampling import TE_GAP_DEFAULT

_CASES = {
    "naca4412": {"te": "round"},                                          # every default: ~600 points
    "s1223":    {"te": "round", "dx_mid": 0.004, "te_ratio": 0.1},        # the steepest TE stored; te clamped
    "mid012r":  {"te": "round", "le_ratio": 0.02, "cluster_width": 0.08},  # the sharpest nose stored
    "e387":     {"te": "sharp", "cluster_width": 0.1},                    # the c2d C-grid's TE: no cap
    "s4096":    {"te": "blunt", "gap": 0.006},                            # the gmsh C-mesh's TE: left open
}

# Per view: the x window and the y it is centred on, the y window following from the plot's shape; None
# fits the window to the case's TE pins (tip and welds, or the open ends)
_VIEWS = [("whole", (-0.05, 1.05), 0.0),
          ("LE close-up", (-0.01, 0.05), 0.0),
          ("TE close-up", (0.97, 1.006), 0.0),
          ("TE cap close-up", None, None)]


class _Case(NamedTuple):
    raw: np.ndarray
    raw_normalized: np.ndarray
    processed: np.ndarray
    handled: np.ndarray
    pins: np.ndarray
    sampled: np.ndarray  # empty if the config was rejected
    caption: str


def _load(cases: dict) -> list[_Case]:
    """Each case's four layers, from the manifest and the sampler."""
    paths = Geom_paths()
    if not paths.db.is_file():
        pytest.skip(f"no geometry manifest at {paths.db}")
    loaded = []
    with GeometryStore.open(paths.db, read_only=True) as store:
        for name, settings in cases.items():
            geo = store.get_geometry(name)
            shape = store.get_shape(geo.geometry_id, SHAPE_VERSION) if geo is not None else None
            assert shape is not None, f"{name} has no stored shape under {SHAPE_VERSION}; edit _CASES"
            knobs = dict(settings)
            te, gap = TETreatment(knobs.pop("te", "round")), knobs.pop("gap", TE_GAP_DEFAULT)
            fields = SamplingFields(**knobs)
            contour = Geom_build_contour(shape, te, gap)
            smp = Geom_sample(contour, fields)

            raw = Geom_read_dat(paths.from_rel(geo.raw_path)).coords
            r = smp.report
            handling = te.value + (f", gap {contour.gap:.2%}c" if contour.gap else "")
            te_text = f"{r.te_spacing:.1e}" + (" (clamped)" if r.te_clamped else "")
            if smp.accepted:
                outcome = (f"sampled: {r.n_points} points, panels {r.min_panel:.1e}..{r.max_panel:.1e}, "
                           f"max turn {r.max_turn_deg:.1f} deg, neighbour ratio {r.max_panel_ratio:.3f}")
            else:
                outcome = (f"REJECTED ({r.rejection.value}): curvature x spacing {r.precheck_turn:+.1%} over the "
                           f"limit at x {r.x_at_precheck_turn:.4f}; nothing sampled")
            caption = (f"{name}: TE {handling}; dx_mid {fields.dx_mid:.1e}, LE {r.le_spacing:.1e}, TE {te_text}, "
                       f"cluster width {fields.cluster_width}\n{outcome}; {len(raw)} .dat points")
            loaded.append(_Case(
                raw=raw, raw_normalized=Geom_normalize(raw)[0], processed=Geom_cst_contour(shape.cst, 2001),
                handled=contour.evaluate(np.union1d(contour.table_s, [p.s for p in contour.pins])),
                pins=np.array([p.xy for p in contour.pins]),
                sampled=smp.points if smp.accepted else np.empty((0, 2)), caption=caption))
    return loaded


def test_viewer_cases_sample():
    cases = _load(_CASES)

    assert len(cases) == len(_CASES) <= 9
    assert all(len(c.sampled) for c in cases), "a case's config was rejected; edit _CASES"


@pytest.mark.interactive
def test_sampling_viewer():
    import matplotlib.pyplot as plt

    cases = _load(_CASES)
    state = {"index": 0, "normalized": False, "view": 0}
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)
    fig.text(0.01, 0.015, "keys  1-9: case   n: .dat as read / normalized   z: whole / LE / TE / TE cap",
             fontsize=9, color="0.35")

    def te_window(c, shape):
        """An x window ending just past the TE and a y centre, holding the TE pins at the plot's aspect."""
        te = c.pins[c.pins[:, 0] > 0.5]
        margin = 0.25 * max(np.ptp(te[:, 0]), np.ptp(te[:, 1]), 4e-3)
        width = max(np.ptp(te[:, 0]) + 2 * margin, (np.ptp(te[:, 1]) + 2 * margin) / shape)
        right = te[:, 0].max() + 0.2 * margin
        return (right - width, right), (te[:, 1].min() + te[:, 1].max()) / 2

    def draw():
        c = cases[state["index"]]
        view, xlim, yc = _VIEWS[state["view"]]
        pos, (w, h) = ax.get_position(), fig.get_size_inches()
        shape = (pos.height * h) / (pos.width * w)
        if xlim is None:
            xlim, yc = te_window(c, shape)
        half_y = (xlim[1] - xlim[0]) * shape / 2  # equal aspect, exactly
        raw = c.raw_normalized if state["normalized"] else c.raw
        ax.clear()
        ax.plot(raw[:, 0], raw[:, 1], "o-", color="0.55", lw=0.6, ms=4, mfc="none",
                label=".dat: " + ("normalized, not closed" if state["normalized"] else "as read"))
        ax.plot(c.processed[:, 0], c.processed[:, 1], "--", color="C0", lw=1.2, label="processed: stored CST")
        ax.plot(c.handled[:, 0], c.handled[:, 1], "-", color="C2", lw=1.6, label="handled: with its TE")
        if len(c.sampled):
            ax.plot(c.sampled[:, 0], c.sampled[:, 1], ".-", color="C1", lw=0.5, ms=5,
                    label=f"sampled: {len(c.sampled)} points")
        ax.plot(c.pins[:, 0], c.pins[:, 1], "s", color="k", ms=5, mfc="none", label="pinned: tip/ends, welds, LE")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(*xlim)
        ax.set_ylim(yc - half_y, yc + half_y)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"[{state['index'] + 1}] {c.caption}\n{view}", fontsize=10, loc="left")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key is not None and len(event.key) == 1 and event.key in "123456789":
            if int(event.key) > len(cases):
                return
            state["index"] = int(event.key) - 1
        elif event.key == "n":
            state["normalized"] = not state["normalized"]
        elif event.key == "z":
            state["view"] = (state["view"] + 1) % len(_VIEWS)
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
