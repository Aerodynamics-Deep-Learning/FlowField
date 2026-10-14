"""
Manual check of the geometry manifest: one matplotlib window showing an airfoil's source points
(undigested) over its stored CST shape (digested), on one x/c, y/c plot, so a person can eyeball what
the ingest did to it. Same convention as the meshing viewer: opt in with `pytest -m interactive`.

Keys:
    1-9, 0   airfoil 1-10 of `_SELECTION`
    n        source points as read / moved into the digested frame (normalized, not closed)
    z        whole airfoil / LE close-up / TE close-up
The toolbar zoom and pan work as usual.

Reads the repo's own manifest under `data/`; `test_viewer_selection_is_in_the_manifest` keeps the
selection in step with it in every batch run.
"""

from typing import NamedTuple

import numpy as np
import pytest

from src.datagen.geometry.common.io import Geom_read_dat
from src.datagen.geometry.build.normalize import Geom_normalize
from src.datagen.geometry.build.parametrization import Geom_cst_contour
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.manifest.store import GeometryStore

_SELECTION = [
    "naca4412",    # 4-digit family, built from its formula; its nose point is not the geometric LE
    "naca001066",  # modified 4-digit, formula against a sparse 33-point file
    "s4096",       # the largest gap pried closed (6.2%c), slanted TE base
    "sc21010",     # source tilted 2.6 deg
    "n24",         # source LE ~0.05c off (0, 0)
    "fx69274",     # slanted TE base, stretched to x = 1
    "goe09k",      # the sparsest file, 29 points
    "s1221",       # high-lift section
    "e387",        # a plain closed tabulated airfoil, the control case
    "goe217",      # the worst accepted fit, 9.7e-4c
]

# Per view: the x window and the y it is centred on; the y window follows from the plot's shape
_VIEWS = [("whole", (-0.05, 1.05), 0.0),
          ("LE close-up", (-0.06, 0.14), 0.0),
          ("TE close-up", (0.78, 1.04), 0.0)]


class _Airfoil(NamedTuple):
    name: str
    raw: np.ndarray
    raw_normalized: np.ndarray
    digested: np.ndarray
    caption: str


def _load(names) -> list[_Airfoil]:
    """Each airfoil's source points (as read and normalized) and its stored shape, densely rendered."""
    paths = Geom_paths()
    if not paths.db.is_file():
        pytest.skip(f"no geometry manifest at {paths.db}")
    airfoils = []
    with GeometryStore.open(paths.db, read_only=True) as store:
        missing = [n for n in names if store.get_geometry(n) is None]
        assert not missing, f"not in the manifest: {missing}; edit _SELECTION"
        for name in names:
            geo = store.get_geometry(name)
            shape = store.get_shape(geo.geometry_id, SHAPE_VERSION)
            raw = Geom_read_dat(paths.from_rel(geo.raw_path)).coords
            raw_normalized, tf = Geom_normalize(raw)
            family = f"{geo.family.value} family" if geo.family else "tabulated"
            caption = (f"{geo.name}: {family}, closure {shape.closure.value}, source gap {shape.root_te_gap:.2%}c, "
                       f"fit max {shape.fit_max:.1e}c\nundigested: {len(raw)} points from {geo.raw_path}, "
                       f"normalizing rotates {tf.rotation_deg:+.2f} deg; digested: CST order {shape.cst.order} "
                       f"({shape.shape_version})")
            airfoils.append(_Airfoil(geo.name, raw, raw_normalized, Geom_cst_contour(shape.cst, 2001), caption))
    return airfoils


def test_viewer_selection_is_in_the_manifest():
    airfoils = _load(_SELECTION)

    assert len(airfoils) == 10
    assert all(a.digested[0].tolist() == [1.0, 0.0] for a in airfoils)


@pytest.mark.interactive
def test_geometry_viewer():
    import matplotlib.pyplot as plt

    airfoils = _load(_SELECTION)
    state = {"index": 0, "normalized": False, "view": 0}
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)
    fig.text(0.01, 0.015, "keys  1-9, 0: airfoil   n: undigested as read / normalized   z: whole / LE / TE",
             fontsize=9, color="0.35")

    def draw():
        a = airfoils[state["index"]]
        view, xlim, yc = _VIEWS[state["view"]]
        pos, (w, h) = ax.get_position(), fig.get_size_inches()
        half_y = (xlim[1] - xlim[0]) * (pos.height * h) / (pos.width * w) / 2  # equal aspect, exactly
        raw = a.raw_normalized if state["normalized"] else a.raw
        ax.clear()
        ax.plot(a.digested[:, 0], a.digested[:, 1], "-", color="C0", lw=1.4, label="digested: stored CST")
        ax.plot(raw[:, 0], raw[:, 1], "o-", color="C3", lw=0.6, ms=3.5, mfc="none",
                label="undigested: " + ("normalized, not closed" if state["normalized"] else "as read"))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(*xlim)
        ax.set_ylim(yc - half_y, yc + half_y)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        key = (state["index"] + 1) % 10
        ax.set_title(f"[{key}] {a.caption}\n{view}", fontsize=10, loc="left")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key is not None and len(event.key) == 1 and event.key.isdigit():
            state["index"] = (int(event.key) - 1) % 10
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
