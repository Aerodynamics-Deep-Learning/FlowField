"""
Manual check of the geometry manifest: one matplotlib window showing an airfoil's source points
(undigested) over its stored CST shape (digested), on one x/c, y/c plot, so a person can eyeball what
the ingest did to it. Same convention as the meshing viewer: opt in with `pytest -m interactive`.

The airfoils come from `test_manual_geometry_viewer.yaml` beside this file (gitignored, this machine's
own), else the committed `test_manual_geometry_viewer.example.yaml`. A name is looked up among the
accepted airfoils first, then the rejected ones, so the selection can list either; a rejection stopped
before its fit has no digested shape.

Keys:
    1-9, 0   airfoil 1-10 of the selection
    n        source points as read / moved into the digested frame (normalized, not closed)
    z        whole airfoil / LE close-up / TE close-up
The toolbar zoom and pan work as usual.

Reads the repo's own manifest under `data/`; `test_viewer_selection_is_in_the_manifest` keeps the
committed selection in step with it in every batch run.

    python -m pytest tests/integration/datagen/geometry/test_manual_geometry_viewer.py -m interactive -n 0
"""

from typing import NamedTuple, Optional

import numpy as np
import pytest

from src.datagen.geometry.common.io import Geom_read_dat
from src.datagen.geometry.build.normalize import Geom_normalize
from src.datagen.geometry.build.parametrization import Geom_cst_contour
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.schemas import RejectionRecord
from src.datagen.geometry.manifest.store import GeometryStore
from tests.viewer_config import load_selection


def _selection(example: bool = False) -> list[str]:
    return [str(name) for name in load_selection(__file__, example=example)["airfoils"]]

# Per view: the x window and the y it is centred on; the y window follows from the plot's shape
_VIEWS = [("whole", (-0.05, 1.05), 0.0),
          ("LE close-up", (-0.06, 0.14), 0.0),
          ("TE close-up", (0.78, 1.04), 0.0)]


class _Airfoil(NamedTuple):
    name: str
    raw: np.ndarray
    raw_normalized: np.ndarray
    digested: Optional[np.ndarray]
    caption: str


def _lookup(store: GeometryStore, name: str):
    """(the airfoil's identity, its shape or rejection) under the current version, or None."""
    geo = store.get_geometry(name)
    if geo is None:
        rej = store.get_rejection_named(name, SHAPE_VERSION)
        return (rej, rej) if rej is not None else None
    shape = store.get_shape(geo.geometry_id, SHAPE_VERSION)  # None if this version rejected a stored airfoil
    return geo, shape if shape is not None else store.get_rejection(geo.raw_hash, SHAPE_VERSION)


def _load(names) -> list[_Airfoil]:
    """Each airfoil's source points (as read and normalized) and its stored shape, densely rendered."""
    paths = Geom_paths()
    if not paths.db.is_file():
        pytest.skip(f"no geometry manifest at {paths.db}")
    airfoils = []
    with GeometryStore.open(paths.db, read_only=True) as store:
        found = {n: _lookup(store, n) for n in names}
        missing = [n for n, f in found.items() if f is None or f[1] is None]
        assert not missing, f"not in the manifest: {missing}; edit the viewer's .yaml"
        for name in names:
            geo, shape = found[name]
            raw = Geom_read_dat(paths.from_rel(geo.raw_path)).coords
            raw_normalized, tf = Geom_normalize(raw)
            family = f"{geo.family.value} family" if geo.family else "tabulated"
            verdict = f"rejected ({shape.reason.value}), " if isinstance(shape, RejectionRecord) else ""
            fit = f"fit max {shape.fit_max:.1e}c" if shape.fit_max is not None else "no fit"
            digested = f"CST order {shape.cst.order}" if shape.cst is not None else "none, rejected before the fit"
            caption = (f"{geo.name}: {verdict}{family}, closure {shape.closure.value}, "
                       f"source gap {shape.root_te_gap:.2%}c, {fit}\nundigested: {len(raw)} points from {geo.raw_path}, "
                       f"normalizing rotates {tf.rotation_deg:+.2f} deg; digested: {digested} ({shape.shape_version})")
            contour = Geom_cst_contour(shape.cst, 2001) if shape.cst is not None else None
            airfoils.append(_Airfoil(geo.name, raw, raw_normalized, contour, caption))
    return airfoils


def test_viewer_selection_is_in_the_manifest():
    names = _selection(example=True)
    airfoils = _load(names)

    assert len(airfoils) == len(names) <= 10
    assert all(a.digested[0].tolist() == [1.0, 0.0] for a in airfoils if a.digested is not None)


@pytest.mark.interactive
def test_geometry_viewer():
    import matplotlib.pyplot as plt

    names = _selection()
    assert len(names) <= 10, f"{len(names)} airfoils; keys 1-9, 0 reach ten"
    airfoils = _load(names)
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
        if a.digested is not None:
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
            if (int(event.key) - 1) % 10 >= len(airfoils):
                return
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
