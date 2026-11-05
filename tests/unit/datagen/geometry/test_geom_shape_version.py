"""
Tests that `SHAPE_VERSION` names the shapes stored under it: three fixed airfoils, one per path through
`add.py`, are run through the pipeline and compared with what was pinned for the current version, i.e.:

Step 1: Ensure a change to the stored shapes comes with a new shape version
    - test_shapes_match_their_version

A failure means the normalize -> close -> fit -> render pipeline now makes different shapes. If that
was meant, bump the revision in `SHAPE_VERSION` (common/constants.py) and pin the values the failure
prints under the new version; otherwise undo the change.
"""

import numpy as np

from src.datagen.geometry.build.add import Geom_add_airfoil
from src.datagen.geometry.build.families import _family_contour
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.io import Geom_write_dat
from src.datagen.geometry.common.schemas import Family

# Per version, per airfoil: (upper, lower, LE weight, sum of |rendered coordinates|)
_PINNED = {
    'cst8-r1': {
        'naca4412': ([0.175878592394, 0.209458346212, 0.252775346535, 0.284584406339, 0.214726461646, 0.282921071682, 0.251979902025, 0.271932518374, 0.279186633556],
            [-0.175878592394, -0.098441827015, -0.102274325026, 0.034493837882, -0.142092789791, 0.040371064685, -0.059924631248, -0.004315916708, -0.016542607033],
            0.124416913311, 533.937541546),
        'closedfoil': ([0.174052317561, 0.181250543224, 0.21487740974, 0.205856955089, 0.195830663432, 0.202268626635, 0.203858890994, 0.204404398103, 0.212640713808],
            [-0.174052317561, -0.123640774381, -0.14245447258, -0.042907654552, -0.16328229586, -0.037800838345, -0.109554827977, -0.069983779435, -0.081529742173],
            0.060134675136, 533.837953344),
        'tiltfoil': ([0.138210618516, 0.15720216674, 0.405545637402, 0.146973435227, 0.305823451498, 0.266773398452, 0.24672168346, 0.281338579085, 0.274022522567],
            [-0.138210618516, -0.07720487043, 0.152813419818, -0.042666634027, 0.030493359171, 0.103168556038, 0.003730765064, 0.083649062884, 0.055177476722],
            0.355110639382, 532.282772361),
    },
}


def _sources() -> dict[str, np.ndarray]:
    """A family, a closed tabulated airfoil, and an open one away from the normalized frame."""
    closed = _family_contour(Family.NACA4, {"m": 2, "p": 4, "t": 12}, 81, True)
    closed[-1] = closed[0]
    a = np.radians(1.5)
    turn = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    tilted = 1.3 * _family_contour(Family.NACA4, {"m": 6, "p": 3, "t": 9}, 81, False) @ turn + [0.2, -0.1]
    return {"naca4412": _family_contour(Family.NACA4, {"m": 4, "p": 4, "t": 12}, 81, False),
            "closedfoil": closed, "tiltfoil": tilted}


def _as_pins(current: dict) -> str:
    lines = [f"    {SHAPE_VERSION!r}: {{"]
    for name, (upper, lower, le_weight, rendered) in current.items():
        lines += [f"        {name!r}: ({[round(v, 12) for v in upper]},",
                  f"            {[round(v, 12) for v in lower]},",
                  f"            {round(le_weight, 12)}, {round(rendered, 9)}),"]
    return "\n".join(lines + ["    },"])


# region Step 1
def test_shapes_match_their_version(tmp_path):
    current = {}
    for name, coords in _sources().items():
        path = tmp_path / f"{name}.dat"
        Geom_write_dat(path, name, coords)
        res = Geom_add_airfoil(path)
        assert res.accepted, res.reason
        current[name] = (res.cst.upper, res.cst.lower, res.cst.le_weight, float(np.abs(res.norm_coords).sum()))
    pins = f"Pin these in _PINNED:\n{_as_pins(current)}"

    assert SHAPE_VERSION in _PINNED, f"nothing is pinned for {SHAPE_VERSION}. {pins}"
    for name, (upper, lower, le_weight, rendered) in _PINNED[SHAPE_VERSION].items():
        got = current[name]
        same = (np.allclose(got[0], upper, rtol=0, atol=1e-9) and np.allclose(got[1], lower, rtol=0, atol=1e-9)
                and abs(got[2] - le_weight) <= 1e-9 and abs(got[3] - rendered) <= 1e-9 * rendered)
        assert same, (f"{name} now makes a different shape under {SHAPE_VERSION}: bump the revision in "
                      f"SHAPE_VERSION (common/constants.py), or undo the change. {pins}")
# endregion
