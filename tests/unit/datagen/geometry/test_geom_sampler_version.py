"""
Tests that `SAMPLER_VERSION` names the samples made under it: three fixed shapes, each with every TE, are
sampled at the default knobs and compared with what was pinned for the current version, i.e.:

Step 1: Ensure a change to the samples comes with a new sampler version
    - test_samples_match_their_version

A failure means the contour -> spacing -> placement now puts points elsewhere. If that was meant, bump the
revision in `SAMPLER_VERSION` (common/constants.py) and pin the values the failure prints under the new
version; otherwise undo the change. The shapes are fixed coefficients, so a shape-version change can't
trip this test.
"""

import numpy as np

from src.datagen.geometry.common.constants import CST_N1, CST_N2, SAMPLER_VERSION
from src.datagen.geometry.common.schemas import CSTParams, TETreatment
from src.datagen.geometry.sample.contour import Geom_build_contour
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from tests.geometry_shapes import as_record

# Per version, per (shape, TE): (points, le_idx, sum of |coordinates|)
_PINNED = {
    'samp-r1': {
        ('naca4412', 'sharp'): (715, 359, 444.197985565),
        ('naca4412', 'blunt'): (715, 359, 445.028358846),
        ('naca4412', 'round'): (772, 388, 501.730137018),
        ('tiltfoil', 'sharp'): (714, 359, 443.523618758),
        ('tiltfoil', 'blunt'): (714, 359, 443.848080034),
        ('tiltfoil', 'round'): (771, 389, 500.164828712),
        ('cusp', 'sharp'): (712, 356, 438.850996041),
        ('cusp', 'blunt'): (712, 356, 439.382331823),
        ('cusp', 'round'): (767, 384, 493.642948231),
    },
}

# A cambered NACA formula and a tilted tabulated section as stored under cst8-r1, and a cusp
_SHAPES = {
    "naca4412": CSTParams(n1=CST_N1, n2=CST_N2, le_weight=0.124416913311,
                          upper=[0.175878592394, 0.209458346212, 0.252775346535, 0.284584406339, 0.214726461646,
                                 0.282921071682, 0.251979902025, 0.271932518374, 0.279186633556],
                          lower=[-0.175878592394, -0.098441827015, -0.102274325026, 0.034493837882, -0.142092789791,
                                 0.040371064685, -0.059924631248, -0.004315916708, -0.016542607033]),
    "tiltfoil": CSTParams(n1=CST_N1, n2=CST_N2, le_weight=0.355110639382,
                          upper=[0.138210618516, 0.15720216674, 0.405545637402, 0.146973435227, 0.305823451498,
                                 0.266773398452, 0.24672168346, 0.281338579085, 0.274022522567],
                          lower=[-0.138210618516, -0.07720487043, 0.152813419818, -0.042666634027, 0.030493359171,
                                 0.103168556038, 0.003730765064, 0.083649062884, 0.055177476722]),
    "cusp": CSTParams(n1=CST_N1, n2=CST_N2, le_weight=0.02,
                      upper=[0.17, 0.15, 0.20, 0.18, 0.16, 0.20, 0.19, 0.12, 0.05],
                      lower=[-0.17, -0.10, -0.08, -0.05, -0.06, -0.02, 0.01, 0.03, 0.05]),
}


def _as_pins(current: dict) -> str:
    lines = [f"    {SAMPLER_VERSION!r}: {{"]
    lines += [f"        {key!r}: ({n}, {le_idx}, {round(total, 9)})," for key, (n, le_idx, total) in current.items()]
    return "\n".join(lines + ["    },"])


# region Step 1
def test_samples_match_their_version():
    current = {}
    for name, cst in _SHAPES.items():
        for te in TETreatment:
            smp = Geom_sample(Geom_build_contour(as_record(name, cst), te, 0.004), SamplingFields())
            assert smp.accepted, smp.report.rejection
            current[(name, te.value)] = (len(smp.points), smp.le_idx, float(np.abs(smp.points).sum()))
    pins = f"Pin these in _PINNED:\n{_as_pins(current)}"

    assert SAMPLER_VERSION in _PINNED, f"nothing is pinned for {SAMPLER_VERSION}. {pins}"
    for key, (n, le_idx, total) in _PINNED[SAMPLER_VERSION].items():
        got = current[key]
        assert got[:2] == (n, le_idx) and abs(got[2] - total) <= 1e-9 * total, (
            f"{key} now samples differently under {SAMPLER_VERSION}: bump the revision in SAMPLER_VERSION "
            f"(common/constants.py), or undo the change. {pins}")
# endregion
