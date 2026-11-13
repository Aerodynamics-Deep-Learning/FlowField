"""
Tests for `active_learning.rules`: the rules a design must pass before p_mesh is asked about it, i.e.:

Step 1: Ensure the shape-only rules pass a design and refuse a crossing, and a non-design is a caller bug
    - test_cusp_passes_both_groups
    - test_crossing_inside_fails_thickness
    - test_non_design_raises
    - test_rules_agree_with_the_ingest_on_stored_shapes
Step 2: Ensure the settings-dependent rules read the TE and knobs from the production settings
    - test_cap_leaving_its_side_fails_contour_under_a_round_te_only
    - test_precheck_rejection_depends_on_the_knobs
    - test_unsampled_pairing_raises
"""

import pytest

from src.datagen.active_learning.rules import RULES_VERSION, AL_geometry_rules, AL_mesh_rules
from src.datagen.geometry.build.checks import Geom_check_thickness
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import CSTParams, Nose
from src.datagen.geometry.manifest.store import GeometryStore
from tests.active_learning_rows import KNOBS, PRODUCTION, production
from tests.geometry_shapes import CUSP


def _cusp_with(upper=None, lower=None) -> CSTParams:
    """The cusp with some coefficients replaced, as {index: value}; lower A_0 follows the nose."""
    up, lo = list(CUSP.upper), list(CUSP.lower)
    for i, v in (upper or {}).items():
        up[i] = v
    for i, v in (lower or {}).items():
        lo[i] = v
    return CSTParams(n1=CUSP.n1, n2=CUSP.n2, upper=up, lower=lo[1:], le_weight=CUSP.le_weight)


# The TE dives ~60 deg below the chord, both surfaces alike: thick enough, but a round cap can't stay above y = 0
DROOPED = _cusp_with(upper={8: -2.0}, lower={8: -2.5})


# region Step 1
def test_cusp_passes_both_groups():
    assert AL_geometry_rules(CUSP).ok and AL_mesh_rules(CUSP, PRODUCTION).ok
    assert RULES_VERSION == "rules-r1"


def test_crossing_inside_fails_thickness():
    crossed = _cusp_with(upper={4: -0.8})
    verdict = AL_geometry_rules(crossed)

    assert not verdict.ok and verdict.rule == "thickness" and "cross at x=" in verdict.reason


@pytest.mark.parametrize("update, match", [({"nose": Nose.SEPARATE}, "designs are LEM"),
                                           ({"n1": 0.6}, "class exponents")])
def test_non_design_raises(update, match):
    with pytest.raises(ValueError, match=match):
        AL_geometry_rules(CUSP.model_copy(update=update))


def test_rules_agree_with_the_ingest_on_stored_shapes():
    db = Geom_paths().db
    if not db.is_file():
        pytest.skip(f"no committed manifest at {db}")
    with GeometryStore.open(db, read_only=True) as store:
        csts = [r.cst for r in store.shapes(SHAPE_VERSION)]

    assert csts and all(AL_geometry_rules(c).ok == Geom_check_thickness(c)[0] for c in csts)
# endregion


# region Step 2
def test_cap_leaving_its_side_fails_contour_under_a_round_te_only():
    assert AL_geometry_rules(DROOPED).ok  # the shape itself is fine
    verdict = AL_mesh_rules(DROOPED, PRODUCTION)  # c2d O-grid: a round TE
    assert not verdict.ok and verdict.rule == "contour" and "cap" in verdict.reason

    sharp = AL_mesh_rules(DROOPED, production(topology="CGRID"))  # c2d C-grid: a sharp TE needs no cap
    assert sharp.rule != "contour"


def test_precheck_rejection_depends_on_the_knobs():
    coarse_nose = AL_mesh_rules(CUSP, production({**KNOBS, "le_ratio": 1.0}))
    assert not coarse_nose.ok and coarse_nose.rule == "precheck" and "turn" in coarse_nose.reason

    too_many = AL_mesh_rules(CUSP, production(cell_ceiling=10 * int(KNOBS["jmax"])))  # 10 points allowed
    assert not too_many.ok and too_many.rule == "precheck" and "points" in too_many.reason


def test_unsampled_pairing_raises():
    with pytest.raises(ValueError, match="not sampled"):
        AL_mesh_rules(CUSP, production(backend="gmsh", topology="CGRID"))
# endregion
