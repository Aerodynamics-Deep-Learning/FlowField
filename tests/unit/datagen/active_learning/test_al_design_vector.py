"""
Tests for `active_learning.design_vector`: the 18 free values a design is learned as, i.e.:

Step 1: Ensure only LEM designs with the pipeline's order and class exponents become vectors, and back
    - test_other_designs_refused
    - test_free_vector_round_trips_stored_shapes
    - test_non_finite_values_refused
Step 2: Ensure the TE wedge gap reads as the wedge angle does, and a crossed TE is refused
    - test_gap_sign_matches_the_angle_on_stored_shapes
    - test_gap_sign_matches_the_angle_at_the_edges
    - test_zero_gap_is_a_cusp
    - test_gap_off_by_rounding_is_snapped_to_a_cusp
    - test_crossed_te_refused
"""

import numpy as np
import pytest

from src.datagen.active_learning.design_vector import N_FREE, AL_cst_to_free, AL_free_to_cst, AL_te_wedge_gap
from src.datagen.geometry.build.checks import Geom_check_thickness
from src.datagen.geometry.build.parametrization import Geom_cst_te_wedge
from src.datagen.geometry.common.constants import CST_ORDER, SHAPE_VERSION, WEDGE_ATOL
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import Nose
from src.datagen.geometry.manifest.store import GeometryStore
from tests.geometry_shapes import CUSP


def _stored_shapes():
    db = Geom_paths().db
    if not db.is_file():
        pytest.skip(f"no committed manifest at {db}")
    with GeometryStore.open(db, read_only=True) as store:
        return [r.cst for r in store.shapes(SHAPE_VERSION)]


def _with_gap(gap: float) -> np.ndarray:
    """CUSP's free values with lower A_N set to upper A_N - gap."""
    x = AL_cst_to_free(CUSP)
    x[N_FREE - 2] = x[CST_ORDER] - gap  # lower A_N sits just before the LE weight, upper A_N at index N
    return x


# region Step 1
@pytest.mark.parametrize("update, match", [({"nose": Nose.SEPARATE}, "designs are LEM"),
                                           ({"n1": 0.6}, "class exponents"),
                                           ({"n2": 0.9}, "class exponents")])
def test_other_designs_refused(update, match):
    # n1/n2 aren't in the vector: unchecked, a shape differing only there would share another's 18 values
    with pytest.raises(ValueError, match=match):
        AL_cst_to_free(CUSP.model_copy(update=update))
    with pytest.raises(ValueError, match="need 18 free values"):
        AL_free_to_cst(np.zeros(19))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("i", [0, 16])  # upper A_0; lower A_N, where a NaN also slips past the wedge rule
def test_non_finite_values_refused(bad, i):
    x = AL_cst_to_free(CUSP)
    x[i] = bad

    with pytest.raises(ValueError, match=rf"must be finite, .* at index \[{i}\]"):
        AL_free_to_cst(x)


def test_free_vector_round_trips_stored_shapes():
    csts = _stored_shapes()

    assert csts and all(AL_free_to_cst(AL_cst_to_free(c)) == c for c in csts)
# endregion


# region Step 2
def test_gap_sign_matches_the_angle_on_stored_shapes():
    csts = _stored_shapes()

    assert csts and all(np.sign(AL_te_wedge_gap(c)) == np.sign(Geom_cst_te_wedge(c)) for c in csts)
    assert all(AL_te_wedge_gap(c) >= 0 for c in csts)  # the fit's bound: no stored shape is crossed


@pytest.mark.parametrize("gap", [0.06, 0.0, -1e-12, -0.06])
def test_gap_sign_matches_the_angle_at_the_edges(gap):
    # Built directly, not through AL_free_to_cst, so the crossed ones keep their sign
    lower = CUSP.lower[:-1] + [CUSP.upper[-1] - gap]
    cst = CUSP.model_copy(update={"lower": lower})

    assert AL_te_wedge_gap(cst) == pytest.approx(gap, abs=1e-15)
    assert np.sign(AL_te_wedge_gap(cst)) == np.sign(Geom_cst_te_wedge(cst)) == np.sign(gap)


def test_zero_gap_is_a_cusp():
    cst = AL_free_to_cst(_with_gap(0.0))

    assert AL_te_wedge_gap(cst) == 0.0 and Geom_cst_te_wedge(cst) == 0.0
    assert Geom_check_thickness(cst)[0]


def test_gap_off_by_rounding_is_snapped_to_a_cusp():
    cst = AL_free_to_cst(_with_gap(-1e-12))

    assert cst.lower[-1] == cst.upper[-1]  # exactly, not just close
    assert Geom_check_thickness(cst)[0]  # which refuses any crossing, however small


@pytest.mark.parametrize("gap", [-10 * WEDGE_ATOL, -0.06])
def test_crossed_te_refused(gap):
    with pytest.raises(ValueError, match="TE wedge gap must be >= 0") as err:
        AL_free_to_cst(_with_gap(gap))
    assert "crossed at the TE" in str(err.value)
# endregion
