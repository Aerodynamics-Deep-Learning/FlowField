"""
Tests for `geometry.common.schemas.CSTParams`: one nose radius under a shared nose, i.e.:

Step 1: Ensure a shared nose's lower A_0 is derived, snapped, or refused
    - test_left_out_nose_is_derived
    - test_nose_off_by_rounding_is_snapped
    - test_mismatched_nose_rejected
    - test_separate_nose_keeps_both_radii
    - test_wrong_length_lower_rejected
Step 2: Ensure stored shapes, written before `nose` existed, still load as LEM
    - test_json_without_nose_loads_as_lem
    - test_stored_shapes_load_with_one_nose

A generated design (the sampler, the run log) has one nose dial, upper A_0: the lower one follows it.
"""

import pytest
from pydantic import ValidationError

from src.datagen.geometry.common.constants import CST_N1, CST_N2, NOSE_ATOL, SHAPE_VERSION
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import CSTParams, Nose
from src.datagen.geometry.manifest.store import GeometryStore

_UPPER = [0.17, 0.15, 0.20, 0.18, 0.16]
_LOWER_REST = [-0.10, -0.08, -0.05, 0.01]  # the lower A_1..A_N


def _cst(lower, nose: Nose = Nose.LEM) -> CSTParams:
    return CSTParams(n1=CST_N1, n2=CST_N2, upper=_UPPER, lower=lower, nose=nose)


# region Step 1
@pytest.mark.parametrize("nose", [Nose.LEM, Nose.SHARED])
def test_left_out_nose_is_derived(nose):
    cst = _cst(_LOWER_REST, nose)

    assert cst.lower == [-0.17] + _LOWER_REST and cst.order == len(_UPPER) - 1


def test_nose_off_by_rounding_is_snapped():
    cst = _cst([-0.17 + 0.5 * NOSE_ATOL] + _LOWER_REST)

    assert cst.lower[0] == -cst.upper[0]  # exactly, not just close
    assert cst.lower[1:] == _LOWER_REST


@pytest.mark.parametrize("nose", [Nose.LEM, Nose.SHARED])
def test_mismatched_nose_rejected(nose):
    with pytest.raises(ValidationError, match="shared nose") as err:
        _cst([-0.12] + _LOWER_REST, nose)
    assert "lower[0] must equal -upper[0]" in str(err.value) and "nose='separate'" in str(err.value)

    with pytest.raises(ValidationError, match="shared nose"):
        _cst([-0.17 - 10 * NOSE_ATOL] + _LOWER_REST, nose)  # past rounding: refused, not snapped


def test_separate_nose_keeps_both_radii():
    assert _cst([-0.12] + _LOWER_REST, Nose.SEPARATE).lower[0] == -0.12

    with pytest.raises(ValidationError, match="same, nonzero number"):
        _cst(_LOWER_REST, Nose.SEPARATE)  # two radii: nothing to derive the lower one from


@pytest.mark.parametrize("lower", [_LOWER_REST[:-1], [-0.17] + _LOWER_REST + [0.0]])
def test_wrong_length_lower_rejected(lower):
    with pytest.raises(ValidationError, match="same, nonzero number"):
        _cst(lower)
# endregion


# region Step 2
def test_json_without_nose_loads_as_lem():
    cst = CSTParams.model_validate_json('{"n1": 0.5, "n2": 1.0, "upper": [0.17, 0.2], "lower": [-0.17, 0.1]}')

    assert cst.nose is Nose.LEM and cst.lower == [-0.17, 0.1]
    assert CSTParams.model_validate_json(cst.model_dump_json()) == cst
    with pytest.raises(ValidationError, match="shared nose"):  # JSON goes through the same rule
        CSTParams.model_validate_json('{"n1": 0.5, "n2": 1.0, "upper": [0.17, 0.2], "lower": [-0.12, 0.1]}')


def test_stored_shapes_load_with_one_nose():
    db = Geom_paths().db
    if not db.is_file():
        pytest.skip(f"no committed manifest at {db}")
    with GeometryStore.open(db, read_only=True) as store:  # every row is validated as it is read
        csts = [r.cst for r in store.shapes(SHAPE_VERSION)] + \
               [r.cst for r in store.rejections(SHAPE_VERSION) if r.cst is not None]

    assert csts
    assert all(c.nose is Nose.LEM and c.lower[0] == -c.upper[0] for c in csts)
# endregion
