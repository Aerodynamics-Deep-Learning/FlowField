"""
Tests for `geometry.sample.fields`: the sampling knobs a mesh config inherits, i.e.:

Step 1: Ensure the defaults and bounds
    - test_defaults
    - test_out_of_bounds_refused
    - test_uniform_spacing_allowed
"""

import pytest
from pydantic import ValidationError

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.geometry.sample.fields import SamplingFields


# region Step 1
def test_defaults():
    f = SamplingFields()

    assert (f.dx_mid, f.le_ratio, f.te_ratio, f.cluster_width) == (DX_MAX, 0.025, 0.05, 0.05)


@pytest.mark.parametrize("field, value", [("dx_mid", 0.0), ("dx_mid", DX_MAX * 1.001), ("le_ratio", 0.0),
                                          ("le_ratio", 1.01), ("te_ratio", -0.1), ("te_ratio", 1.5),
                                          ("cluster_width", 0.0), ("cluster_width", 0.6), ("n_surface", 400)])
def test_out_of_bounds_refused(field, value):
    with pytest.raises(ValidationError):
        SamplingFields(**{field: value})


def test_uniform_spacing_allowed():
    f = SamplingFields(dx_mid=1e-3, le_ratio=1.0, te_ratio=1.0)  # the coarse end: dx_mid everywhere

    assert f.le_ratio * f.dx_mid == f.te_ratio * f.dx_mid == f.dx_mid
# endregion
