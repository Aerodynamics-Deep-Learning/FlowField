"""
Tests that every airfoil in the repo's geometry manifest takes each TE its meshers need, and samples
within the rules anywhere in the optimizer's search space, i.e.:

Step 1: Ensure every stored shape builds a contour
    - test_every_stored_shape_builds_a_contour
Step 2: Ensure sampling keeps its guarantees on every stored shape
    - test_random_configs_keep_the_rules

Reads the repo's own manifest under `data/`; skipped without it.
"""

import numpy as np
import pytest

from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.geometry.sample.checks import SamplerAssertionError
from src.datagen.geometry.sample.contour import ContourError, Geom_build_contour
from src.datagen.geometry.sample.sample import Geom_sample
from tests.geometry_shapes import search_space_configs

_GAP = 0.004  # the studies' default TE gap


@pytest.fixture(scope="module")
def contours() -> dict:
    """Per stored airfoil name, per TE: its contour, or the refusal's message."""
    paths = Geom_paths()
    if not paths.db.is_file():
        pytest.skip(f"no geometry manifest at {paths.db}")
    built = {}
    with GeometryStore.open(paths.db, read_only=True) as store:
        for shape in store.shapes(SHAPE_VERSION):
            per_te = built.setdefault(store.name_of(shape.geometry_id), {})
            for te in TETreatment:
                try:
                    per_te[te] = Geom_build_contour(shape, te, _GAP)
                except ContourError as e:
                    per_te[te] = str(e)
    return built


# region Step 1
def test_every_stored_shape_builds_a_contour(contours):
    refused = [f"{name} ({te.value}): {c}" for name, per_te in contours.items() for te, c in per_te.items()
               if isinstance(c, str)]
    limits = [per_te[TETreatment.ROUND].te_cap_limit for per_te in contours.values()]

    assert contours and not refused, refused
    assert 1e-4 < min(limits) and max(limits) < 1e-3  # cap peak curvature ~2-7/g at g = 0.4%c
# endregion


# region Step 2
def test_random_configs_keep_the_rules(contours):
    rng = np.random.default_rng(2026)
    broken, on_caps, counts = [], [], {"accepted": 0, "turn": 0}
    for name, per_te in contours.items():
        for te, k in ((TETreatment.ROUND, 3), (TETreatment.SHARP, 1)):
            contour = per_te[te]
            for fields in search_space_configs(rng, k):
                try:
                    smp = Geom_sample(contour, fields)
                except SamplerAssertionError as e:
                    broken.append(f"{name}: {e} ({fields})")
                    continue
                counts["accepted" if smp.accepted else smp.report.rejection.value] += 1
                if not smp.accepted and contour.weld_x is not None and smp.report.x_at_precheck_turn >= contour.weld_x:
                    on_caps.append(f"{name}: {fields}")

    assert not broken, broken
    assert not on_caps, f"the clamped caps failed the pre-check: {on_caps}"
    assert counts["accepted"] > counts["turn"] > 0  # both outcomes exercised
# endregion
