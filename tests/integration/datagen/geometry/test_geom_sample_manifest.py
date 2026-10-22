"""
Tests that every airfoil in the repo's geometry manifest takes each TE its meshers need, i.e.:

Step 1: Ensure every stored shape builds a contour
    - test_every_stored_shape_builds_a_contour

Reads the repo's own manifest under `data/`; skipped without it.
"""

import pytest

from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.geometry.sample.contour import ContourError, Geom_build_contour

_GAP = 0.004  # the studies' default TE gap


# region Step 1
def test_every_stored_shape_builds_a_contour():
    paths = Geom_paths()
    if not paths.db.is_file():
        pytest.skip(f"no geometry manifest at {paths.db}")
    refused, limits = [], []
    with GeometryStore.open(paths.db, read_only=True) as store:
        shapes = store.shapes(SHAPE_VERSION)
        for shape in shapes:
            for te in TETreatment:
                try:
                    contour = Geom_build_contour(shape, te, _GAP)
                except ContourError as e:
                    refused.append(f"{store.name_of(shape.geometry_id)} ({te.value}): {e}")
                    continue
                if te is TETreatment.ROUND:
                    limits.append(contour.te_cap_limit)

    assert shapes and not refused, refused
    assert 1e-4 < min(limits) and max(limits) < 1e-3  # cap peak curvature ~2-7/g at g = 0.4%c
# endregion
