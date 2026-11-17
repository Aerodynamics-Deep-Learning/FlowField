"""Fixtures every tier shares, and the suite's xdist seeding."""

from pathlib import Path

import pytest

from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.manifest.ingest import Geom_ingest_folder

pytest_plugins = ("tests.xdist_seeding",)

AIRFOILS = Path(__file__).with_name("airfoils")  # naca0012, naca0040, naca4412, s1223: copies of their root files


@pytest.fixture(scope="session")
def airfoils_manifest(tmp_path_factory) -> Path:
    """A geometry manifest of `AIRFOILS`, ingested once a session, so no test reads the repo's data/."""
    root = tmp_path_factory.mktemp("airfoils_manifest")
    Geom_ingest_folder(AIRFOILS, root)
    return Geom_paths(root).db
