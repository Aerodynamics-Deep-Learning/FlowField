"""
Tests for `geometry.paths`: the file layouts, the repo's under `data/` and the flat one elsewhere, i.e.:

Step 1: Ensure the layouts and the manifest's relative paths
    - test_the_repo_layout
    - test_the_flat_layout
    - test_relative_paths_round_trip
    - test_default_root_is_the_repo_data_folder
    - test_source_label_hides_local_paths
"""

from pathlib import Path

from src.datagen.geometry.common.paths import DATA_ROOT, REPO_ROOT, GeometryPaths, Geom_paths, Geom_source_label


# region Step 1
def test_the_repo_layout(tmp_path):
    p = GeometryPaths(tmp_path, repo=True)
    files = tmp_path / "raw" / "datagen" / "geometry"

    assert p.db == tmp_path / "manifest" / "datagen" / "geometry" / "geometry.sqlite"
    assert p.raw_dir == files / "raw"
    assert p.rejected_raw("goe08k", "0752aa19ff") == files / "rejected" / "0752aa19" / "goe08k.dat"
    assert p.norm_dir("cst8-r1") == files / "normalised" / "cst8-r1"
    assert p.report_dir("20261012T1830Z_3f9a") == files / "report" / "20261012T1830Z_3f9a"
    assert p.calibration_dir == files / "report" / "calibration"


def test_the_flat_layout(tmp_path):
    p = Geom_paths(tmp_path)  # any folder but the repo's data/
    files = tmp_path / "raw"

    assert p.db == tmp_path / "geometry_ingest.sqlite" and p.archive_db == tmp_path / "geometry_archive.sqlite"
    assert p.raw_dir == files / "input"
    assert p.rejected_raw("goe08k", "0752aa19ff") == files / "rejected" / "0752aa19" / "goe08k.dat"
    assert p.rejected_norm("cst8-r1", "goe08k", "0752aa19ff") == \
        files / "normalised" / "cst8-r1" / "rejected" / "0752aa19" / "goe08k.dat"
    assert p.report_dir("20261012T1830Z_3f9a") == files / "20261012T1830Z_3f9a"


def test_relative_paths_round_trip(tmp_path):
    p = Geom_paths(tmp_path)
    path = p.raw_dir / "naca2412.dat"

    assert p.to_rel(path) == "raw/input/naca2412.dat"
    assert p.from_rel(p.to_rel(path)) == path


def test_default_root_is_the_repo_data_folder():
    assert DATA_ROOT.name == "data"
    assert (DATA_ROOT.parent / "src" / "datagen" / "geometry" / "common" / "paths.py").is_file()
    assert Geom_paths().data_root == Path(DATA_ROOT)
    repo_db = DATA_ROOT / "manifest" / "datagen" / "geometry" / "geometry.sqlite"
    assert Geom_paths().db == repo_db and Geom_paths(DATA_ROOT / "scratchpad" / "..").db.resolve() == repo_db


def test_source_label_hides_local_paths(tmp_path):
    assert Geom_source_label(tmp_path / "for_cst") == "for_cst"  # outside the repo: the name only
    assert Geom_source_label(Geom_paths().raw_dir) == "data/raw/datagen/geometry/raw"
    assert Geom_source_label(REPO_ROOT) == "."
# endregion
