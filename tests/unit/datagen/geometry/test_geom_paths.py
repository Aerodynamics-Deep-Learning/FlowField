"""
Tests for `geometry.paths`: the file layout under `data/`, i.e.:

Step 1: Ensure the layout and the manifest's relative paths
    - test_layout
    - test_relative_paths_round_trip
    - test_default_root_is_the_repo_data_folder
    - test_source_label_hides_local_paths
"""

from pathlib import Path

from src.datagen.geometry.common.paths import DATA_ROOT, REPO_ROOT, Geom_paths, Geom_source_label


# region Step 1
def test_layout(tmp_path):
    p = Geom_paths(tmp_path)

    assert p.db == tmp_path / "manifest" / "datagen" / "geometry" / "geometry.sqlite"
    assert p.raw_dir == tmp_path / "raw" / "datagen" / "geometry" / "raw"
    assert p.norm_dir("cst8-r1") == tmp_path / "raw" / "datagen" / "geometry" / "normalised" / "cst8-r1"
    assert p.report_dir("20261012T1830Z_3f9a") == tmp_path / "raw" / "datagen" / "geometry" / "report" / "20261012T1830Z_3f9a"
    assert p.calibration_dir == tmp_path / "raw" / "datagen" / "geometry" / "report" / "calibration"


def test_relative_paths_round_trip(tmp_path):
    p = Geom_paths(tmp_path)
    path = p.raw_dir / "naca2412.dat"

    assert p.to_rel(path) == "raw/datagen/geometry/raw/naca2412.dat"
    assert p.from_rel(p.to_rel(path)) == path


def test_default_root_is_the_repo_data_folder():
    assert DATA_ROOT.name == "data"
    assert (DATA_ROOT.parent / "src" / "datagen" / "geometry" / "common" / "paths.py").is_file()
    assert Geom_paths().data_root == Path(DATA_ROOT)


def test_source_label_hides_local_paths(tmp_path):
    assert Geom_source_label(tmp_path / "for_cst") == "for_cst"  # outside the repo: the name only
    assert Geom_source_label(Geom_paths().raw_dir) == "data/raw/datagen/geometry/raw"
    assert Geom_source_label(REPO_ROOT) == "."
# endregion
