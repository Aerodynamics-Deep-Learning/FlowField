"""
Where the geometry pipeline's files live, all under the repo's committed `data/`:

    data/manifest/datagen/geometry/geometry.sqlite, README.md
    data/manifest/datagen/geometry/geometry_archive.sqlite            shapes of earlier shape versions
    data/raw/datagen/geometry/raw/<name>.dat                          verbatim copies of accepted sources
    data/raw/datagen/geometry/normalised/<shape_version>/<name>.dat   renderings of the fitted shapes
    data/raw/datagen/geometry/report/<ingest_id>/                     one report per file seen, plus summary.txt
    data/raw/datagen/geometry/report/calibration/                     calibration tables

The manifest stores paths relative to `data/` with forward slashes, so a clone elsewhere resolves them.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]  # src/datagen/geometry: the code a shape depends on
DATA_ROOT = REPO_ROOT / "data"
ARCHIVE_NAME = "geometry_archive.sqlite"  # next to the manifest


class GeometryPaths:
    """The layout above, rooted at `data_root` (tests point it at a temporary folder)."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.manifest_dir = self.data_root / "manifest" / "datagen" / "geometry"
        self.db = self.manifest_dir / "geometry.sqlite"
        self.archive_db = self.manifest_dir / ARCHIVE_NAME
        self.files_dir = self.data_root / "raw" / "datagen" / "geometry"
        self.raw_dir = self.files_dir / "raw"
        self.calibration_dir = self.files_dir / "report" / "calibration"

    def norm_dir(self, shape_version: str) -> Path:
        return self.files_dir / "normalised" / shape_version

    def report_dir(self, ingest_id: str) -> Path:
        return self.files_dir / "report" / ingest_id

    def to_rel(self, path) -> str:
        """A path under `data_root` as the manifest stores it."""
        return Path(path).resolve().relative_to(self.data_root.resolve()).as_posix()

    def from_rel(self, rel: str) -> Path:
        return self.data_root / rel


def Geom_paths(data_root=DATA_ROOT) -> GeometryPaths:
    """
    The geometry file layout.

    Args:
        data_root (str | Path): The `data/` folder, the repo's own by default

    Returns:
        GeometryPaths: Every location the pipeline reads or writes
    """
    return GeometryPaths(data_root)


def Geom_source_label(folder) -> str:
    """
    A source folder as committed files name it: relative to the repo inside it, else only its name, so
    no one's local paths end up in the repo.

    Args:
        folder (str | Path): An ingested or calibrated folder

    Returns:
        str: The label
    """
    folder = Path(folder).resolve()
    try:
        return folder.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return folder.name


def Geom_git_version() -> str:
    """The commit the code runs at, '-dirty' with uncommitted changes under CODE_DIR, 'unknown' outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", CODE_DIR.relative_to(REPO_ROOT).as_posix()],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")
