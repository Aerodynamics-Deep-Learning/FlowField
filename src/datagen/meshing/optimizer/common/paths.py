"""
Where the mesh study keeps its files, under the repo's `data/` (geometry's `DATA_ROOT`):

    data/manifest/datagen/meshing/meshing.sqlite        the run store, committed
    data/raw/datagen/meshing/report/calibration/        the calibration's dated reports
"""

from src.datagen.geometry.common.paths import DATA_ROOT

MESHING_DB = DATA_ROOT / "manifest" / "datagen" / "meshing" / "meshing.sqlite"
CALIBRATION_DIR = DATA_ROOT / "raw" / "datagen" / "meshing" / "report" / "calibration"
