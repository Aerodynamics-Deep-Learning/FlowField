"""
The data-generation pipeline's command lines: one script per stage, each parsing its options and calling its stage
function from `src/datagen` in this process.

    geometry_ingest      .dat folder -> geometry manifest              (geometry/manifest/ingest.py)
    geometry_calibrate   .dat folder -> CST calibration report         (geometry/build/calibrate.py)
    mesh_calibrate       manifest -> the ladder's calibration report    (meshing/optimizer/batch/calibrate.py)
    mesh_campaign        manifest -> run store, each airfoil's best    (meshing/optimizer/batch/campaign.py)
    mesh_remesh          run store -> .su2 (.vtk) per airfoil          (meshing/optimizer/batch/remesh.py)

Run each as `python -m scripts.datagen.<script>` from the repo root; `--help` lists its options.
"""
