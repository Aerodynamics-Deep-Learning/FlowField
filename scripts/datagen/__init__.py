"""
The data-generation pipeline's command lines (README.md in this folder is the guide), one folder per domain, one
script per stage, each parsing its options and calling its stage function from `src/datagen` in this process:

    geometry/calibrate     .dat folder -> CST calibration report          (geometry/build/calibrate.py)
    geometry/ingest        .dat folder -> geometry manifest               (geometry/manifest/ingest.py)
    meshing/machine_check  does this machine sample and mesh like the reference?   (meshing/parity.py)
    meshing/calibrate      manifest -> the ladder's calibration report     (meshing/optimizer/batch/calibrate.py)
    meshing/campaign       manifest -> run store, each airfoil's best     (meshing/optimizer/batch/campaign.py)
    meshing/remesh         run store -> .su2 (.vtk) per airfoil           (meshing/optimizer/batch/remesh.py)
    meshing/master         the stages a YAML config enables, in order

`engine.py` is what they all share: the plan made before anything runs, the output rule (the run's folder, new,
existing, shared) and the run's one log. Run each as `python -m scripts.datagen.<folder>.<script>` from the repo root;
`--help` lists its options.
"""
