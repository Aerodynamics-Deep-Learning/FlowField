"""
The data-generation pipeline's command lines (README.md in this folder is the guide). One script per stage, each
parsing its options and calling its stage function from `src/datagen` in this process:

    machine_check        does this machine sample and mesh like the reference?   (parity.py)
    geometry_calibrate   .dat folder -> CST calibration report                  (geometry/build/calibrate.py)
    geometry_ingest      .dat folder -> geometry manifest                       (geometry/manifest/ingest.py)
    mesh_calibrate       manifest -> the ladder's calibration report             (meshing/optimizer/batch/calibrate.py)
    mesh_campaign        manifest -> run store, each airfoil's best             (meshing/optimizer/batch/campaign.py)
    mesh_remesh          run store -> .su2 (.vtk) per airfoil                   (meshing/optimizer/batch/remesh.py)

and `mesh_master`, which runs the stages a YAML config enables. `engine.py` is what they share: the plan made
before anything runs, the output rule (scratchpad, new, existing, shared) and the logs. Run each as
`python -m scripts.datagen.<script>` from the repo root; `--help` lists its options.
"""
