"""
Mesh optimization per airfoil study.

Layout:
    common/    the study's fixed choices (constants.py), search space (space.py), the settings row recording
               them (settings.py) and its data/ paths (paths.py); imports nothing else in the optimizer
    evaluate/  one config: the sampled input it meshes (sampling.py) and the executor turning it into one run
               row (executor.py)
    search/    one airfoil: the NOMAD driver searching a rung's knobs (mads.py), the sampler-only n* search
               (nstar.py), and the ladder from n* up the rungs (ladder.py)
    manifest/  meshing.sqlite, the run store every row lands in (store.py, at common/paths.py's MESHING_DB)
"""
