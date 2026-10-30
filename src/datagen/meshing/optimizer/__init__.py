"""
Mesh optimization per airfoil study: the sampled input each evaluation meshes (sampling.py), the
study's fixed choices (constants.py), search space (space.py) and the settings row recording them
(settings.py), the executor turning one config into one run row (executor.py), the NOMAD driver searching
a rung's knobs through it (mads.py), the sampler-only n* search (nstar.py), the per-airfoil ladder from
n* up the rungs (ladder.py), and the run store every row lands in (store.py, at paths.py's MESHING_DB).
"""
