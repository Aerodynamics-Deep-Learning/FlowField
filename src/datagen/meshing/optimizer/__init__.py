import importlib

_EXPORTS = {
    # Every airfoil
    "Optim_load_shapes": "batch.runner",            # the stored shapes to study, by name, from the geometry manifest
    "Optim_run_airfoils": "batch.runner",           # one task per airfoil over isolated worker slots, one store writer
    "BatchStopped": "batch.runner",           # a batch ended early: a setup error, or aborts in a row
    "Optim_slot_platform": "batch.slots",           # this machine's slots: job objects on Windows, cgroups on Linux
    "Optim_run_campaign": "batch.campaign",         # the ladder over every airfoil not finished under the constants
    "Optim_campaign_summary": "batch.campaign",     # a campaign's outcome, read off the run store
    "Optim_calibrate": "batch.calibrate",           # the ladder's calibration: jmax*, rung 3's plateau, rung 2's budget
    "Optim_format_report": "batch.calibrate",       # the calibration's report, read off its run store
    "Optim_run_calibration": "batch.calibrate",     # a dated calibration: its database and report in one folder
    "Optim_remesh_airfoils": "batch.remesh",        # each airfoil's best run meshed again into its own folder
    "Optim_remesh_run": "batch.remesh",             # one run by id meshed again into its airfoil's folder
    # One airfoil
    "Optim_run_ladder": "search.ladder",            # n*, the golden run and the rungs it routes to, reported as events
    "LadderOutcome": "search.events",         # how a ladder ended, its error handed up rather than raised
    "Ladder": "search.ladder",                # the climb's steps, for a ladder that climbs differently
    "Optim_record": "search.events",                # one ladder event written to the run store
    "Optim_n_star_search": "search.nstar",          # the fewest surface points the sampler accepts
    "NStarResult": "search.nstar",
    "Optim_run_mads": "search.mads",                # one NOMAD search over a rung's knobs, to its budget or plateau
    "RungResult": "search.mads",
    # One config
    "RunExecutor": "evaluate.executor",       # one config sampled, meshed and judged into one run row
    "RunContext": "evaluate.context",         # what every run of a batch shares: settings, host, git state
    "RunResult": "evaluate.executor",
    "StudySetupError": "evaluate.context",    # c2d missing or the input refused: a setup to fix, not a config
    "Optim_build_study_contour": "evaluate.sampling",  # a stored shape with the TE its mesher needs
    "Optim_sample_for_mesher": "evaluate.sampling",    # one config's surface points as the mesher's input
    "Optim_sample_for_study": "evaluate.sampling",     # the same, the rest of the input from the settings row
    # One stored run
    "Optim_remesh": "evaluate.remesh",              # a run meshed again, delivered once it hashes to the stored mesh
    "Remeshed": "evaluate.remesh",
    "RemeshRefused": "evaluate.remesh",       # no mesh to rebuild, or something it was meshed by has moved
    "RemeshMismatch": "evaluate.remesh",      # the rebuilt mesh is not the stored one; nothing delivered
    # The study's definition and record
    "Optim_build_settings": "common.settings",      # the hashed settings row every run is meshed and judged by
    "RunStore": "manifest.store",             # meshing.sqlite: settings, ladders, runs and the best run per airfoil
}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_EXPORTS)
