"""
How a ladder reports, kept apart from the ladder so that the runner, and with it a slot's worker, load neither
NOMAD nor the meshing stack:

    LadderOutcome  how a ladder ended, with the error that ended it, for the caller to act on
    Optim_record   one ladder event written to the run store, by the campaign's one writer
"""

from typing import NamedTuple, Optional

from src.datagen.meshing.optimizer.manifest.store import RunStore


class LadderOutcome(NamedTuple):
    """How a ladder ended. Picklable, so a pool's worker can return it."""
    ladder_id: str
    status: str                       # 'passed', 'failed' or 'aborted'
    route: list[str]                  # the stages entered, e.g. ['golden', 'jmax_bump']
    best_row: Optional[dict]          # the fewest-node pass
    error: Optional[BaseException]    # what aborted it, for the caller to act on
    notes: Optional[dict] = None      # what a Ladder subclass measured beyond its rows


def Optim_record(store: RunStore, event: tuple) -> None:
    """
    Writes one ladder event to the run store: a start inserts the ladder's row, a run its own row, and the
    end completes the ladder's row and repoints the airfoil's best run, together.

    Args:
        store (RunStore): The store, holding the settings row the ladder ran under
        event (tuple): ("ladder_start", row), ("run", row) or ("ladder_end", ladder_id, fields)

    Raises:
        ValueError: On an unknown event, or a write the store refuses (an unknown column, a ladder never
            started)
    """
    kind = event[0]
    if kind == "ladder_start":
        store.start_ladder(event[1])
    elif kind == "run":
        store.insert_run(event[1])
    elif kind == "ladder_end":
        _, ladder_id, fields = event
        with store.transaction():
            store.finish_ladder(ladder_id, fields)
            ladder = store.ladder(ladder_id)
            store.update_best(ladder["geometry_id"], ladder["settings_id"])
    else:
        raise ValueError(f"unknown ladder event {kind!r}")
