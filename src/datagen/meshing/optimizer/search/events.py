"""
How a ladder reports, kept apart from the ladder so that the runner, and with it a slot's worker, load neither
NOMAD nor the meshing stack:

    LadderOutcome         how a ladder ended, with the error that ended it, for the caller to act on
    Optim_record          one ladder event written to the run store, by the campaign's one writer
    Optim_failure_values  why a run failed, in numbers against its gates, for a batch's log
"""

import json
from typing import NamedTuple, Optional

from src.datagen.meshing.optimizer.manifest.store import RunStore

# A worst-cell gate: its value's column, its limit's column, whether above the limit fails, the value's format
_WORST = {"skew": ("q_skew_max", "gate_skew_max", True, ".3f"),
          "ortho": ("q_ortho_min", "gate_ortho_min", False, ".3f"),
          "jac": ("q_jac_min", "gate_jac_min", False, ".3f"),
          "aspect_ratio": ("q_ar_max", "gate_ar_max", True, ".3g"),
          "size_ratio": ("q_size_ratio_max", "gate_size_ratio_max", True, ".4f")}


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


def Optim_failure_values(row: dict, settings: dict) -> str:
    """
    Why a run failed, in numbers: each failed gate's value against its limit, as the executor judged it
    ('skew 0.262 > 0.22, size_ratio 1.2502 > 1.25'), or the run's status when it never meshed.

    Args:
        row (dict): A run row, emitted or read back (its `failed_gates` a list or the stored JSON)
        settings (dict): The settings row it ran under

    Returns:
        str: The failed gates with their values, or the status
    """
    if row["status"] != "meshed":
        return row["status"]
    gates = row["failed_gates"] or []
    out = []
    for gate in json.loads(gates) if isinstance(gates, str) else gates:
        if gate in _WORST:
            column, limit, above, spec = _WORST[gate]
            out.append(f"{gate} {row[column]:{spec}} {'>' if above else '<'} {settings[limit]:g}")
        elif gate == "inverted":
            out.append(f"inverted: {row['q_neg_cells'] or 0} cells, jac min {row['q_jac_min']:.3f}")
        elif gate == "geo_dev":
            geo = row["g_max_mesh_to_input"]
            out.append("geo_dev unmeasured" if geo is None else
                       f"geo_dev {max(geo, row['g_max_input_to_mesh']):.2e} > {settings['gate_geo_dev_max']:g}")
        else:
            out.append(gate)
    return ", ".join(out)
