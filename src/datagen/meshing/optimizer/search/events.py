"""
How a ladder reports, kept apart from the ladder so that the runner, and with it a slot's worker, load neither
NOMAD nor the meshing stack:

    LadderOutcome         how a ladder ended, with the error that ended it, for the caller to act on
    Optim_record          one ladder event written to the run store, by the campaign's one writer
    Optim_failure_values  why a run failed, in numbers against its gates, for a batch's log
    Optim_gate_value      a gate's measured value on a run
    Optim_violation       a run's worst gate violation, as rung 2's search scores it
    Optim_ladder_line     how a ladder ended and where each rung before stopped short, for a batch's log
"""

import json
import math
from typing import NamedTuple, Optional, Sequence

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
            out.append(f"inverted: {row['q_neg_cells'] or 0} cells (jac min {row['q_jac_min']:.3f})")
        elif gate == "geo_dev":
            geo = Optim_gate_value(row, "geo_dev")
            out.append("geo_dev unmeasured" if geo is None else f"geo_dev {geo:.2e} > {settings['gate_geo_dev_max']:g}")
        else:
            out.append(gate)
    return ", ".join(out)


def Optim_gate_value(row: dict, gate: str) -> Optional[float]:
    """A gate's measured value on a run: its worst cell's, or the worse deviation for geo_dev; None if unmeasured."""
    if gate == "geo_dev":
        geo = row.get("g_max_mesh_to_input")
        return None if geo is None else max(geo, row["g_max_input_to_mesh"])
    return row.get(_WORST[gate][0]) if gate in _WORST else None


def Optim_failures_by_gate(runs: Sequence[tuple[str, dict]]) -> list[str]:
    """
    Failed runs counted by gate, most often failed first: each gate's count, its range and the airfoil with the
    worst value ('skew 21 (0.250-0.271, worst e376)'); a gate with no value (inverted, a mesher's status), its count.

    Args:
        runs (Sequence[tuple[str, dict]]): (airfoil, run row) for each failed run
    """
    by_gate: dict[str, list[tuple[Optional[float], str]]] = {}
    for name, row in runs:
        gates = row["failed_gates"] if row["status"] == "meshed" else [row["status"]]
        for gate in json.loads(gates) if isinstance(gates, str) else gates or []:
            by_gate.setdefault(gate, []).append((Optim_gate_value(row, gate), name))
    out = []
    for gate, hits in sorted(by_gate.items(), key=lambda item: -len(item[1])):
        values = [(v, n) for v, n in hits if v is not None]
        if not values:
            out.append(f"{gate} {len(hits)}")
            continue
        spec = _WORST[gate][3] if gate in _WORST else ".2e"
        worst = (max if gate not in _WORST or _WORST[gate][2] else min)(values)
        lo, hi = min(v for v, _ in values), max(v for v, _ in values)
        out.append(f"{gate} {len(hits)} ({lo:{spec}}-{hi:{spec}}, worst {worst[1]})")
    return out


def Optim_violation(row: dict, settings: dict) -> Optional[float]:
    """
    A run's worst gate violation as rung 2's search scores it (`executor.Optim_nomad_outputs`, mirrored here
    since the executor loads the meshers): positive past a gate, so the smallest is the closest to passing.

    Args:
        row (dict): A run row
        settings (dict): The settings row it ran under

    Returns:
        float | None: The worst violation; None for a run without the quality measured
    """
    if row["status"] != "meshed" or row.get("q_skew_max") is None:
        return None
    s, geo = settings, Optim_gate_value(row, "geo_dev")
    return max(row["q_skew_max"] / s["gate_skew_max"] - 1, 1 - row["q_ortho_min"] / s["gate_ortho_min"],
               1 - row["q_jac_min"] / s["gate_jac_min"], math.log(row["q_ar_max"] / s["gate_ar_max"]),
               math.log(row["q_size_ratio_max"] / s["gate_size_ratio_max"]),
               1e3 if geo is None else geo / s["gate_geo_dev_max"] - 1)


# The rungs as a log names them
_RUNG = {"golden": "the golden run", "jmax_bump": "rung 1", "smoothing": "rung 2", "mads": "rung 3"}


def Optim_ladder_line(outcome: LadderOutcome, runs: Sequence[dict], settings: dict) -> str:
    """
    How a ladder ended, for a batch's log: where it passed or stopped, its best and c2d runs, then where each
    rung before stopped short (the golden run only when it went straight to rung 3, the bumps only when they
    ran out, rungs 2 and 3 by their closest mesh).

    Args:
        outcome (LadderOutcome): How it ended
        runs (Sequence[dict]): Its runs, in order
        settings (dict): The settings row it ran under

    Returns:
        str: E.g. 'passed in rung 3, 68,820 nodes, 120 c2d runs; rung 2: no pass in 50 meshes at jmax 148,
            closest size_ratio 1.2513 > 1.25'
    """
    route = outcome.route
    if route == ["nstar"]:
        return "failed: n* accepted no grid point"
    ran = [r for r in runs if r.get("entry_flag") is not None]
    last = next((step for step in reversed(route) if any(r["rung"] == step for r in ran)), route[-1] if route else "-")
    text = f"passed in {_RUNG.get(last, last)}" if outcome.status == "passed" else \
        f"{outcome.status} after {_RUNG.get(last, last)}"
    if outcome.best_row and outcome.best_row.get("num_nodes"):
        text += f", {outcome.best_row['num_nodes']:,} nodes"
    text += f", {len(ran)} c2d runs"
    for i, step in enumerate(route):
        rows = [r for r in ran if r["rung"] == step]
        handed_to = route[i + 1] if i + 1 < len(route) else None
        if not rows or any(r.get("passed") for r in rows) or (step in ("golden", "jmax_bump") and handed_to != "mads"):
            continue
        text += f"; {_stopped(step, rows, settings)}"
    return text


def _stopped(step: str, rows: list[dict], settings: dict) -> str:
    """Where one rung stopped short: the golden run's failure, the last bump's, or a NOMAD rung's closest mesh."""
    if step == "golden":
        return f"the golden run: {Optim_failure_values(rows[0], settings)}"
    if step == "jmax_bump":
        return f"rung 1: no pass by jmax {rows[-1]['jmax']}, {Optim_failure_values(rows[-1], settings)}"
    scored = [(v, r) for r in rows if (v := Optim_violation(r, settings)) is not None]
    closest = f"closest {Optim_failure_values(min(scored, key=lambda p: p[0])[1], settings)}" if scored else "none meshed"
    at = f" at jmax {rows[0]['jmax']}" if step == "smoothing" else ""
    return f"{_RUNG.get(step, step)}: no pass in {len(rows)} mesh{'es' if len(rows) != 1 else ''}{at}, {closest}"
