"""
The mesh study's executor: one config in, one run row out. Everything that meshes for the study (the
ladder, NOMAD, the calibration) goes through `RunExecutor.run`.

    RunContext       what every run of a batch shares: settings, ids, provenance, the work root
    RunExecutor      one per ladder: samples, meshes, scores and judges a config, and emits its row
    RunResult        that row, and whether it came from the ladder's cache
    failed_gates     the study's verdict on a meshed row
    nomad_outputs    what NOMAD is told about a row
    StudySetupError  the study's setup is broken (no c2d, a bad input), not the config

The executor never writes the store: `emit` hands each row on, to the campaign's single writer or to a
list in tests. A row holds only the runs table's columns, and no mesh file outlives its run.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import socket
import time
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple, Optional

from pydantic import ValidationError

from src.datagen.geometry.sample.contour import Contour
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.entry import Common_GenerateMesh
from src.datagen.meshing.common.schemas import MeshBackend, MeshExitFlag, MeshTopology, QualityLimits
from src.datagen.meshing.optimizer.constants import CHORD, PINNED
from src.datagen.meshing.optimizer.sampling import sample_for_mesher
from src.datagen.meshing.optimizer.settings import config_schema_fp
from src.datagen.meshing.optimizer.store import git_state, su2_sha256, utc_now
from src.datagen.schemas import Freestream

# The gates a meshed row can fail, in the order `failed_gates` lists them
GATE_ORDER = ("conversion", "inverted", "skew", "ortho", "jac", "aspect_ratio", "size_ratio", "geo_dev")

# Mesher flags that mean the study's setup is broken, not the config
_SETUP_FLAGS = {MeshExitFlag.EXECUTABLE_NOT_FOUND, MeshExitFlag.INPUT_TENSOR_FAIL, MeshExitFlag.INPUT_CLOSURE_FAIL,
                MeshExitFlag.INPUT_FREESTREAM_FAIL, MeshExitFlag.INPUT_UNKNOWN_SOLVER}

# The settings a mesh depends on, part of every cache key
_MESH_AFFECTING = ("backend", "topology", "design_re", "shape_version", "sampler_version", "mesher_sha256",
                   "te_gap", "cell_ceiling", "dx_max", "theta_max_deg", "precheck_margin")

# Outcomes a retry cannot change; dominated depends on the best so far, a crash may not recur
_CACHEABLE = {"meshed", "rejected", "mesher_failed"}

# The meshing code logs some failures' reasons instead of returning them; listened to while a mesh runs
_MESHING_LOGGER = "src.datagen.meshing"

_KNOB_COLUMNS = ("dx_mid", "le_ratio", "te_ratio", "cluster_width", "jmax", "asmt", "epsi", "epse", "funi", "alfa")
_REPORT_FIELDS = ("n_required", "rejection", "precheck_turn", "x_at_precheck_turn", "le_spacing", "te_spacing",
                  "te_clamped", "te_cap_limit", "n_points", "max_panel", "min_panel", "max_panel_ratio",
                  "max_turn_deg", "x_at_max_turn", "max_dev")
_QUALITY_COLUMNS = {"ncell": "num_cells", "neg": "q_neg_cells", "max_skew": "q_skew_max", "min_ortho": "q_ortho_min",
                    "min_jac": "q_jac_min", "max_ar": "q_ar_max", "max_size_ratio": "q_size_ratio_max"}
_DISTRIBUTIONS = {"skew_dist": "skew", "ortho_dist": "ortho", "jac_dist": "jac", "size_ratio_dist": "sr"}
_DIST_FIELDS = ("p50", "t99", "t999", "n_bad", "x", "y", "dwall")
_GEO_COLUMNS = {"max_dev_mesh_to_input": "g_max_mesh_to_input", "max_dev_input_to_mesh": "g_max_input_to_mesh",
                "rms_dev_mesh_to_input": "g_rms_mesh_to_input", "rms_dev_input_to_mesh": "g_rms_input_to_mesh",
                "n_boundary_nodes": "g_n_boundary_nodes"}


class StudySetupError(RuntimeError):
    """The study cannot mesh anything as set up (no c2d, an input the gates refuse): abort the ladder."""


class _Captured(logging.Handler):
    """The meshing code's warnings and errors, tracebacks included, while one mesh runs."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class RunResult(NamedTuple):
    """A run's row, and whether it came from the ladder's cache (no new row, no budget spent)."""
    row: dict
    cached: bool


@dataclass(frozen=True)
class RunContext:
    """What every run of a batch shares. Picklable, so a pool's workers can each take one."""
    settings: dict
    batch_id: str
    host: str
    git_commit: str
    git_dirty: Optional[int]
    config_schema_fp: str
    work_root: str

    @classmethod
    def create(cls, settings: dict, work_root, batch_id: Optional[str] = None) -> "RunContext":
        """
        A batch's context, with its host, git state and config fingerprint taken now.

        Args:
            settings (dict): `settings.build_settings()`'s row
            work_root (str | Path): The folder each run meshes in a subfolder of
            batch_id (str | None): The batch's id, a new uuid4 if None
        """
        commit, dirty = git_state()
        return cls(settings=settings, batch_id=batch_id or str(uuid.uuid4()), host=socket.gethostname(),
                   git_commit=commit, git_dirty=dirty, config_schema_fp=config_schema_fp(), work_root=str(work_root))


class RunExecutor:
    """Runs one ladder's configs; its cache of finished configs lives as long as it does."""

    def __init__(self, ctx: RunContext, emit: Callable[[dict], None]):
        self.ctx = ctx
        self.emit = emit
        self._cache: dict[str, dict] = {}
        s = ctx.settings
        self._limits = QualityLimits(skew_max=s["gate_skew_max"], ortho_min=s["gate_ortho_min"],
                                     jac_min=s["gate_jac_min"], size_ratio_max=s["gate_size_ratio_max"])
        self._backend, self._topology = MeshBackend(s["backend"]), MeshTopology(s["topology"])

    def run(self, *, ladder_id: Optional[str], geometry_id: str, airfoil_name: str, contour: Contour, knobs: dict,
            rung: str, seq: int, parent_run_id: Optional[str] = None,
            skip_if_nodes_ge: Optional[int] = None) -> RunResult:
        """
        Samples, meshes, scores and judges one config, and emits its row; a config this executor has
        already finished returns that row instead, emitting nothing.

        Args:
            ladder_id (str | None): The ladder's id; None for a manual run
            geometry_id (str): The airfoil's id in the geometry manifest
            airfoil_name (str): Its name; the mesher's files take a filesystem-safe copy
            contour (Contour): The study's contour, from `sampling.build_study_contour`
            knobs (dict): The config's searched fields; `PINNED` fills the rest
            rung (str): 'golden', 'jmax_bump', 'smoothing', 'mads', 'bisect' or 'manual'
            seq (int): The run's place in its rung
            parent_run_id (str | None): The run this one follows from
            skip_if_nodes_ge (int | None): Don't mesh a config with at least this many nodes (dominated)

        Returns:
            RunResult: The row, and whether it was cached

        Raises:
            StudySetupError: On a setup flag from the mesher, after emitting the run as crashed
        """
        ctx = self.ctx
        row = dict(run_id=str(uuid.uuid4()), settings_id=ctx.settings["settings_id"], batch_id=ctx.batch_id,
                   ladder_id=ladder_id, geometry_id=geometry_id, airfoil_name=airfoil_name, rung=rung, seq=seq,
                   parent_run_id=parent_run_id, config_schema_fp=ctx.config_schema_fp, started_at=utc_now(),
                   host=ctx.host, git_commit=ctx.git_commit, git_dirty=ctx.git_dirty)
        t0 = time.perf_counter()
        try:
            config = C2D_MeshingConfig(**PINNED, **knobs)
        except ValidationError as e:
            # Knob columns stay NULL: STRICT tables would refuse the very values pydantic did
            asked = json.dumps(knobs, sort_keys=True, default=str)
            row.update(status="config_invalid", error_text=f"knobs: {asked}\n{e}")
            return self._finish(row, t0)

        values = config.model_dump()
        row["config_json"] = json.dumps(values, sort_keys=True)
        row.update({name: values[name] for name in _KNOB_COLUMNS})
        row["cache_key"] = self._cache_key(geometry_id, values)
        if row["cache_key"] in self._cache:
            return RunResult(self._cache[row["cache_key"]], cached=True)

        work_dir = os.path.join(ctx.work_root, f"{_safe(airfoil_name)}_{row['run_id'][:8]}")
        setup_error = None
        try:
            os.makedirs(work_dir)
            self._mesh(row, config, contour, airfoil_name, work_dir, skip_if_nodes_ge)
        except StudySetupError as e:
            setup_error = e
            row.update(status="crashed", error_text=f"setup: {e}")
        except Exception:
            row.update(status="crashed", error_text=traceback.format_exc())
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        result = self._finish(row, t0)
        if setup_error is not None:
            raise setup_error
        return result

    def _mesh(self, row: dict, config: C2D_MeshingConfig, contour: Contour, airfoil_name: str, work_dir: str,
              skip_if_nodes_ge: Optional[int]) -> None:
        """Fills the row from sampling on; leaves `status` set unless something raises."""
        # c2d reads only Re; Freestream requires an alpha and a mach regardless
        freestream = Freestream(alpha=0.0, Re=self.ctx.settings["design_re"], mach=0.5)
        sampled = sample_for_mesher(config, contour, airfoil_name=_safe(airfoil_name), chord=CHORD,
                                    freestream=freestream, backend=self._backend, topology=self._topology,
                                    working_dir=work_dir, count_limits=self._limits, write_vtk=False)
        report = sampled.report
        for name in _REPORT_FIELDS:
            value = getattr(report, name)
            row[f"s_{name}"] = value.value if isinstance(value, Enum) else value
        if sampled.mesh_in is None:
            row["status"] = "rejected"
            return
        if skip_if_nodes_ge is not None and (report.n_points - 1) * config.jmax >= skip_if_nodes_ge:
            row["status"] = "dominated"
            return

        # One mesh at a time per process, so what the handler hears is this run's
        captured, meshing_logger = _Captured(), logging.getLogger(_MESHING_LOGGER)
        meshing_logger.addHandler(captured)
        try:
            out = Common_GenerateMesh(sampled.mesh_in)
        finally:
            meshing_logger.removeHandler(captured)
        row["entry_flag"] = out.flag.name
        row["mesher_log"] = _read_verbose(out.verbose_list, 0)
        if out.flag is MeshExitFlag.FATAL_ERROR:
            row["error_text"] = _read_verbose(out.verbose_list, 2)  # the runner's exception, before the folder goes
        if out.flag in _SETUP_FLAGS:
            raise StudySetupError("\n".join([out.flag.name, *captured.lines]))
        if out.quality is None or not out.mesh_path or not os.path.isfile(out.mesh_path):
            row["status"] = "mesher_failed"
            row["error_text"] = "\n".join(filter(None, [row.get("error_text"), *captured.lines])) or None
            return

        quality, geo = out.quality, out.geo_dev
        row.update(status="meshed", num_nodes=out.num_nodes)
        row.update({column: getattr(quality, name) for name, column in _QUALITY_COLUMNS.items()})
        for name, short in _DISTRIBUTIONS.items():
            dist = getattr(quality, name)
            if dist is not None:
                row.update({f"q_{short}_{f}": getattr(dist, f) for f in _DIST_FIELDS})
        if geo is not None:
            row.update({column: getattr(geo, name) for name, column in _GEO_COLUMNS.items()})
        row["su2_sha256"], row["su2_bytes"] = su2_sha256(out.mesh_path)
        failed = failed_gates(row, self.ctx.settings)
        row.update(passed=int(not failed), failed_gates=failed)

    def _cache_key(self, geometry_id: str, config: dict) -> str:
        key = {**{name: self.ctx.settings[name] for name in _MESH_AFFECTING}, "geometry_id": geometry_id, "config": config}
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    def _finish(self, row: dict, t0: float) -> RunResult:
        row.update(wall_s=time.perf_counter() - t0, finished_at=utc_now())
        if row["status"] in _CACHEABLE:
            self._cache[row["cache_key"]] = row
        self.emit(row)
        return RunResult(row, cached=False)


def failed_gates(row: dict, settings: dict) -> list[str]:
    """
    The gates a meshed row fails, in `GATE_ORDER`, against the settings row's limits: a conversion
    flag, inverted cells, the five worst-cell gates, and geometric deviation (missing counts as failed).
    """
    geo = _geo_dev(row)
    failing = {
        "conversion": row.get("entry_flag") == MeshExitFlag.CONVERSION_FAIL.name,
        "inverted": (row.get("q_neg_cells") or 0) > 0 or row["q_jac_min"] <= 0,
        "skew": row["q_skew_max"] > settings["gate_skew_max"],
        "ortho": row["q_ortho_min"] < settings["gate_ortho_min"],
        "jac": row["q_jac_min"] < settings["gate_jac_min"],
        "aspect_ratio": row["q_ar_max"] > settings["gate_ar_max"],
        "size_ratio": row["q_size_ratio_max"] > settings["gate_size_ratio_max"],
        "geo_dev": geo is None or geo > settings["gate_geo_dev_max"],
    }
    return [gate for gate in GATE_ORDER if failing[gate]]


def nomad_outputs(row: dict, objective: str, settings: dict) -> Optional[list[float]]:
    """
    What NOMAD is told about a row, for `BB_OUTPUT_TYPE OBJ EB EB PB PB PB PB PB PB`.

    The two EBs reject a point outright when positive: the turn pre-check, and n * jmax over the cell
    ceiling, minus 1. The six PBs are the gates, positive when violated: skew, orthogonality and the
    Jacobian as fractions of their gate, aspect and size ratio as logs, deviation as a fraction (1e3
    when unmeasured). OBJ is the node count, (n - 1) * jmax, or the worst PB.

    Args:
        row (dict): A run row
        objective (str): "nodes" or "violation"
        settings (dict): The settings row it ran under

    Returns:
        list[float] | None: OBJ, EB, EB, then the six PBs (all zero for a rejected row); None for a
            failed evaluation: any status but meshed or rejected, or a mesh that failed the conversion
            gate, which no PB measures

    Raises:
        ValueError: On an unknown objective
    """
    if objective not in ("nodes", "violation"):
        raise ValueError(f"objective is 'nodes' or 'violation', not {objective!r}")
    status = row["status"]
    if status not in ("meshed", "rejected") or row.get("entry_flag") == MeshExitFlag.CONVERSION_FAIL.name:
        return None
    s = settings
    jmax = row["jmax"]
    eb = [row["s_precheck_turn"], row["s_n_required"] * jmax / s["cell_ceiling"] - 1]
    if status == "rejected":
        pb = [0.0] * 6
    else:
        geo = _geo_dev(row)
        pb = [row["q_skew_max"] / s["gate_skew_max"] - 1,
              1 - row["q_ortho_min"] / s["gate_ortho_min"],
              1 - row["q_jac_min"] / s["gate_jac_min"],
              math.log(row["q_ar_max"] / s["gate_ar_max"]),
              math.log(row["q_size_ratio_max"] / s["gate_size_ratio_max"]),
              1e3 if geo is None else geo / s["gate_geo_dev_max"] - 1]
    n = row["s_n_points"] or row["s_n_required"]
    obj = (n - 1) * jmax if objective == "nodes" else max(pb)
    return [float(v) for v in (obj, *eb, *pb)]


def _geo_dev(row: dict) -> Optional[float]:
    """The worse of the two deviation directions, or None if unmeasured."""
    if row.get("g_max_mesh_to_input") is None:
        return None
    return max(row["g_max_mesh_to_input"], row["g_max_input_to_mesh"])


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _read_verbose(verbose_list: list, index: int) -> Optional[str]:
    """
    A file the mesher listed in `verbose_list`: c2d puts its log first and, on FATAL_ERROR, its
    exception file third. None if that slot is missing, empty, or names no file.
    """
    path = verbose_list[index] if len(verbose_list) > index else None
    if not path or not os.path.isfile(path):
        return None
    with open(path, errors="replace") as f:
        return f.read()
