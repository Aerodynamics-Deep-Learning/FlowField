"""
The mesh study's run store, `meshing.sqlite`: the one module that reads and writes it.

    settings  one row per settings_id: every value a run is meshed and judged by (`settings.py`)
    ladders   one row per airfoil per campaign: its n*, route through the rungs, counts and outcome
    runs      one row per evaluation, meshed or not: config, sample, quality, deviation and verdict
    best      per (airfoil, settings), its fewest-node passing run; triggers keep it pointing at one

No mesh file is kept: a run's `config_json` re-meshes it, and `su2_sha256` checks the result.

Schema changes land as numbered migrations in `_MIGRATIONS`, tracked by `PRAGMA user_version`, as in the
geometry manifest. The file is committed, so it runs in rollback-journal mode. One process writes, and
every write commits at once, so a crash loses at most the row in flight. Timestamps are ISO-8601 UTC
with milliseconds (`utc_now`).
"""

import hashlib
import json
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.datagen.geometry.common.paths import REPO_ROOT
from src.datagen.meshing.optimizer.paths import MESHING_DB

# Never edit a released migration; append a new one
_MIGRATIONS = [
    """
    CREATE TABLE settings (
        settings_id TEXT PRIMARY KEY,       -- settings.build_settings: hash of every column but created_at, label
        created_at TEXT NOT NULL, label TEXT,
        ladder_version TEXT NOT NULL, backend TEXT NOT NULL, topology TEXT NOT NULL, design_re REAL NOT NULL,
        shape_version TEXT NOT NULL, sampler_version TEXT NOT NULL, mesher_sha256 TEXT NOT NULL,
        dx_max REAL NOT NULL, theta_max_deg REAL NOT NULL, precheck_margin REAL NOT NULL,
        te_gap REAL NOT NULL, cell_ceiling INTEGER NOT NULL,
        gate_skew_max REAL NOT NULL, gate_ortho_min REAL NOT NULL, gate_jac_min REAL NOT NULL,
        gate_ar_max REAL NOT NULL, gate_size_ratio_max REAL NOT NULL, gate_geo_dev_max REAL NOT NULL,
        golden_jmax INTEGER NOT NULL, jmax_bump_step INTEGER NOT NULL, jmax_bump_max INTEGER NOT NULL,
        rung2_budget INTEGER NOT NULL, rung3_budget INTEGER NOT NULL, max_bb_eval_factor INTEGER NOT NULL,
        nomad_seed INTEGER NOT NULL,
        nstar_json TEXT NOT NULL, search_space_json TEXT NOT NULL, pinned_json TEXT NOT NULL
    ) STRICT;

    CREATE TABLE ladders (
        ladder_id TEXT PRIMARY KEY,
        settings_id TEXT NOT NULL REFERENCES settings(settings_id),
        batch_id TEXT NOT NULL, geometry_id TEXT NOT NULL, airfoil_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'passed', 'failed', 'aborted')),
        final_rung TEXT, route TEXT, golden_failed_gates TEXT,  -- JSON lists
        n_star INTEGER, n_star_knobs_json TEXT, n_star_grid_evals INTEGER, n_star_s REAL,
        meshes_golden INTEGER, meshes_bump INTEGER, meshes_smoothing INTEGER, meshes_mads INTEGER,  -- c2d runs
        n_rejected INTEGER, n_dominated INTEGER,
        nomad_json TEXT,                    -- per NOMAD rung: run_flag, stop_reason, seed, budget, calls, mesher_runs, x0s
        best_run_id TEXT,                   -- this ladder's own fewest-node pass
        error_text TEXT, host TEXT, git_commit TEXT, git_dirty INTEGER,
        started_at TEXT NOT NULL, finished_at TEXT, wall_s REAL
    ) STRICT;

    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        settings_id TEXT NOT NULL REFERENCES settings(settings_id),
        batch_id TEXT NOT NULL,
        ladder_id TEXT REFERENCES ladders(ladder_id),  -- NULL for a manual run or a re-mesh
        geometry_id TEXT NOT NULL, airfoil_name TEXT NOT NULL,
        rung TEXT NOT NULL CHECK (rung IN ('golden', 'jmax_bump', 'smoothing', 'mads', 'bisect', 'manual')),
        seq INTEGER NOT NULL, parent_run_id TEXT REFERENCES runs(run_id),
        config_json TEXT,                   -- NULL only if the config did not validate
        config_schema_fp TEXT NOT NULL, cache_key TEXT,
        dx_mid REAL, le_ratio REAL, te_ratio REAL, cluster_width REAL, jmax INTEGER,
        asmt INTEGER, epsi REAL, epse REAL, funi REAL, alfa REAL,
        s_n_required INTEGER, s_rejection TEXT CHECK (s_rejection IN ('turn', 'points')),
        s_precheck_turn REAL, s_x_at_precheck_turn REAL, s_le_spacing REAL, s_te_spacing REAL,
        s_te_clamped INTEGER, s_te_cap_limit REAL, s_n_points INTEGER, s_max_panel REAL, s_min_panel REAL,
        s_max_panel_ratio REAL, s_max_turn_deg REAL, s_x_at_max_turn REAL, s_max_dev REAL,
        status TEXT NOT NULL
            CHECK (status IN ('rejected', 'dominated', 'meshed', 'mesher_failed', 'crashed', 'config_invalid')),
        entry_flag TEXT,                    -- the MeshExitFlag name, on constants.py's looser gates
        started_at TEXT, finished_at TEXT, wall_s REAL, host TEXT, git_commit TEXT, git_dirty INTEGER,
        num_nodes INTEGER, num_cells INTEGER, q_neg_cells INTEGER,
        q_skew_max REAL, q_ortho_min REAL, q_jac_min REAL, q_ar_max REAL, q_size_ratio_max REAL,
        -- per metric: median, the values 99% / 99.9% are at least as good as, count past the study's
        -- gate, and the worst value's x, y and distance to the wall
        q_skew_p50 REAL, q_skew_t99 REAL, q_skew_t999 REAL, q_skew_n_bad INTEGER,
        q_skew_x REAL, q_skew_y REAL, q_skew_dwall REAL,
        q_ortho_p50 REAL, q_ortho_t99 REAL, q_ortho_t999 REAL, q_ortho_n_bad INTEGER,
        q_ortho_x REAL, q_ortho_y REAL, q_ortho_dwall REAL,
        q_jac_p50 REAL, q_jac_t99 REAL, q_jac_t999 REAL, q_jac_n_bad INTEGER,
        q_jac_x REAL, q_jac_y REAL, q_jac_dwall REAL,
        q_sr_p50 REAL, q_sr_t99 REAL, q_sr_t999 REAL, q_sr_n_bad INTEGER,
        q_sr_x REAL, q_sr_y REAL, q_sr_dwall REAL,
        g_max_mesh_to_input REAL, g_max_input_to_mesh REAL, g_rms_mesh_to_input REAL, g_rms_input_to_mesh REAL,
        g_n_boundary_nodes INTEGER,
        passed INTEGER CHECK (passed IN (0, 1)),  -- NULL unless meshed
        failed_gates TEXT,                  -- JSON list
        su2_sha256 TEXT, su2_bytes INTEGER, mesher_log TEXT, error_text TEXT
    ) STRICT;

    CREATE INDEX runs_by_airfoil ON runs (geometry_id, settings_id);
    CREATE INDEX runs_by_cache_key ON runs (cache_key);
    CREATE INDEX runs_by_ladder ON runs (ladder_id);

    CREATE TABLE best (
        geometry_id TEXT NOT NULL,
        settings_id TEXT NOT NULL REFERENCES settings(settings_id),
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        chosen_at TEXT NOT NULL,
        PRIMARY KEY (geometry_id, settings_id)
    ) STRICT;

    CREATE TRIGGER best_insert_must_pass BEFORE INSERT ON best
    WHEN NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = NEW.run_id AND r.passed = 1
                     AND r.geometry_id = NEW.geometry_id AND r.settings_id = NEW.settings_id)
    BEGIN SELECT RAISE(ABORT, 'best must point at a passed run of the same airfoil and settings'); END;

    CREATE TRIGGER best_update_must_pass BEFORE UPDATE ON best
    WHEN NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = NEW.run_id AND r.passed = 1
                     AND r.geometry_id = NEW.geometry_id AND r.settings_id = NEW.settings_id)
    BEGIN SELECT RAISE(ABORT, 'best must point at a passed run of the same airfoil and settings'); END;

    CREATE VIEW v_best AS
    SELECT b.geometry_id, r.airfoil_name, b.settings_id, r.run_id, r.rung, r.num_nodes, r.s_n_points, r.jmax,
           r.dx_mid, r.le_ratio, r.te_ratio, r.cluster_width, r.q_skew_max, r.q_ortho_min, r.q_jac_min,
           r.q_size_ratio_max, r.su2_sha256, r.config_json
    FROM best b JOIN runs r USING (run_id);

    CREATE VIEW v_airfoil_outcome AS
    SELECT l.geometry_id, l.airfoil_name, l.settings_id, l.status, l.final_rung, l.route, l.n_star,
           l.meshes_golden + l.meshes_bump + l.meshes_smoothing + l.meshes_mads AS meshes,
           rb.num_nodes AS best_nodes, rb.jmax AS best_jmax
    FROM ladders l
    LEFT JOIN best b ON b.geometry_id = l.geometry_id AND b.settings_id = l.settings_id
    LEFT JOIN runs rb ON rb.run_id = b.run_id
    WHERE l.started_at = (SELECT MAX(started_at) FROM ladders l2
                          WHERE l2.geometry_id = l.geometry_id AND l2.settings_id = l.settings_id);
    """,
]

# The runs table's columns, in its order
RUN_COLUMNS = (
    "run_id", "settings_id", "batch_id", "ladder_id", "geometry_id", "airfoil_name", "rung", "seq",
    "parent_run_id", "config_json", "config_schema_fp", "cache_key",
    "dx_mid", "le_ratio", "te_ratio", "cluster_width", "jmax", "asmt", "epsi", "epse", "funi", "alfa",
    "s_n_required", "s_rejection", "s_precheck_turn", "s_x_at_precheck_turn", "s_le_spacing", "s_te_spacing",
    "s_te_clamped", "s_te_cap_limit", "s_n_points", "s_max_panel", "s_min_panel", "s_max_panel_ratio",
    "s_max_turn_deg", "s_x_at_max_turn", "s_max_dev",
    "status", "entry_flag", "started_at", "finished_at", "wall_s", "host", "git_commit", "git_dirty",
    "num_nodes", "num_cells", "q_neg_cells",
    "q_skew_max", "q_ortho_min", "q_jac_min", "q_ar_max", "q_size_ratio_max",
    *(f"q_{m}_{f}" for m in ("skew", "ortho", "jac", "sr") for f in ("p50", "t99", "t999", "n_bad", "x", "y", "dwall")),
    "g_max_mesh_to_input", "g_max_input_to_mesh", "g_rms_mesh_to_input", "g_rms_input_to_mesh", "g_n_boundary_nodes",
    "passed", "failed_gates", "su2_sha256", "su2_bytes", "mesher_log", "error_text",
)

# Ties among equally small passes go to the cheapest rung that found one
_BEST_SQL = """
    SELECT run_id FROM runs
    WHERE geometry_id = ? AND settings_id = ? AND passed = 1 AND rung NOT IN ('manual', 'bisect')
    ORDER BY num_nodes,
             CASE rung WHEN 'golden' THEN 0 WHEN 'jmax_bump' THEN 1 WHEN 'smoothing' THEN 2 ELSE 3 END,
             started_at
    LIMIT 1
"""

#region main

class RunStore:
    """An open run store. Use as a context manager, or `close()` it."""

    def __init__(self, conn: sqlite3.Connection, path: Path):
        self._conn = conn
        self.path = path
        self._columns = {table: [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                         for table in ("settings", "ladders", "runs")}

    @classmethod
    def open(cls, db=None, read_only: bool = False) -> "RunStore":
        """
        Opens the store, creating or migrating it as needed.

        Args:
            db (str | Path | None): The database file, the repo's own (`paths.MESHING_DB`) by default
            read_only (bool): Open an existing, current database without writing anything

        Returns:
            RunStore: The open store

        Raises:
            RuntimeError: If the database was written by newer code (a later schema version), or a
                read-only open finds one that needs migrating
            FileNotFoundError: If a read-only open finds no database
        """
        path = Path(db) if db is not None else MESHING_DB
        if read_only:
            if not path.is_file():
                raise FileNotFoundError(f"{path} does not exist")
            conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != len(_MIGRATIONS):
                conn.close()
                raise RuntimeError(f"{path} is at schema version {version}, this code at {len(_MIGRATIONS)}; "
                                   f"open it for writing once to migrate it")
            return cls(conn, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None)  # transactions are explicit
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = DELETE")
        try:
            _migrate(conn, path)
            return cls(conn, path)
        except BaseException:
            conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        """All writes inside commit together or not at all. Nests: an inner block joins the outer one."""
        if self._conn.in_transaction:
            yield
            return
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

#endregion

#region writes

    def ensure_settings(self, values: dict, label: Optional[str] = None) -> str:
        """
        Records a settings row unless its id is already there; a repeat keeps the first label and time.

        Args:
            values (dict): `settings.build_settings()`'s row
            label (str | None): A note for people, not part of the id

        Returns:
            str: The settings_id

        Raises:
            ValueError: If `values` is not exactly the settings columns but created_at and label
        """
        expected = set(self._columns["settings"]) - {"created_at", "label"}
        if set(values) != expected:
            raise ValueError(f"settings row is off: missing {sorted(expected - set(values))}, "
                             f"unknown {sorted(set(values) - expected)}")
        row = {**values, "created_at": utc_now(), "label": label}
        with self.transaction():
            self._insert("settings", row, on_conflict="ON CONFLICT(settings_id) DO NOTHING")
        return values["settings_id"]

    def start_ladder(self, row: dict) -> None:
        """Records a ladder as it starts (status 'running'), before any of its runs."""
        with self.transaction():
            self._insert("ladders", row)

    def finish_ladder(self, ladder_id: str, fields: dict) -> None:
        """
        Writes a ladder's outcome onto its row.

        Raises:
            ValueError: On an unknown column, or a ladder that was never started
        """
        _check_columns("ladders", fields, self._columns["ladders"])
        if "ladder_id" in fields:
            raise ValueError("a ladder's id cannot change")
        sets = ", ".join(f"{name} = ?" for name in fields)
        with self.transaction():
            cur = self._conn.execute(f"UPDATE ladders SET {sets} WHERE ladder_id = ?",
                                     [_sql(v) for v in fields.values()] + [ladder_id])
            if cur.rowcount != 1:
                raise ValueError(f"no ladder {ladder_id}")

    def insert_run(self, row: dict) -> None:
        """
        Records one evaluation. Columns left out are NULL; lists and dicts are stored as JSON.

        Raises:
            ValueError: On a column the runs table does not have
        """
        with self.transaction():
            self._insert("runs", row)

    def update_best(self, geometry_id: str, settings_id: str) -> Optional[str]:
        """
        Points `best` at the airfoil's fewest-node passing run under these settings: ties go to golden,
        then jmax_bump, smoothing, mads, then the earliest; manual and bisect runs never count.

        Returns:
            str | None: The chosen run_id, or None (and no row) if nothing has passed
        """
        with self.transaction():
            hit = self._conn.execute(_BEST_SQL, (geometry_id, settings_id)).fetchone()
            if hit is None:
                return None
            self._conn.execute(
                "INSERT INTO best (geometry_id, settings_id, run_id, chosen_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(geometry_id, settings_id) DO UPDATE SET run_id = excluded.run_id, "
                "chosen_at = excluded.chosen_at WHERE run_id != excluded.run_id",
                (geometry_id, settings_id, hit[0], utc_now()))
        return hit[0]

    def abort_running(self) -> int:
        """
        Marks every ladder still 'running' as aborted, as a campaign starts: one left running was cut
        short. Its runs stay; the campaign restarts that airfoil from scratch.

        Returns:
            int: How many ladders it aborted
        """
        with self.transaction():
            cur = self._conn.execute(
                "UPDATE ladders SET status = 'aborted', finished_at = ?, "
                "error_text = COALESCE(error_text, 'still running when a campaign started') "
                "WHERE status = 'running'", (utc_now(),))
        return cur.rowcount

#endregion

#region reads

    def settings(self, settings_id: str) -> Optional[dict]:
        """A settings row by column name, or None."""
        return self._one("SELECT * FROM settings WHERE settings_id = ?", settings_id)

    def ladder(self, ladder_id: str) -> Optional[dict]:
        """A ladder row by column name, or None."""
        return self._one("SELECT * FROM ladders WHERE ladder_id = ?", ladder_id)

    def run(self, run_id: str) -> Optional[dict]:
        """A run row by column name, JSON columns as stored, or None."""
        return self._one("SELECT * FROM runs WHERE run_id = ?", run_id)

    def best(self, geometry_id: str, settings_id: str) -> Optional[str]:
        """The airfoil's best run_id under these settings, or None."""
        hit = self._conn.execute("SELECT run_id FROM best WHERE geometry_id = ? AND settings_id = ?",
                                 (geometry_id, settings_id)).fetchone()
        return hit[0] if hit else None

    def finished_airfoils(self, settings_id: str) -> set[str]:
        """The airfoils with a ladder that passed or failed under these settings: a resume skips them."""
        return {r[0] for r in self._conn.execute(
            "SELECT DISTINCT geometry_id FROM ladders WHERE settings_id = ? AND status IN ('passed', 'failed')",
            (settings_id,))}

#endregion

#region helpers

    def _insert(self, table: str, row: dict, on_conflict: str = "") -> None:
        _check_columns(table, row, self._columns[table])
        names = ", ".join(row)
        marks = ", ".join("?" * len(row))
        self._conn.execute(f"INSERT INTO {table} ({names}) VALUES ({marks}) {on_conflict}",
                           [_sql(v) for v in row.values()])

    def _one(self, sql: str, *params) -> Optional[dict]:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return None if row is None else dict(zip((d[0] for d in cur.description), row))


def _migrate(conn: sqlite3.Connection, path: Path) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(_MIGRATIONS):
        raise RuntimeError(f"{path} is at schema version {version}; this code knows up to "
                           f"{len(_MIGRATIONS)}. Update the code before opening it.")
    if 0 < version < len(_MIGRATIONS):
        with sqlite3.connect(path.with_name(f"{path.name}.bak-v{version}")) as backup:
            conn.backup(backup)
    for n in range(version, len(_MIGRATIONS)):
        try:
            conn.executescript(f"BEGIN;\n{_MIGRATIONS[n]}\nPRAGMA user_version = {n + 1};\nCOMMIT;")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


def _check_columns(table: str, row: dict, columns: list[str]) -> None:
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ValueError(f"{table} has no column {', '.join(unknown)}")


def _sql(value):
    """A value as SQLite stores it: numpy scalars as Python's, bools as 0/1, lists and dicts as JSON."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def utc_now() -> str:
    """Now, as ISO-8601 UTC with milliseconds: '2026-10-28T14:03:12.481Z'."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def su2_sha256(path) -> tuple[str, int]:
    """
    A `.su2`'s SHA-256 with CRLF folded to LF, so a re-mesh on any platform checks against it, and its
    size in bytes as written.
    """
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(), len(data)


def git_state() -> tuple[str, Optional[int]]:
    """
    The commit the code runs at, and whether `src/datagen` differs from it, untracked files included
    (the store itself lives under `data/`, so writing it never counts). ("unknown", None) outside git.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--", "src/datagen"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit, int(bool(changes))

#endregion
