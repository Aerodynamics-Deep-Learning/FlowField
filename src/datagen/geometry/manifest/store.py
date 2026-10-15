"""
The geometry manifest, `geometry.sqlite`: the one module that reads and writes it.

Schema changes land as numbered migrations in `_MIGRATIONS`, tracked by `PRAGMA user_version`; an
existing database is backed up next to itself before it is migrated. The file is committed, so it
runs in rollback-journal mode: between writes it is the only file in its folder.

Rejected files are kept in as much detail as accepted ones, in `rejected_shapes`, apart from the
accepted tables so nothing that reads shapes can pick one up by mistake; the `geometry_outcomes`
view puts both together under a `status` column.

The manifest holds shapes (and rejections) of the current shape version only. Older ones move to
`geometry_archive.sqlite` next to it, one JSON row each, so the archive never needs migrating; the
move is one transaction across both files.

What each table and column means is in the README next to the database.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.datagen.geometry.common.paths import ARCHIVE_NAME, Geom_paths
from src.datagen.geometry.common.schemas import (
    AddResult, CSTParams, GeometryRecord, ItemStatus, RejectionRecord, ShapeRecord,
)

# Never edit a released migration; append a new one
_MIGRATIONS = [
    """
    CREATE TABLE ingests (
        ingest_id      TEXT PRIMARY KEY,
        started_at     TEXT NOT NULL,
        source_root    TEXT NOT NULL,
        shape_version  TEXT NOT NULL,
        git_commit     TEXT NOT NULL,
        report_path    TEXT NOT NULL
    ) STRICT;

    CREATE TABLE geometries (
        geometry_id    TEXT PRIMARY KEY,
        name           TEXT NOT NULL UNIQUE COLLATE NOCASE,
        family         TEXT,
        family_params  TEXT,
        raw_path       TEXT NOT NULL,
        raw_hash       TEXT NOT NULL,
        raw_points     INTEGER NOT NULL,
        ingest_id      TEXT NOT NULL REFERENCES ingests (ingest_id)
    ) STRICT;

    CREATE TABLE geometry_shapes (
        geometry_id    TEXT NOT NULL REFERENCES geometries (geometry_id),
        shape_version  TEXT NOT NULL,
        cst            TEXT NOT NULL,
        rotation_deg   REAL NOT NULL,
        closure        TEXT NOT NULL,
        root_te_gap    REAL NOT NULL,
        source_dev     REAL NOT NULL,
        fit_max        REAL NOT NULL,
        te_wedge_deg   REAL NOT NULL,
        norm_path      TEXT NOT NULL,
        norm_hash      TEXT NOT NULL,
        PRIMARY KEY (geometry_id, shape_version)
    ) STRICT;
    CREATE INDEX geometry_shapes_norm_hash ON geometry_shapes (norm_hash);

    CREATE TABLE ingest_items (
        ingest_id      TEXT NOT NULL REFERENCES ingests (ingest_id),
        source_relpath TEXT NOT NULL,
        name           TEXT NOT NULL,
        status         TEXT NOT NULL,
        reason         TEXT,
        geometry_id    TEXT REFERENCES geometries (geometry_id),
        report_path    TEXT NOT NULL,
        PRIMARY KEY (ingest_id, source_relpath)
    ) STRICT;
    """,
    """
    CREATE TABLE rejected_shapes (
        raw_hash       TEXT NOT NULL,
        shape_version  TEXT NOT NULL,
        geometry_id    TEXT NOT NULL,
        name           TEXT NOT NULL,
        family         TEXT,
        family_params  TEXT,
        raw_path       TEXT NOT NULL,
        raw_points     INTEGER NOT NULL,
        reason         TEXT NOT NULL,
        detail         TEXT NOT NULL,
        cst            TEXT,
        rotation_deg   REAL,
        closure        TEXT,
        root_te_gap    REAL,
        source_dev     REAL,
        fit_max        REAL,
        te_wedge_deg   REAL,
        norm_path      TEXT,
        norm_hash      TEXT,
        ingest_id      TEXT NOT NULL REFERENCES ingests (ingest_id),
        report_path    TEXT NOT NULL,
        PRIMARY KEY (raw_hash, shape_version)
    ) STRICT;

    CREATE VIEW geometry_outcomes AS
        SELECT 'accepted' AS status, NULL AS reason, s.shape_version, g.geometry_id, g.raw_hash, g.name,
               g.family, g.family_params, g.raw_path, g.raw_points, s.cst, s.rotation_deg, s.closure,
               s.root_te_gap, s.source_dev, s.fit_max, s.te_wedge_deg, s.norm_path, s.norm_hash
        FROM geometries g JOIN geometry_shapes s USING (geometry_id)
        UNION ALL
        SELECT 'rejected', reason, shape_version, geometry_id, raw_hash, name,
               family, family_params, raw_path, raw_points, cst, rotation_deg, closure,
               root_te_gap, source_dev, fit_max, te_wedge_deg, norm_path, norm_hash
        FROM rejected_shapes;
    """,
]

# Live table -> its archive table (created on first use) and the column naming a row besides shape_version
_ARCHIVED = {
    "geometry_shapes": ("archived_shapes", "geometry_id"),
    "rejected_shapes": ("archived_rejections", "raw_hash"),
}
_ARCHIVE_TABLE = """
    CREATE TABLE IF NOT EXISTS archive.{table} (
        {key}          TEXT NOT NULL,
        shape_version  TEXT NOT NULL,
        archived_at    TEXT NOT NULL,
        ingest_id      TEXT NOT NULL,
        shape          TEXT NOT NULL,
        PRIMARY KEY ({key}, shape_version)
    ) STRICT
"""


class GeometryStore:
    """An open manifest. Use as a context manager, or `close()` it."""

    def __init__(self, conn: sqlite3.Connection, path: Path):
        self._conn = conn
        self.path = path

    @classmethod
    def open(cls, db=None, read_only: bool = False) -> "GeometryStore":
        """
        Opens the manifest, creating or migrating it as needed.

        Args:
            db (str | Path | None): The database file, the repo's own by default
            read_only (bool): Open an existing, current database without writing anything (a dry run)

        Returns:
            GeometryStore: The open store

        Raises:
            RuntimeError: If the database was written by newer code (a later schema version), or a
                read-only open finds one that needs migrating
            FileNotFoundError: If a read-only open finds no database
        """
        path = Path(db) if db is not None else Geom_paths().db
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
        store = cls(conn, path)
        try:
            store._migrate()
        except BaseException:
            conn.close()
            raise
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GeometryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version > len(_MIGRATIONS):
            raise RuntimeError(f"{self.path} is at schema version {version}; this code knows up to "
                               f"{len(_MIGRATIONS)}. Update the code before opening it.")
        if 0 < version < len(_MIGRATIONS):
            with sqlite3.connect(self.path.with_name(f"{self.path.name}.bak-v{version}")) as backup:
                self._conn.backup(backup)
        for n in range(version, len(_MIGRATIONS)):
            try:
                self._conn.executescript(f"BEGIN;\n{_MIGRATIONS[n]}\nPRAGMA user_version = {n + 1};\nCOMMIT;")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

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

    # ------------------------------------------------------------------------------- writes

    def begin_ingest(self, ingest_id: str, started_at: str, source_root: str, shape_version: str,
                     git_commit: str, report_path: str) -> None:
        """Records an ingest run, before any of its files."""
        with self.transaction():
            self._conn.execute("INSERT INTO ingests VALUES (?, ?, ?, ?, ?, ?)",
                               (ingest_id, started_at, source_root, shape_version, git_commit, report_path))

    def add_geometry(self, result: AddResult, ingest_id: str, raw_path: str, norm_path: str,
                     shape_version: str) -> None:
        """
        Stores an accepted airfoil and its shape, together.

        Args:
            result (AddResult): An accepted `Geom_add_airfoil` result
            ingest_id (str): The ingest adding it
            raw_path (str): Its source copy, relative to `data/`
            norm_path (str): Its rendering, relative to `data/`
            shape_version (str): The pipeline version that produced the shape
        """
        if not result.accepted:
            raise ValueError(f"{result.path} was rejected ({result.reject.value}); only accepted airfoils are stored")
        ident = result.identity
        with self.transaction():
            self._conn.execute(
                "INSERT INTO geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ident.geometry_id, ident.name, ident.family.value if ident.family else None,
                 json.dumps(ident.family_params) if ident.family_params is not None else None,
                 raw_path, ident.raw_hash, ident.raw_points, ingest_id))
            self.add_shape(result, ident.geometry_id, norm_path, shape_version)

    def add_shape(self, result: AddResult, geometry_id: str, norm_path: str, shape_version: str) -> None:
        """
        Stores an airfoil's shape under one shape version: with `add_geometry` for a new airfoil, alone
        when a stored one is refitted under a new version.

        Args:
            result (AddResult): An accepted `Geom_add_airfoil` result
            geometry_id (str): The stored airfoil it belongs to
            norm_path (str): Its rendering, relative to `data/`
            shape_version (str): The pipeline version that produced it
        """
        if not result.accepted:
            raise ValueError(f"{result.path} was rejected ({result.reject.value}); only accepted shapes are stored")
        with self.transaction():
            self._conn.execute(
                "INSERT INTO geometry_shapes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (geometry_id, shape_version, result.cst.model_dump_json(), result.transform.rotation_deg,
                 result.closure.value, result.root_te_gap, result.source_dev, result.fit.max_dev,
                 result.te_wedge_deg, norm_path, result.norm_hash))

    def add_rejection(self, result: AddResult, reason: str, detail: str, ingest_id: str, raw_path: str,
                      norm_path: Optional[str], shape_version: str, report_path: str) -> None:
        """
        Stores a rejected file with everything measured on it, replacing its row under this version if
        an earlier run judged it already.

        Args:
            result (AddResult): A `Geom_add_airfoil` result that got as far as identifying the file
                (accepted only when the ingest refused it, e.g. for its name)
            reason (str): The `RejectReason` value
            detail (str): The reason in words
            ingest_id (str): The ingest judging it
            raw_path (str): Its source copy, relative to `data/`
            norm_path (str | None): Its rendering, relative to `data/`, if it has a fit
            shape_version (str): The pipeline version that judged it
            report_path (str): Its report, relative to `data/`
        """
        ident = result.identity
        if ident is None:
            raise ValueError(f"{result.path} was never identified; only identified files are stored as rejections")
        tf, fit = result.transform, result.fit
        with self.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO rejected_shapes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ident.raw_hash, shape_version, ident.geometry_id, ident.name,
                 ident.family.value if ident.family else None,
                 json.dumps(ident.family_params) if ident.family_params is not None else None,
                 raw_path, ident.raw_points, getattr(reason, "value", reason), detail,
                 result.cst.model_dump_json() if result.cst is not None else None,
                 tf.rotation_deg if tf is not None else None,
                 result.closure.value if result.closure is not None else None,
                 result.root_te_gap, result.source_dev, fit.max_dev if fit is not None else None,
                 result.te_wedge_deg, norm_path, result.norm_hash if norm_path is not None else None,
                 ingest_id, report_path))

    def remove_rejection(self, raw_hash: str, shape_version: str) -> None:
        """Drops a file's rejection under one version: it has been stored as accepted since."""
        with self.transaction():
            self._conn.execute("DELETE FROM rejected_shapes WHERE raw_hash = ? AND shape_version = ?",
                               (raw_hash, shape_version))

    def archive_old_shapes(self, keep_version: str, ingest_id: str, archived_at: str) -> list[str]:
        """
        Moves every shape and rejection not under `keep_version` into the archive database, in one
        transaction across both files. Creates the archive only when there is something to move.

        Args:
            keep_version (str): The current shape version, left in place
            ingest_id (str): The ingest moving them
            archived_at (str): When, as UTC ISO text

        Returns:
            list[str]: The shape versions moved, sorted
        """
        moved = sorted({r[0] for table in _ARCHIVED for r in self._conn.execute(
            f"SELECT DISTINCT shape_version FROM {table} WHERE shape_version != ?", (keep_version,))})
        if not moved:
            return []
        self._conn.execute("ATTACH DATABASE ? AS archive", (str(self.path.with_name(ARCHIVE_NAME)),))
        try:
            for archive_table, key in _ARCHIVED.values():
                self._conn.execute(_ARCHIVE_TABLE.format(table=archive_table, key=key))
            with self.transaction():
                for table, (archive_table, key) in _ARCHIVED.items():
                    cur = self._conn.execute(f"SELECT * FROM {table} WHERE shape_version != ?", (keep_version,))
                    columns = [c[0] for c in cur.description]
                    for row in cur.fetchall():
                        shape = dict(zip(columns, row))
                        self._conn.execute(f"INSERT OR REPLACE INTO archive.{archive_table} VALUES (?, ?, ?, ?, ?)",
                                           (shape[key], shape["shape_version"], archived_at, ingest_id,
                                            json.dumps(shape)))
                    self._conn.execute(f"DELETE FROM {table} WHERE shape_version != ?", (keep_version,))
        finally:
            self._conn.execute("DETACH DATABASE archive")
        return moved

    def add_item(self, ingest_id: str, source_relpath: str, name: str, status: ItemStatus,
                 reason: Optional[str], geometry_id: Optional[str], report_path: str) -> None:
        """
        Records what an ingest did with one file.

        Args:
            ingest_id (str): The ingest
            source_relpath (str): The file, relative to the ingested folder
            name (str): The airfoil name (the file stem when the file could not be read)
            status (ItemStatus): Added, duplicate or rejected
            reason (str | None): A `RejectReason` value, or what matched for a duplicate
            geometry_id (str | None): The airfoil it added or duplicated
            report_path (str): Its report, relative to `data/`
        """
        with self.transaction():
            self._conn.execute("INSERT INTO ingest_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (ingest_id, source_relpath, name, ItemStatus(status).value,
                                getattr(reason, "value", reason), geometry_id, report_path))

    # -------------------------------------------------------------------------------- reads

    def has_geometry(self, geometry_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM geometries WHERE geometry_id = ?", (geometry_id,)).fetchone() is not None

    def find_norm_hash(self, norm_hash: str) -> Optional[str]:
        """The airfoil whose stored shape renders to exactly these coordinates, if any."""
        row = self._conn.execute("SELECT geometry_id FROM geometry_shapes WHERE norm_hash = ?", (norm_hash,)).fetchone()
        return row[0] if row else None

    def name_of(self, geometry_id: str) -> Optional[str]:
        row = self._conn.execute("SELECT name FROM geometries WHERE geometry_id = ?", (geometry_id,)).fetchone()
        return row[0] if row else None

    def name_owner(self, name: str) -> Optional[str]:
        """The airfoil stored under `name`, ignoring case."""
        row = self._conn.execute("SELECT geometry_id FROM geometries WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def get_geometry(self, name: str) -> Optional[GeometryRecord]:
        """A stored airfoil by name, ignoring case."""
        row = self._conn.execute("SELECT * FROM geometries WHERE name = ?", (name,)).fetchone()
        return _geometry(row) if row is not None else None

    def geometries_without_shape(self, shape_version: str) -> list[GeometryRecord]:
        """Stored airfoils with no shape under `shape_version`, by name: what a version bump leaves to refit."""
        rows = self._conn.execute(
            "SELECT * FROM geometries WHERE geometry_id NOT IN "
            "(SELECT geometry_id FROM geometry_shapes WHERE shape_version = ?) ORDER BY name", (shape_version,))
        return [_geometry(row) for row in rows]

    def get_shape(self, geometry_id: str, shape_version: str) -> Optional[ShapeRecord]:
        """An airfoil's stored shape under one shape version."""
        row = self._conn.execute("SELECT * FROM geometry_shapes WHERE geometry_id = ? AND shape_version = ?",
                                 (geometry_id, shape_version)).fetchone()
        if row is None:
            return None
        geometry_id, version, cst, rotation, closure, gap, source_dev, fit_max, wedge, norm_path, norm_hash = row
        return ShapeRecord(geometry_id=geometry_id, shape_version=version, cst=CSTParams.model_validate_json(cst),
                           rotation_deg=rotation, closure=closure, root_te_gap=gap, source_dev=source_dev,
                           fit_max=fit_max, te_wedge_deg=wedge, norm_path=norm_path, norm_hash=norm_hash)

    def get_rejection(self, raw_hash: str, shape_version: str) -> Optional[RejectionRecord]:
        """A file's rejection under one shape version."""
        cur = self._conn.execute("SELECT * FROM rejected_shapes WHERE raw_hash = ? AND shape_version = ?",
                                 (raw_hash, shape_version))
        return next(_rejections(cur), None)

    def find_rejection(self, raw_hash: str) -> Optional[RejectionRecord]:
        """A file's rejection under any shape version: the last one written."""
        cur = self._conn.execute("SELECT * FROM rejected_shapes WHERE raw_hash = ? ORDER BY rowid DESC LIMIT 1",
                                 (raw_hash,))
        return next(_rejections(cur), None)

    def get_rejection_named(self, name: str, shape_version: str) -> Optional[RejectionRecord]:
        """A rejection by name under one shape version, ignoring case; names repeat among rejections, so the last written."""
        cur = self._conn.execute("SELECT * FROM rejected_shapes WHERE name = ? COLLATE NOCASE AND shape_version = ? "
                                 "ORDER BY rowid DESC LIMIT 1", (name, shape_version))
        return next(_rejections(cur), None)

    def rejections(self, shape_version: str) -> list[RejectionRecord]:
        """Every rejection under one shape version, by name."""
        cur = self._conn.execute("SELECT * FROM rejected_shapes WHERE shape_version = ? ORDER BY name, raw_hash",
                                 (shape_version,))
        return list(_rejections(cur))

    def rejections_to_recheck(self) -> list[RejectionRecord]:
        """
        One row per rejected file whose airfoil is not stored, by name: what an ingest judges again
        from its copy. A stored airfoil's own rejection comes back through `geometries_without_shape`.
        """
        cur = self._conn.execute(
            "SELECT * FROM rejected_shapes WHERE geometry_id NOT IN (SELECT geometry_id FROM geometries) "
            "ORDER BY name, raw_hash, rowid DESC")  # a replaced row gets a new rowid: the last written first
        latest = {}
        for rec in _rejections(cur):
            latest.setdefault(rec.raw_hash, rec)
        return list(latest.values())


def _rejections(cur: sqlite3.Cursor):
    columns = [c[0] for c in cur.description]
    for row in cur:
        rec = dict(zip(columns, row))
        rec["family_params"] = json.loads(rec["family_params"]) if rec["family_params"] is not None else None
        rec["cst"] = CSTParams.model_validate_json(rec["cst"]) if rec["cst"] is not None else None
        yield RejectionRecord(**rec)


def _geometry(row: tuple) -> GeometryRecord:
    geometry_id, name, family, params, raw_path, raw_hash, raw_points, ingest_id = row
    return GeometryRecord(geometry_id=geometry_id, name=name, family=family,
                          family_params=json.loads(params) if params is not None else None,
                          raw_path=raw_path, raw_hash=raw_hash, raw_points=raw_points, ingest_id=ingest_id)
