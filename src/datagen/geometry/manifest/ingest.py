"""
Geom addition driver: adds every root .dat under a folder to the geometry manifest.

    `python -m src.datagen.geometry.manifest.ingest [<folder>] [--dry-run] [--data-root <data folder>]`

First, every stored airfoil without a shape under the current `SHAPE_VERSION` (all of them, after a
version bump) is refitted from its raw copy; then the shapes of other versions move to the archive
database and their renderings are deleted. Without a folder, that is all a run does.

Then per file, in sorted path order: identify it (an airfoil already stored is a duplicate, skipped
before fitting), run `add.Geom_add_airfoil`, then check its rendering and name against what is stored.
An added airfoil's files are written first and its rows last, in one transaction, so an interrupted
run leaves at most files the next run rewrites. Every file gets a report and the run a summary.

A file that fails is recorded and the run goes on; only a missing folder or an unwritable manifest
stops it. A dry run does everything but write, against the manifest as it stands.
"""

import argparse
import datetime
import secrets
import shutil
import traceback
from pathlib import Path
from typing import Optional

from src.datagen.geometry.build.add import Geom_add_airfoil, Geom_identify
from src.datagen.geometry.build.checks import Geom_check_suspect_point
from src.datagen.geometry.common.constants import CST_FIT_TOL, FAMILY_MATCH_TOL, SHAPE_VERSION
from src.datagen.geometry.common.io import GeometryFormatError, Geom_write_dat
from src.datagen.geometry.common.paths import DATA_ROOT, GeometryPaths, Geom_git_version, Geom_paths, Geom_source_label
from src.datagen.geometry.common.schemas import (
    AddResult, GeometryRecord, IngestItem, IngestSummary, ItemStatus, RejectReason,
)
from .store import GeometryStore


def Geom_ingest_folder(folder=None, data_root=DATA_ROOT, dry_run: bool = False) -> IngestSummary:
    """
    Brings the stored airfoils to the current shape version, then adds every .dat under `folder`.

    Args:
        folder (str | Path | None): A root folder, searched recursively; None to only bring the stored
            airfoils to the current shape version
        data_root (str | Path): The `data/` folder holding the manifest and its files
        dry_run (bool): Do everything but write

    Returns:
        IngestSummary: What happened to each stored airfoil refitted and each file

    Raises:
        NotADirectoryError: If `folder` is not a folder
    """
    if folder is not None:
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a folder")
    paths = Geom_paths(data_root)
    now = datetime.datetime.now(datetime.timezone.utc)
    summary = IngestSummary(ingest_id=f"{now:%Y%m%dT%H%MZ}_{secrets.token_hex(2)}",
                            started_at=f"{now:%Y-%m-%dT%H:%M:%SZ}",
                            source_root=Geom_source_label(folder) if folder is not None else "",
                            shape_version=SHAPE_VERSION, git_commit=Geom_git_version(), dry_run=dry_run)
    report_dir = paths.report_dir(summary.ingest_id)

    if dry_run:
        store = GeometryStore.open(paths.db, read_only=True) if paths.db.is_file() else None
    else:
        store = GeometryStore.open(paths.db)
        summary.summary_path = paths.to_rel(report_dir / "summary.txt")
        store.begin_ingest(summary.ingest_id, summary.started_at, summary.source_root, SHAPE_VERSION,
                           summary.git_commit, summary.summary_path)
    known = _Known(store)
    try:
        if store is not None:
            for geo in store.geometries_without_shape(SHAPE_VERSION):
                item = _assess_reshape(geo, paths)
                if not dry_run:
                    _record(item, paths.from_rel(geo.raw_path), summary, paths, store)
                if item.status is ItemStatus.RESHAPED:
                    known.add(item)
                summary.items.append(item)
            if not dry_run:
                for version in store.archive_old_shapes(SHAPE_VERSION, summary.ingest_id, summary.started_at):
                    shutil.rmtree(paths.norm_dir(version), ignore_errors=True)
        for path in sorted(folder.rglob("*.dat")) if folder is not None else []:
            item = _assess(path, folder, known)
            if not dry_run:
                _record(item, path, summary, paths, store)
            if item.status is ItemStatus.ADDED:
                known.add(item)
            summary.items.append(item)
    finally:
        if store is not None:
            store.close()

    if not dry_run:
        _write_text(paths.from_rel(summary.summary_path), Geom_format_summary(summary))
    return summary


class _Known:
    """What is stored plus what this run added, so a dry run deduplicates like a real one."""

    def __init__(self, store: Optional[GeometryStore]):
        self.store = store
        self.names_by_id, self.ids_by_hash, self.ids_by_name = {}, {}, {}

    def add(self, item: IngestItem) -> None:
        self.names_by_id[item.geometry_id] = item.name
        self.ids_by_hash[item.result.norm_hash] = item.geometry_id
        self.ids_by_name[item.name.lower()] = item.geometry_id

    def name_of(self, geometry_id: str) -> Optional[str]:
        if geometry_id in self.names_by_id:
            return self.names_by_id[geometry_id]
        return self.store.name_of(geometry_id) if self.store else None

    def norm_owner(self, norm_hash: str) -> Optional[str]:
        if norm_hash in self.ids_by_hash:
            return self.ids_by_hash[norm_hash]
        return self.store.find_norm_hash(norm_hash) if self.store else None

    def name_owner(self, name: str) -> Optional[str]:
        if name.lower() in self.ids_by_name:
            return self.ids_by_name[name.lower()]
        return self.store.name_owner(name) if self.store else None


def _assess(path: Path, folder: Path, known: _Known) -> IngestItem:
    """Decides what happens to one file; writes nothing."""
    rel = path.relative_to(folder).as_posix()
    try:
        ident = Geom_identify(path)
    except GeometryFormatError as e:
        return IngestItem(source_relpath=rel, name=path.stem, status=ItemStatus.REJECTED,
                          reason=RejectReason.FORMAT.value, detail=str(e))
    except Exception:
        return _error(rel, path.stem)

    stored_as = known.name_of(ident.geometry_id)
    if stored_as is not None:
        return IngestItem(source_relpath=rel, name=ident.name, status=ItemStatus.DUPLICATE, reason="geometry_id",
                          detail=f"the same airfoil is stored as '{stored_as}'", geometry_id=ident.geometry_id)
    try:
        res = Geom_add_airfoil(path)
    except Exception:
        return _error(rel, ident.name)

    item = IngestItem(source_relpath=rel, name=ident.name, status=ItemStatus.ADDED,
                      geometry_id=ident.geometry_id, result=res)
    if not res.accepted:
        item.status, item.reason, item.detail, item.geometry_id = ItemStatus.REJECTED, res.reject.value, res.reason, None
        return item
    owner = known.norm_owner(res.norm_hash)
    if owner is not None:
        item.status, item.reason, item.geometry_id = ItemStatus.DUPLICATE, "norm_hash", owner
        item.detail = f"its shape renders identically to the stored '{known.name_of(owner)}'"
        return item
    owner = known.name_owner(ident.name)
    if owner is not None:
        item.status, item.reason, item.geometry_id = ItemStatus.REJECTED, RejectReason.NAME_TAKEN.value, None
        item.detail = f"a different airfoil is stored as '{known.name_of(owner)}'; rename this file"
    return item


def _assess_reshape(geo: GeometryRecord, paths: GeometryPaths) -> IngestItem:
    """Refits a stored airfoil from its raw copy under the current shape version; writes nothing."""
    item = IngestItem(source_relpath=geo.raw_path, name=geo.name, status=ItemStatus.RESHAPED,
                      geometry_id=geo.geometry_id)
    try:
        item.result = res = Geom_add_airfoil(paths.from_rel(geo.raw_path))
    except Exception:
        return _error(geo.raw_path, geo.name, geo.geometry_id)
    if res.identity is None or res.identity.raw_hash != geo.raw_hash:
        item.status, item.reason = ItemStatus.REJECTED, RejectReason.ERROR.value
        item.detail = f"the raw copy {geo.raw_path} no longer holds the coordinates stored for '{geo.name}'"
    elif not res.accepted:
        item.status, item.reason, item.detail = ItemStatus.REJECTED, res.reject.value, res.reason
    return item


def _error(rel: str, name: str, geometry_id: Optional[str] = None) -> IngestItem:
    return IngestItem(source_relpath=rel, name=name, status=ItemStatus.REJECTED, reason=RejectReason.ERROR.value,
                      detail=traceback.format_exc(), geometry_id=geometry_id)


def _record(item: IngestItem, path: Path, summary: IngestSummary, paths: GeometryPaths,
            store: GeometryStore) -> None:
    """Writes one outcome: its files, its report, then its rows in one transaction."""
    report = paths.report_dir(summary.ingest_id) / Path(item.source_relpath).with_suffix(".txt")
    item.report_path = paths.to_rel(report)
    raw_rel = norm_rel = None
    if item.status in (ItemStatus.ADDED, ItemStatus.RESHAPED):
        raw = paths.raw_dir / f"{item.name}.dat"
        norm = paths.norm_dir(SHAPE_VERSION) / f"{item.name}.dat"
        _copy(path, raw)
        Geom_write_dat(norm, item.name, item.result.norm_coords)
        raw_rel, norm_rel = paths.to_rel(raw), paths.to_rel(norm)
    _write_text(report, Geom_format_report(item, summary, raw_rel, norm_rel))
    with store.transaction():
        if item.status is ItemStatus.ADDED:
            store.add_geometry(item.result, summary.ingest_id, raw_rel, norm_rel, SHAPE_VERSION)
        elif item.status is ItemStatus.RESHAPED:
            store.add_shape(item.result, item.geometry_id, norm_rel, SHAPE_VERSION)
        store.add_item(summary.ingest_id, item.source_relpath, item.name, item.status, item.reason,
                       item.geometry_id, item.report_path)


def _copy(src: Path, dst: Path) -> None:
    """Copies byte for byte; a source that already is the copy (re-ingesting `raw/`) is left alone."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.samefile(src):
        return
    shutil.copyfile(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------ reports

def Geom_format_report(item: IngestItem, summary: IngestSummary, raw_rel: Optional[str] = None,
                       norm_rel: Optional[str] = None) -> str:
    """
    One file's report: what was decided and everything measured on the way.

    Args:
        item (IngestItem): The file's outcome
        summary (IngestSummary): The run it belongs to
        raw_rel (str | None): Its stored source copy, if added
        norm_rel (str | None): Its stored rendering, if added

    Returns:
        str: The report text
    """
    head = f"{item.status.value}" + (f" ({item.reason})" if item.reason else "")
    out = [f"{item.name}: {head}",
           f"source   {item.source_relpath}",
           f"ingest   {summary.ingest_id}, shape version {summary.shape_version}, code {summary.git_commit}"]
    if item.detail:
        out += ["", *item.detail.rstrip().splitlines()]
    res: Optional[AddResult] = item.result
    if res is None or res.identity is None:
        if item.geometry_id:
            out += ["", f"geometry_id  {item.geometry_id}"]
        return "\n".join(out) + "\n"

    ident = res.identity
    family = (f"{ident.family.value} " + " ".join(f"{k}={v}" for k, v in ident.family_params.items())
              + ", built from the name in closed-TE form" if ident.family else "none (tabulated)")
    out += ["", "identity",
            f"  geometry_id  {ident.geometry_id}",
            f"  raw_hash     {ident.raw_hash}, {ident.raw_points} points",
            f"  family       {family}"]
    if res.transform is not None:
        tf = res.transform
        frame = "the formula contour" if ident.family else "the file"
        out += ["", f"normalization (of {frame})",
                f"  LE           point {tf.le_index}, {'interpolated between two input points' if tf.le_inserted else 'an input point'}, "
                f"at ({tf.translate[0]:.6f}, {tf.translate[1]:.6f})",
                f"  rotation     {tf.rotation_deg:+.4f} deg, scale {tf.scale:.6f}, "
                f"x stretch upper {tf.stretch_upper:.6f} lower {tf.stretch_lower:.6f}",
                "", "TE",
                f"  closure      {res.closure.value}, source gap {res.root_te_gap:.3e} c"]
        if res.family_dev is not None:
            out.append(f"  family check the file is within {res.family_dev:.2e} c of the formula shape "
                       f"(limit {FAMILY_MATCH_TOL:g})")
    if res.cst is not None:
        cst = res.cst
        out += ["", f"CST (order {cst.order}, n1 {cst.n1:g}, n2 {cst.n2:g})",
                "  upper        " + " ".join(f"{a:+.6f}" for a in cst.upper),
                "  lower        " + " ".join(f"{a:+.6f}" for a in cst.lower),
                f"  LE term      {cst.le_weight:+.6f}",
                f"  fit          max {res.fit.max_dev:.2e} c at x={res.fit.x_at_max:.4f} (limit {CST_FIT_TOL:g}), "
                f"rms {res.fit.rms_dev:.2e} c, over {res.fit.n_points} points",
                f"  source       the file's points before closing are within {res.source_dev:.2e} c",
                f"  TE wedge     {res.te_wedge_deg:.2f} deg"]
        ok, _, note = Geom_check_suspect_point(res.fit)
        if not ok:
            out.append(f"  suspect      {note}")
    if raw_rel:
        out += ["", "files",
                f"  raw          {raw_rel}",
                f"  normalised   {norm_rel}",
                f"  norm_hash    {res.norm_hash}"]
    return "\n".join(out) + "\n"


def Geom_format_summary(summary: IngestSummary) -> str:
    """
    The run's summary: counts, then one line per file.

    Args:
        summary (IngestSummary): The run

    Returns:
        str: The summary text
    """
    rejected = [i.reason for i in summary.items if i.status is ItemStatus.REJECTED]
    by_reason = ", ".join(f"{r} {rejected.count(r)}" for r in sorted(set(rejected)))
    reshaped = summary.count(ItemStatus.RESHAPED)
    out = [f"Geometry ingest {summary.ingest_id}" + (" (dry run, nothing written)" if summary.dry_run else ""),
           f"started {summary.started_at}, code {summary.git_commit}, shape version {summary.shape_version}",
           f"source  {summary.source_root or '(none: stored airfoils only)'}",
           f"{len(summary.items)} files: {summary.count(ItemStatus.ADDED)} added, "
           + (f"{reshaped} reshaped, " if reshaped else "")
           + f"{summary.count(ItemStatus.DUPLICATE)} duplicate, {summary.count(ItemStatus.REJECTED)} rejected"
           + (f" ({by_reason})" if by_reason else ""),
           "",
           f"{'status':9} {'reason':15} {'fit max':>8} {'src dev':>8} {'closure':8} source"]
    for item in summary.items:
        res = item.result
        fitted = res is not None and res.fit is not None
        fit = f"{res.fit.max_dev:8.1e} {res.source_dev:8.1e}" if fitted else f"{'':8} {'':8}"
        closure = res.closure.value if res is not None and res.closure is not None else ""
        suspect = "" if not fitted or Geom_check_suspect_point(res.fit)[0] else f"  (suspect point x={res.fit.x_at_max:.4f})"
        out.append(f"{item.status.value:9} {item.reason or '':15} {fit} {closure:8} {item.source_relpath}{suspect}")
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, nargs="?",
                        help="root folder of .dat files, searched recursively; leave out to only bring the stored "
                             "airfoils to the current shape version")
    parser.add_argument("--dry-run", action="store_true", help="do everything but write")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="the data/ folder (default: the repo's)")
    args = parser.parse_args(argv)

    summary = Geom_ingest_folder(args.folder, args.data_root, args.dry_run)
    print(Geom_format_summary(summary))
    if summary.summary_path:
        print(f"Summary written to {Geom_paths(args.data_root).from_rel(summary.summary_path)}")


if __name__ == "__main__":
    main()
