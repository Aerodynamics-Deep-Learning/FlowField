"""
Manual look at any mesh a run store recorded. Each is re-meshed from its stored config and kept only once its
.su2 hashes to what the run stored (`meshing/optimizer/evaluate/remesh.py`), then shown in the shared mesh viewer
(`tests/mesh_viewer.py`: keys 1-5 pick a folder, the left and right arrows move within it, 't' / 'c' / 'n' for
edges, metric and colour scale), its "gate" scale at the gates its own settings row judged it by.

What to show comes from `test_manual_remesh_viewer.yaml` beside this file (gitignored, this machine's own), else
the committed `test_manual_remesh_viewer.example.yaml`, which documents the format: up to five folders, each a
run store and the airfoils (`best`, `all_runs`) or run ids (`manual`) to show from it. With no folders, the five
newest run stores under data/, every best run in each.

    python -m pytest tests/integration/datagen/mesh/optimizer/test_manual_remesh_viewer.py -m interactive -n 0 -s

Everything named is resolved before the window opens: an unknown store, airfoil or run id stops the test, with
the closest matches. A run that cannot be re-meshed (meshed by code that has since moved, never meshed, a hash
that no longer matches, a store too old to read) shows why in its slot. Re-meshes run in a background process
pool, into pytest's tmp folder; the stores are only read.
"""

import difflib
import json
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

import pytest

from src.datagen.geometry.common.paths import DATA_ROOT, Geom_paths
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.manifest.store import RunStore
from tests.mesh_viewer import ORTHO, Entry, Folder, MeshViewer, scored_fluid_domain
from tests.viewer_config import load_selection

pytestmark = [
    pytest.mark.interactive,
    pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment"),
]

MODES = ("best", "all_runs", "manual")
GATE_COLUMNS = {"skew": "gate_skew_max", ORTHO: "gate_ortho_min", "scaled_jacobian": "gate_jac_min",
                "aspect_ratio": "gate_ar_max", "size_ratio": "gate_size_ratio_max"}
TOPOLOGIES = {"OGRID": "O-grid", "CGRID": "C-grid"}


def remesh_job(store_path, run_id, out_dir, geometry_db):
    """In the viewer's pool: the run re-meshed, verified and scored into a .vtu; or why not."""
    import logging
    from src.datagen.meshing.optimizer.evaluate.remesh import Optim_remesh

    logging.getLogger("src.datagen").setLevel(logging.ERROR)  # the slot reports; keep the terminal to the viewer
    try:
        with RunStore.open(store_path, read_only=True) as store:
            done = Optim_remesh(store, run_id, out_dir, geometry_db=geometry_db)
        vtu = Path(done.su2_path).with_suffix(".vtu")
        scored_fluid_domain(done.su2_path).save(vtu)
        return str(vtu), None
    except Exception as e:  # shown in its slot; the pool must outlive any one run
        return None, f"{type(e).__name__}: {e}"


def test_remesh_viewer(tmp_path):
    specs = load_selection(__file__).get("folders") or _newest_stores(5)
    if len(specs) > 5:
        pytest.fail(f"{len(specs)} folders; keys 1-5 reach five", pytrace=False)
    MeshViewer([_folder(spec, tmp_path / str(f + 1)) for f, spec in enumerate(specs)]).show()


# What to show

def _folder(spec: dict, out_root: Path) -> Folder:
    unknown = set(spec) - {"store", "geometry", *MODES}
    modes = [key for key in spec if key in MODES]  # in the order written
    if unknown or "store" not in spec or not modes:
        pytest.fail(f"folder {spec}: needs a store and any of {', '.join(MODES)}, plus an optional geometry"
                    + (f"; unknown: {', '.join(sorted(unknown))}" if unknown else ""), pytrace=False)
    store = _store_path(spec["store"])
    name = f"{_show(store)} ({', '.join(modes)})"
    try:
        RunStore.open(store, read_only=True).close()
    except Exception as e:  # a store a migration must write to first, or no run store at all
        return Folder(name, [Entry(message=f"this store can't be read: {e}")])

    with closing(sqlite3.connect(f"{store.resolve().as_uri()}?mode=ro", uri=True)) as conn, ExitStack() as stack:
        conn.row_factory = sqlite3.Row
        picks = []
        for mode in modes:
            items = [str(item) for item in (spec[mode] or [])]
            picks += {"best": _best, "all_runs": _all_runs, "manual": _manual}[mode](conn, items, store)
        best = {row[0] for row in conn.execute("SELECT run_id FROM best")}
        settings = {row["settings_id"]: row for row in conn.execute("SELECT * FROM settings")}
        geometries = _geometry_candidates(store, spec.get("geometry"))
        opened = {db: stack.enter_context(GeometryStore.open(db, read_only=True)) for db in geometries}

        entries, seen = [], set()
        for pick in picks:
            if isinstance(pick, Entry):
                entries.append(pick)
                continue
            if pick in seen:  # a run picked twice is shown once
                continue
            seen.add(pick)
            run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (pick,)).fetchone()
            s = settings[run["settings_id"]]
            caption = _caption(run, s, pick in best)
            gates = {metric: s[column] for metric, column in GATE_COLUMNS.items()}
            holder = next((db for db, g in opened.items() if g.get_shape(run["geometry_id"], s["shape_version"])), None)
            if holder is None:
                entries.append(Entry(caption, gates, message=f"no geometry manifest holds {run['airfoil_name']}'s "
                                     f"shape {s['shape_version']}; looked in {', '.join(map(_show, geometries))}"))
            else:
                entries.append(Entry(caption, gates, job=(remesh_job, str(store), pick, str(out_root / pick),
                                                          str(holder))))
    return Folder(name, entries)


def _best(conn, names, store) -> list:
    """Each airfoil's best run per settings row (a calibration store has a row per rung); none named, every one."""
    sql = ("SELECT b.run_id FROM best b JOIN runs r USING (run_id) JOIN settings s ON s.settings_id = b.settings_id "
           "{} ORDER BY s.created_at, r.airfoil_name")
    if not names:
        found = [row[0] for row in conn.execute(sql.format(""))]
        return found or [Entry(message="no best run in this store: nothing in it passed")]
    _check_names(conn, names, store)
    picks = []
    for name in names:
        found = [row[0] for row in conn.execute(sql.format("WHERE r.airfoil_name = ?"), (name,))]
        picks += found or [Entry(name, message=f"{name} has no best run here: nothing of it passed, or its ladder "
                                               f"never finished")]
    return picks


def _all_runs(conn, names, store) -> list:
    """Every run of each airfoil that produced a mesh, passed or not, in the order they ran."""
    if not names:
        pytest.fail(f"{_show(store)}: all_runs needs airfoil names", pytrace=False)
    _check_names(conn, names, store)
    picks = []
    for name in names:
        found = [row[0] for row in conn.execute(
            "SELECT run_id FROM runs WHERE airfoil_name = ? AND su2_sha256 IS NOT NULL ORDER BY rowid", (name,))]
        picks += found or [Entry(name, message=f"no run of {name} produced a mesh")]
    return picks


def _manual(conn, ids, store) -> list:
    """The runs named, by the whole id or its first characters."""
    if not ids:
        pytest.fail(f"{_show(store)}: manual needs run ids", pytrace=False)
    picks = []
    for prefix in ids:
        found = [row[0] for row in conn.execute("SELECT run_id FROM runs WHERE substr(run_id, 1, ?) = ?",
                                                (len(prefix), prefix))]
        if len(found) != 1:
            pytest.fail(f"{_show(store)}: {'no run' if not found else f'{len(found)} runs'} with an id starting "
                        f"{prefix!r}", pytrace=False)
        picks += found
    return picks


def _check_names(conn, names, store):
    known = [row[0] for row in conn.execute("SELECT DISTINCT airfoil_name FROM runs")]
    missing = [n for n in names if n not in known]
    if missing:
        pytest.fail(f"{_show(store)} has no runs of: " + "; ".join(
            f"{n} (closest: {', '.join(difflib.get_close_matches(n, known, 3)) or 'none'})" for n in missing),
            pytrace=False)


def _caption(run, settings, is_best) -> str:
    pipeline = f"{settings['backend']} {TOPOLOGIES.get(settings['topology'], settings['topology'])}"
    if run["status"] != "meshed":
        verdict = run["status"]
    elif run["passed"]:
        verdict = "passed"
    else:
        verdict = "failed: " + ", ".join(json.loads(run["failed_gates"] or "[]"))
    size = f"jmax {run['jmax']} x {run['s_n_points']} points  -  " if run["jmax"] and run["s_n_points"] else ""
    return (f"{run['airfoil_name']}  -  {pipeline}  -  {'best, ' if is_best else ''}rung {run['rung']} "
            f"#{run['seq']}  -  run {run['run_id'][:8]}\n"
            f"{size}{verdict}\n"
            f"settings {settings['settings_id'][:8]}: {settings['label'] or 'no label'}")


# Where things are

def _store_path(value) -> Path:
    path = Path(value) if Path(value).is_absolute() else DATA_ROOT / value
    if path.is_dir():
        found = sorted(path.glob("*.sqlite"))
        if len(found) != 1:
            pytest.fail(f"{_show(path)} holds {len(found)} .sqlite files; name the store", pytrace=False)
        path = found[0]
    if not path.is_file():
        newest = "\n  ".join(_show(p) for p in _run_stores()[:10]) or "none"
        pytest.fail(f"no run store at {_show(path)}; the newest under data/:\n  {newest}", pytrace=False)
    return path


def _geometry_candidates(store: Path, given) -> list:
    """The manifest named; else the store's run folder's own ingest, then the shared one."""
    if given is not None:
        path = Path(given) if Path(given).is_absolute() else DATA_ROOT / given
        path = Geom_paths(path).db if path.is_dir() else path
        if not path.is_file():
            pytest.fail(f"no geometry manifest at {_show(path)}", pytrace=False)
        return [path]
    own = [Geom_paths(folder / "geometry_ingest").db for folder in list(store.parents)[:3]]
    return [db for db in own if db.is_file()] + [Geom_paths().db]


def _run_stores() -> list:
    """Every run store under data/, newest first."""
    found = []
    for path in DATA_ROOT.rglob("*.sqlite"):
        try:
            with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'").fetchone():
                    found.append(path)
        except sqlite3.Error:
            pass
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _newest_stores(count) -> list:
    """The newest run stores this code can read, every best run in each."""
    specs = []
    for path in _run_stores():
        try:
            RunStore.open(path, read_only=True).close()
        except Exception as e:
            print(f"[viewer] skipped {_show(path)}: {e}")
            continue
        specs.append({"store": str(path), "best": None})
        if len(specs) == count:
            break
    if not specs:
        pytest.skip("no readable run store under data/; run a campaign first")
    return specs


def _show(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
