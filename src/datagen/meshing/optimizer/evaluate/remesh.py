"""
A stored run meshed again: the study keeps no mesh files, only each run's config and its `.su2`'s hash, so a
mesh is rebuilt on the executor's own input path and delivered only if it hashes to what the run stored. At
another Re there is no stored hash: that mesh is delivered only if it passes the run's settings' gates.

    Optim_remesh       one run: checked, meshed, verified (or judged, at another Re), delivered
    Remeshed           what it delivered
    RemeshRefused      the run cannot be rebuilt by this code: it has no mesh, or something it was meshed by moved
    RemeshMismatch     the rebuilt mesh is not the stored one; nothing was delivered
    RemeshFailedGates  the mesh at another Re fails the gates; nothing was delivered

A run is refused before meshing when its shape revision, sampler version, config schema or c2d executable
differs from the code's now: rebuilding it then needs its commit checked out. Anything else that moved (the
machine, a dirty tree) is caught by the hash. The run store is only read.
"""

import contextlib
import json
import os
import shutil
import socket
from pathlib import Path
from typing import NamedTuple, Optional

from src.datagen.geometry.common.constants import SAMPLER_VERSION, SHAPE_REVISION
from src.datagen.geometry.common.schemas import ShapeRecord
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.entry import Common_GenerateMesh
from src.datagen.meshing.optimizer.common.settings import Optim_config_schema_fp, Optim_mesher_sha256
from src.datagen.meshing.optimizer.evaluate.executor import Optim_failed_gates, Optim_mesh_columns
from src.datagen.meshing.optimizer.evaluate.sampling import (
    Optim_contour_for, Optim_safe_name, Optim_sample_for_study,
)
from src.datagen.meshing.optimizer.manifest.store import Optim_git_state, Optim_su2_sha256, RunStore


class Remeshed(NamedTuple):
    """A delivered re-mesh."""
    run_id: str
    airfoil_name: str
    su2_path: str
    vtk_path: Optional[str]  # None unless asked for
    num_nodes: int
    su2_sha256: str
    mesh_re: float           # the Re it was meshed at
    measured: dict           # the delivered mesh's `Optim_mesh_columns`; empty if the mesher measured nothing


class RemeshRefused(ValueError):
    """The run cannot be rebuilt by this code: it has no mesh, or something it was meshed by has moved."""


class RemeshMismatch(RuntimeError):
    """The rebuilt mesh is not the stored one, or no mesh came out; nothing was delivered."""


class RemeshFailedGates(RuntimeError):
    """The mesh at another Re fails the run's settings' gates; nothing was delivered."""

    def __init__(self, message: str, failed: list[str], measured: dict):
        super().__init__(message)
        self.failed = failed      # the gates it fails, in `GATE_ORDER`
        self.measured = measured  # its `Optim_mesh_columns`


def Optim_remesh(store: RunStore, run_id: str, out_dir, *, with_vtk: bool = False, geometry_db=None,
                 mesh_re: Optional[float] = None) -> Remeshed:
    """
    Meshes a stored run again and delivers its `.su2` (and `.vtk`) into `out_dir`: at the run's design Re once its
    hash matches; at another `mesh_re`, which has no stored hash, once it passes the settings' gates.

    c2d runs in `out_dir/.remesh_<run id[:8]>/`, which is deleted whatever happens; the accepted files move up
    as `<name>.su2` and `<name>.vtk`, replacing any there, the name file-safe. An `out_dir` this call made is
    removed again when nothing was delivered.

    Args:
        store (RunStore): The run store holding the run, read-only is enough
        run_id (str): The run
        out_dir (str | Path): Where the files go; created if missing
        with_vtk (bool): Also deliver the diagnostic .vtk
        geometry_db (str | Path | None): The geometry manifest, the repo's own by default
        mesh_re (float | None): The Re to mesh at; None or the settings' design Re for the stored mesh

    Returns:
        Remeshed: The delivered files

    Raises:
        RemeshRefused: Before meshing, if the run is unknown, has no mesh, or its shape revision, sampler
            version, config schema or c2d executable differs from now, or its shape is not stored
        RemeshMismatch: After meshing, if the sampler rejected the config, no .su2 (or asked-for .vtk) was
            written, the .su2's hash differs from the stored one, or a mesh at another Re was not measured
        RemeshFailedGates: After meshing at another Re, if the mesh fails a gate
        ValidationError: If `mesh_re` is outside the freestream's range
        FileNotFoundError: If no c2d executable is configured
    """
    run = store.run(run_id)
    if run is None:
        raise RemeshRefused(f"no run {run_id} in {store.path}")
    settings = store.settings(run["settings_id"])
    _refuse_if_moved(run, settings)
    name = run["airfoil_name"]
    shape = _load_shape(run["geometry_id"], settings["shape_version"], geometry_db)
    if shape is None:
        raise RemeshRefused(f"run {run_id}: {name} has no stored shape under {settings['shape_version']}")
    contour = Optim_contour_for(shape, settings)
    if contour is None:
        raise RemeshRefused(f"run {run_id}: {settings['backend']}/{settings['topology']} is not sampled")
    config = C2D_MeshingConfig(**json.loads(run["config_json"]))
    stored = mesh_re is None or mesh_re == settings["design_re"]  # the stored mesh, so its hash decides
    mesh_re = settings["design_re"] if mesh_re is None else mesh_re

    out_dir = Path(out_dir)
    made = not out_dir.exists()
    work = out_dir / f".remesh_{run_id[:8]}"
    shutil.rmtree(work, ignore_errors=True)  # a killed re-mesh's leftovers
    work.mkdir(parents=True)
    try:
        sampled = Optim_sample_for_study(settings, config, contour, name, str(work), write_vtk=with_vtk,
                                         mesh_re=mesh_re)
        if sampled.mesh_in is None:
            raise RemeshMismatch(f"run {run_id} ({name}): the sampler now rejects its config "
                                 f"({sampled.report.rejection}); {_provenance(run)}")
        out = Common_GenerateMesh(sampled.mesh_in)
        if not out.mesh_path or not os.path.isfile(out.mesh_path):
            raise RemeshMismatch(f"run {run_id} ({name}): no .su2 written ({out.flag.name}); {_provenance(run)}")
        sha, _ = Optim_su2_sha256(out.mesh_path)
        measured = Optim_mesh_columns(out) if out.quality is not None else {}
        if stored and sha != run["su2_sha256"]:
            raise RemeshMismatch(f"run {run_id} ({name}): the .su2 hashes to {sha}, the run stored "
                                 f"{run['su2_sha256']}; {_provenance(run)}")
        if not stored:
            if not measured:
                raise RemeshMismatch(f"run {run_id} ({name}) at Re {mesh_re:.4g}: the mesh was not measured "
                                     f"({out.flag.name})")
            failed = Optim_failed_gates({"entry_flag": out.flag.name, **measured}, settings)
            if failed:
                raise RemeshFailedGates(f"run {run_id} ({name}) at Re {mesh_re:.4g} fails {', '.join(failed)}; "
                                        f"it passed at its design Re {settings['design_re']:.4g}", failed, measured)
        if with_vtk and not (out.mesh_path_vtk and os.path.isfile(out.mesh_path_vtk)):
            raise RemeshMismatch(f"run {run_id} ({name}): no .vtk written ({out.flag.name})")

        safe = Optim_safe_name(name)
        su2_path = out_dir / f"{safe}.su2"
        os.replace(out.mesh_path, su2_path)
        vtk_path = None
        if with_vtk:
            vtk_path = out_dir / f"{safe}.vtk"
            os.replace(out.mesh_path_vtk, vtk_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if made:
            with contextlib.suppress(OSError):  # only an empty folder goes: one this call made, nothing delivered
                out_dir.rmdir()
    return Remeshed(run_id, name, str(su2_path), str(vtk_path) if vtk_path else None, out.num_nodes, sha, mesh_re,
                    measured)


def _refuse_if_moved(run: dict, settings: dict) -> None:
    """Raises RemeshRefused naming everything the run was meshed by that differs now."""
    if run["su2_sha256"] is None:
        raise RemeshRefused(f"run {run['run_id']} has no mesh to rebuild (status {run['status']})")
    shapes = settings["shape_version"]  # its order and nose are the run's own; only the pipeline's revision must hold
    checks = (("shape version", shapes, f"{shapes.rpartition('-')[0]}-{SHAPE_REVISION}"),
              ("sampler version", settings["sampler_version"], SAMPLER_VERSION),
              ("config schema", run["config_schema_fp"], Optim_config_schema_fp()),
              ("c2d executable", settings["mesher_sha256"], Optim_mesher_sha256()))
    moved = [f"{what} {then[:16]} then, {now[:16]} now" for what, then, now in checks if then != now]
    if moved:
        raise RemeshRefused(f"run {run['run_id']} was meshed by other code ({'; '.join(moved)}): re-mesh it at "
                            f"its commit {run['git_commit']}, with the c2d it ran")


def _load_shape(geometry_id: str, shape_version: str, db) -> Optional[ShapeRecord]:
    with GeometryStore.open(db, read_only=True) as geometry:
        return geometry.get_shape(geometry_id, shape_version)


def _provenance(run: dict) -> str:
    """Where and at what commit the run was meshed, against now."""
    commit, dirty = Optim_git_state()
    then = f"{run['host']} at {run['git_commit'][:12]}{' (dirty)' if run['git_dirty'] else ''}"
    now = f"{socket.gethostname()} at {commit[:12]}{' (dirty)' if dirty else ''}"
    return f"meshed on {then}, re-meshed on {now}"
