"""
Whether this machine makes the mesh study's inputs and meshes bit for bit as the reference machine did, i.e.:

Step 1: Ensure the sampler gives every manifest airfoil the reference's points and figures
    - test_this_machine_samples_like_the_reference
Step 2: Ensure c2d turns three of them into the reference's meshes
    - test_this_machine_meshes_like_the_reference

Run it on a new machine before a campaign or a re-mesh: a failure means meshes made here will not hash like
the reference's, so runs stored elsewhere will not re-mesh here. The reference, `machine_parity.json` beside
this file, holds per manifest airfoil the SHA-256 of its samples at three configs (points and reports), the n*
grid's, and the .su2 SHA-256 of three golden meshes with the c2d executable's own. Step 2 skips without c2d, or
with another executable, whose meshes belong to another settings row. After a sampler or shape version change,
or a new manifest, rewrite the reference on the reference machine:
    `python -m tests.integration.datagen.mesh.optimizer.test_machine_parity`
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.datagen.geometry.common.constants import DX_MAX, SAMPLER_VERSION, SHAPE_VERSION
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.constants import (
    CELL_CEILING, CLUSTER_WIDTH_MAX, CLUSTER_WIDTH_MIN, NSTAR_CW_GRID, NSTAR_LE_GRID, RATIO_MIN, SMOOTHING_DEFAULTS,
)
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings, Optim_mesher_sha256
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.evaluate.executor import RunExecutor
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_build_study_contour
from src.datagen.meshing.optimizer.search.nstar import _grid

REFERENCE = Path(__file__).with_name("machine_parity.json")

# The golden config, a coarse nose with a clamped TE, and a fine one
CONFIGS = ({"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026},
           {"dx_mid": 0.004, "le_ratio": 0.3, "te_ratio": 0.2, "cluster_width": 0.25},
           {"dx_mid": 0.002, "le_ratio": 0.01, "te_ratio": 0.05, "cluster_width": 0.1})
JMAX = 147
MESHED = ("naca0012", "s1223", "sc21010")


def _nstar_grid_hash() -> str:
    grids = [_grid(RATIO_MIN, 1.0, NSTAR_LE_GRID, log=True), _grid(CLUSTER_WIDTH_MIN, CLUSTER_WIDTH_MAX, NSTAR_CW_GRID)]
    return hashlib.sha256(json.dumps(grids).encode()).hexdigest()


def _sample_hashes() -> dict:
    """Per manifest airfoil, the SHA-256 over its samples' points and reports at CONFIGS."""
    out = {}
    for name, shape in Optim_load_shapes():
        contour = Optim_build_study_contour(shape, MeshBackend.C2D, MeshTopology.OGRD)
        h = hashlib.sha256()
        for knobs in CONFIGS:
            smp = Geom_sample(contour, SamplingFields(**knobs), max_points=CELL_CEILING // JMAX)
            if smp.points is not None:
                h.update(np.ascontiguousarray(smp.points, dtype="<f8").tobytes())
            h.update(smp.report.model_dump_json().encode())
        out[name] = h.hexdigest()
    return out


def _mesh_hashes(work_root) -> dict:
    """The .su2 SHA-256 of MESHED's golden meshes, through the study's executor."""
    rows = []
    executor = RunExecutor(RunContext.create(Optim_build_settings(), work_root), rows.append)
    for name, shape in Optim_load_shapes(MESHED):
        contour = Optim_build_study_contour(shape, MeshBackend.C2D, MeshTopology.OGRD)
        executor.run(ladder_id=None, geometry_id=shape.geometry_id, airfoil_name=name, contour=contour,
                     knobs={**CONFIGS[0], "jmax": JMAX, **SMOOTHING_DEFAULTS}, rung="manual", seq=0)
    return {row["airfoil_name"]: row["su2_sha256"] for row in rows}


def _reference() -> dict:
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    ref = json.loads(REFERENCE.read_text())
    assert (ref["sampler_version"], ref["shape_version"]) == (SAMPLER_VERSION, SHAPE_VERSION), (
        f"the reference is for {ref['sampler_version']}/{ref['shape_version']}, the code is at "
        f"{SAMPLER_VERSION}/{SHAPE_VERSION}: rewrite it on the reference machine (see this module's docstring)")
    return ref


# region Step 1
def test_this_machine_samples_like_the_reference():
    ref = _reference()
    got = _sample_hashes()

    assert set(got) == set(ref["samples"]), "the manifest's airfoils are not the reference's: rewrite the reference"
    assert _nstar_grid_hash() == ref["nstar_grid"], "n*'s grid points differ from the reference's"
    differ = sorted(name for name in got if got[name] != ref["samples"][name])
    assert not differ, f"{len(differ)} of {len(got)} airfoils sample differently here: {', '.join(differ)}"
# endregion


# region Step 2
@pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")
def test_this_machine_meshes_like_the_reference(tmp_path):
    ref = _reference()
    if "c2d" not in ref:
        pytest.skip("the reference was written without c2d, so holds no meshes")
    exe, ref_exe = Optim_mesher_sha256(), ref["c2d"]["mesher_sha256"]
    if exe != ref_exe:
        pytest.skip(f"this c2d executable ({exe[:12]}) is not the reference's ({ref_exe[:12]})")

    assert _mesh_hashes(tmp_path) == ref["c2d"]["meshes"]
# endregion


if __name__ == "__main__":
    reference = {"sampler_version": SAMPLER_VERSION, "shape_version": SHAPE_VERSION,
                 "nstar_grid": _nstar_grid_hash(), "samples": _sample_hashes()}
    if C2D_find_exe() is not None:
        with tempfile.TemporaryDirectory() as tmp:
            reference["c2d"] = {"mesher_sha256": Optim_mesher_sha256(), "meshes": _mesh_hashes(tmp)}
    REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    print(f"{REFERENCE}: {len(reference['samples'])} airfoils" + (", 3 meshes" if "c2d" in reference else ""))
