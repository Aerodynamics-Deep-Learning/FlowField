"""
Tests for `meshing.optimizer.sampling`: a study's contour, and each config's sample turned into a MeshIn, i.e.:

Step 1: Ensure each pairing gets the TE its mesher's own gate asks for
    - test_te_map_agrees_with_the_closure_gate
    - test_build_study_contour
    - test_sampled_input_passes_the_mesher_gates
Step 2: Ensure the MeshIn carries exactly the sample
    - test_mesh_in_carries_exactly_the_sample
Step 3: Ensure rejections build nothing, and setup mistakes raise
    - test_rejected_config_builds_no_mesh_in
    - test_cell_ceiling_follows_jmax
    - test_setup_mismatches_raise
"""

import numpy as np
import pytest
import torch

from src.datagen.geometry.common.schemas import Rejection, TETreatment
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.common.utils import (
    _TE_REQUIREMENTS, Common_te_shape_warnings, Common_validate_te_closure, TEClosure, TEShape,
)
from src.datagen.meshing.gmsh.schemas import GMSH_CMeshingConfig, GMSH_OMeshingConfig
from src.datagen.meshing.optimizer.sampling import (
    CELL_CEILING, TE_FOR_PAIRING, TE_GAP_DEFAULT, build_study_contour, sample_for_mesher,
)
from src.datagen.schemas import Freestream
from tests.geometry_shapes import SHAPE_NAMES, stored_shape

C2D_O, C2D_C = (MeshBackend.C2D, MeshTopology.OGRD), (MeshBackend.C2D, MeshTopology.CGRD)
GMSH_O, GMSH_C = (MeshBackend.GMSH, MeshTopology.OGRD), (MeshBackend.GMSH, MeshTopology.CGRD)
_CONFIG = {C2D_O: C2D_MeshingConfig, C2D_C: C2D_MeshingConfig, GMSH_O: GMSH_OMeshingConfig}


def _sampled(pairing, config=None, name="naca4412", contour=None, tmp_path="."):
    contour = contour or build_study_contour(stored_shape(name), *pairing)
    return sample_for_mesher(config or _CONFIG[pairing](), contour, airfoil_name=name, chord=0.5,
                             freestream=Freestream(alpha=2.0, Re=5e6, mach=0.5), backend=pairing[0],
                             topology=pairing[1], working_dir=str(tmp_path))


# region Step 1
def test_te_map_agrees_with_the_closure_gate():
    # The gate's shape column is the shape a pairing meshes poorly, so the sampled TE must not be it
    assert set(TE_FOR_PAIRING) == set(_TE_REQUIREMENTS)
    for pairing, te in TE_FOR_PAIRING.items():
        closure, warned = _TE_REQUIREMENTS[pairing]
        if te is None:  # not sampled: left to the blunt TE its C-mesh needs
            assert closure is TEClosure.OPEN
        else:
            assert closure is TEClosure.CLOSED
            assert warned is (TEShape.SHARP if te is TETreatment.ROUND else TEShape.BLUNT)


def test_build_study_contour():
    shape = stored_shape("naca0012")

    assert build_study_contour(shape, *C2D_O).te is TETreatment.ROUND
    assert build_study_contour(shape, *C2D_O).gap == TE_GAP_DEFAULT
    assert build_study_contour(shape, *C2D_O, gap=0.006).gap == 0.006
    assert build_study_contour(shape, *C2D_C).te is TETreatment.SHARP
    assert build_study_contour(shape, *GMSH_O).te is TETreatment.ROUND
    assert build_study_contour(shape, *GMSH_C) is None
    with pytest.raises(ValueError, match="no TE"):
        build_study_contour(shape, MeshBackend.UNKNOWN, MeshTopology.OGRD)


@pytest.mark.parametrize("name", SHAPE_NAMES)
@pytest.mark.parametrize("pairing", [C2D_O, C2D_C, GMSH_O])
def test_sampled_input_passes_the_mesher_gates(name, pairing):
    mesh_in = _sampled(pairing, name=name).mesh_in

    assert Common_validate_te_closure(mesh_in.airfoil.coords_tensor, *pairing) == (True, "")
    assert Common_te_shape_warnings(mesh_in.airfoil.coords_tensor, *pairing) == []  # no shape or corner warning
# endregion


# region Step 2
@pytest.mark.parametrize("pairing", [C2D_O, C2D_C, GMSH_O])
def test_mesh_in_carries_exactly_the_sample(pairing, tmp_path):
    config = _CONFIG[pairing](dx_mid=0.005, le_ratio=0.02)
    before = config.model_dump()
    contour = build_study_contour(stored_shape("naca4412"), *pairing)
    sampled = _sampled(pairing, config=config, contour=contour, tmp_path=tmp_path)
    alone = Geom_sample(contour, config, max_points=CELL_CEILING // config.jmax if pairing[0] is MeshBackend.C2D else None)
    mesh_in = sampled.mesh_in

    assert mesh_in.airfoil.coords_tensor.dtype == torch.float64
    assert np.array_equal(mesh_in.airfoil.coords_tensor.numpy(), alone.points)  # the sample, unaltered
    assert mesh_in.airfoil.le_idx == alone.le_idx == sampled.report.le_idx
    assert (mesh_in.airfoil.airfoil_name, mesh_in.airfoil.chord) == ("naca4412", 0.5)
    assert (mesh_in.backend, mesh_in.topology, mesh_in.working_dir) == (*pairing, str(tmp_path))
    assert mesh_in.mesh_config is config and config.model_dump() == before  # passed through, untouched
    assert sampled.report == alone.report
# endregion


# region Step 3
def test_rejected_config_builds_no_mesh_in():
    sampled = _sampled(C2D_O, config=C2D_MeshingConfig(le_ratio=1.0))  # a coarse nose

    assert sampled.mesh_in is None and sampled.report.rejection is Rejection.TURN


def test_cell_ceiling_follows_jmax():
    n = _sampled(C2D_O).report.n_points
    tall = C2D_MeshingConfig(jmax=CELL_CEILING // n + 1)  # n points x this jmax is over the ceiling

    assert _sampled(C2D_O, config=tall).report.rejection is Rejection.POINTS
    assert _sampled(C2D_O, config=C2D_MeshingConfig(jmax=CELL_CEILING // n)).mesh_in is not None
    assert _sampled(GMSH_O).mesh_in is not None  # no jmax, no ceiling


def test_setup_mismatches_raise():
    round_ = build_study_contour(stored_shape("naca0012"), *C2D_O)

    with pytest.raises(ValueError, match="not sampled"):
        _sampled(GMSH_C, config=GMSH_CMeshingConfig(), contour=round_)
    with pytest.raises(ValueError, match="needs sharp"):
        _sampled(C2D_C, contour=round_)  # the O-grid's contour handed to the C-grid
    with pytest.raises(TypeError, match="sampling knobs"):
        _sampled(C2D_O, config=GMSH_CMeshingConfig(), contour=round_)
# endregion
