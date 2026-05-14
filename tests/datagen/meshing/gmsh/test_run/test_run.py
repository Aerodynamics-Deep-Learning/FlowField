import pytest
import torch
import numpy as np

from src.datagen.schemas import *
from src.datagen.meshing.gmsh import *

@pytest.fixture
def dummy_blunted_airfoil():
    coords_tensor = torch.tensor(
        ([1.0, 0.001],
        [0.5, 0.05],
        [0.0, 0.0],
        [0.5, -0.05],
        [1.0, -0.001])
    , dtype=torch.float32)

    return Airfoil(
        airfoil_name="test_wedge",
        coords_tensor=coords_tensor,
        chord=1.0,
        le_idx=2
    )

@pytest.fixture
def dummy_freestream():
    return Freestream(
        alpha=0.0,
        Re=6e6,
        mach=0.15,
        altitude_m=500.0
    )

def test_gmsh_mesh_generator_success(dummy_blunted_airfoil, dummy_freestream, tmp_path):
    
    meshing_config = GMSH_MeshingConfig()

    gmsh_in = GMSH_In(
        airfoil=dummy_blunted_airfoil,
        freestream=dummy_freestream,
        working_dir=str(tmp_path),
        meshing_config=meshing_config
    )

    out = GMSH_MeshGenerator(gmsh_in)

    error_msg = "No exception file generated."
    
    if out.flag == GMSH_ExitFlag.FATAL_ERROR and out.verbose_list[2] is not None:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    assert out.mesh_path is not None
    assert out.min_mesh_quality > 0.0, f"Mesh generated but quality is strictly negative/poor: {out.min_mesh_quality}"

    mesh_file = tmp_path / "test_wedge_mesh.su2"
    brep_file = tmp_path / "test_wedge.brep"
    
    assert mesh_file.exists(), "SU2 mesh file was not written to disk"
    assert mesh_file.stat().st_size > 0, "SU2 mesh file is empty"
    assert brep_file.exists(), "BREP dump was not written to disk"

def test_gmsh_mesh_generator_singularity_abort(dummy_freestream, tmp_path):
    """
    Tests the defensive geometry check
    """
    # Sharp TE (Coordinates converge exactly at [1.0, 0.0])
    sharp_coords = np.array([
        [1.0, 0.0],
        [0.5, 0.05],
        [0.0, 0.0],
        [0.5, -0.05],
        [1.0, 0.0]
    ], dtype=np.float32)
    
    sharp_airfoil = Airfoil(
        airfoil_name="sharp_wedge",
        coords_tensor=torch.from_numpy(sharp_coords),
        le_idx=2,
        chord=1.0
    )
    
    gmsh_in = GMSH_In(
        airfoil=sharp_airfoil,
        freestream=dummy_freestream,
        meshing_config=GMSH_MeshingConfig(),
        working_dir=str(tmp_path)
    )
    
    out = GMSH_MeshGenerator(gmsh_in)
    
    assert out.flag == GMSH_ExitFlag.FATAL_ERROR, "Failed to catch sharp TE singularity."
    assert out.mesh_path is None, "Mesh path should be None upon failure."