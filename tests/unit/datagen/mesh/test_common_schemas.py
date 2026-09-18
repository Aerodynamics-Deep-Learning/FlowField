"""
Tests for `common.schemas.MeshIn`'s single `mesh_config` field, i.e.:

Step 1: Ensure the (backend, topology) pairing resolves the config class
    - test_mesh_config_defaults_per_pairing
    - test_mesh_config_dict_is_coerced_to_the_pairings_class
    - test_mesh_config_instance_passes_through
Step 2: Ensure a config that doesn't belong to the pairing is rejected
    - test_mesh_config_wrong_class_rejected
Step 3: Ensure the resolution holds when the model is validated from a raw payload
    - test_mesh_config_resolved_when_validated_from_payload
Step 4: Ensure a mistyped keyword can't silently fall back to the default config
    - test_unknown_keyword_rejected
"""

import tempfile

import pytest
import torch
from pydantic import ValidationError

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import GMSH_CMeshingConfig, GMSH_OMeshingConfig
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshIn, MeshBackend, MeshTopology


def _mesh_in(backend, topology, **kwargs) -> MeshIn:
    coords = torch.tensor([[1.0, 0.0063], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], [1.0, -0.0063]],
                          dtype=torch.float32)
    return MeshIn(
        airfoil=Airfoil(airfoil_name="naca0012", coords_tensor=coords, chord=1.0, le_idx=2),
        freestream=Freestream(alpha=0.0, Re=5e6, mach=0.3, altitude_m=1.0),
        working_dir=tempfile.gettempdir(),
        backend=backend,
        topology=topology,
        **kwargs,
    )


# region Step 1
@pytest.mark.parametrize("backend, topology, expected", [
    (MeshBackend.GMSH, MeshTopology.CGRD, GMSH_CMeshingConfig),
    (MeshBackend.GMSH, MeshTopology.OGRD, GMSH_OMeshingConfig),
    (MeshBackend.C2D, MeshTopology.CGRD, C2D_MeshingConfig),
    (MeshBackend.C2D, MeshTopology.OGRD, C2D_MeshingConfig),
])
def test_mesh_config_defaults_per_pairing(backend, topology, expected):
    # Omitting mesh_config entirely has to yield the class *this* pairing meshes with; gmsh's two
    # topologies take structurally different configs, so a wrong default is a wrong mesh.
    data = _mesh_in(backend, topology)

    assert isinstance(data.mesh_config, expected)


def test_mesh_config_dict_is_coerced_to_the_pairings_class():
    # Every config class is all-defaults, so a bare union would resolve {} by first match. The
    # pairing is what disambiguates it.
    data = _mesh_in(MeshBackend.GMSH, MeshTopology.OGRD, mesh_config={"nx_afoil": 55})

    assert isinstance(data.mesh_config, GMSH_OMeshingConfig)
    assert data.mesh_config.nx_afoil == 55


def test_mesh_config_instance_passes_through():
    config = GMSH_OMeshingConfig(nx_afoil=200)
    data = _mesh_in(MeshBackend.GMSH, MeshTopology.OGRD, mesh_config=config)

    assert isinstance(data.mesh_config, GMSH_OMeshingConfig)
    assert data.mesh_config.nx_afoil == 200
# endregion


# region Step 2
def test_mesh_config_wrong_class_rejected():
    with pytest.raises(ValidationError, match="requires GMSH_OMeshingConfig, got GMSH_CMeshingConfig"):
        _mesh_in(MeshBackend.GMSH, MeshTopology.OGRD, mesh_config=GMSH_CMeshingConfig())
# endregion


# region Step 3
def test_mesh_config_resolved_when_validated_from_payload():
    # `MeshIn` can't round-trip through JSON (Airfoil.coords_tensor is a torch.Tensor, hence the
    # model's arbitrary_types_allowed), but it is still validated from raw payloads whose configs
    # arrive as plain dicts. Without the pairing-aware resolver those land on the union's first
    # member, silently turning a stored O-mesh config into a C-mesh one.
    coords = torch.tensor([[1.0, 0.0063], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], [1.0, -0.0063]],
                          dtype=torch.float32)
    restored = MeshIn.model_validate({
        "airfoil": Airfoil(airfoil_name="naca0012", coords_tensor=coords, chord=1.0, le_idx=2),
        "freestream": Freestream(alpha=0.0, Re=5e6, mach=0.3, altitude_m=1.0),
        "working_dir": tempfile.gettempdir(),
        "backend": "gmsh",
        "topology": "OGRID",
        "mesh_config": {"nx_afoil": 77},
    })

    assert isinstance(restored.mesh_config, GMSH_OMeshingConfig)
    assert restored.mesh_config.nx_afoil == 77
# endregion


# region Step 4
def test_unknown_keyword_rejected():
    # mesh_config defaults when omitted, so without extra="forbid" a stale or mistyped keyword is
    # dropped silently and the mesh is built from defaults instead of the config that was passed.
    with pytest.raises(ValidationError, match="gmsh_cmesh_config"):
        _mesh_in(MeshBackend.GMSH, MeshTopology.CGRD, gmsh_cmesh_config=GMSH_CMeshingConfig())
# endregion
