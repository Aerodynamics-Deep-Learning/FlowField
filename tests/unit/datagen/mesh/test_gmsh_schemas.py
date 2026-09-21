"""
Tests for `gmsh.schemas.GMSH_In`'s topology-driven `meshing_config` resolver, i.e.:

Step 1: Ensure `topology` resolves the config class when none is given
    - test_meshing_config_defaults_per_topology
    - test_meshing_config_dict_is_coerced_to_the_topologys_class
Step 2: Ensure a correctly-matched topology/config pair is accepted
    - test_cmesh_topology_with_cmesh_config_accepted
    - test_omesh_topology_with_omesh_config_accepted
Step 3: Ensure a config belonging to the other topology is rejected
    - test_meshing_config_wrong_class_rejected
Step 4: Ensure a mistyped keyword can't silently fall back to the default config
    - test_unknown_keyword_rejected
"""

import torch
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import (
    GMSH_In, GMSH_Topology, GMSH_CMeshingConfig, GMSH_OMeshingConfig
)


def _airfoil() -> Airfoil:
    coords = torch.tensor([[1.0, 0.0], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, 0.0]], dtype=torch.float32)
    return Airfoil(airfoil_name="stub", coords_tensor=coords, chord=1.0, le_idx=2)


def _freestream() -> Freestream:
    return Freestream(alpha=0.0, Re=5e6, mach=0.5)


# region Step 1
@pytest.mark.parametrize("topology, expected", [
    (GMSH_Topology.CGRD, GMSH_CMeshingConfig),
    (GMSH_Topology.OGRD, GMSH_OMeshingConfig),
])
def test_meshing_config_defaults_per_topology(topology, expected):
    data = GMSH_In(
        airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp", topology=topology,
    )
    assert isinstance(data.meshing_config, expected)


def test_meshing_config_dict_is_coerced_to_the_topologys_class():
    # Both config classes are all-defaults, so a bare union would resolve a dict by first match.
    data = GMSH_In(
        airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
        topology=GMSH_Topology.OGRD, meshing_config={"nx_afoil": 55},
    )
    assert isinstance(data.meshing_config, GMSH_OMeshingConfig)
    assert data.meshing_config.nx_afoil == 55
# endregion


# region Step 2
def test_cmesh_topology_with_cmesh_config_accepted():
    data = GMSH_In(
        airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
        topology=GMSH_Topology.CGRD, meshing_config=GMSH_CMeshingConfig(),
    )
    assert isinstance(data.meshing_config, GMSH_CMeshingConfig)


def test_omesh_topology_with_omesh_config_accepted():
    data = GMSH_In(
        airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
        topology=GMSH_Topology.OGRD, meshing_config=GMSH_OMeshingConfig(),
    )
    assert isinstance(data.meshing_config, GMSH_OMeshingConfig)
# endregion


# region Step 3
def test_meshing_config_wrong_class_rejected():
    with pytest.raises(ValidationError, match="topology=OGRD requires GMSH_OMeshingConfig"):
        GMSH_In(
            airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
            topology=GMSH_Topology.OGRD, meshing_config=GMSH_CMeshingConfig(),
        )
# endregion


# region Step 4
def test_unknown_keyword_rejected():
    # meshing_config defaults when omitted, so without extra="forbid" the old `cmesh_config=`
    # keyword is dropped silently and the mesh is built from defaults, not the config passed.
    with pytest.raises(ValidationError, match="cmesh_config"):
        GMSH_In(
            airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
            topology=GMSH_Topology.CGRD, cmesh_config=GMSH_CMeshingConfig(),
        )
# endregion
