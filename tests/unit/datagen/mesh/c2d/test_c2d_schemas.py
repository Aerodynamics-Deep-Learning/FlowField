"""
Tests for `c2d.schemas.C2D_In`'s topology/topo-syncing validator, i.e.:

Step 1: Ensure an unset `meshing_config.topo` is filled in from `topology`
    - test_unset_topo_is_filled_in_from_topology
Step 2: Ensure a `meshing_config.topo` matching `topology` is accepted as-is
    - test_matching_topo_is_accepted
Step 3: Ensure a `meshing_config.topo` conflicting with `topology` is rejected
    - test_mismatched_topo_is_rejected
Step 4: Ensure `topology` itself is required
    - test_missing_topology_rejected
Step 5: Ensure the sync leaves the caller's config untouched
    - test_one_config_serves_both_topologies
"""

import torch
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.c2d.schemas import C2D_In, C2D_MeshingConfig, C2D_Topology


def _airfoil() -> Airfoil:
    coords = torch.tensor([[1.0, 0.001], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, -0.001]], dtype=torch.float32)
    return Airfoil(airfoil_name="stub", coords_tensor=coords, chord=1.0, le_idx=2)


def _freestream() -> Freestream:
    return Freestream(alpha=0.0, Re=2e6, mach=0.3)


# region Step 1
def test_unset_topo_is_filled_in_from_topology():
    data = C2D_In(
        airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
        topology=C2D_Topology.CGRD, meshing_config=C2D_MeshingConfig(),
    )
    assert data.meshing_config.topo == "CGRD"
# endregion


# region Step 2
def test_matching_topo_is_accepted():
    data = C2D_In(
        airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
        topology=C2D_Topology.OGRD, meshing_config=C2D_MeshingConfig(topo="OGRD"),
    )
    assert data.meshing_config.topo == "OGRD"
# endregion


# region Step 3
def test_mismatched_topo_is_rejected():
    with pytest.raises(ValidationError, match="does not match"):
        C2D_In(
            airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
            topology=C2D_Topology.OGRD, meshing_config=C2D_MeshingConfig(topo="CGRD"),
        )
# endregion


# region Step 4
def test_missing_topology_rejected():
    with pytest.raises(ValidationError):
        C2D_In(
            airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
            meshing_config=C2D_MeshingConfig(),
        )
# endregion


# region Step 5
def test_one_config_serves_both_topologies():
    # pydantic keeps a passed model instance as-is, so filling `topo` in place used to pin the
    # caller's config to the first topology and fail the second as a conflict
    config = C2D_MeshingConfig()
    for topology in (C2D_Topology.OGRD, C2D_Topology.CGRD):
        data = C2D_In(
            airfoil=_airfoil(), freestream=_freestream(), working_dir="/tmp",
            topology=topology, meshing_config=config,
        )
        assert data.meshing_config.topo == topology.value

    assert config.topo is None
# endregion
