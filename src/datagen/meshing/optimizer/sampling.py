"""
The meshing side of Sequence 3 (see `geometry/__init__.py`): a study's contour, and each config's sample
turned into the mesher's input.

    build_study_contour  once per study: the stored shape with the TE its (backend, topology) needs, or
                         None for a pairing that is not sampled
    sample_for_mesher    each evaluation: samples the contour with the config's knobs, under the cell
                         ceiling, and builds the MeshIn from the points; a rejected config builds none

The MeshIn is only ever built here, from sampled points: there is no placeholder tensor to forget to
replace. Sits above both geometry and meshing, since the mesh configs import the knobs from geometry.
"""

from typing import Optional

import torch
from pydantic import BaseModel

from src.datagen.geometry.common.schemas import SampleReport, ShapeRecord, TETreatment
from src.datagen.geometry.sample.contour import Contour, Geom_build_contour
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshBackend, MeshIn, MeshingConfig, MeshTopology
from src.datagen.schemas import Airfoil, Freestream

# The TE each pairing meshes; None: not sampled (the gmsh C-mesh, until its anchor indices become x/c)
TE_FOR_PAIRING = {
    (MeshBackend.C2D, MeshTopology.OGRD): TETreatment.ROUND,
    (MeshBackend.GMSH, MeshTopology.OGRD): TETreatment.ROUND,
    (MeshBackend.C2D, MeshTopology.CGRD): TETreatment.SHARP,
    (MeshBackend.GMSH, MeshTopology.CGRD): None,
}

# The study's TE gap [c] for a round or blunt TE: a study hyperparameter, recorded with the study
TE_GAP_DEFAULT = 0.004

# Most surface points x jmax a config may ask for; more is rejected before meshing
CELL_CEILING = 300_000


class SampledInput(BaseModel):
    """One evaluation's input: the MeshIn to mesh, or none if the config was rejected, and the sample's report."""
    model_config = {"arbitrary_types_allowed": True}

    mesh_in: Optional[MeshIn] = None
    report: SampleReport


def build_study_contour(shape: ShapeRecord, backend: MeshBackend, topology: MeshTopology,
                        gap: float = TE_GAP_DEFAULT) -> Optional[Contour]:
    """
    The contour a study samples, once.

    Args:
        shape (ShapeRecord): The airfoil's stored shape under the current shape version
        backend (MeshBackend): The study's mesher
        topology (MeshTopology): Its topology
        gap (float): The study's TE gap [c]

    Returns:
        Contour: The shape with the pairing's TE, or None if the pairing is not sampled

    Raises:
        ValueError: If no TE is known for the pairing
        ContourError: If the shape cannot take that TE: the study's setup fails
    """
    if (backend, topology) not in TE_FOR_PAIRING:
        raise ValueError(f"no TE is known for backend={backend}/topology={topology}")
    te = TE_FOR_PAIRING[(backend, topology)]
    return None if te is None else Geom_build_contour(shape, te, gap)


def sample_for_mesher(config: MeshingConfig, contour: Contour, *, airfoil_name: str, chord: float,
                      freestream: Freestream, backend: MeshBackend, topology: MeshTopology,
                      working_dir: str) -> SampledInput:
    """
    Samples a study's contour with one config's knobs and builds the mesher's input from the points.

    Args:
        config (MeshingConfig): The evaluation's validated mesh config, carrying the sampling knobs
        contour (Contour): The study's contour, from `build_study_contour`
        airfoil_name (str): The name the mesher writes its files under
        chord (float): The airfoil's physical chord
        freestream (Freestream): The study's freestream
        backend (MeshBackend): The study's mesher
        topology (MeshTopology): Its topology
        working_dir (str): The evaluation's own, existing run directory

    Returns:
        SampledInput: The MeshIn (None if a pre-check rejected the config) and the sample's report

    Raises:
        ValueError: If the pairing is not sampled, or the contour has another pairing's TE (setup bugs)
        TypeError: If the config carries no sampling knobs
        SamplerAssertionError: If the sampler broke a rule it guarantees (a bug)
    """
    te = TE_FOR_PAIRING.get((backend, topology))
    if te is None:
        raise ValueError(f"backend={backend}/topology={topology} is not sampled")
    if contour.te is not te:
        raise ValueError(f"the contour has a {contour.te.value} TE, {backend.value}/{topology.value} needs {te.value}")
    if not isinstance(config, SamplingFields):
        raise TypeError(f"{type(config).__name__} carries no sampling knobs")

    max_points = CELL_CEILING // config.jmax if isinstance(config, C2D_MeshingConfig) else None
    sample = Geom_sample(contour, config, max_points=max_points)
    if not sample.accepted:
        return SampledInput(report=sample.report)
    airfoil = Airfoil(airfoil_name=airfoil_name, coords_tensor=torch.tensor(sample.points, dtype=torch.float64),
                      chord=chord, le_idx=sample.le_idx)
    mesh_in = MeshIn(airfoil=airfoil, freestream=freestream, working_dir=working_dir, backend=backend,
                     topology=topology, mesh_config=config)
    return SampledInput(mesh_in=mesh_in, report=sample.report)
