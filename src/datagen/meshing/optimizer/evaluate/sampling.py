"""
The meshing side of Sequence 3 (see `geometry/__init__.py`): a study's contour, and each config's sample
turned into the mesher's input.

    Optim_build_study_contour  once per study: the stored shape with the TE its (backend, topology) needs,
                               or None for a pairing that is not sampled
    Optim_sample_for_mesher    each evaluation: samples the contour with the config's knobs, under the cell
                               ceiling, and builds the MeshIn from the points; a rejected config builds none
    Optim_sample_for_study     the same, with the rest of the input from the settings row: what the executor
                               and a re-mesh both call, so the two cannot build different inputs
    Optim_safe_name            an airfoil name as the mesher's files take it

The MeshIn is only ever built here, from sampled points: there are no placeholder coords to forget to
replace. Sits above both geometry and meshing, since the mesh configs import the knobs from geometry.
"""

import re
from typing import Optional

from pydantic import BaseModel

from src.datagen.geometry.common.schemas import SampleReport, ShapeRecord, TETreatment
from src.datagen.geometry.sample.contour import Contour, Geom_build_contour
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshBackend, MeshIn, MeshingConfig, MeshTopology, QualityLimits
from src.datagen.meshing.optimizer.common.constants import CELL_CEILING, CHORD, TE_GAP_DEFAULT
from src.datagen.schemas import Airfoil, Freestream

# The TE each pairing meshes; None: not sampled (the gmsh C-mesh, until its anchor indices become x/c)
TE_FOR_PAIRING = {
    (MeshBackend.C2D, MeshTopology.OGRD): TETreatment.ROUND,
    (MeshBackend.GMSH, MeshTopology.OGRD): TETreatment.ROUND,
    (MeshBackend.C2D, MeshTopology.CGRD): TETreatment.SHARP,
    (MeshBackend.GMSH, MeshTopology.CGRD): None,
}


class SampledInput(BaseModel):
    """One evaluation's input: the MeshIn to mesh, or none if the config was rejected, and the sample's report."""
    model_config = {"arbitrary_types_allowed": True}

    mesh_in: Optional[MeshIn] = None
    report: SampleReport


def Optim_build_study_contour(shape: ShapeRecord, backend: MeshBackend, topology: MeshTopology,
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


def Optim_sample_for_mesher(config: MeshingConfig, contour: Contour, *, airfoil_name: str, chord: float,
                            freestream: Freestream, backend: MeshBackend, topology: MeshTopology,
                            working_dir: str, count_limits: Optional[QualityLimits] = None,
                            write_vtk: bool = True) -> SampledInput:
    """
    Samples a study's contour with one config's knobs and builds the mesher's input from the points.

    Args:
        config (MeshingConfig): The evaluation's validated mesh config, carrying the sampling knobs
        contour (Contour): The study's contour, from `Optim_build_study_contour`
        airfoil_name (str): The name the mesher writes its files under
        chord (float): The airfoil's physical chord
        freestream (Freestream): The study's freestream
        backend (MeshBackend): The study's mesher
        topology (MeshTopology): Its topology
        working_dir (str): The evaluation's own, existing run directory
        count_limits (QualityLimits | None): Limits the quality summary counts bad cells against;
            None counts against constants.py
        write_vtk (bool): Whether the mesher also writes the diagnostic .vtk

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
    airfoil = Airfoil(airfoil_name=airfoil_name, coords=sample.points, chord=chord, le_idx=sample.le_idx)
    mesh_in = MeshIn(airfoil=airfoil, freestream=freestream, working_dir=working_dir, backend=backend,
                     topology=topology, mesh_config=config, count_limits=count_limits or QualityLimits(),
                     write_vtk=write_vtk)
    return SampledInput(mesh_in=mesh_in, report=sample.report)


def Optim_sample_for_study(settings: dict, config: MeshingConfig, contour: Contour, airfoil_name: str,
                           working_dir: str, write_vtk: bool = False) -> SampledInput:
    """
    `Optim_sample_for_mesher` as the study calls it: the freestream at the settings' design Re, the
    settings' gates as the count limits, its backend and topology, `CHORD`, and a file-safe name.

    Args:
        settings (dict): The settings row the run is meshed under
        config (MeshingConfig): The run's validated mesh config
        contour (Contour): The study's contour, from `Optim_build_study_contour`
        airfoil_name (str): The airfoil's name; the mesher's files take `Optim_safe_name`'s copy
        working_dir (str): The run's own, existing directory
        write_vtk (bool): Whether the mesher also writes the diagnostic .vtk

    Returns:
        SampledInput: As `Optim_sample_for_mesher` returns it, raising as it does
    """
    s = settings
    # c2d reads only Re; Freestream requires an alpha and a mach regardless
    freestream = Freestream(alpha=0.0, Re=s["design_re"], mach=0.5)
    limits = QualityLimits(skew_max=s["gate_skew_max"], ortho_min=s["gate_ortho_min"], jac_min=s["gate_jac_min"],
                           size_ratio_max=s["gate_size_ratio_max"])
    return Optim_sample_for_mesher(config, contour, airfoil_name=Optim_safe_name(airfoil_name), chord=CHORD,
                                   freestream=freestream, backend=MeshBackend(s["backend"]),
                                   topology=MeshTopology(s["topology"]), working_dir=working_dir,
                                   count_limits=limits, write_vtk=write_vtk)


def Optim_safe_name(name: str) -> str:
    """`name` with every character but letters, digits and underscores made an underscore."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)
