"""
The rules a design must pass before it is meshed, so before p_mesh is asked about it: p_mesh is
P(meshes | passed these rules), so the campaign sets `MeshAttempt.geometry_rejected` from these same
functions, and the classifier runs them before predicting.

    AL_geometry_rules   the shape alone: a design of the pipeline's form, positive thickness, no crossed TE
    AL_mesh_rules       the production settings too: the contour with the pairing's TE and gap, then the
                        sampler's pre-check with the frozen knobs (turn, point count)

The ingest's other gates (fit tolerance, a suspect point, the family match) judge a fit to a file's points,
which a generated design doesn't have.
"""

from typing import NamedTuple, Optional

from src.datagen.active_learning.design_vector import AL_cst_to_free
from src.datagen.active_learning.labels import ProductionMesh
from src.datagen.geometry.build.checks import Geom_check_thickness
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.schemas import CSTParams
from src.datagen.geometry.sample.contour import ContourError, Geom_build_cst_contour
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.optimizer.common.constants import PINNED
from src.datagen.meshing.optimizer.evaluate.sampling import TE_FOR_PAIRING

# Names the rules below; bump it on any change to either group, so rows judged under other rules aren't mixed
RULES_VERSION = "rules-r1"


class RuleVerdict(NamedTuple):
    """Whether a design passed, and if not the first rule it failed and why."""
    ok: bool
    rule: Optional[str] = None  # "thickness", "contour" or "precheck"
    reason: str = ""


def AL_geometry_rules(cst: CSTParams) -> RuleVerdict:
    """
    The rules on the shape alone.

    Args:
        cst (CSTParams): The design

    Returns:
        RuleVerdict: ok, or "thickness" with the crossing

    Raises:
        ValueError: If it isn't a design (another nose, order or class exponents): a caller bug
    """
    AL_cst_to_free(cst)
    ok, _, reason = Geom_check_thickness(cst)
    if not ok:
        return RuleVerdict(False, "thickness", reason)
    return RuleVerdict(True)


def AL_mesh_rules(cst: CSTParams, production: ProductionMesh) -> RuleVerdict:
    """
    The rules that depend on the production settings: the contour the pairing's TE makes, then the sampler's
    pre-check, as `Optim_sample_for_mesher` runs them before c2d.

    Args:
        cst (CSTParams): A design that passed `AL_geometry_rules`
        production (ProductionMesh): The settings row (pairing, TE gap, cell ceiling) and the frozen knobs

    Returns:
        RuleVerdict: ok, "contour" if the shape can't take the TE, or "precheck" if the sampler refuses it

    Raises:
        ValueError: If the pairing isn't sampled
    """
    te = TE_FOR_PAIRING.get((production.backend, production.topology))
    if te is None:
        raise ValueError(f"{production.backend.value}/{production.topology.value} is not sampled")
    try:
        contour = Geom_build_cst_contour(cst, te, production.te_gap, geometry_id="design",
                                         shape_version=SHAPE_VERSION)
    except ContourError as e:
        return RuleVerdict(False, "contour", str(e))

    config = C2D_MeshingConfig(**{**PINNED, **production.knobs})
    report = Geom_sample(contour, config, max_points=production.cell_ceiling // config.jmax).report
    if report.rejection is not None:
        return RuleVerdict(False, "precheck", f"sampler pre-check: {report.rejection.value} "
                                              f"(turn {report.precheck_turn:.3g}, {report.n_required} points)")
    return RuleVerdict(True)
