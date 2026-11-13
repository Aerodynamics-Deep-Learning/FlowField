"""
Inputs for the active-learning tests: a production settings row hashed as the optimizer hashes one, and
synthetic p_mesh rows around the cusp, labelled by a known smooth boundary so a classifier's answer can
be checked.
"""

from typing import Optional

import numpy as np

from src.datagen.active_learning.design_vector import AL_cst_to_free, AL_free_to_cst
from src.datagen.active_learning.labels import MeshRow, ProductionMesh
from src.datagen.active_learning.rules import RULES_VERSION, AL_geometry_rules
from src.datagen.meshing.optimizer.common.constants import CELL_CEILING, DESIGN_RE, TE_GAP_DEFAULT
from src.datagen.meshing.optimizer.common.settings import Optim_settings_hash
from tests.geometry_shapes import CUSP

KNOBS = {"jmax": 147.0, "dx_mid": 0.006}
BASE_X = AL_cst_to_free(CUSP)
SPREAD = 0.02                 # perturbation of each free value: small enough that the shapes stay valid
BOUNDARY = np.zeros(len(BASE_X))
BOUNDARY[[1, 2, 3, 10]] = 1.0  # label 1 where these perturbations sum above zero: a smooth, learnable edge


def production_settings(**over) -> dict:
    """A settings row with the columns the rules read, hashed by `Optim_settings_hash`."""
    values = {"backend": "c2d", "topology": "OGRID", "te_gap": TE_GAP_DEFAULT, "cell_ceiling": CELL_CEILING,
              "design_re": DESIGN_RE, **over}
    values["settings_id"] = Optim_settings_hash(values)
    return values


def production(knobs: Optional[dict] = None, **settings) -> ProductionMesh:
    return ProductionMesh(settings=production_settings(**settings), knobs=dict(KNOBS if knobs is None else knobs))


PRODUCTION = production()


def boundary_score(x: np.ndarray) -> float:
    """Where a design sits against the synthetic boundary, in units of SPREAD: positive is label 1."""
    return float(BOUNDARY @ (np.asarray(x) - BASE_X)) / SPREAD


def design_at(score: float) -> np.ndarray:
    """The free values at `score` along the boundary's normal, through the cusp."""
    return BASE_X + score * SPREAD * BOUNDARY / (BOUNDARY @ BOUNDARY)


def synthetic_rows(n: int, seed: int = 0, flip: float = 0.0, *, production_mesh: ProductionMesh = PRODUCTION,
                   rules_version: str = RULES_VERSION, prefix: str = "g") -> list[MeshRow]:
    """
    `n` rows around the cusp that pass the geometry rules, labelled by the boundary, a fraction `flip`
    of labels flipped (noise).
    """
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        x = BASE_X + rng.normal(0.0, SPREAD, len(BASE_X))
        if x[8] < x[16] or not AL_geometry_rules(AL_free_to_cst(x)).ok:  # keep the TE uncrossed
            continue
        label = int(boundary_score(x) > 0)
        if rng.random() < flip:
            label = 1 - label
        rows.append(MeshRow(geometry_id=f"{prefix}{len(rows):04d}", settings_id=production_mesh.settings_id,
                            rules_version=rules_version, attempt_id=f"{prefix}{len(rows):04d}-a1", x=x.tolist(),
                            label=label))
    return rows
