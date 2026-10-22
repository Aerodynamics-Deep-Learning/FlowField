"""
The sampling knobs a mesh config carries (c2d, gmsh O-mesh), defined once here and inherited flat by the
configs. Imports only pydantic and the constants, so a config pays nothing else for them.
"""

from pydantic import BaseModel, Field

from src.datagen.geometry.common.constants import DX_MAX


class SamplingFields(BaseModel):
    """
    How densely, and where, `Geom_sample` places a mesher's surface points. Lengths in chords. The defaults
    pass the pre-checks on every stored airfoil (~600 points); the sharpest noses (r ~6e-4c) set le_ratio.
    """
    model_config = {"extra": "forbid"}

    dx_mid: float = Field(DX_MAX, gt=0.0, le=DX_MAX, description="Panel length away from the LE and TE clustering; "
                          "no panel is longer, so it is bounded by DX_MAX")
    le_ratio: float = Field(0.025, gt=0.0, le=1.0, description="Panel length at the LE, as a share of dx_mid")
    te_ratio: float = Field(0.05, gt=0.0, le=1.0, description="Panel length at the TE, as a share of dx_mid; held "
                            "under a round TE's cap limit")
    cluster_width: float = Field(0.05, gt=0.0, le=0.5, description="Arc length over which the LE and TE spacing "
                                 "grows back to dx_mid")
