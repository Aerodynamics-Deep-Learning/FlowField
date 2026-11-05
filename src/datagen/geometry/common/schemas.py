"""
The types passed between the geometry modules.

Summary:
    - Family, Closure, ItemStatus, RejectReason: what an airfoil is, how its TE was closed, what an
      ingest did with it, why it was refused
    - Nose: how a CST fit represents the nose
    - RootAirfoil: a root .dat as read
    - Identity: the IDs an airfoil is deduplicated by, computed without fitting
    - Transform: what `normalize.Geom_normalize` did to the points
    - CSTParams, FitMetrics: the fitted shape and how well it matches its points
    - AddResult: everything `add.Geom_add_airfoil` found out about one file
    - IngestItem, IngestSummary: what `ingest.Geom_ingest_folder` did with each file, and the run
    - GeometryRecord, ShapeRecord, RejectionRecord: rows of the manifest, as `store.GeometryStore`
      reads them back
    - TETreatment: the TE a mesher needs, which `contour.Geom_build_contour` gives a stored shape
    - Rejection, SampleReport, Sample: what `sample.Geom_sample` placed, or why it refused, and how it measures
"""

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, model_validator


class Family(str, Enum):
    """Airfoil families built from their name instead of their file's points."""
    NACA4 = "naca4"
    NACA4_MOD = "naca4_mod"
    NACA5 = "naca5"
    NACA16 = "naca16"


class Closure(str, Enum):
    """How a stored shape's TE was closed."""
    NONE = "none"        # the source was already closed
    PRIED = "pried"      # an open source, closed by the linear TE term
    FORMULA = "formula"  # a family, built with its closed-TE formula


class ItemStatus(str, Enum):
    """What an ingest did with one file."""
    ADDED = "added"
    RESHAPED = "reshaped"  # a stored airfoil refitted from its raw copy under a new shape version
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class RejectReason(str, Enum):
    """Why an airfoil was not stored."""
    FORMAT = "format"
    FAMILY_MISMATCH = "family_mismatch"
    TOO_FEW_POINTS = "too_few_points"
    SURFACES_CROSS = "surfaces_cross"
    FIT_TOLERANCE = "fit_tolerance"
    NAME_TAKEN = "name_taken"  # set by the ingest, which knows the stored names
    ERROR = "error"            # set by the ingest, on an exception


class Nose(str, Enum):
    """How a CST fit represents the nose: the radius is A_0^2 / 2 per surface."""
    SEPARATE = "separate"  # each surface its own A_0: two radii, a curvature jump at the LE
    SHARED = "shared"      # one A_0: one radius, but a cambered nose's lean is out of reach
    LEM = "lem"            # one A_0 plus Kulfan's leading-edge modification term: one radius, and the lean


class TETreatment(str, Enum):
    """The TE a mesher needs, given to a stored (closed) shape for sampling."""
    SHARP = "sharp"  # the stored CST as is, closed at (1, 0)
    BLUNT = "blunt"  # opened to the study's gap and left open
    ROUND = "round"  # opened, then closed at (1, 0) by a Bezier cap on each side


class Rejection(str, Enum):
    """Why a sampling config was refused before any point was placed, so before meshing."""
    TURN = "turn"      # its spacing would turn more than THETA_MAX_DEG somewhere
    POINTS = "points"  # it needs more points than the caller allows


class RootAirfoil(BaseModel):
    """A root .dat as read: Selig-ordered (n, 2) float64 coordinates."""
    model_config = {"arbitrary_types_allowed": True}

    name: str
    coords: np.ndarray


class Identity(BaseModel):
    """
    The IDs of one root file. `geometry_id` hashes what defines the airfoil: its family string for a
    family, else its raw coordinates (then it equals `raw_hash`).
    """
    name: str
    raw_hash: str
    raw_points: int
    family: Optional[Family] = None
    family_params: Optional[dict[str, int]] = None
    geometry_id: str


class Transform(BaseModel):
    """What `Geom_normalize` did: translate by -`translate`, rotate, scale, then stretch each surface's x."""
    le_index: int
    le_inserted: bool  # the LE fell between the input's points and was added
    translate: tuple[float, float]
    rotation_deg: float
    scale: float
    stretch_upper: float
    stretch_lower: float


class CSTParams(BaseModel):
    """
    Class-function exponents, the Bernstein coefficients of each surface (LE first), and the weight of
    the leading-edge modification term both surfaces share (see `parametrization.py`).
    """
    n1: float
    n2: float
    upper: list[float]
    lower: list[float]
    le_weight: float = 0.0

    @model_validator(mode="after")
    def _same_order(self):
        if not self.upper or len(self.upper) != len(self.lower):
            raise ValueError(f"upper and lower need the same, nonzero number of coefficients, "
                             f"got {len(self.upper)} and {len(self.lower)}")
        return self

    @property
    def order(self) -> int:
        return len(self.upper) - 1


class FitMetrics(BaseModel):
    """Perpendicular distances from the fitted points to the CST, each measured on its own surface."""
    max_dev: float
    rms_dev: float
    x_at_max: float
    n_points: int
    isolation: float  # the worst distance over the median of its up-to-6 neighbours': high = a lone bad point
    zigzag: bool      # the points turn against both neighbours at the worst one: it pokes out of the curve


class AddResult(BaseModel):
    """One file through `Geom_add_airfoil`. Fields stay None past the step that rejected it."""
    model_config = {"arbitrary_types_allowed": True}

    path: str
    identity: Optional[Identity] = None
    reject: Optional[RejectReason] = None
    reason: str = ""
    closure: Optional[Closure] = None
    root_te_gap: Optional[float] = None  # the source file's gap, before closing
    transform: Optional[Transform] = None
    family_dev: Optional[float] = None
    cst: Optional[CSTParams] = None
    fit: Optional[FitMetrics] = None
    source_dev: Optional[float] = None    # worst distance from the source's own points, before closing, to the CST
    te_wedge_deg: Optional[float] = None  # the CST's TE wedge angle
    norm_coords: Optional[np.ndarray] = None
    norm_hash: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.reject is None


class IngestItem(BaseModel):
    """What an ingest did with one file."""
    model_config = {"arbitrary_types_allowed": True}

    source_relpath: str
    name: str
    status: ItemStatus
    reason: Optional[str] = None       # a RejectReason value, or what matched for a duplicate
    detail: str = ""                   # the reason in words
    geometry_id: Optional[str] = None  # the airfoil added or duplicated
    report_path: Optional[str] = None  # relative to data/; None on a dry run
    result: Optional[AddResult] = None


class IngestSummary(BaseModel):
    """One ingest run."""
    ingest_id: str
    started_at: str
    source_root: str
    shape_version: str
    git_commit: str
    dry_run: bool
    items: list[IngestItem] = []
    summary_path: Optional[str] = None  # relative to data/; None on a dry run

    def count(self, status: ItemStatus) -> int:
        return sum(item.status is status for item in self.items)


class GeometryRecord(BaseModel):
    """A `geometries` row: one stored airfoil."""
    geometry_id: str
    name: str
    family: Optional[Family] = None
    family_params: Optional[dict[str, int]] = None
    raw_path: str
    raw_hash: str
    raw_points: int
    ingest_id: str


class ShapeRecord(BaseModel):
    """A `geometry_shapes` row: an airfoil's stored shape under one shape version."""
    geometry_id: str
    shape_version: str
    cst: CSTParams
    rotation_deg: float
    closure: Closure
    root_te_gap: float
    source_dev: float
    fit_max: float
    te_wedge_deg: float
    norm_path: str
    norm_hash: str


class RejectionRecord(BaseModel):
    """
    A `rejected_shapes` row: a file the ingest refused under one shape version, with the columns of
    `geometries` and `geometry_shapes`; the measurements stay None past the step that rejected it.
    """
    raw_hash: str
    shape_version: str
    geometry_id: str
    name: str
    family: Optional[Family] = None
    family_params: Optional[dict[str, int]] = None
    raw_path: str
    raw_points: int
    reason: RejectReason
    detail: str
    cst: Optional[CSTParams] = None
    rotation_deg: Optional[float] = None
    closure: Optional[Closure] = None
    root_te_gap: Optional[float] = None
    source_dev: Optional[float] = None
    fit_max: Optional[float] = None
    te_wedge_deg: Optional[float] = None
    norm_path: Optional[str] = None
    norm_hash: Optional[str] = None
    ingest_id: str  # the run that last judged it
    report_path: str


class SampleReport(BaseModel):
    """
    What one sampling produced and how it measures, in plain numbers so it stores as JSON. Lengths in
    chords. The measurements stay None for a rejected config, which placed no points.
    """
    sampler_version: str
    shape_version: str
    geometry_id: str
    te: TETreatment
    gap: Optional[float]
    weld_x: Optional[float]
    te_cap_limit: Optional[float]
    dx_mid: float
    le_spacing: float
    te_spacing: float
    te_clamped: bool                         # te_ratio * dx_mid was more than the caps let pass the pre-check
    cluster_width: float
    precheck_turn: float                     # largest curvature x spacing over its limit, minus 1: > 0 rejects
    x_at_precheck_turn: float                # where that is: the nose, or wherever a rejection came from
    n_required: int                          # points the spacing calls for, known before any is placed
    rejection: Optional[Rejection] = None
    n_points: Optional[int] = None
    le_idx: Optional[int] = None
    stretch_panels: Optional[list[int]] = None  # panels between consecutive pins
    max_panel: Optional[float] = None
    x_at_max_panel: Optional[float] = None
    min_panel: Optional[float] = None
    max_turn_deg: Optional[float] = None
    x_at_max_turn: Optional[float] = None
    max_dev: Optional[float] = None          # farthest the curve strays from its panels
    x_at_max_dev: Optional[float] = None
    max_panel_ratio: Optional[float] = None  # larger over smaller of neighbouring panels


class Sample(BaseModel):
    """The points one sampling placed (Selig order, closed TEs repeating the first last), or none if rejected."""
    model_config = {"arbitrary_types_allowed": True}

    points: Optional[np.ndarray] = None
    le_idx: Optional[int] = None
    report: SampleReport

    @property
    def accepted(self) -> bool:
        return self.report.rejection is None
