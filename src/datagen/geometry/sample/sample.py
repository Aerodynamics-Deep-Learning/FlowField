"""
Sequence 3: one sampling of a study's contour for its mesher, from a config's knobs to the points.

    target spacing -> turn pre-check -> panels per stretch, so n -> point cap
    -> stations -> points evaluated on the contour -> post-checks -> report

A config failing a pre-check is rejected with no points placed: cheap, and never meshed. Pure and
deterministic: the same contour and knobs always give the same points.
"""

from typing import Optional

from src.datagen.geometry.common.constants import SAMPLER_VERSION
from src.datagen.geometry.common.schemas import Rejection, Sample, SampleReport
from src.datagen.geometry.sample.checks import Geom_sample_checks
from src.datagen.geometry.sample.contour import Contour, PinKind
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.place import (
    Geom_stations, Geom_stretch_panels, Geom_target_spacing, Geom_turn_precheck,
)


def Geom_sample(contour: Contour, fields: SamplingFields, max_points: Optional[int] = None) -> Sample:
    """
    Places a mesher's surface points on a contour, as a config's sampling knobs ask.

    Args:
        contour (Contour): The study's contour
        fields (SamplingFields): The config's sampling knobs
        max_points (int): Most points the caller accepts (its cell ceiling), or no limit

    Returns:
        Sample: The points, le_idx and report; no points if a pre-check rejected the config

    Raises:
        SamplerAssertionError: If the placed points break a rule the construction guarantees
    """
    spacing = Geom_target_spacing(contour, fields)
    precheck, precheck_x = Geom_turn_precheck(contour, spacing)
    panels = Geom_stretch_panels(contour, spacing)
    n = 1 + sum(panels)
    if precheck > 0.0:
        rejection = Rejection.TURN
    elif max_points is not None and n > max_points:
        rejection = Rejection.POINTS
    else:
        rejection = None
    report = dict(sampler_version=SAMPLER_VERSION, shape_version=contour.shape_version,
                  geometry_id=contour.geometry_id, te=contour.te, gap=contour.gap, weld_x=contour.weld_x,
                  te_cap_limit=contour.te_cap_limit, dx_mid=fields.dx_mid, le_spacing=spacing.le,
                  te_spacing=spacing.te, te_clamped=spacing.te_clamped, cluster_width=fields.cluster_width,
                  precheck_turn=precheck, x_at_precheck_turn=precheck_x, n_required=n, rejection=rejection)
    if rejection is not None:
        return Sample(report=SampleReport(**report))

    s = Geom_stations(contour, spacing, panels)
    points = contour.evaluate(s)
    le_idx = sum(panels[:[p.kind for p in contour.pins].index(PinKind.LE)])
    measures = Geom_sample_checks(contour, s, points, fields.dx_mid)
    return Sample(points=points, le_idx=le_idx,
                  report=SampleReport(**report, n_points=len(points), le_idx=le_idx, stretch_panels=panels, **measures))
