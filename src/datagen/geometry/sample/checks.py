"""
Sequence 3, step 6: measures of a placed sample, held to the rules its construction guarantees. A broken
rule is a sampler bug, never a bad config, so it raises.

    panel      no panel longer than dx_mid (so than DX_MAX): each stretch's count was rounded up
    turn       no point turns more than THETA_MAX_DEG: the pre-check held curvature x spacing under it.
               A sharp TE's closure point is a corner by design, and skipped
    deviation  the curve strays from its panels by at most DEV_MAX, the sagitta of a chord of DX_MAX that
               turns THETA_MAX_DEG on a circle: the two rules above bound every panel by it
"""

import math

import numpy as np

from src.datagen.geometry.common.constants import DX_MAX, THETA_MAX_DEG
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.sample.contour import Contour

DEV_MAX = (DX_MAX / 2) * math.tan(math.radians(THETA_MAX_DEG) / 4)

_ALONG = np.arange(1, 6) / 6  # where along each panel the curve is compared with it


class SamplerAssertionError(AssertionError):
    """A sample broke a rule its construction guarantees: a sampler bug, not a bad config."""


def Geom_sample_checks(contour: Contour, s: np.ndarray, points: np.ndarray, dx_mid: float) -> dict:
    """
    Measures a placed sample's panels, turns and deviation from its contour, and holds them to the rules.

    Args:
        contour (Contour): The contour sampled
        s (np.ndarray): (n,) arc length of each point
        points (np.ndarray): (n, 2) the points, Selig order
        dx_mid (float): The config's mid-chord spacing [c]

    Returns:
        dict: The `SampleReport` measurement fields

    Raises:
        SamplerAssertionError: If a rule is broken
    """
    e = np.diff(points, axis=0)
    panel = np.hypot(e[:, 0], e[:, 1])

    a, b, vertex = e[:-1], e[1:], np.arange(1, len(points) - 1)
    if contour.te is TETreatment.ROUND:  # its closure point is smooth, so checked too
        a, b, vertex = np.vstack([e[-1:], a]), np.vstack([e[:1], b]), np.concatenate([[0], vertex])
    turn = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))))

    curve = contour.evaluate((s[:-1, None] + _ALONG * np.diff(s)[:, None]).ravel()).reshape(len(e), len(_ALONG), 2)
    rel = curve - points[:-1, None, :]
    t = np.clip(np.sum(rel * e[:, None, :], axis=2) / np.maximum(panel ** 2, 1e-300)[:, None], 0.0, 1.0)
    off = rel - t[..., None] * e[:, None, :]
    dev = np.hypot(off[..., 0], off[..., 1]).max(axis=1)

    k_panel, k_turn, k_dev = int(np.argmax(panel)), int(np.argmax(turn)), int(np.argmax(dev))
    measures = dict(max_panel=float(panel[k_panel]), x_at_max_panel=float(points[k_panel:k_panel + 2, 0].mean()),
                    min_panel=float(panel.min()), max_turn_deg=float(turn[k_turn]),
                    x_at_max_turn=float(points[vertex[k_turn], 0]), max_dev=float(dev[k_dev]),
                    x_at_max_dev=float(points[k_dev:k_dev + 2, 0].mean()),
                    max_panel_ratio=float(np.maximum(panel[1:] / panel[:-1], panel[:-1] / panel[1:]).max()))

    broken = []
    if measures["max_panel"] > dx_mid * (1 + 1e-6):
        broken.append(f"panel {measures['max_panel']:.3e} > dx_mid {dx_mid:.3e}")
    if measures["max_turn_deg"] > THETA_MAX_DEG:
        broken.append(f"turn {measures['max_turn_deg']:.2f} deg > {THETA_MAX_DEG} deg at x {measures['x_at_max_turn']:.5f}")
    if measures["max_dev"] > DEV_MAX:
        broken.append(f"deviation {measures['max_dev']:.3e} > DEV_MAX {DEV_MAX:.3e} at x {measures['x_at_max_dev']:.5f}")
    if broken:
        raise SamplerAssertionError(f"{contour.geometry_id} ({contour.te.value}): " + "; ".join(broken))
    return measures
