"""
Closed NACA0012 contours for the meshing tests, standing in for the orchestrator's geometry step.

c2d meshes the input points as given (BUFF), so the input has to carry what c2d no longer provides:
an exactly closed TE and graded surface spacing. Plain cosine spacing will not do, since its end
panels jump 3:1 and BUFF meshes that jump straight into a size-ratio failure.
"""

from typing import Literal

import numpy as np
import torch

from src.datagen.schemas import Airfoil


def _naca0012_half_thickness(x, closed_te: bool):
    """NACA 4-digit symmetric half-thickness; the -0.1036 coefficient closes the TE to zero."""
    a4 = -0.1036 if closed_te else -0.1015
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)


def _dense_contour(te: Literal["round", "sharp"]) -> np.ndarray:
    """A densely sampled closed Selig loop starting at the TE closure point, unit chord."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 4001)))
    if te == "sharp":
        y = _naca0012_half_thickness(x, closed_te=True)
        y[-1] = 0.0
        return np.vstack([np.stack([x[::-1], y[::-1]], 1), np.stack([x[1:], -y[1:]], 1)])

    # Open section closed by a semicircle on its TE gap, closure point at the apex
    y = _naca0012_half_thickness(x, closed_te=False)
    r = y[-1]
    th = np.linspace(0.0, np.pi / 2, 200)
    cap_up = np.stack([1.0 + r * np.cos(th), r * np.sin(th)], 1)[:-1]
    cap_lo = cap_up[::-1] * np.array([1.0, -1.0])
    contour = np.vstack([cap_up, np.stack([x[::-1], y[::-1]], 1), np.stack([x[1:], -y[1:]], 1), cap_lo])
    return contour / (1.0 + r)


def _resample(contour: np.ndarray, h_max: float, h_le: float, h_te: float, width: float) -> np.ndarray:
    """Resamples a closed loop by arc length, spacing dipping from `h_max` to `h_le`/`h_te` over `width`."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))])
    d_le = np.abs(s - s[np.argmin(contour[:, 0])])
    d_te = np.minimum(s, s[-1] - s)
    h = h_max - (h_max - h_le) * np.exp(-(d_le / width) ** 2) - (h_max - h_te) * np.exp(-(d_te / width) ** 2)

    phi = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / h[1:] + 1.0 / h[:-1]) * np.diff(s))])
    n = 2 * int(round(phi[-1] / 2.0))  # even, so the symmetric middle sample lands on the LE
    s_new = np.interp(np.linspace(0.0, phi[-1], n + 1), phi, s)
    pts = np.stack([np.interp(s_new, s, contour[:, 0]), np.interp(s_new, s, contour[:, 1])], 1)
    pts[-1] = pts[0]
    return pts


def closed_naca0012(name: str, te: Literal["round", "sharp"] = "round", h_max: float = 8e-3,
                    h_le: float = 5e-4, h_te: float = 3e-4, width: float = 0.08) -> Airfoil:
    """A closed NACA0012 with graded spacing: a rounded TE (open section capped by a semicircle) or a
    sharp one. Last point identical to the first."""
    pts = _resample(_dense_contour(te), h_max, h_le, h_te, width)
    coords = torch.tensor(pts, dtype=torch.float64)
    coords[-1] = coords[0]
    le_idx = int(torch.argmin(coords[:, 0]).item())
    return Airfoil(airfoil_name=name, coords_tensor=coords, chord=1.0, le_idx=le_idx)
