"""
Reading and writing root .dat files, and hashing coordinates.

The format is the root folder's: line 1 is the airfoil name, equal to the file stem; every following
line is one `x y` pair in Selig order (from the TE over the upper surface to the LE, then back along
the lower surface).
"""

import hashlib
from pathlib import Path

import numpy as np

from .constants import DAT_DECIMALS, MIN_DAT_POINTS
from .schemas import RootAirfoil


class GeometryFormatError(ValueError):
    """A .dat file that is not in the root-folder format."""


def Geom_read_dat(path) -> RootAirfoil:
    """
    Reads a root .dat file, refusing anything not in the root-folder format.

    Args:
        path (str | Path): The .dat file

    Returns:
        RootAirfoil: The name and the coordinates as read

    Raises:
        GeometryFormatError: On anything else, with the reason
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as e:
        raise GeometryFormatError(f"{path.name}: not UTF-8 text ({e.reason})") from None
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise GeometryFormatError(f"{path.name}: empty file")

    name = lines[0].strip()
    if name != path.stem:
        raise GeometryFormatError(f"{path.name}: name line '{name}' differs from the file name '{path.stem}'")

    rows = []
    for k, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 2:
            raise GeometryFormatError(f"{path.name}: line {k} has {len(parts)} columns, expected 2")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            raise GeometryFormatError(f"{path.name}: line {k} is not two numbers: '{line.strip()}'") from None

    coords = np.array(rows, dtype=np.float64).reshape(-1, 2)
    _check_selig(coords, path.name)
    return RootAirfoil(name=name, coords=coords)


def _check_selig(coords: np.ndarray, fname: str) -> None:
    """Raises unless `coords` reads as one Selig loop: TE first and last, upper surface first."""
    if len(coords) < MIN_DAT_POINTS:
        raise GeometryFormatError(f"{fname}: {len(coords)} points, at least {MIN_DAT_POINTS} needed")
    if not np.isfinite(coords).all():
        raise GeometryFormatError(f"{fname}: has NaN or Inf values")
    if (np.diff(coords, axis=0) == 0).all(axis=1).any():
        raise GeometryFormatError(f"{fname}: has consecutive duplicate points")

    x = coords[:, 0]
    near_te = x.max() - 0.1 * (x.max() - x.min())
    if x[0] < near_te or x[-1] < near_te:
        raise GeometryFormatError(f"{fname}: does not start and end at the TE (not Selig order)")
    # Shoelace: positive for the counter-clockwise loop Selig order is
    area = 0.5 * np.sum(x * np.roll(coords[:, 1], -1) - np.roll(x, -1) * coords[:, 1])
    if area <= 0:
        raise GeometryFormatError(f"{fname}: runs clockwise (lower surface first)")


def Geom_write_dat(path, name: str, coords: np.ndarray, decimals: int = DAT_DECIMALS) -> None:
    """
    Writes coordinates in the root-folder format.

    Args:
        path (str | Path): The .dat file to write
        name (str): The name line, which should equal the file stem
        coords (np.ndarray): (n, 2) Selig-ordered coordinates
        decimals (int): Decimals per value
    """
    vals = np.round(np.asarray(coords, dtype=np.float64), decimals) + 0.0  # no "-0.000"
    lines = [name] + [f"{x: .{decimals}f} {y: .{decimals}f}" for x, y in vals]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def Geom_as_written(coords: np.ndarray, decimals: int = DAT_DECIMALS) -> np.ndarray:
    """
    The coordinates exactly as `Geom_write_dat` writes them and `Geom_read_dat` reads them back, so a
    hash of them also verifies the file.

    Args:
        coords (np.ndarray): (n, 2) coordinates
        decimals (int): Decimals per value

    Returns:
        np.ndarray: (n, 2) float64, rounded through the text form
    """
    vals = np.round(np.asarray(coords, dtype=np.float64), decimals) + 0.0
    return np.array([[float(f"{v:.{decimals}f}") for v in row] for row in vals]).reshape(-1, 2) + 0.0


def Geom_coords_hash(coords: np.ndarray) -> str:
    """
    SHA-256 of the coordinates as little-endian float64, independent of the dtype they arrive in.

    Args:
        coords (np.ndarray): (n, 2) coordinates

    Returns:
        str: The hex digest
    """
    return hashlib.sha256(np.ascontiguousarray(coords, dtype="<f8").tobytes()).hexdigest()
