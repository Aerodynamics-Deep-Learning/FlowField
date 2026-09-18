import pytest
import numpy as np
import torch
from pathlib import Path

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.common.quality import _cell_metrics, _orthogonal_quality
from src.datagen.meshing.gmsh.schemas import GMSH_CMeshingConfig, GMSH_OMeshingConfig, GMSH_In, GMSH_Topology
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig, C2D_In, C2D_Topology

# The reference mesh provided by Emre; both backends' integration tests score their own output
# against this baseline (see `MeshQualityBaseline` below).
EXAMPLE_MESH_PATH = Path(__file__).parent / "example_naca0012.vtk"

# Note: only the x-stations below are literal data (a reasonable cosine-like distribution,
# denser near the leading and trailing edges); the y-values are computed from the standard NACA
# 4-digit symmetric-thickness formula (`_naca0012_thickness`), not hand-transcribed.
_NACA0012_UPPER_X = [
    1.0000e+00, 9.9329e-01, 9.8206e-01, 9.6938e-01, 9.5536e-01, 9.4019e-01, 9.2413e-01,
    9.0746e-01, 8.9036e-01, 8.7299e-01, 8.5545e-01, 8.3779e-01, 8.2005e-01, 8.0226e-01,
    7.8443e-01, 7.6657e-01, 7.4868e-01, 7.3078e-01, 7.1286e-01, 6.9494e-01, 6.7701e-01,
    6.5908e-01, 6.4115e-01, 6.2323e-01, 6.0532e-01, 5.8742e-01, 5.6954e-01, 5.5168e-01,
    5.3385e-01, 5.1605e-01, 4.9830e-01, 4.8060e-01, 4.6296e-01, 4.4542e-01, 4.2799e-01,
    4.1072e-01, 3.9369e-01, 3.7690e-01, 3.6029e-01, 3.4382e-01, 3.2749e-01, 3.1128e-01,
    2.9519e-01, 2.7924e-01, 2.6344e-01, 2.4780e-01, 2.3234e-01, 2.1708e-01, 2.0207e-01,
    1.8733e-01, 1.7290e-01, 1.5886e-01, 1.4526e-01, 1.3218e-01, 1.1970e-01, 1.0790e-01,
    9.6855e-02, 8.6616e-02, 7.7212e-02, 6.8641e-02, 6.0877e-02, 5.3873e-02, 4.7569e-02,
    4.1901e-02, 3.6804e-02, 3.2217e-02, 2.8085e-02, 2.4357e-02, 2.0994e-02, 1.7957e-02,
    1.5218e-02, 1.2750e-02, 1.0534e-02, 8.5530e-03, 6.7940e-03, 5.2480e-03, 3.9090e-03,
    2.7710e-03, 1.8320e-03, 1.0900e-03, 5.4200e-04, 1.8700e-04, 1.7000e-05,
]
_NACA0012_LOWER_X = [
    2.7000e-05, 2.2400e-04, 6.3000e-04, 1.2620e-03, 2.1340e-03, 3.2490e-03, 4.6090e-03,
    6.2090e-03, 8.0430e-03, 1.0108e-02, 1.2403e-02, 1.4929e-02, 1.7694e-02, 2.0711e-02,
    2.3996e-02, 2.7574e-02, 3.1474e-02, 3.5734e-02, 4.0399e-02, 4.5524e-02, 5.1177e-02,
    5.7433e-02, 6.4383e-02, 7.2125e-02, 8.0766e-02, 9.0407e-02, 1.0114e-01, 1.1301e-01,
    1.2603e-01, 1.4016e-01, 1.5529e-01, 1.7130e-01, 1.8806e-01, 2.0543e-01, 2.2330e-01,
    2.4158e-01, 2.6020e-01, 2.7909e-01, 2.9820e-01, 3.1747e-01, 3.3684e-01, 3.5622e-01,
    3.7557e-01, 3.9487e-01, 4.1412e-01, 4.3337e-01, 4.5265e-01, 4.7199e-01, 4.9138e-01,
    5.1082e-01, 5.3031e-01, 5.4985e-01, 5.6943e-01, 5.8901e-01, 6.0859e-01, 6.2815e-01,
    6.4769e-01, 6.6721e-01, 6.8670e-01, 7.0618e-01, 7.2565e-01, 7.4511e-01, 7.6455e-01,
    7.8398e-01, 8.0340e-01, 8.2279e-01, 8.4214e-01, 8.6141e-01, 8.8053e-01, 8.9939e-01,
    9.1781e-01, 9.3552e-01, 9.5217e-01, 9.6743e-01, 9.8105e-01, 9.9296e-01, 1.0000e+00,
]


def _naca0012_thickness(x, t=0.12):
    """Standard NACA 4-digit symmetric half-thickness at chord fraction `x` (open trailing edge)."""
    x = np.asarray(x, dtype=np.float64)
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                       + 0.2843 * x ** 3 - 0.1015 * x ** 4)


_NACA0012_UPPER = [[x, y] for x, y in zip(_NACA0012_UPPER_X, _naca0012_thickness(_NACA0012_UPPER_X))]
_NACA0012_LOWER = [[x, -y] for x, y in zip(_NACA0012_LOWER_X, _naca0012_thickness(_NACA0012_LOWER_X))]


def _naca0012_airfoil(name: str) -> Airfoil:
    coords_tensor = torch.tensor(_NACA0012_UPPER + _NACA0012_LOWER, dtype=torch.float32)
    le_idx = int(torch.argmin(coords_tensor[:, 0]).item())
    return Airfoil(airfoil_name=name, coords_tensor=coords_tensor, chord=1.0, le_idx=le_idx)


def _naca0012_sharp_thickness(x, t=0.12):
    """Same NACA 4-digit symmetric half-thickness as `_naca0012_thickness`, but with the
    closed-trailing-edge coefficient (-0.1036 instead of -0.1015) so thickness is exactly 0 at
    x=1 (required for GMSH_generate_omesh, which merges upper/lower into a single TE point)."""
    x = np.asarray(x, dtype=np.float64)
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                       + 0.2843 * x ** 3 - 0.1036 * x ** 4)


_NACA0012_SHARP_UPPER = [[x, y] for x, y in zip(_NACA0012_UPPER_X, _naca0012_sharp_thickness(_NACA0012_UPPER_X))]
_NACA0012_SHARP_LOWER = [[x, -y] for x, y in zip(_NACA0012_LOWER_X, _naca0012_sharp_thickness(_NACA0012_LOWER_X))]


def _naca0012_sharp_airfoil(name: str) -> Airfoil:
    coords_tensor = torch.tensor(_NACA0012_SHARP_UPPER + _NACA0012_SHARP_LOWER, dtype=torch.float32)
    le_idx = int(torch.argmin(coords_tensor[:, 0]).item())
    return Airfoil(airfoil_name=name, coords_tensor=coords_tensor, chord=1.0, le_idx=le_idx)


def _naca0012_freestream() -> Freestream:
    # NACA0012, M=0.5, AOA=0, Re=5M
    return Freestream(alpha=0.0, Re=5e6, mach=0.5, altitude_m=1.0)


@pytest.fixture(scope="function")
def integration_mesh_workspace(integration_root: Path) -> Path:
    """
    Takes the parent 'integration_root' fixture and builds a
    mesh-specific subdirectory structure inside it
    """
    mesh_workspace = integration_root / "mesh_workspace"
    mesh_workspace.mkdir()

    print(f"\n[Mesh Integration Workspace]: {mesh_workspace.resolve()}")

    return mesh_workspace

@pytest.fixture(scope="function")
def sterile_gmsh_input(integration_mesh_workspace: Path):

    airfoil = _naca0012_airfoil("naca0012_test")
    freestream = _naca0012_freestream()

    meshing_config = GMSH_CMeshingConfig(
            # Airfoil specific params
            upper_anchor_idx = 55,
            lower_anchor_idx = 100,

            # Overall params
            wake_length = 20.0,
            farfield_radius = 15.0,
            bl_thickness = 1.5,
            target_yplus = 1.0,

            # BL meshing configs
            nx_le1 = 60, #60
            nx_le2 = 60, #60
            nx_upper = 150, #150
            nx_lower = 150, #150
            nx_wake = 75,
            bl_growth_ratio = 1.05,
            wake_progression = 1.175,
            te_coarsen_factor = 60.0,
            chord_bump = 0.75,

            # Farfield meshing configs
            ff_growth_ratio = 1.1
        )

    gmsh_in = GMSH_In(
            airfoil=airfoil,
            freestream=freestream,
            working_dir=str(integration_mesh_workspace),
            topology=GMSH_Topology.CGRD,
            meshing_config=meshing_config
        )

    return gmsh_in


@pytest.fixture(scope="function")
def sterile_gmsh_omesh_input(integration_mesh_workspace: Path):
    """The O-mesh counterpart to `sterile_gmsh_input`; same freestream, but a sharp-TE airfoil
    (`_naca0012_sharp_airfoil`), since GMSH_generate_omesh requires upper/lower to meet at one
    point."""

    airfoil = _naca0012_sharp_airfoil("naca0012_omesh_test")
    freestream = _naca0012_freestream()

    gmsh_in = GMSH_In(
            airfoil=airfoil,
            freestream=freestream,
            working_dir=str(integration_mesh_workspace),
            topology=GMSH_Topology.OGRD,
            meshing_config=GMSH_OMeshingConfig()
        )

    return gmsh_in


@pytest.fixture(scope="function")
def sterile_c2d_input(integration_mesh_workspace: Path) -> C2D_In:
    """Bundles c2d's `C2D_MeshGenerator` input the same way `sterile_gmsh_input` bundles
    gmsh's `GMSH_In`, so the two integration test modules read the same way. Blunt-TE airfoil with
    `topology=C2D_Topology.OGRD` (O-grid) -- c2d's own `_te_is_sharp`-based recommendation
    for a blunt TE (see `c2d/run.py`), so this pairing doesn't trigger the topo/bluntness-mismatch
    auto-confirm path. Not to be conflated with gmsh's "cmesh"/"omesh" naming -- c2d's
    OGRD/CGRD and gmsh's cmesh/omesh are two independent backends' topology choices, and an O-grid
    here is paired with a *blunt* airfoil, the opposite of what gmsh's O-mesh requires. See
    `sterile_c2d_cgrd_input` for the CGRD/sharp-TE counterpart. `meshing_config.topo` is left
    unset here -- `C2D_In`'s validator fills it in from `topology` automatically."""
    return C2D_In(
        airfoil=_naca0012_airfoil("naca0012_test_c2d"),
        freestream=_naca0012_freestream(),
        working_dir=str(integration_mesh_workspace),
        topology=C2D_Topology.OGRD,
        meshing_config=C2D_MeshingConfig(),
    )


@pytest.fixture(scope="function")
def sterile_c2d_cgrd_input(integration_mesh_workspace: Path) -> C2D_In:
    """The `topology=C2D_Topology.CGRD` (C-grid) counterpart to `sterile_c2d_input` (OGRD), on a
    sharp-TE airfoil (c2d's own recommendation for a sharp TE, so this pairing doesn't
    trigger the topo/bluntness-mismatch auto-confirm path either). Named for c2d's own
    "CGRD" (not "cmesh"/"omesh") deliberately (see `sterile_c2d_input`'s docstring: this isn't
    equivalent to gmsh's C-mesh or O-mesh, it's c2d's own C-grid option)."""
    return C2D_In(
        airfoil=_naca0012_sharp_airfoil("naca0012_test_c2d_cgrd"),
        freestream=_naca0012_freestream(),
        working_dir=str(integration_mesh_workspace),
        topology=C2D_Topology.CGRD,
        meshing_config=C2D_MeshingConfig(),
    )


def _mesh_quality_summary(vtk_path) -> dict:
    """Loads a 2D mesh `.vtk` and scores its interior (fluid) tri/quad cells with `common.quality`'s
    own metrics; the same functions, and therefore the same definitions, the pipeline renders its
    verdict with.

    Deliberately not pyvista's `cell_quality`, which this used to call. VTK's `scaled_jacobian` and
    `area` do agree with the pipeline to machine precision, but its `skew` is a principal-axis
    measure rather than equiangle skewness (reading up to 0.20 higher), and its `aspect_ratio` is a
    perimeter/area measure that lands at half the edge ratio on near-rectangular cells and 3.66x the
    other way on an O-mesh. Bounds tuned against those numbers cannot be read against
    `common.constants`, which is what made the old thresholds here impossible to reason about.

    Still reads `.vtk` rather than `.su2`, because Emre's baseline only exists as a
    `.vtk` and both sides of the comparison have to be measured the same way. That costs no accuracy:
    `common.quality` promotes pyvista's float32 points to float64 itself, so these numbers match the
    `.su2`-derived ones exactly.
    """
    import pyvista as pv
    import vtk

    grid = pv.read(str(vtk_path))
    interior_indices = np.where(
        (grid.celltypes == vtk.VTK_TRIANGLE) | (grid.celltypes == vtk.VTK_QUAD)
    )[0]
    fluid_domain = grid.extract_cells(interior_indices)

    nodes = np.asarray(fluid_domain.points)[:, :2]
    blocks = [(np.asarray(conn, int), np.asarray(conn, int).shape[1])
              for conn in fluid_domain.cells_dict.values()]
    _area, skew, aspect, jacobian = (np.concatenate(values) for values in
                                     zip(*(_cell_metrics(nodes[conn]) for conn, _k in blocks)))

    summary = {"n_cells": int(fluid_domain.n_cells)}
    for name, values in (("scaled_jacobian", jacobian), ("skew", skew), ("aspect_ratio", aspect),
                         ("orthogonal_quality", _orthogonal_quality(nodes, blocks))):
        summary[name] = dict(min=float(values.min()), mean=float(values.mean()),
                             max=float(values.max()))
    return summary


class MeshQualityBaseline:
    """Scores a generated mesh's `.vtk` against Emre's `example_naca0012.vtk`
    baseline. The baseline was produced by a different resolution/growth-parameter recipe than
    either backend's test config, so exact equivalence isn't a meaningful bar; these bounds are
    intentionally loose, only meant to catch a generated mesh regressing far below the reference's
    quality (e.g. a bug that silently tanks mesh quality while `analyze_su2`'s own thresholds still
    pass it). Aspect ratio is checked for validity only (not compared ratio-wise): a boundary-layer
    mesh's aspect ratio is dominated by first-cell-height/growth-ratio choices that legitimately
    differ a lot by resolution, and a high aspect ratio near the wall is expected, not a defect.

    All four bounds are on `common.quality`'s scale, so they can be read directly against
    `common.constants`, this stays the stricter of the two, tripping on a regression before the
    pipeline's own absolute gate would. Measured 2026-09-18, baseline vs the two meshes actually
    checked here (gmsh C-mesh, c2d O-grid):

                      skew max/mean     jacobian min/mean    ortho min/mean
        baseline       0.166 / 0.017      0.966 / 0.9990      0.967 / 0.9990
        gmsh C-mesh    0.446 / 0.049      0.765 / 0.9908      0.765 / 0.9908
        c2d  O-grid    0.231 / 0.026      0.935 / 0.9977      0.937 / 0.9978
    """

    MIN_SCALED_JACOBIAN_RATIO = 0.5  # generated mean scaled_jacobian >= half the baseline's
    MIN_ORTHO_RATIO = 0.5            # ...and the same bar for mean orthogonal quality
    MAX_SKEW_RATIO = 3.0             # generated max skew <= baseline's max skew * this ratio...
    MAX_SKEW_FLOOR = 0.6             # ...or this floor, whichever is larger (baseline skew can be tiny)
    # The floor is what binds at this baseline (3.0 * 0.166 = 0.50) and was 0.5 while the numbers
    # came from VTK's skew. On equiangle skew that would have left the gmsh C-mesh, at 0.446, only
    # 11% of headroom (far too tight for a bound whose whole point is being loose). 0.6 restores
    # ~35% while staying well under SKEWNESS_LIMIT (0.85), so this still fires first.

    def __init__(self, baseline: dict):
        self.baseline = baseline

    @staticmethod
    def summarize(vtk_path) -> dict:
        return _mesh_quality_summary(vtk_path)

    def assert_not_worse_than_baseline(self, candidate_vtk_path) -> dict:
        candidate = self.summarize(candidate_vtk_path)
        baseline = self.baseline

        assert candidate["n_cells"] > 0, "Generated mesh has no interior tri/quad cells"
        assert candidate["scaled_jacobian"]["min"] > 0.0, (
            f"Generated mesh has inverted/degenerate cells "
            f"(min scaled_jacobian={candidate['scaled_jacobian']['min']:.4g})"
        )
        assert candidate["orthogonal_quality"]["min"] > 0.0, (
            f"Generated mesh has folded cells "
            f"(min orthogonal_quality={candidate['orthogonal_quality']['min']:.4g})"
        )
        assert np.isfinite(candidate["aspect_ratio"]["max"]), "Generated mesh has a non-finite aspect ratio"

        min_sj = baseline["scaled_jacobian"]["mean"] * self.MIN_SCALED_JACOBIAN_RATIO
        assert candidate["scaled_jacobian"]["mean"] >= min_sj, (
            f"Generated mesh mean scaled_jacobian ({candidate['scaled_jacobian']['mean']:.4f}) is "
            f"more than {1 - self.MIN_SCALED_JACOBIAN_RATIO:.0%} worse than the baseline "
            f"({baseline['scaled_jacobian']['mean']:.4f})"
        )
        min_oq = baseline["orthogonal_quality"]["mean"] * self.MIN_ORTHO_RATIO
        assert candidate["orthogonal_quality"]["mean"] >= min_oq, (
            f"Generated mesh mean orthogonal_quality ({candidate['orthogonal_quality']['mean']:.4f}) "
            f"is more than {1 - self.MIN_ORTHO_RATIO:.0%} worse than the baseline "
            f"({baseline['orthogonal_quality']['mean']:.4f})"
        )
        max_skew = max(baseline["skew"]["max"] * self.MAX_SKEW_RATIO, self.MAX_SKEW_FLOOR)
        assert candidate["skew"]["max"] <= max_skew, (
            f"Generated mesh max skew ({candidate['skew']['max']:.4f}) exceeds the "
            f"baseline-derived limit ({max_skew:.4f}); baseline max skew is "
            f"{baseline['skew']['max']:.4f}"
        )
        return candidate


@pytest.fixture(scope="session")
def mesh_quality_baseline() -> MeshQualityBaseline:
    return MeshQualityBaseline(_mesh_quality_summary(EXAMPLE_MESH_PATH))


@pytest.fixture(scope="session")
def baseline_mesh_path() -> Path:
    """Path to Emre's reference mesh (`example_naca0012.vtk`), for tests that want the
    raw file rather than `mesh_quality_baseline`'s numeric summary (e.g. a manual visual diff.)"""
    return EXAMPLE_MESH_PATH
