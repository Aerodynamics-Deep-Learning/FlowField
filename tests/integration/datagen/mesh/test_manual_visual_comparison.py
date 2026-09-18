"""Manual/visual counterpart to `test_gmsh_integration.py` / `test_c2d_integration.py`.

Runs 5 meshes: Emre's baseline, gmsh's two topologies (cmesh/omesh), and
c2d's two topologies (OGRD/CGRD); through an interactive pyvista window showing one at a
time (switch with number keys 1-5), so a person can eyeball the differences directly. Same
edge/quality-metric toggles as `gmsh.utils.GMSH_plot_mesh`.

Note: gmsh's cmesh/omesh and c2d's OGRD/CGRD are each backend's own independent topology
choice, not a matched pair across backends (in particular gmsh's O-mesh requires a sharp TE,
while c2d's O-grid (OGRD) is c2d's own recommendation for a blunt TE). Don't read
"gmsh omesh" and "c2d OGRD" as the same case; see `conftest.py`'s `sterile_c2d_input`/
`sterile_c2d_cgrd_input` docstrings.

Just run it directly:

    python -m pytest tests/integration/datagen/mesh/test_manual_visual_comparison.py -v -s

Controls (once the window opens):
    left-click + drag        pan
    right-click + drag/scroll zoom
    '1'..'5'                  jump to a mesh: 1=baseline, 2=gmsh cmesh, 3=gmsh omesh,
                              4=c2d OGRD, 5=c2d CGRD
    't'                       toggle mesh edges
    'c'                       cycle quality metric: solid -> scaled_jacobian -> skew ->
                              aspect_ratio -> area -> orthogonal_quality -> solid
    'n'                       cycle colour scale: gate -> absolute -> auto -> gate

Every metric is `common.quality`'s own (`_cell_metrics` for area/skew/aspect ratio/scaled Jacobian
and `_orthogonal_quality` for orthogonality) rather than pyvista's same-named `cell_quality`
measures, so the shading matches the thresholds in `common.constants` exactly. See the note by
QUALITY_METRICS for where the two families disagree.

The colour scale matters as much as the metric, hence 'n'. "gate" puts the `constants.py` limit at
the red end so red is what the pipeline rejects; "absolute" uses the metric's definitional range so
a colour means the same number everywhere; "auto" fits this mesh's own range, which is the only way
to see structure in a mesh that passes everything comfortably. Red is always the bad end, whichever
metric is up. The active metric and scale are drawn on screen, and the worst cell of each metric is
printed when the window opens.
"""

import numpy as np
import pytest

pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag
from src.datagen.meshing.c2d.run import C2D_MeshGenerator, C2D_find_exe
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag
from src.datagen.meshing.common.quality import _cell_metrics, _orthogonal_quality
from src.datagen.meshing.common.constants import (
    SKEWNESS_LIMIT, ORTHO_MIN, JACOBIAN_LIMIT, ASPECT_RATIO_LIMIT,
)
# The four acceptance thresholds `common.quality` renders its verdict against, used here as one of
# the three colour scales: pinned to the worst-case end of the colormap, they turn a cell's colour
# into a reading of how close it is to being rejected.

pytestmark = [
    # Out of default runs; opt in with `-m interactive`.
    pytest.mark.interactive,
    pytest.mark.skipif(
        C2D_find_exe() is None, reason="c2d executable not built in this environment"
    ),
]

ORTHOGONALITY_METRIC = "orthogonal_quality"
QUALITY_METRICS = ["scaled_jacobian", "skew", "aspect_ratio", "area", ORTHOGONALITY_METRIC]

# All five come from `common.quality` rather than pyvista's `cell_quality`, so what is shaded here is
# exactly what the pipeline gates on. Two of them genuinely disagree with VTK's same-named measure:
# `skew` is equiangle skewness, where VTK's is a principal-axis measure reading up to 0.20 higher,
# and `aspect_ratio` is longest/shortest edge, where VTK's is a perimeter/area measure that happens
# to land at half the edge ratio on near-rectangular cells but diverges badly elsewhere (they
# correlate at 0.43 on the O-mesh). `scaled_jacobian` and `area` do agree with VTK to machine
# precision (taking them from here too just keeps one source for the lot).

# How each metric is coloured, per scale mode ('n' cycles them):
#   "gate"     : the `constants.py` limit pinned to the worst-case end of the colormap, with
#                 anything past it clamped there, so whatever shows red is what the pipeline would
#                 reject. The mode to answer "is this mesh acceptable, and where is it marginal".
#   "absolute" : the metric's own definitional range, independent of both mesh and threshold, so a
#                 colour means the same number in every mesh and every session.
#   "auto"     : the range actually present in this mesh. The only one that shows structure once
#                 every cell sits comfortably inside the limits, which is the usual case: a mesh
#                 spanning 0.967..1.000 orthogonality is flat in the other two modes by design.
# `worst` marks which end is the bad one, and the colormap is reversed for the higher-is-better
# metrics so that red means bad whichever metric is on screen. Aspect ratio is log-scaled in every
# mode, since it spans six decades on a real mesh and is unreadable linearly. `area` has neither a
# definitional range nor a threshold, so it autoscales in all three.
SCALE_MODES = ["gate", "absolute", "auto"]
METRIC_SCALES = {
    "scaled_jacobian":    dict(worst="low",  absolute=(-1.0, 1.0), gate=(JACOBIAN_LIMIT, 1.0)),
    "skew":               dict(worst="high", absolute=(0.0, 1.0),  gate=(0.0, SKEWNESS_LIMIT)),
    "aspect_ratio":       dict(worst="high", absolute=(1.0, 1e10), gate=(1.0, ASPECT_RATIO_LIMIT),
                               log=True),
    "area":               dict(worst=None),
    ORTHOGONALITY_METRIC: dict(worst="low",  absolute=(-1.0, 1.0), gate=(ORTHO_MIN, 1.0)),
}


def _scalar_bar_title(metric, scale, clim):
    """'Skew  [gate: 0..0.85]'. The mode has to be on screen -- two scales of one metric otherwise
    look like two different meshes."""
    pretty = metric.replace("_", " ").title()
    if clim is None:
        return f"{pretty}  [{scale}: this mesh]"
    return f"{pretty}  [{scale}: {clim[0]:g}..{clim[1]:g}]"


def _fmt_node_count(n_nodes):
    """59800 -> '59.8k', 60000 -> '60k', 412 -> '412'."""
    if n_nodes < 1000:
        return str(n_nodes)
    thousands = f"{n_nodes / 1000:.1f}"
    if thousands.endswith(".0"):
        thousands = thousands[:-2]
    return f"{thousands}k"


def _pipeline_quality_arrays(fluid_domain):
    """`common.quality`'s metrics over a pyvista grid, back in the grid's own cell order.

    Both helpers take cells grouped by vertex count and return them in that grouping, so the values
    have to be scattered back to where each cell actually sits -- otherwise a mixed tri/quad mesh
    would shade the wrong cells.
    """
    nodes = np.asarray(fluid_domain.points)[:, :2]
    blocks, positions = [], []
    for cell_type, conn in fluid_domain.cells_dict.items():
        conn = np.asarray(conn, int)
        blocks.append((conn, conn.shape[1]))
        positions.append(np.flatnonzero(fluid_domain.celltypes == cell_type))
    order = np.concatenate(positions)

    per_block = [_cell_metrics(nodes[conn]) for conn, _ in blocks]
    area, skew, aspect, jacobian = (np.concatenate(values) for values in zip(*per_block))

    arrays = {}
    for name, values in (("scaled_jacobian", jacobian), ("skew", skew), ("aspect_ratio", aspect),
                         ("area", area), (ORTHOGONALITY_METRIC, _orthogonal_quality(nodes, blocks))):
        scattered = np.empty(fluid_domain.n_cells, dtype=float)
        scattered[order] = values
        arrays[name] = scattered
    return arrays


def _fluid_domain_with_quality_metrics(vtk_path):
    """Same fluid-domain extraction as `gmsh.utils.GMSH_plot_mesh`, scored by `common.quality` with
    every metric precomputed up front so cycling between them is just a scalars swap."""
    import pyvista as pv
    import vtk

    grid = pv.read(str(vtk_path))
    interior_indices = np.where(
        (grid.celltypes == vtk.VTK_TRIANGLE) | (grid.celltypes == vtk.VTK_QUAD)
    )[0]
    fluid_domain = grid.extract_cells(interior_indices)

    for metric, values in _pipeline_quality_arrays(fluid_domain).items():
        fluid_domain.cell_data[metric] = values
    return fluid_domain


def test_visual_comparison_gmsh_vs_c2d_vs_baseline(
    sterile_gmsh_input, sterile_gmsh_omesh_input,
    sterile_c2d_input, sterile_c2d_cgrd_input,
    baseline_mesh_path,
):
    import pyvista as pv

    gmsh_cmesh_out = GMSH_MeshGenerator(sterile_gmsh_input)
    assert gmsh_cmesh_out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh cmesh failed with flag {gmsh_cmesh_out.flag}"

    gmsh_omesh_out = GMSH_MeshGenerator(sterile_gmsh_omesh_input)
    assert gmsh_omesh_out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh omesh failed with flag {gmsh_omesh_out.flag}"

    c2d_ogrd_out = C2D_MeshGenerator(sterile_c2d_input)
    assert c2d_ogrd_out.flag == C2D_ExitFlag.SUCCESS, (
        f"c2d OGRD failed with flag {c2d_ogrd_out.flag}"
    )

    c2d_cgrd_out = C2D_MeshGenerator(sterile_c2d_cgrd_input)
    assert c2d_cgrd_out.flag == C2D_ExitFlag.SUCCESS, (
        f"c2d CGRD failed with flag {c2d_cgrd_out.flag}"
    )

    meshes = [
        ("1: baseline (example_naca0012.vtk)", baseline_mesh_path),
        ("2: gmsh cmesh", gmsh_cmesh_out.mesh_path_vtk),
        ("3: gmsh omesh", gmsh_omesh_out.mesh_path_vtk),
        ("4: c2d OGRD", c2d_ogrd_out.mesh_path_vtk),
        ("5: c2d CGRD", c2d_cgrd_out.mesh_path_vtk),
    ]
    fluid_domains = [_fluid_domain_with_quality_metrics(vtk_path) for _, vtk_path in meshes]
    # Node count of the extracted fluid domain, i.e. of what's actually on screen. Matches the
    # `.su2` NPOIN for both backends, and works for the baseline too (it has no MeshOut to read).
    labels = [f"{label}  -  {_fmt_node_count(fluid_domain.n_points)} nodes"
              for (label, _), fluid_domain in zip(meshes, fluid_domains)]

    plotter = pv.Plotter()
    modes = [None] + QUALITY_METRICS
    mode_state = {"index": 0}
    mesh_state = {"index": 0}
    scale_state = {"index": 0}
    edges_visible = {"value": False}

    def render_edges():
        wireframe_mesh = fluid_domains[mesh_state["index"]].copy()
        wireframe_mesh.translate((0.0, 0.0, 0.000001), inplace=True)
        edge_actor = plotter.add_mesh(
            wireframe_mesh, style="wireframe", color="black", line_width=0.5,
            name="elevated_edges", reset_camera=False,
        )
        edge_actor.SetVisibility(edges_visible["value"])

    def render_main_actor():
        label = labels[mesh_state["index"]]
        fluid_domain = fluid_domains[mesh_state["index"]]
        current_mode = modes[mode_state["index"]]

        for existing_title in list(plotter.scalar_bars.keys()):
            plotter.remove_scalar_bar(existing_title)
        if current_mode is None:
            plotter.add_mesh(
                fluid_domain, color="purple", show_edges=False,
                name="main_fluid_surface", reset_camera=False,
            )
        else:
            spec = METRIC_SCALES[current_mode]
            scale = SCALE_MODES[scale_state["index"]]
            clim = spec.get(scale)   # absent -> pyvista autoscales to this mesh's own range
            plotter.add_mesh(
                fluid_domain, scalars=current_mode, show_edges=False,
                cmap="jet_r" if spec["worst"] == "low" else "jet",
                log_scale=bool(spec.get("log")),
                name="main_fluid_surface", reset_camera=False, clim=clim,
                scalar_bar_args={"title": _scalar_bar_title(current_mode, scale, clim)},
            )
            label = f"{label}\n{current_mode}  |  scale: {scale}"
        plotter.add_text(label, name="mesh_label", font_size=12)
        render_edges()  # re-add so the wireframe stays on top of the freshly (re)added surface

    def toggle_edges():
        edges_visible["value"] = not edges_visible["value"]
        render_edges()
        plotter.render()

    def toggle_quality():
        mode_state["index"] = (mode_state["index"] + 1) % len(modes)
        render_main_actor()
        active = modes[mode_state["index"]] or "solid (no metric)"
        print(f"Active quality metric: {active}")

    def cycle_scale():
        scale_state["index"] = (scale_state["index"] + 1) % len(SCALE_MODES)
        render_main_actor()
        print(f"Colour scale: {SCALE_MODES[scale_state['index']]}")

    def select_mesh(index):
        def _select():
            mesh_state["index"] = index
            render_main_actor()
            print(f"Now showing: {labels[mesh_state['index']]}")
        return _select

    plotter.view_xy()
    plotter.enable_2d_style()
    # Same fixed viewport `common.quality.plot_su2_quality` uses, zoomed on the airfoil rather
    # than the whole farfield domain (makes the five meshes comparable at a glance).
    plotter.reset_camera(bounds=(-0.6, 1.8, -0.9, 0.9, 0.0, 0.0))
    render_main_actor()

    for i in range(len(meshes)):
        plotter.add_key_event(str(i + 1), select_mesh(i))
    plotter.add_key_event('t', toggle_edges)
    plotter.add_key_event('c', toggle_quality)
    # 'n' rather than the more obvious 's': pyvista already binds s/w/r/v/f/e/q for its own surface,
    # wireframe, camera and exit actions, and `add_key_event` stacks onto those instead of replacing.
    plotter.add_key_event('n', cycle_scale)

    print("\n[Manual mesh viewer] Left-click to pan, right-click/scroll to zoom.")
    print("Press 1-5 to jump to a mesh, 't' to toggle edges, 'c' to cycle quality metrics,")
    print(f"'n' to cycle the colour scale ({' -> '.join(SCALE_MODES)}).")
    # The worst cell per metric, so the picture comes with the numbers the pipeline actually gates on
    for label, fluid_domain in zip(labels, fluid_domains):
        worst = {m: (np.asarray(fluid_domain.cell_data[m]).min() if m in ("scaled_jacobian", ORTHOGONALITY_METRIC)
                     else np.asarray(fluid_domain.cell_data[m]).max())
                 for m in QUALITY_METRICS}
        print(f"  {label}")
        print(f"      worst: min ortho {worst[ORTHOGONALITY_METRIC]:.4f}, "
              f"min scaled_jacobian {worst['scaled_jacobian']:.4f}, "
              f"max skew {worst['skew']:.4f}, max aspect_ratio {worst['aspect_ratio']:.4g}")
    plotter.show()
