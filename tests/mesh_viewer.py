"""
The pyvista window the interactive meshing tests share: meshes in up to five folders, every cell shaded by
`meshing/common/quality.py`'s own metrics rather than pyvista's same-named `cell_quality` measures, so a colour
reads against the numbers the pipeline gates on.

Keys:
    1-5            a folder (each keeps its place)
    left / right   the previous / next mesh in the folder, wrapping
    t              mesh edges on / off
    c              the metric: solid -> scaled_jacobian -> skew -> aspect_ratio -> area -> size_ratio ->
                   orthogonal_quality -> solid
    n              the colour scale: gate -> absolute -> auto
    left-drag pans, right-drag or the wheel zooms; the view holds across meshes.

The colour scales:
    gate      the mesh's own limit at the red end, anything past it clamped there, so red is what its gate
              rejects. An entry's gates are `meshing/common/constants.py`'s unless it brings its own.
    absolute  the metric's definitional range, so a colour means one number in every mesh.
    auto      this mesh's own range: the only one showing structure in a mesh well inside its limits.
Red is the bad end whichever metric is up. Aspect ratio is log-scaled. Area has neither a gate nor a definitional
range and size ratio no definitional range, so those autoscale.

Two metrics differ from VTK's of the same name: `skew` is equiangle skewness (VTK's principal-axis measure reads
up to 0.20 higher), `aspect_ratio` is longest over shortest edge (VTK's is perimeter/area based).
`quality.py` measures size ratio per interior face; a cell here shows its worst face's.

A mesh comes as a file (scored when the window opens), or as a job for a background process pool that returns a
scored .vtu or why there is none. The window stays live while jobs run; the mesh on screen goes first, then its
neighbours and each folder's current mesh. A .su2 is read in float64, as `quality.py` reads it, so the shading
is exactly what was gated on. A .vtk works too, but VTK reads c2d's `POINTS float` as float32, which moves thin
boundary-layer cells' aspect ratio by up to ~1%.
"""

import multiprocessing
import textwrap
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.datagen.meshing.common.constants import (
    ASPECT_RATIO_LIMIT, JACOBIAN_LIMIT, ORTHO_MIN, SIZE_RATIO_LIMIT, SKEWNESS_LIMIT,
)
from src.datagen.meshing.common.quality import _cell_metrics, _faces, _orthogonal_quality, _size_ratio
from src.datagen.meshing.common.su2io import Common_read_su2

ORTHO = "orthogonal_quality"
METRICS = ["scaled_jacobian", "skew", "aspect_ratio", "area", "size_ratio", ORTHO]
SCALE_MODES = ["gate", "absolute", "auto"]

# worst: the bad end; best: the ideal value, the gate scale's other end; absolute: the definitional range
METRIC_SCALES = {
    "scaled_jacobian": dict(worst="low", best=1.0, absolute=(-1.0, 1.0)),
    "skew":            dict(worst="high", best=0.0, absolute=(0.0, 1.0)),
    "aspect_ratio":    dict(worst="high", best=1.0, absolute=(1.0, 1e10), log=True),
    "area":            dict(worst=None),
    "size_ratio":      dict(worst="high", best=1.0),
    ORTHO:             dict(worst="low", best=1.0, absolute=(-1.0, 1.0)),
}
PIPELINE_GATES = {"scaled_jacobian": JACOBIAN_LIMIT, "skew": SKEWNESS_LIMIT, "aspect_ratio": ASPECT_RATIO_LIMIT,
                  "size_ratio": SIZE_RATIO_LIMIT, ORTHO: ORTHO_MIN}

VIEWPORT = (-0.6, 1.8, -0.9, 0.9, 0.0, 0.0)  # `quality.plot_su2_quality`'s, on the airfoil
WORKERS = 2
POLL_MS = 250
GRIDS_KEPT = 6  # meshes read back from their .vtu, most recent first


@dataclass
class Entry:
    """One mesh: the lines drawn above it, its gates, and where it comes from (a file, a job, or only a message)."""
    caption: str = ""
    gates: dict = field(default_factory=lambda: dict(PIPELINE_GATES))
    mesh_path: Optional[str] = None  # a .su2, or a .vtk
    job: Optional[tuple] = None    # (function, *args) for the pool, returning (.vtu path or None, why not)
    message: Optional[str] = None  # no mesh, only this


@dataclass
class Folder:
    name: str
    entries: list


def quality_arrays(fluid_domain) -> dict:
    """`quality.py`'s metrics per cell, in the grid's own cell order.

    Its helpers take cells grouped by vertex count, so values are scattered back to where each cell sits;
    otherwise a mixed tri/quad mesh would shade the wrong cells.
    """
    nodes = np.asarray(fluid_domain.points)[:, :2]
    blocks, positions = [], []
    for cell_type, conn in fluid_domain.cells_dict.items():
        conn = np.asarray(conn, int)
        blocks.append((conn, conn.shape[1]))
        positions.append(np.flatnonzero(fluid_domain.celltypes == cell_type))
    order = np.concatenate(positions)

    area, skew, aspect, jacobian = (np.concatenate(v) for v in zip(*(_cell_metrics(nodes[c]) for c, _ in blocks)))
    faces = _faces(len(nodes), blocks)
    _fa, _fb, owner, i0, i1 = faces
    jump = _size_ratio(len(nodes), blocks, area, faces)
    size_ratio = np.ones(len(area))
    np.maximum.at(size_ratio, owner[i0], jump)
    np.maximum.at(size_ratio, owner[i1], jump)

    values = {"scaled_jacobian": jacobian, "skew": skew, "aspect_ratio": aspect, "area": area,
              "size_ratio": size_ratio, ORTHO: _orthogonal_quality(nodes, blocks, faces)}
    arrays = {}
    for name, v in values.items():
        scattered = np.empty(fluid_domain.n_cells, dtype=float)
        scattered[order] = v
        arrays[name] = scattered
    return arrays


def scored_fluid_domain(mesh_path):
    """The mesh's triangles and quads (its fluid domain), every metric attached as cell data."""
    import pyvista as pv

    if str(mesh_path).lower().endswith(".su2"):
        nodes, quads, tris, _ = Common_read_su2(mesh_path)
        cells = np.concatenate([np.column_stack([np.full(len(quads), 4), quads]).ravel(),
                                np.column_stack([np.full(len(tris), 3), tris]).ravel()])
        types = np.concatenate([np.full(len(quads), pv.CellType.QUAD), np.full(len(tris), pv.CellType.TRIANGLE)])
        domain = pv.UnstructuredGrid(cells, types.astype(np.uint8), np.column_stack([nodes, np.zeros(len(nodes))]))
    else:
        grid = pv.read(str(mesh_path))
        domain = grid.extract_cells(np.flatnonzero(np.isin(grid.celltypes, (pv.CellType.TRIANGLE, pv.CellType.QUAD))))
    for name, values in quality_arrays(domain).items():
        domain.cell_data[name] = values
    return domain


def worst_line(grid) -> str:
    worst = {m: (grid.cell_data[m].min() if s["worst"] == "low" else grid.cell_data[m].max())
             for m, s in METRIC_SCALES.items() if s["worst"]}
    return (f"min ortho {worst[ORTHO]:.4f}, min scaled_jacobian {worst['scaled_jacobian']:.4f}, "
            f"max skew {worst['skew']:.4f}, max aspect_ratio {worst['aspect_ratio']:.4g}, "
            f"max size_ratio {worst['size_ratio']:.4f}")


def _node_count(n: int) -> str:
    """59800 -> '59.8k', 60000 -> '60k', 412 -> '412'."""
    if n < 1000:
        return str(n)
    k = f"{n / 1000:.1f}"
    return f"{k[:-2] if k.endswith('.0') else k}k"


def _bar_title(metric, scale, clim) -> str:
    """'Skew  [gate: 0..0.85]': two scales of one metric otherwise look like two meshes."""
    pretty = metric.replace("_", " ").title()
    return f"{pretty}  [{scale}: this mesh]" if clim is None else f"{pretty}  [{scale}: {clim[0]:g}..{clim[1]:g}]"


def _wrapped(text: str, width: int = 110) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


class MeshViewer:
    def __init__(self, folders: list, workers: int = WORKERS):
        assert 1 <= len(folders) <= 5, f"{len(folders)} folders; keys 1-5 reach five"
        assert all(f.entries for f in folders), "a folder with nothing in it"
        self.folders, self.workers = folders, workers
        self.folder, self.at = 0, [0] * len(folders)
        self.metric, self.scale, self.edges = 0, 0, False
        self.results = {}  # (folder, index) -> a scored grid, a .vtu Path, or a message
        self.futures = {}
        self.grids = OrderedDict()
        self.pool = self.plotter = None

    def show(self):
        import pyvista as pv

        for (f, i), entry in self._entries():
            if entry.mesh_path is not None:
                self.results[(f, i)] = grid = scored_fluid_domain(entry.mesh_path)
                print(f"  [{f + 1}] {self.folders[f].name}: {_node_count(grid.n_points)} nodes; worst: "
                      f"{worst_line(grid)}")
        if any(entry.job for _, entry in self._entries()):
            self._new_pool()

        p = self.plotter = pv.Plotter()
        p.view_xy()
        p.enable_2d_style()
        p.reset_camera(bounds=VIEWPORT)
        for f in range(len(self.folders)):
            p.add_key_event(str(f + 1), self._goto(f))
        p.add_key_event("Left", self._step(-1))
        p.add_key_event("Right", self._step(+1))
        p.add_key_event("t", self._toggle_edges)
        p.add_key_event("c", self._next_metric)
        # 'n' rather than 's': pyvista binds s/w/r/v/f/e/q itself, and `add_key_event` stacks onto those
        p.add_key_event("n", self._next_scale)
        if self.pool is not None and p.iren is not None:  # off screen there is no interactor
            p.iren.add_observer("TimerEvent", self._poll)
            p.iren.create_timer(POLL_MS)

        print("\n[mesh viewer] 1-5: folder, left/right: mesh in the folder, t: edges, c: metric, "
              f"n: colour scale ({' -> '.join(SCALE_MODES)}); left-drag pans, right-drag/wheel zooms.")
        for f, folder in enumerate(self.folders):
            print(f"  [{f + 1}] {folder.name}: {len(folder.entries)} mesh{'es' if len(folder.entries) > 1 else ''}")
        self._schedule()
        self._draw()
        try:
            p.show()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)

    # Navigation

    def _goto(self, f):
        def go():
            self.folder = f
            self._moved()
        return go

    def _step(self, by):
        def step():
            n = len(self.folders[self.folder].entries)
            self.at[self.folder] = (self.at[self.folder] + by) % n
            self._moved()
        return step

    def _moved(self):
        self._schedule()
        self._draw()

    def _toggle_edges(self):
        self.edges = not self.edges
        self._draw()

    def _next_metric(self):
        self.metric = (self.metric + 1) % (len(METRICS) + 1)
        print(f"metric: {([None] + METRICS)[self.metric] or 'solid'}")
        self._draw()

    def _next_scale(self):
        self.scale = (self.scale + 1) % len(SCALE_MODES)
        print(f"colour scale: {SCALE_MODES[self.scale]}")
        self._draw()

    # Jobs

    def _entries(self):
        for f, folder in enumerate(self.folders):
            for i, entry in enumerate(folder.entries):
                yield (f, i), entry

    def _key(self):
        return self.folder, self.at[self.folder]

    def _new_pool(self):
        self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"))

    def _schedule(self):
        """The mesh on screen first, then its neighbours and each folder's current one; the rest dropped."""
        if self.pool is None:
            return
        f, i = self._key()
        n = len(self.folders[f].entries)
        wanted = [(f, i), (f, (i + 1) % n), (f, (i - 1) % n)]
        wanted += [(g, self.at[g]) for g in range(len(self.folders)) if g != f]
        wanted = list(dict.fromkeys(wanted))
        for key in [k for k in self.futures if k not in wanted]:
            if self.futures[key].cancel():  # not started yet
                del self.futures[key]
        for key in wanted:
            entry = self.folders[key[0]].entries[key[1]]
            if entry.job and key not in self.results and key not in self.futures:
                fn, *args = entry.job
                try:
                    self.futures[key] = self.pool.submit(fn, *args)
                except BrokenProcessPool:  # a worker died; the jobs it took are reported as such
                    self._new_pool()
                    self.futures[key] = self.pool.submit(fn, *args)

    def _poll(self, *_):
        future = self.futures.get(self._key())
        if future is not None and future.done():
            self._draw()

    def _state(self, key):
        """A scored grid, a message, or None while its job runs."""
        if key not in self.results:
            entry = self.folders[key[0]].entries[key[1]]
            future = self.futures.get(key)
            if entry.message is not None:
                self.results[key] = entry.message
            elif future is not None and future.done():
                del self.futures[key]
                try:
                    path, why = future.result()
                except Exception as e:  # the job itself died, e.g. a broken pool
                    path, why = None, f"{type(e).__name__}: {e}"
                self.results[key] = Path(path) if path else why
                if path:
                    grid = self._grid(key, Path(path))
                    print(f"  [{key[0] + 1}] {key[1] + 1}: {_node_count(grid.n_points)} nodes; worst: "
                          f"{worst_line(grid)}")
            else:
                return None
        result = self.results[key]
        return self._grid(key, result) if isinstance(result, Path) else result

    def _grid(self, key, vtu: Path):
        import pyvista as pv

        if key not in self.grids:
            self.grids[key] = pv.read(str(vtu))
            while len(self.grids) > GRIDS_KEPT:
                self.grids.popitem(last=False)
        self.grids.move_to_end(key)
        return self.grids[key]

    # Drawing

    def _draw(self):
        p = self.plotter
        key = self._key()
        folder, entry = self.folders[key[0]], self.folders[key[0]].entries[key[1]]
        state = self._state(key)
        header = f"[{key[0] + 1}] {folder.name}"
        if len(folder.entries) > 1:
            header += f"  -  {key[1] + 1}/{len(folder.entries)}"

        for title in list(p.scalar_bars.keys()):
            p.remove_scalar_bar(title)
        if state is None or isinstance(state, str):
            p.remove_actor("main_fluid_surface")
            p.remove_actor("elevated_edges")
            body = _wrapped(state) if state is not None else "re-meshing in the background ..."
            lines = [header, entry.caption, "", body]
        else:
            header += f"  -  {_node_count(state.n_points)} nodes"
            lines = [header, entry.caption, self._add_surface(state, entry)]
            self._add_edges(state)
        if not entry.caption:
            del lines[1]
        # VTK's text renderer reads '|' as a table's column break
        p.add_text("\n".join(lines).replace("|", "/"), name="mesh_label", font_size=10)
        p.render()

    def _add_surface(self, grid, entry) -> str:
        p = self.plotter
        metric = ([None] + METRICS)[self.metric]
        if metric is None:
            p.add_mesh(grid, color="purple", show_edges=False, name="main_fluid_surface", reset_camera=False)
            return "solid"
        spec, scale = METRIC_SCALES[metric], SCALE_MODES[self.scale]
        clim = None
        if scale == "gate" and entry.gates.get(metric) is not None and "best" in spec:
            clim = tuple(sorted((spec["best"], entry.gates[metric])))
        elif scale == "absolute":
            clim = spec.get("absolute")
        p.add_mesh(grid, scalars=metric, show_edges=False, cmap="jet_r" if spec["worst"] == "low" else "jet",
                   log_scale=bool(spec.get("log")), clim=clim, name="main_fluid_surface", reset_camera=False,
                   scalar_bar_args={"title": _bar_title(metric, scale, clim)})
        return f"{metric}  -  scale: {scale}"

    def _add_edges(self, grid):
        wire = grid.copy()
        wire.translate((0.0, 0.0, 1e-6), inplace=True)  # just above the surface, so it stays on top
        actor = self.plotter.add_mesh(wire, style="wireframe", color="black", line_width=0.5,
                                      name="elevated_edges", reset_camera=False)
        actor.SetVisibility(self.edges)
