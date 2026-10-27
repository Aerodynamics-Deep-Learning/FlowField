"""
Convert a C2D grid (.p3d + .nmf boundary file) into an SU2 .su2 mesh.

One of the most complicated code for C2D.

SU2 can't read Plot3D, so this reads the structured node block and the .nmf
boundary faces (VISCOUS->airfoil, FARFIELD->farfield, ONE_TO_ONE->wake cut),
builds quad cells, welds the coincident wake-cut nodes (cut becomes interior,
mesh watertight), enforces CCW ordering, and writes a 2D .su2 with
MARKER_AIRFOIL/MARKER_FARFIELD markers. Also exports .vtk (su2_to_vtk). C- or
O-grid, sharp or blunt TE, all are within capabilities.
"""

import numpy as np

from src.datagen.meshing.common.constants import MARKER_AIRFOIL, MARKER_FARFIELD
# The SU2 boundary names this writer stamps onto the mesh, shared with gmsh's physical groups and
# with `solvers/su2/schemas.py`'s marker defaults.

def read_plot3d_2d(path):
    """
    Read a C2D single-block 2D Plot3D grid.

    Header line: "imax jmax".  Then imax*jmax x-values followed by imax*jmax
    y-values.  C2D writes the i-index in reverse (imax..1) to keep
    positive cell volumes; we keep the file's storage order and call the stored
    position 'p' (p = 0 corresponds to internal i = imax).

    Returns X, Y arrays of shape (imax, jmax) indexed [p, j].
    """
    with open(path, "r") as f:
        tokens = f.read().split()

    # first integers are the header. 2D single-block C2D writes "imax jmax".
    ints = []
    for tok in tokens[:3]:
        try:
            ints.append(int(tok))
        except ValueError:
            break

    if len(ints) < 2:
        raise ValueError("Could not parse Plot3D header in %s" % path)

    imax, jmax = ints[0], ints[1]
    start = 2
    # guard against a possible leading block count "1"
    if imax == 1 and len(ints) >= 3:
        imax, jmax = ints[1], ints[2]
        start = 3

    n = imax * jmax
    vals = np.array(tokens[start:start + 2 * n], dtype=float)
    if vals.size < 2 * n:
        raise ValueError(
            "Plot3D file %s: expected %d coords, found %d" % (path, 2 * n, vals.size)
        )

    # storage order: outer loop j = 1..jmax, inner loop p = 0..imax-1
    X = vals[:n].reshape(jmax, imax).T          # shape (imax, jmax) -> [p, j]
    Y = vals[n:2 * n].reshape(jmax, imax).T
    return X, Y, imax, jmax


def read_nmf(path):
    """
    Parse a C2D 2D .nmf file into a list of boundary faces.

    Each returned face is a dict: {type, face, r1, r2} using 1-based *internal*
    indices, with the face number meaning:
        face 1 -> i = imin (=1)      face 2 -> i = imax
        face 3 -> j = jmin (=1)      face 4 -> j = jmax
    r1 is the in-plane index span; r2 is the k-span (1..1 in 2D, ignored).
    """
    faces = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            btype = parts[0].upper()
            if btype not in ("VISCOUS", "FARFIELD", "ONE_TO_ONE"):
                continue
            # columns after Type: B1 F1 S1 E1 S2 E2 [...]. The unpack is inside the try because a
            # short row is just as unusable as a non-numeric one, and both are skipped, not raised.
            try:
                nums = [int(x) for x in parts[1:7]]
                b1, f1, s1, e1, s2, e2 = nums
            except (ValueError, IndexError):
                continue
            faces.append(dict(type=btype, face=f1, r1=(s1, e1), r2=(s2, e2)))
    return faces


def face_edges(face, imax, jmax):
    """
    Return a list of ((i,j),(i,j)) node index pairs (1-based internal) forming
    the boundary edges of a given face.  For 2D the in-plane range is r1.
    """
    f = face["face"]
    a, b = face["r1"]
    lo, hi = min(a, b), max(a, b)
    edges = []
    if f == 1:      # i = imin, vary j
        for j in range(lo, hi):
            edges.append(((1, j), (1, j + 1)))
    elif f == 2:    # i = imax, vary j
        for j in range(lo, hi):
            edges.append(((imax, j), (imax, j + 1)))
    elif f == 3:    # j = jmin, vary i
        for i in range(lo, hi):
            edges.append(((i, 1), (i + 1, 1)))
    elif f == 4:    # j = jmax, vary i
        for i in range(lo, hi):
            edges.append(((i, jmax), (i + 1, jmax)))
    return edges


def _pj(i, j, imax):
    """
    internal 1-based (i,j) -> stored (p, j0) 0-based; p = imax - i.
    """
    return imax - i, j - 1


def convert(p3d_path, nmf_path, su2_path, weld_tol=1e-8, verbose=False):
    X, Y, imax, jmax = read_plot3d_2d(p3d_path)

    def log(*a):
        if verbose:
            print(*a)

    log("Read grid: imax=%d  jmax=%d  (%d nodes, %d quad cells)"
        % (imax, jmax, imax * jmax, (imax - 1) * (jmax - 1)))

    def nid(p, j0):
        return j0 * imax + p

    # order="F": flat index = p + j0*imax  -> matches nid(p, j0)
    coords = np.column_stack([X.reshape(-1, order="F"), Y.reshape(-1, order="F")])

    # weld coincident nodes (closes the wake cut): each maps to the first node sharing its rounded
    # key; + 0.0 folds -0.0 into 0.0, as a tuple key does
    decimals = max(0, int(round(-np.log10(weld_tol))))
    _, first, inverse = np.unique(np.round(coords, decimals) + 0.0, axis=0, return_index=True,
                                  return_inverse=True)
    remap = first[inverse.ravel()]
    n_welded = int(np.sum(remap != np.arange(imax * jmax)))
    log("Welded %d coincident node(s) along cut/seam." % n_welded)

    # build quad cells (no i-wrap), j0 outer and p inner
    j0, p = np.divmod(np.arange((jmax - 1) * (imax - 1)), imax - 1)
    base = nid(p, j0)
    n0, n1, n2, n3 = remap[base], remap[base + 1], remap[base + imax + 1], remap[base + imax]
    # degenerate (collapsed at cut) -- skip
    keep = (n0 != n1) & (n0 != n2) & (n0 != n3) & (n1 != n2) & (n1 != n3) & (n2 != n3)
    n0, n1, n2, n3 = n0[keep], n1[keep], n2[keep], n3[keep]
    x, y = coords[:, 0], coords[:, 1]
    area = 0.5 * ((x[n0]*y[n1]-x[n1]*y[n0]) + (x[n1]*y[n2]-x[n2]*y[n1]) +
                  (x[n2]*y[n3]-x[n3]*y[n2]) + (x[n3]*y[n0]-x[n0]*y[n3]))
    neg = area < 0
    n1, n3 = np.where(neg, n3, n1), np.where(neg, n1, n3)  # reverse -> CCW
    elems = np.stack([n0, n1, n2, n3], axis=1)
    neg_before = int(neg.sum())
    log("Built %d quads (%d reversed to CCW)." % (len(elems), neg_before))

    # markers from .nmf 
    faces = read_nmf(nmf_path)
    marker_edges = {MARKER_AIRFOIL: [], MARKER_FARFIELD: []}
    for fc in faces:
        if fc["type"] == "VISCOUS":
            tag = MARKER_AIRFOIL
        elif fc["type"] == "FARFIELD":
            tag = MARKER_FARFIELD
        else:
            continue  # ONE_TO_ONE -> interior after weld
        for (ia, ja), (ib, jb) in face_edges(fc, imax, jmax):
            pa, j0a = _pj(ia, ja, imax)
            pb, j0b = _pj(ib, jb, imax)
            na = int(remap[nid(pa, j0a)])
            nb = int(remap[nid(pb, j0b)])
            if na != nb:
                marker_edges[tag].append((na, nb))
    log("Marker edges: %s=%d  %s=%d"
        % (MARKER_AIRFOIL, len(marker_edges[MARKER_AIRFOIL]),
           MARKER_FARFIELD, len(marker_edges[MARKER_FARFIELD])))

    # compact node numbering
    used_mask = np.zeros(imax * jmax, dtype=bool)
    used_mask[elems.ravel()] = True
    used = np.flatnonzero(used_mask)
    old2new = np.full(imax * jmax, -1)
    old2new[used] = np.arange(len(used))
    new_coords = coords[used]

    # A collapsed cell can leave a marker edge on a node no surviving cell uses. That piece of
    # boundary no longer exists, so drop it; keeping it would either emit an orphan node or
    # (before this) raise KeyError mid-write and leave a truncated .su2 on disk.
    n_orphan = 0
    for tag, edges in marker_edges.items():
        e = np.array(edges, dtype=int).reshape(-1, 2)
        kept = (old2new[e] >= 0).all(axis=1)
        n_orphan += int((~kept).sum())
        marker_edges[tag] = old2new[e[kept]]
    if n_orphan:
        log("Dropped %d marker edge(s) left on collapsed cells." % n_orphan)

    # write SU2: one %-format per block over a flat tuple, since per-row lists cost more in GC than the writing
    cells = old2new[elems]
    xs, ys = new_coords[:, 0].tolist(), new_coords[:, 1].tolist()
    tags = [t for t in (MARKER_AIRFOIL, MARKER_FARFIELD) if len(marker_edges[t])]
    out = ["%% SU2 mesh converted from C2D by plot3d_to_su2.py\n", "NDIME= 2\n",
           "NELEM= %d\n" % len(cells), ("9 %d %d %d %d\n" * len(cells)) % tuple(cells.ravel().tolist()),
           "NPOIN= %d\n" % len(used),
           ("%.16e %.16e %d\n" * len(used)) % tuple(v for row in zip(xs, ys, range(len(used))) for v in row),
           "NMARK= %d\n" % len(tags)]
    for t in tags:
        e = marker_edges[t]
        out.append("MARKER_TAG= %s\nMARKER_ELEMS= %d\n" % (t, len(e)))
        out.append(("3 %d %d\n" * len(e)) % tuple(e.ravel().tolist()))
    with open(su2_path, "w") as f:
        f.write("".join(out))

    log("Wrote %s  (%d nodes, %d elems, markers: %s)"
        % (su2_path, len(used), len(elems), ", ".join(tags)))
    return dict(imax=imax, jmax=jmax, npoin=len(used), nelem=len(elems),
                welded=n_welded, reversed_cells=neg_before, dropped_marker_edges=n_orphan,
                markers={t: len(marker_edges[t]) for t in tags})


#  VTK export (legacy ASCII UNSTRUCTURED_GRID), almost a must-have for visualization

def write_vtk(nodes, quads, tris, path):
    """
    Write a 2D mesh (nodes Nx2, quads Mx4, tris Kx3) to a legacy .vtk file.
    """
    nodes = np.asarray(nodes, float)
    quads = np.asarray(quads, int).reshape(-1, 4) if len(quads) else np.zeros((0, 4), int)
    tris = np.asarray(tris, int).reshape(-1, 3) if len(tris) else np.zeros((0, 3), int)
    ncell = len(quads) + len(tris)
    size = len(quads) * 5 + len(tris) * 4        # count + indices per cell
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nC2D mesh\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS %d float\n" % len(nodes))
        for x, y in nodes:
            f.write("%.9e %.9e 0.0\n" % (x, y))
        f.write("CELLS %d %d\n" % (ncell, size))
        for q in quads:
            f.write("4 %d %d %d %d\n" % (q[0], q[1], q[2], q[3]))
        for t in tris:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))
        f.write("CELL_TYPES %d\n" % ncell)
        for _ in quads:
            f.write("9\n")     # VTK_QUAD
        for _ in tris:
            f.write("5\n")     # VTK_TRIANGLE
    return dict(nodes=len(nodes), cells=ncell)


def su2_to_vtk(su2_path, vtk_path):
    """
    Read a 2D .su2 mesh and write it as .vtk.
    """
    nodes, quads, tris = [], [], []
    with open(su2_path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("%")]
    i, n = 0, len(lines)
    while i < n:
        t = lines[i].split(); key = t[0].upper()
        if key == "NELEM=":
            ne = int(t[1]); i += 1
            for _ in range(ne):
                p = lines[i].split(); et = int(p[0])
                if et == 9:
                    quads.append([int(p[1]), int(p[2]), int(p[3]), int(p[4])])
                elif et == 5:
                    tris.append([int(p[1]), int(p[2]), int(p[3])])
                i += 1
            continue
        if key == "NPOIN=":
            npn = int(t[1]); i += 1
            for _ in range(npn):
                p = lines[i].split(); nodes.append([float(p[0]), float(p[1])]); i += 1
            continue
        i += 1
    return write_vtk(np.array(nodes), quads, tris, vtk_path)

