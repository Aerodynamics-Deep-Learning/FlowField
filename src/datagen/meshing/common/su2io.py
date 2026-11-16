"""
The 2D `.su2` reader shared by the quality and deviation passes and by c2d's `.vtk` writer. Imports only
numpy, so a backend can use it without loading the rest of `meshing/common`.
"""

import numpy as np


def Common_read_su2(path):
    """Reads a 2D .su2 mesh. Returns nodes (N,2), quads (M,4), tris (K,3), and markers {tag: [(a,b),...]}.
    Each NELEM/NPOIN block is parsed in one `np.loadtxt` call; one that isn't a clean table (mixed widths,
    malformed rows, cut short) goes through the per-line loop, so it parses, or raises, as that always did."""
    nodes = []
    quads = []
    tris = []
    markers = {}
    with open(path) as f:
        # Strip before testing for '%', or an indented comment parses as data
        lines = [s for s in (ln.strip() for ln in f) if s and not s.startswith("%")]
    i = 0
    n = len(lines)
    while i < n:
        t = lines[i].split()
        key = t[0].upper()
        if key == "NELEM=":
            ne = int(t[1]); i += 1
            if ne:
                _read_elems(lines[i:i + ne], ne, quads, tris)
            i += ne
            continue
        if key == "NPOIN=":
            npn = int(t[1]); i += 1
            if npn:
                _read_nodes(lines[i:i + npn], npn, nodes)
            i += npn
            continue
        if key == "MARKER_TAG=":
            tag = t[1]; i += 1
            me = int(lines[i].split()[1]); i += 1
            seg = []
            for _ in range(me):
                p = lines[i].split(); seg.append((int(p[1]), int(p[2]))); i += 1
            markers[tag] = seg
            continue
        i += 1
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), int)
    tris = np.concatenate(tris) if tris else np.zeros((0, 3), int)
    return (np.concatenate(nodes) if nodes else np.asarray([], float),
            quads if len(quads) else np.zeros((0, 4), int),
            tris if len(tris) else np.zeros((0, 3), int),
            markers)


def _read_elems(block, ne, quads, tris):
    """One NELEM block of `ne` rows into `quads`/`tris` (lists of arrays). A quad or tri row takes its
    first 5 columns (a tri's 5th is its index); anything loadtxt refuses goes through the per-line loop."""
    if len(block) == ne:
        try:
            a = np.loadtxt(block, dtype=int, comments=None, ndmin=2, usecols=(0, 1, 2, 3, 4))
        except ValueError:
            pass
        else:
            quads.append(a[a[:, 0] == 9, 1:5])
            tris.append(a[a[:, 0] == 5, 1:4])
            return
    q, tr = [], []
    for line in block:
        p = line.split(); et = int(p[0])
        if et == 9:
            q.append([int(p[1]), int(p[2]), int(p[3]), int(p[4])])
        elif et == 5:
            tr.append([int(p[1]), int(p[2]), int(p[3])])
    quads.append(np.array(q, dtype=int).reshape(-1, 4))
    tris.append(np.array(tr, dtype=int).reshape(-1, 3))
    if len(block) < ne:
        raise IndexError("list index out of range")  # the block runs past the end of the file


def _read_nodes(block, npn, nodes):
    """One NPOIN block of `npn` rows into `nodes` (a list of (n,2) arrays), as `_read_elems` does."""
    if len(block) == npn:
        try:
            nodes.append(np.loadtxt(block, dtype=float, comments=None, ndmin=2, usecols=(0, 1)))
            return
        except ValueError:
            pass
    nodes.append(np.array([[float(p[0]), float(p[1])] for p in map(str.split, block)], dtype=float).reshape(-1, 2))
    if len(block) < npn:
        raise IndexError("list index out of range")
