"""
Backend-agnostic `.su2` mesh-quality analysis and plotting. Works on any 2D quad/tri `.su2` mesh.
"""

import os
import logging

import numpy as np

from .schemas import MeshQualitySummary, MeshExitFlag
from .constants import SKEWNESS_LIMIT, ORTHO_MIN, JACOBIAN_LIMIT, ASPECT_RATIO_LIMIT
# The four cell-shape gates `_analyze_su2` applies together, all as worst-cell values: equiangle
# skewness, orthogonal quality, scaled Jacobian, and aspect ratio `constants.py` carries the logic 
# for meshes each was calibrated against.

logger = logging.getLogger(__name__)


def Common_evaluate_mesh_quality(mesh_path: str | None) -> tuple[MeshExitFlag, MeshQualitySummary | None, int | None]:
    """
    The single `.su2` based quality verdict both backends converge on. This reads the `.su2` file directly
    and owns UNACCEPTABLE_QUALITY/LOW_QUALITY/SUCCESS.

    Args:
        mesh_path (str | none): Mesh path either string or none
    
    Returns:
        MeshExitFlag, MeshQualitySummary | None, int | None: Mesh success flag, quality summary dict, num of nodes
    """
    # An unparseable .su2 is a conversion failure, not a crash. Handling it here rather than letting it reach `entry.py`
    if not mesh_path or not os.path.isfile(mesh_path):
        return MeshExitFlag.CONVERSION_FAIL, None, None

    try:
        a = _analyze_su2(mesh_path)
        if a["ncell"] == 0:
            logger.warning("Mesh at %s parsed but declares no cells.", mesh_path)
            return MeshExitFlag.CONVERSION_FAIL, None, None
        quality = MeshQualitySummary(**_summary_from_analysis(a))
    except Exception:
        logger.exception("Could not analyze the .su2 mesh at %s", mesh_path)
        return MeshExitFlag.CONVERSION_FAIL, None, None

    # A folded corner is as unusable to a solver as an inverted cell, so it joins `neg` on the severe
    # verdict rather than the graded one; the other three gates only ever grade down to LOW_QUALITY.
    if a["neg"] > 0 or a["min_jac"] <= 0.0:
        return MeshExitFlag.UNACCEPTABLE_QUALITY, quality, a["nnode"]
    if not a["acceptable"]:
        return MeshExitFlag.LOW_QUALITY, quality, a["nnode"]
    return MeshExitFlag.SUCCESS, quality, a["nnode"]


def _read_su2(path):
    """Reads a 2D .su2 mesh. Returns nodes (N,2), quads (list of 4-idx),
    tris (list of 3-idx), and markers {tag: [(a,b),...]}."""
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
                p = lines[i].split()
                nodes.append([float(p[0]), float(p[1])])
                i += 1
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
    return (np.asarray(nodes, float),
            np.asarray(quads, int) if quads else np.zeros((0, 4), int),
            np.asarray(tris, int) if tris else np.zeros((0, 3), int),
            markers)


def _cell_metrics(P):
    """P: (M, k, 2) corner coords for M cells with k vertices.
    Returns signed area, equiangle-skewness, aspect ratio, scaled Jacobian (all length M)."""
    P = np.asarray(P, dtype=np.float64)   # see `_orthogonal_quality` on why this is not optional
    M, k, _ = P.shape
    # signed area (shoelace)
    x = P[:, :, 0]; y = P[:, :, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    # interior angles at each vertex
    prev = np.roll(P, 1, axis=1); nxt = np.roll(P, -1, axis=1)
    v1 = prev - P; v2 = nxt - P
    d = np.sum(v1 * v2, axis=2)
    n1 = np.linalg.norm(v1, axis=2); n2 = np.linalg.norm(v2, axis=2)
    cosang = np.clip(d / np.maximum(n1 * n2, 1e-30), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))            # (M,k)
    the = 180.0 * (k - 2) / k                       # ideal angle: tri 60, quad 90
    amax = ang.max(axis=1); amin = ang.min(axis=1)
    skew = np.maximum((amax - the) / (180.0 - the), (the - amin) / the)
    # aspect ratio = longest edge / shortest edge
    edges = np.linalg.norm(np.roll(P, -1, axis=1) - P, axis=2)   # (M,k)
    ar = edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-30)
    # scaled Jacobian: worst corner's normalized edge cross product
    cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    jac = (-cross / np.maximum(n1 * n2, 1e-30)).min(axis=1)
    return area, skew, ar, jac


def _cos(u, v):
    """Row-wise cosine between two (N,2) vector stacks, guarded against a zero-length row."""
    return np.sum(u * v, axis=1) / np.maximum(
        np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)


def _orthogonal_quality(nodes, blocks):
    """
    Orthogonal quality per cell, in the order `blocks` lists them.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    centroids = []; face_a = []; face_b = []; face_cell = []
    offset = 0
    for conn, _k in blocks:
        M, k = conn.shape
        centroids.append(nodes[conn].mean(axis=1))
        face_a.append(conn.ravel())
        face_b.append(np.roll(conn, -1, axis=1).ravel())
        face_cell.append(np.repeat(np.arange(offset, offset + M), k))
        offset += M
    centroid = np.concatenate(centroids)
    fa = np.concatenate(face_a); fb = np.concatenate(face_b); fc = np.concatenate(face_cell)

    A = nodes[fa]; B = nodes[fb]
    edge = B - A
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)   # outward while the cell is wound CCW
    score = _cos(normal, 0.5 * (A + B) - centroid[fc])

    # Neighbour term. Two faces carrying the same node pair are the two sides of one interior face;
    # sorting by that pair puts them adjacent. A non-manifold edge (>2 uses) would only get its first
    # two sides paired, which neither backend's structured output produces.
    key = np.minimum(fa, fb).astype(np.int64) * len(nodes) + np.maximum(fa, fb)
    order = np.argsort(key, kind="stable")
    shared = np.flatnonzero(key[order][:-1] == key[order][1:])
    i0 = order[shared]; i1 = order[shared + 1]
    joins = centroid[fc[i1]] - centroid[fc[i0]]
    score[i0] = np.minimum(score[i0], _cos(normal[i0], joins))
    score[i1] = np.minimum(score[i1], _cos(normal[i1], -joins))

    out = []; start = 0
    for conn, _k in blocks:
        M, k = conn.shape
        out.append(score[start:start + M * k].reshape(M, k).min(axis=1))
        start += M * k
    return np.concatenate(out)


def _analyze_su2(su2_path):
    """
    Reads a `.su2` and reduces it to the per-mesh quality metrics the verdict is built on.
    """
    nodes, quads, tris, markers = _read_su2(su2_path)
    # One list, in the order everything below is concatenated in `_orthogonal_quality` indexes
    # cells by that same order, so the two must not drift apart
    blocks = [(conn, k) for conn, k in ((quads, 4), (tris, 3)) if len(conn)]
    skew_all = []; ar_all = []; area_all = []; jac_all = []; polys = []
    for conn, k in blocks:
        P = nodes[conn]                      # (M,k,2)
        area, skew, ar, jac = _cell_metrics(P)
        skew_all.append(skew); ar_all.append(ar); area_all.append(area); jac_all.append(jac)
        polys.extend(list(P))
    skew = np.concatenate(skew_all) if skew_all else np.zeros(0)
    ar = np.concatenate(ar_all) if ar_all else np.zeros(0)
    area = np.concatenate(area_all) if area_all else np.zeros(0)
    jac = np.concatenate(jac_all) if jac_all else np.zeros(0)
    ortho = _orthogonal_quality(nodes, blocks) if blocks else np.zeros(0)
    ncell = len(skew)
    neg = int(np.sum(area <= 0))
    max_skew = float(skew.max()) if ncell else 0.0
    min_ortho = float(ortho.min()) if ncell else 1.0
    max_ar = float(ar.max()) if ncell else 0.0
    min_jac = float(jac.min()) if ncell else 1.0
    # `ncell > 0` first: with no cells the rest pass vacuously (0 skew/ar, 0 neg, 1.0 ortho/jac)
    ok = (ncell > 0 and (neg == 0) and (max_skew <= SKEWNESS_LIMIT) and (min_ortho >= ORTHO_MIN)
          and (min_jac >= JACOBIAN_LIMIT) and (max_ar <= ASPECT_RATIO_LIMIT))
    return dict(nodes=nodes, polys=polys, skew=skew, ar=ar, area=area, jac=jac,
                ncell=ncell, nnode=len(nodes), nquad=len(quads), ntri=len(tris),
                neg=neg, max_skew=max_skew, min_ortho=min_ortho, max_ar=max_ar, min_jac=min_jac,
                markers={k: len(v) for k, v in markers.items()}, acceptable=ok)

def _summary_from_analysis(a) -> dict:
    """
    Narrow an `analyze_su2()` result to the fields `common.schemas.MeshQualitySummary` needs.
    """
    return dict(ncell=a["ncell"], max_skew=a["max_skew"], min_ortho=a["min_ortho"],
                min_jac=a["min_jac"], max_ar=a["max_ar"], acceptable=a["acceptable"])
