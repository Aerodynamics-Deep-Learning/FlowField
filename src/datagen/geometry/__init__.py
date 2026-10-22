"""
Layout:
    common/    constants, schemas, the data/ layout, .dat I/O; imports nothing else in geometry
    build/     one root .dat -> a closed, fitted CST shape (pure), and the calibration of its limits
    manifest/  geometry.sqlite (store.py) and the folder ingest that fills it
    sample/    a stored shape given the TE its mesher needs (contour.py), and its sampling (Sequence 3)


Sequence 1 (meshing_main): one study = one airfoil x one mesher/topology

1 - define the study (once, by the problem/mesher call;
    meshing/optimizer/sampling.py)
    a - the airfoil by name: its geometry_id, and its shape under the current
        shape version (none: the airfoil can't be studied under it)
    b - the mesher and topology, which fix the TE (Sequence 3, step 1)
    c - the study's TE gap (0.4% chord by default) -> the contour, built once
        (build_study_contour); a refused contour fails the study's setup
    d - the default freestream of the entire run (only Re matters here, and
        should be the worst-case Re); c2d to read it from the freestream too,
        instead of its own `recd`
2 - the mesh config is created (each evaluation) by:
    1 - either the optim procedure,
    2 - or the user specified mesh configs
    3 - or the default mesh configs (they pass on every stored airfoil)
    -> validated as its config class; an invalid config counts as a failed
       evaluation, not a crash
3 - sample the contour with the config's knobs (each evaluation, Sequence 3;
    sample_for_mesher)
    a - a pre-check failure (turn, or n x jmax over the cell ceiling) rejects
        the config before meshing: recorded, directly infeasible, and not
        counted against the budget
    b - otherwise the MeshIn is built from the points and le_idx
4 - get output (each evaluation)
    a - the MeshIn is shoved into the mesher
    b - the outputted mesh is taken with its quality flagging
    c - the sample report from step 3 is attached
    d - store the values in the mesh manifest: config, shape and sampler
        versions, sample report, mesh result
5 - rerun the optim procedure if needed (back to 2.1) until its budget of
    finished meshes is spent; user/default configs run steps 2-4 once
6 - finish: keep the best feasible mesh's files, delete the rest, record the
    study's result


Sequence 2 (add airfoils to the database; manifest/ingest.py, one airfoil: build/add.py)
0 - bring the stored airfoils to the current shape version: any without a
    shape under it (all, after a version bump) is refitted from its raw
    copy, and every stored rejection is judged again from its copy (a
    loosened gate lets it in); shapes and rejections of older versions move
    to geometry_archive.sqlite
    a - then the user's folder, if given, supplies root .dat files
1 - identify each root .dat (sorted path order)
    a - read strictly: name line == file stem, Selig `x y` rows
    b - geometry_id: hash of the family string for a parametric name
        (naca4/5, modified 4-digit, 16-series), else of the raw coords
    c - already in the database (same geometry_id), or the same points
        already stored as a rejection -> duplicate, skip
2 - build the shape
    a - points: the family formula in closed-TE form, else the file's
    b - normalize (rigid): LE where the tangent is perpendicular to the
        line to the TE midpoint (on a spline, inserted if between points)
        -> (0,0); rotate the TE midpoint onto the x axis, scale it to 1,
        stretch each surface's x to end at 1
    c - close: each surface loses x * its TE ordinate, i.e. Kulfan's
        TE-thickness term is removed (the inverse of Sequence 3's
        opening); a family is already closed
    d - family only: the file's own points within 1e-2c of the formula
        shape, else reject (misread name)
    e - fit a CST: order 8, N1 0.5, N2 1, one A_0 for both surfaces (one
        nose radius) plus Kulfan's leading-edge term (the nose's lean);
        least squares bounded so the TE wedge is >= 0 (a cusp at worst)
    f - measure: the fit against the points it was fitted to, and against
        the file's points before closing (what closing moved); TE wedge
3 - gates (reject on fail)
    a - each surface has >= order + 4 points
    b - thickness > 0 everywhere inside, wedge >= 0 at the TE (closing
        can cross the surfaces before the TE)
    c - max fit error vs the points <= 1e-3c
4 - store
    a - same rendering (norm_hash) as a stored airfoil -> duplicate;
        name taken by another airfoil -> reject
    b - files first (raw copy, 1001-point rendering), then rows in one
        transaction: geometries (identity, never changes) and
        geometry_shapes (per shape_version: cst, rotation, closure, gap,
        source deviation, fit, TE wedge)
    c - a rejection is stored just as fully, apart: its copy, its rendering
        if it got a fit, and a rejected_shapes row (the same columns, plus
        the reason; empty past the step that rejected it); only an
        unreadable or crashing file is left with just its report
5 - report: one per file seen plus a run summary; every outcome is an
    ingest_items row; a worst-fit point far off its neighbours where the
    file zig-zags is flagged as a suspect (likely mistyped) coordinate


Sequence 3 (sampling, Sequence 1 step 3): build the shape, then sample once

1 - the TE the mesher/topology requires:
        c2d/gmsh O-grid -> round, c2d C-grid -> sharp, gmsh C-grid -> blunt
2 - build the shape as one continuous curve in Selig order, once per study
    (sample/contour.py); g is the study's TE gap, 0.4% chord by default
    1 - sharp: the CST as is
    2 - blunt: each surface opened by x * g/2 from the LE, i.e. the CST's
        linear TE-thickness term (the inverse of Sequence 2's closing; adds
        no curvature); left open at (1, +-g/2)
    3 - round: the blunt shape cut at the welds x = 1 - g/2, each cut piece
        replaced by a quadratic Bezier cap W -> P1 -> (1, 0), P1 where the
        surface's tangent at W reaches x = 1
        a - the cap starts at W with the surface's own slope and reaches
            (1, 0), mid-gap, with a vertical tangent: no kink anywhere, and
            the chord stays exactly 1
        b - each cap must stay on its own side of y = 0, else the shape is
            refused for the study (ContourError)
    4 - pinned points: the closure point (or the open ends), the welds and
        the LE; the curve is tabulated by arc length and curvature
3 - the spacing wanted along the curve (sample/place.py), from the config's
    knobs: dx_mid at mid-chord, dipping to le_ratio * dx_mid at the LE and
    te_ratio * dx_mid at the TE over about cluster_width; a round TE's
    spacing holds across its caps, clamped to their turn limit
4 - pre-checks, before any point is placed; a failure rejects the config
    (never meshed; directly infeasible for the optimizer)
    a - turn: curvature x spacing stays under 15 deg everywhere
    b - points: no more than the caller allows (its cell ceiling)
5 - place the points: each stretch between pinned points gets ceil(the
    integral of ds/h) panels, so n is derived and no panel exceeds its
    target; positions evaluated on the curve, pinned points exact, first ==
    last for a closed TE (a blunt one stays open)
6 - post-checks a correct sampler cannot fail (SamplerAssertionError, a
    bug): panel <= dx_mid <= DX_MAX, turn <= 15 deg (a sharp TE's closure
    point skipped), deviation from the curve <= DEV_MAX
7 - return the points, le_idx and the report (sample/sample.py)

Note: the gmsh C-grid can only use the blunt branch once its anchor indices
become x/c positions.
"""
