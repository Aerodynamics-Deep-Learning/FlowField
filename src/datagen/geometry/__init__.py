"""
Layout:
    common/    constants, schemas, the data/ layout, .dat I/O; imports nothing else in geometry
    build/     one root .dat -> a closed, fitted CST shape (pure), and the calibration of its limits
    manifest/  geometry.sqlite (store.py) and the folder ingest that fills it
    sample/    a stored shape given the TE its mesher needs (contour.py), and its sampling (Sequence 3)


Sequence 1 (meshing_main): one study = one airfoil x one mesher/topology

1 - define problem (once per study, defined by the problem/mesher call)
    a - mesher gets called with a specific airfoil name
    b - the airfoil's normalised rendering (from the manifest) is appended
        as the tensor for Airfoil config; a placeholder until step 3 replaces it
    c - the default freestream of the entire run gets appended (only Re
        matters here, and should be the worst-case Re); c2d to read it from
        the freestream too, instead of its own `recd`
    d - the mesher and topology get appended, defined by the problem call
2 - the mesh config is created (each evaluation) by:
    1 - either the optim procedure,
    2 - or the user specified mesh configs
    3 - or the default mesh configs (unreliable)
    -> MeshIn is built and validated here; an invalid config counts as a
       failed evaluation, not a crash
3 - get the actual geometry (each evaluation, see Sequence 3)
    a - the geometry module samples the airfoil from its manifest CST, using
        the config's point count and spacing knobs, and the TE the
        mesher/topology requires
    b - the sampled points (and le_idx) replace the root tensor
4 - get output (each evaluation)
    a - the MeshIn is shoved into the mesher
    b - the outputted mesh is taken with its quality flagging
    c - the sampling checks from step 3 are attached
    d - store the values in the mesh manifest: config, sampler version,
        sampling checks, mesh result
5 - rerun the optim procedure if needed (back to 2.1); user/default configs
    run steps 2-4 once
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
3 - sample the shape
    a - exactly n_surface points, placed by arc length from the config's
        spacing knobs (le/te spacing, cluster width), positions evaluated
        on the function
    b - pinned points land exactly; first == last (blunt stays open)
4 - sampling checks against the shape
    a - max turn per point <= 15 deg (pinned points exempt)
    b - max distance from the shape <= the deviation cap
    c - max panel length <= H_MAX (provisional 0.006c)
5 - return the sampled airfoil (points + le_idx) and the check report

Note: the gmsh C-grid can only use the blunt branch once its anchor indices
become x/c positions.
"""
