# Geometry manifest

`geometry.sqlite` records every airfoil in the dataset and how its stored shape was made, and every
file the ingest rejected with the same detail. Only `src/datagen/geometry/manifest/store.py` writes
it, through the ingest; don't edit it by hand.

## Adding airfoils

```
python -m src.datagen.geometry.manifest.ingest [<folder>] [--dry-run]
```

Every `.dat` under `<folder>` is read, normalized, closed, fitted with a CST and checked. The file
format is: line 1 is the airfoil name, equal to the file stem; then one `x y` pair per line in Selig
order (TE, over the upper surface to the LE, back along the lower surface).

- An airfoil already stored (same `geometry_id`) is skipped as a duplicate.
- A different airfoil under a name already stored is rejected; rename its file.
- A rejected file is kept as fully as an accepted one: a copy, its rendering if it got as far as a
  fit, and a `rejected_shapes` row. Only a file that can't be read (`format`) or that crashed the
  pipeline (`error`) has nothing but its report.
- Every run first judges the stored rejections again from their copies, so a loosened gate lets them
  in; a file already stored as a rejection is then skipped as a duplicate.
- Every file seen gets a report, and every run a summary (see Files). A report flags a *suspect
  point* when the worst-fitted point is at least 4x further off than its neighbours and the file's
  points zig-zag there, which is what a mistyped coordinate looks like.

Without a folder, a run only brings the stored airfoils to the current shape version (see Shape
versions); every run does that first.

## Tables

**`ingests`**: one row per ingest run.
- `ingest_id`: UTC start time plus a random suffix, e.g. `20261013T0830Z_3f9a`.
- `started_at`, `source_root` (the folder ingested: its path inside the repo, else only its name),
  `shape_version`, `git_commit` (`-dirty` with uncommitted changes under `src/datagen/geometry`), `report_path` (the run's summary).

**`ingest_items`**: one row per file an ingest saw.
- `status`: `added`, `reshaped` (a stored airfoil refitted from its raw copy under a new shape
  version), `duplicate` or `rejected`.
- `reason`: for a rejection, `format`, `family_mismatch`, `too_few_points`, `surfaces_cross`,
  `fit_tolerance`, `name_taken` or `error`; for a duplicate, what matched: `geometry_id`,
  `norm_hash`, or `rejected` (the same points as a stored rejection).
- `geometry_id`: the stored airfoil the file added or duplicated.
- `report_path`: the file's report, with the details behind the reason.

**`geometries`**: one row per airfoil, never changed once written.
- `geometry_id`: SHA-256 of what defines the airfoil. For a parametric family that is its
  canonical string (e.g. `naca4:m=2:p=4:t=12:te=closed`), so two files of the same NACA are one
  airfoil; otherwise it is the raw coordinates.
- `name`: unique, ignoring case.
- `family`, `family_params`: `naca4`, `naca4_mod`, `naca5` or `naca16`, and its digits (JSON).
  Empty for tabulated airfoils. Family airfoils are built from their name in closed-TE form, and
  their file is only checked against that shape.
- `raw_path`, `raw_hash`, `raw_points`: the verbatim copy of the source file, the SHA-256 of its
  coordinates, and how many points it has. Few points (old tables at standard stations have ~35)
  leave the shape between them a guess, even when the fit passes at the points.
- `ingest_id`: the run that added it.

**`geometry_shapes`**: the stored shape, one row per airfoil per `shape_version`.
- `cst`: the shape (JSON), Kulfan's CST. Each surface is
  `y = x^n1 (1-x)^n2 * sum_i A_i * C(N,i) x^i (1-x)^(N-i) + le_weight * x (1-x)^(N+0.5)`,
  with `upper` and `lower` the coefficients `A_0..A_N`, LE first. The lower `A_0` is minus the
  upper's, so the nose has one radius (`A_0^2 / 2`). `le_weight` is Kulfan's leading-edge
  modification, the same on both surfaces: it lets a cambered nose lean without a second radius.
  Kulfan's TE-thickness term is zero: every shape is closed.
- `rotation_deg`: how far normalizing turned the source (for a family, its formula) to put the chord
  line on the x axis, signed: positive means the source's chord was nose-up, so normalizing turned
  it nose-down. Angles of attack in the dataset are measured from the stored chord line:
  `alpha_dataset = alpha_source + rotation_deg`, where the source's datum is e.g. a published polar's.
- `closure`: how the TE was closed. `none`: the source was closed. `pried`: an open source, closed
  by subtracting x times its TE ordinate from each surface. `formula`: a family, built closed.
- `root_te_gap`: the source's TE gap before closing [chord].
- `source_dev`: worst distance from the source's points, as they were before closing, to the CST
  [chord]: how far the stored shape is from the file, closing included.
- `fit_max`: worst distance from the points the CST was fitted to (closed) to the CST [chord].
- `te_wedge_deg`: the angle between the surfaces where they meet at the TE. The fit is bounded so it
  is never negative; 0 is a cusp (both surfaces arrive at one slope).
- `norm_path`, `norm_hash`: a 1001-point rendering of the CST, and the SHA-256 of its coordinates.

**`rejected_shapes`**: one row per rejected file per `shape_version`, kept apart so nothing that
reads `geometry_shapes` can pick up a rejected shape. Its columns are those of `geometries` and
`geometry_shapes` above, plus:
- `raw_hash` and `shape_version` are the key: names repeat among rejections (a corrected file keeps
  its name), and two files of one NACA name share a `geometry_id`.
- `reason`, `detail`: the rejection's code (as in `ingest_items`) and its words, e.g. where the
  surfaces cross.
- The measurements stop where the pipeline did: `family_mismatch` and `too_few_points` have no
  `cst`, `fit_max`, `source_dev`, `te_wedge_deg` or rendering. A `name_taken` file has them all.
- `raw_path`: its copy (for a stored airfoil a new version rejects, its own raw copy).
- `ingest_id`, `report_path`: the run that last judged it, and that run's report.

**`geometry_outcomes`** (a view): accepted shapes and rejections in one table, with `status`
(`accepted` or `rejected`) and `reason`, over the columns the two share.

Every airfoil is normalized the same way first: LE at (0, 0) where the surface tangent is
perpendicular to the line to the TE midpoint, TE at x = 1, unit chord.

## Files

Paths in the tables are relative to `data/`:

| Path | Holds |
|---|---|
| `manifest/datagen/geometry/geometry_archive.sqlite` | shapes and rejections of earlier shape versions (only once there are any) |
| `raw/datagen/geometry/raw/<name>.dat` | the accepted source files, byte for byte |
| `raw/datagen/geometry/rejected/<hash8>/<name>.dat` | the rejected source files, byte for byte, in a folder named by the first 8 characters of `raw_hash` |
| `raw/datagen/geometry/normalised/<shape_version>/<name>.dat` | the CST renderings |
| `raw/datagen/geometry/normalised/<shape_version>/rejected/<hash8>/<name>.dat` | the renderings of rejected fits |
| `raw/datagen/geometry/report/<ingest_id>/` | `summary.txt`, and a `.txt` report per file seen, at its path in the ingested folder |
| `raw/datagen/geometry/report/calibration/` | how the CST order, nose and fit tolerance were chosen |

The CST in the table is the airfoil's definition downstream; the renderings are for viewing and for
other tools.

## Shape versions

`shape_version` (e.g. `cst8-r1`) names the pipeline that produced a shape: normalization, TE
closure, CST order and fit, and the rendering. A unit test
(`tests/unit/datagen/geometry/test_geom_shape_version.py`) fails when the shapes change but the
version does not.

`geometry_shapes` holds the current version only. After a version bump, run the ingest (with or
without a folder):
1. Every stored airfoil is refitted from its raw copy and gets a row under the new version
   (`reshaped`). One the new version rejects keeps its `geometries` row but no shape; it gets a
   `rejected_shapes` row instead and is judged again every run. Every stored rejection is judged
   again under the new version too.
2. The rows of older versions move to `geometry_archive.sqlite` next to this file, one row per
   shape or rejection, the old row as JSON (`archived_shapes`, `archived_rejections`). Their
   renderings are deleted; the CST in the archive defines the shape.

`geometries` is never touched: IDs and names stay as they were.

## Example queries

```sql
-- why each file of the latest ingest was rejected
SELECT source_relpath, reason, report_path FROM ingest_items
WHERE status = 'rejected' AND ingest_id = (SELECT max(ingest_id) FROM ingests);

-- every rejection under one version, with what was measured and why
SELECT name, reason, fit_max, source_dev, te_wedge_deg, detail FROM rejected_shapes
WHERE shape_version = 'cst8-r1' ORDER BY reason, fit_max;

-- accepted and rejected together, e.g. how the fit error spreads around the gate
SELECT status, name, fit_max FROM geometry_outcomes
WHERE shape_version = 'cst8-r1' AND fit_max BETWEEN 5e-4 AND 2e-3 ORDER BY fit_max;

-- every airfoil with its shape under one version
SELECT g.name, g.family, s.closure, s.root_te_gap, s.source_dev, s.fit_max
FROM geometries g JOIN geometry_shapes s USING (geometry_id)
WHERE s.shape_version = 'cst8-r1';

-- airfoils within 2.5e-3 chord of their source file, closing included, without a near-cusp TE
SELECT g.name FROM geometries g JOIN geometry_shapes s USING (geometry_id)
WHERE s.shape_version = 'cst8-r1' AND s.source_dev <= 2.5e-3 AND s.te_wedge_deg >= 1;
```
