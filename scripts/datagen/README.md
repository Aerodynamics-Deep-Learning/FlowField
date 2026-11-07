# Datagen scripts

The command lines of the data-generation pipeline: airfoil `.dat` files in, a geometry manifest, the mesh
study's best mesh per airfoil, and those meshes back out as `.su2` files. Each script runs one stage; the
master runs several from a YAML config. Run everything from the repo root.

```
scripts/datagen/
├── README.md        this guide
├── engine.py        what every script shares: the plan, the output rule, the logs
├── geometry/        calibrate.py, ingest.py
└── meshing/         machine_check.py (+ parity.py), calibrate.py, campaign.py, remesh.py,
                     master.py (+ master.example.yaml)
```

## 1. First, on a new machine

1. Install the requirements into the project's Python (3.11): `pip install -r requirements.txt`.
2. Copy `tools.example.toml` to `tools.toml` and point `c2d` at your Construct2D executable (gmsh and SU2 too,
   if you use them).
3. Check that this machine samples and meshes exactly as the reference machine did:

   ```
   python -m scripts.datagen.meshing.machine_check
   ```

   It must end with `machine_check passed`, both steps `passed`. **A skip is a failure**: step 2 skips when
   c2d is missing or is a different executable from the reference's, and then meshes made here belong to
   another settings row and will not re-mesh runs stored elsewhere. A pass is stamped in
   `data/.machine_check.json`; until then, and whenever c2d, the parity reference or the code's versions
   change, the mesh scripts warn at the start of every run.

## 2. The pipeline at a glance

```
 .dat folder ──► geometry_calibrate ──► CST calibration report        (recommends; changes nothing)
      │
      └────────► geometry_ingest ──► geometry manifest (geometry.sqlite)
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    ▼                         ▼                          ▼
             mesh_calibrate               mesh_campaign              mesh_remesh
             ladder report +              run store: every   ───►    <out>/<airfoil>/<airfoil>.su2 (.vtk),
             its runs (recommends)        run, each best             delivered only on a matching hash
```

The two calibrations measure; they never change a constant. Their reports tell you what to change in
`src/datagen/constants.py`, where every value that can change lives.

## 3. The scripts

`python -m scripts.datagen.<folder>.<script> --help` lists every option; the table says what each is for. The
stage is the script's name in configs and logs.

| Script (`python -m scripts.datagen.…`) | Stage | What it does | Reads | Writes | Time |
|---|---|---|---|---|---|
| `meshing.machine_check` | `machine_check` | runs the parity test; `--rewrite` writes the parity reference here instead | the shared manifest, c2d | the stamp (`--rewrite`: `machine_parity.json`) | ~1 min |
| `geometry.calibrate <folder>` | `geometry_calibrate` | fits every file at CST orders 4-12 for choosing the CST constants | the `.dat` folder | `--out`: a report file | ~1 min for 190 files |
| `geometry.ingest [folder]` | `geometry_ingest` | adds the folder's airfoils to a manifest; without a folder, brings stored ones to the current shape version | the `.dat` folder | `--data-root`: a data folder (manifest, raw copies, reports) | ~15 s for 190 files |
| `meshing.calibrate` | `mesh_calibrate` | measures the numbers the ladder's rungs run on | `--geometry-db` | `--out`: a folder, `<date>.txt` and `<date>.sqlite` | ~1 h for 158 airfoils on 6 workers (rung 1 ~12 min, rung 2 ~6, rung 3 ~41) |
| `meshing.campaign` | `campaign` | the ladder over every airfoil not finished under the current constants | `--geometry-db` | `--db`: a run store | ~10 min for 158 airfoils on 6 workers, most of it the few that reach rung 3 |
| `meshing.remesh` | `remesh` | rebuilds best runs (`--airfoils`, `--all`) or one run (`--run`), each checked against its stored hash | `--db`, `--geometry-db` | `--out`: a folder per airfoil, and `quality.png` | ~3 s an airfoil |

Exit codes, for every script and the master:

| Code | Meaning |
|---|---|
| 0 | done (the re-mesh: every mesh delivered) |
| 1 | a stage stopped or failed (a campaign stopped early, a mesh not delivered, a machine check not passed) |
| 2 | refused before anything ran: a mistake in what was asked |
| 130 | Ctrl-C |

A script runs its stage in its own process, so you can run it under a debugger and stop anywhere in
`src/datagen`. The campaign's and the calibration's airfoils run in worker processes, though: to stop inside a
ladder, set the debugger to follow subprocesses, or call `Optim_run_ladder` yourself.

## 4. Running several stages: the master

```
python -m scripts.datagen.meshing.master <config.yaml>
```

Copy `scripts/datagen/meshing/master.example.yaml` anywhere and edit it. The stages always run in this order,
whatever order the config lists them in:

```
machine_check -> geometry_calibrate -> geometry_ingest -> mesh_calibrate -> campaign -> remesh
```

- **One section per stage**, each with `enabled: true` or `enabled: false`. `machine_check` is on when its
  section is left out; switching it off warns.
- **A section's other keys are its script's options**, `-` written as `_`: `--settings-label` is
  `settings_label`, `--dry-run` is `dry_run`. `true` turns a flag on; a list becomes `a,b`. No constant can be
  set here, since no script has an option for one.
- **Paths**: a relative path is under `data/`; an absolute one is used as written. `shared` names a committed
  store (section 5).
- **`folder:`**, beside the sections, names the run's folder (section 5); left out, it is a new
  `data/scratchpad/<time>_<config name>/`.
- **A stage reads what an earlier stage of the same run wrote**: the mesh stages read this run's ingested
  manifest, the re-mesh this run's campaign store, unless the config names something else.
- **Everything is checked before anything runs**, even the machine check: every option, every path, every
  named airfoil, each stage's own checks. One problem refuses the whole run, listing all of them.
- **Each stage runs as a child process**; the first one that fails stops the run with its exit code. Ctrl-C
  waits for the running stage to print its summary, then stops with 130.
- **A calibration and the stage that uses its numbers both on** (`geometry_calibrate` with `geometry_ingest`,
  `mesh_calibrate` with `campaign`) warns: the second runs at the constants as they are now, not at what the
  calibration recommends. Edit the constants between two runs.

## 5. Where data and logs go

Every run has a folder of its own: a new `data/scratchpad/<time>_<run name>/`, or the config's `folder:`. It
holds the run's log, `log.log`, a copy of the config, `config_<time>.yaml`, and every output left out, under
`<stage>/`:

```
data/scratchpad/2026-11-07_052637_e2e_for_cst/     the run's folder
├── log.log                                        the run's one log
├── config_2026-11-07_052637.yaml                  the config, copied at the start
├── geometry_ingest/                               outputs left out, one folder per stage: data only
├── campaign/meshing.sqlite
└── remesh/<airfoil>/<airfoil>.su2
```

Every output is decided before anything runs:

| What you give | Where it goes | What you are told, at the start |
|---|---|---|
| nothing | the run's folder, `<stage>/` | a line naming it |
| a path that does not exist | there, made | nothing |
| a path that exists | there: a store is added to, a report file overwritten, a mesh folder's same airfoils replaced | a `!` warning |
| `shared` | the committed store | a `!!` warning |

A `folder:` that exists is used again, with a `!` warning: the log adds this run under its own `===== PLAN =====`,
its config copy joins the earlier runs' (each named by its start), and the outputs left out are the earlier run's,
so a stopped campaign resumes.

- **`shared`** is `data/` for the ingest (the committed manifest), `data/manifest/datagen/meshing/meshing.sqlite`
  for the campaign, and the committed calibration folders for the calibrations. The re-mesh has none: meshes
  are never kept in a committed place. For inputs, `shared` is the default.
- **`data/scratchpad/`** and **`data/experiments/`** are gitignored: put named experiments under
  `data/experiments/`. Anything else you name under `data/` shows up in `git status`. Nothing ever deletes the
  scratchpad; clear it when you like.
- **A campaign resumes only into a store you named**: left out, every run gets a new scratchpad store. Give
  `--db` (or `db:`) a path, or the run a `folder:`, to be able to rerun a stopped campaign; the "exists" warning
  then just says it adds to it.
- **The run store and calibrations are only shared on purpose.** Experiment in your own stores; the campaign
  run into `shared` is the one that gets committed, once the constants are agreed.
- **The log**: one per run, `log.log` in the run's folder; the stage folders hold data only. It is written and
  synced to disk line by line, so a killed run keeps everything up to that moment and an editor showing it (VS
  Code) follows it as it grows. A refused run logs into a new scratchpad folder, never a named one.

## 6. Reading a log

Every line, on screen and in the log:

```
HH:MM:SS [who] [stage]              [level] text
```

- **who**: `run` is the script itself (the plan, starting and ending stages, warnings); `cmd` is the stage's
  own output.
- **stage**: `plan` before anything runs, the stage's name, `summary` at the end.
- **level**: `info`; `!` a warning; `!!` a shared store written; `error`.

Markers open each block, so `START|END` finds every boundary: `===== PLAN =====`, then `===== START <stage> =====`
and `===== END <stage> =====` around each stage, then `===== SUMMARY =====`. Inside `mesh_calibrate`, each rung
has its own `----- START rung N of 3 -----` and `----- END rung N of 3 -----`, the end followed by what it found.

The stages log statistics and edge cases; every table goes to its report file, named at the end:

- **ingest**: the counts, the fit statistics, one line per rejection and duplicate, a `!` per suspect point;
- **CST calibration**: one table, the ingest's configuration and its neighbours at the fit tolerance, and the
  airfoils whose surfaces cross or whose fit strays; every configuration and every airfoil are in the report;
- **campaign and mesh calibration**: a progress line every tenth of the airfoils (`16/158 airfoils (10%), 6.8
  airfoils/min, ends ~05:52`), and a line of its own only for an airfoil that aborted, failed or reached rung 3,
  saying where each rung before stopped short (`e376: passed in rung 3, 68,820 nodes, 120 c2d runs; rung 2: no pass
  in 50 meshes at jmax 148, closest size_ratio 1.2513 > 1.25`). The calibration's rung 1 logs only an airfoil
  without jmax*, its rung 3 only one without a pass; the campaign's summary counts the golden runs' failures by gate;
- **re-mesh**: the same progress line, a `!` per airfoil not delivered saying why, then the delivered meshes'
  quality against each gate (limit, median, worst), with a histogram per gate in `<out>/quality.png`.

```
21:30:12 [run] [plan]               [info]  ===== PLAN =====  master dx003, started 2026-11-06 21:30:12 on Ege_Laptop, code 68ae03b
21:30:12 [run] [plan]               [info]  config D:\runs\dx003.yaml, copied into the run folder as config_2026-11-06_213012.yaml
21:30:12 [run] [plan]               [info]  stages machine_check -> campaign -> remesh (off: mesh_calibrate)
21:30:12 [run] [plan]               [!]     run folder: data/experiments/dx003 exists; this run adds to its log.log and writes the outputs left out into it
21:30:12 [run] [campaign]           [!]     writes db: nothing given, writing into the run folder: data/experiments/dx003/campaign/meshing.sqlite exists; this run adds to it
21:30:12 [run] [remesh]             [info]  reads db: this run's campaign, data/experiments/dx003/campaign/meshing.sqlite
21:30:12 [run] [machine_check]      [info]  ===== START machine_check =====  python -m scripts.datagen.meshing.machine_check
21:31:06 [cmd] [machine_check]      [info]  step 2, c2d: passed
21:31:07 [run] [machine_check]      [info]  ===== END machine_check =====  exit 0, 0.9 min
21:31:07 [run] [campaign]           [info]  ===== START campaign =====  python -m scripts.datagen.meshing.campaign --db ...
21:31:08 [cmd] [campaign]           [!]     2 ladders left running by an earlier campaign marked aborted; their airfoils run again
21:34:20 [run] [campaign]           [info]  ===== END campaign =====  exit 0, 3.2 min
21:34:30 [run] [summary]            [info]  ===== SUMMARY =====  finished in 4.3 min
```

Every warning appears three times: at the top, where it happened, and again in the summary.

## 7. The machine check and its stamp

The parity test (`tests/integration/datagen/mesh/optimizer/test_machine_parity.py`) compares this machine
against `machine_parity.json`, written on the reference machine: step 1 hashes every manifest airfoil's samples
at three configs and n*'s grid; step 2 meshes the ten calibration airfoils with c2d and hashes the `.su2`
files. The sampler gives the same bits on every machine by construction; c2d is the part that can differ.

A pass writes `data/.machine_check.json`: this computer's name, the c2d executable's hash, the reference's,
and the sampler and shape versions. The mesh scripts compare it with the machine as it is now and warn, naming
what moved, until the check passes again. A config with `machine_check` enabled checks at its start instead.

The master's machine check refuses to start when the reference does not fit the code (another sampler or shape
version) or lists other airfoils than the shared manifest, and says how to fix it.

## 8. Changing a constant

The campaign runs at the constants as they are. A changed constant (or another c2d executable) makes a new
`settings_id`, under which every airfoil runs again; the rows of earlier settings stay in the store, and a
re-mesh by name always takes the best run under the current settings. `--settings-label` names a settings row
when the campaign that first records it runs, so you can find it again in a SQLite viewer.

## 9. Ingesting into the shared manifest is a release

The ingest is not machine-portable (its fit runs on LAPACK), so the committed `geometry.sqlite` is what every
machine shares. Writing it is a release:

1. One person ingests at a time, after pulling: `geometry.sqlite` is a binary file git cannot merge.
2. `python -m scripts.datagen.geometry.ingest <folder> --data-root shared` (or `data_root: shared` in a config).
3. The parity reference must list the new airfoils. A master run whose machine check passed earlier in the
   same run rewrites it by itself; otherwise run `python -m scripts.datagen.meshing.machine_check --rewrite` on the
   reference machine.
4. Commit `data/manifest/datagen/geometry/geometry.sqlite` and
   `tests/integration/datagen/mesh/optimizer/machine_parity.json` together.

After a shape or sampler version change in the code the same steps apply, done by hand on the reference
machine; until then every machine check is refused.

## 10. When something goes wrong

- **Refused before anything ran (exit 2)**: the log lists every problem, each under its stage. Fix them all
  and run again.
- **"already holds today's calibration"**: a calibration folder takes one calibration a day; give another
  `--out`, or leave it out for the scratchpad.
- **A campaign stopped (exit 1) or was interrupted (130)**: run it again into the same store; it skips what
  finished and reruns what was cut short.
- **A re-mesh comes out different**: nothing is delivered, and the message gives both hashes and where the run
  was meshed. Run the machine check; a run stored under another c2d or code version is refused outright, with
  the commit to re-mesh it at.
- **"database is locked"**: a SQLite viewer holding the store open; the store waits for it, then gives up.
- **The machine check fails**: the message names the airfoils that differ and the reference's machine. A
  different c2d build is the usual cause.

## 11. YAML, briefly

Configs are YAML, read with `yaml.safe_load`. The traps:

- `yes`, `no`, `on`, `off` unquoted become true/false, and a name like `0012` becomes a number: quote such values.
- `1e-3` is read as text; write `1.0e-3`.
- Indent with spaces only.
- Inside double quotes a backslash starts an escape: write Windows paths with `/`, or in single quotes.
