# Datagen scripts

The command lines of the data-generation pipeline: airfoil `.dat` files in, a geometry manifest, the mesh
study's best mesh per airfoil, and those meshes back out as `.su2` files. Each script runs one stage; the
master runs several from a YAML config. Run everything from the repo root.

## 1. First, on a new machine

1. Install the requirements into the project's Python (3.11): `pip install -r requirements.txt`.
2. Copy `tools.example.toml` to `tools.toml` and point `c2d` at your Construct2D executable (gmsh and SU2 too,
   if you use them).
3. Check that this machine samples and meshes exactly as the reference machine did:

   ```
   python -m scripts.datagen.machine_check
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
`src/datagen/geometry/common/constants.py` or `src/datagen/meshing/optimizer/common/constants.py`.

## 3. The scripts

`python -m scripts.datagen.<script> --help` lists every option; the table says what each is for.

| Script | What it does | Reads | Writes | Time |
|---|---|---|---|---|
| `machine_check` | runs the parity test; `--rewrite` writes the parity reference here instead | the shared manifest, c2d | the stamp (`--rewrite`: `machine_parity.json`) | ~1 min |
| `geometry_calibrate <folder>` | fits every file at CST orders 4-12 for choosing the CST constants | the `.dat` folder | `--out`: a report file | ~1 min for 190 files |
| `geometry_ingest [folder]` | adds the folder's airfoils to a manifest; without a folder, brings stored ones to the current shape version | the `.dat` folder | `--data-root`: a data folder (manifest, raw copies, reports) | a few minutes for 190 files |
| `mesh_calibrate` | measures the numbers the ladder's rungs run on | `--geometry-db` | `--out`: a folder, `<date>.txt` and `<date>.sqlite` | rungs 1-3 ~1.5-2 h on 6 workers (estimated) |
| `mesh_campaign` | the ladder over every airfoil not finished under the current constants | `--geometry-db` | `--db`: a run store | ~12 min for 158 airfoils on 6 workers |
| `mesh_remesh` | rebuilds best runs (`--airfoils`, `--all`) or one run (`--run`), each checked against its stored hash | `--db`, `--geometry-db` | `--out`: a folder per airfoil | ~5 s an airfoil |

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
python -m scripts.datagen.mesh_master <config.yaml>
```

Copy `mesh_master.example.yaml` anywhere and edit it. The stages always run in this order, whatever order the
config lists them in:

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

Every output is decided before anything runs:

| What you give | Where it goes | What you are told, at the start |
|---|---|---|
| nothing | a new folder, `data/scratchpad/<time>_<run name>/<stage>/` | a line naming it |
| a path that does not exist | there, made | nothing |
| a path that exists | there: a store is added to, a report file overwritten, a mesh folder's same airfoils replaced | a `!` warning |
| `shared` | the committed store | a `!!` warning |

- **`shared`** is `data/` for the ingest (the committed manifest), `data/manifest/datagen/meshing/meshing.sqlite`
  for the campaign, and the committed calibration folders for the calibrations. The re-mesh has none: meshes
  are never kept in a committed place. For inputs, `shared` is the default.
- **`data/scratchpad/`** and **`data/experiments/`** are gitignored: put named experiments under
  `data/experiments/`. Anything else you name under `data/` shows up in `git status`. Nothing ever deletes the
  scratchpad; clear it when you like.
- **A campaign resumes only into a store you named**: left out, every call gets a new scratchpad store. Give
  `--db` (or `db:`) a path to be able to rerun a stopped campaign; the "exists" warning then just says it adds
  to it.
- **The run store and calibrations are only shared on purpose.** Experiment in your own stores; the campaign
  run into `shared` is the one that gets committed, once the constants are agreed.
- **The log**: one per run, `<time>_<run name>.log`, an identical copy in every folder the run writes data
  into (beside a store, inside an output folder, beside an ingested manifest). A run that writes no data (a
  machine check, a refused run) logs into its scratchpad folder. It is written line by line, so a killed run
  keeps everything up to that moment.

## 6. Reading a log

Every line, on screen and in the log:

```
HH:MM:SS [who] [stage]              [level] text
```

- **who**: `run` is the script itself (the plan, starting and ending stages, warnings); `cmd` is the stage's
  own output.
- **stage**: `plan` before anything runs, the stage's name, `summary` at the end.
- **level**: `info`; `!` a warning; `!!` a shared store written; `error`.

```
21:30:12 [run] [plan]               [info]  mesh_master dx003, started 2026-11-06 21:30:12 on Ege_Laptop, code 68ae03b
21:30:12 [run] [plan]               [info]  stages machine_check -> campaign -> remesh (off: mesh_calibrate)
21:30:12 [run] [campaign]           [!]     writes db: data/experiments/dx003/meshing.sqlite exists; this run adds to it
21:30:12 [run] [remesh]             [info]  reads db: this run's campaign, data/experiments/dx003/meshing.sqlite
21:30:12 [run] [machine_check]      [info]  start: python -m scripts.datagen.machine_check
21:31:06 [cmd] [machine_check]      [info]  step 2, c2d: passed
21:31:07 [cmd] [campaign]           [!]     2 ladders left running by an earlier campaign marked aborted; their airfoils run again
21:34:20 [run] [campaign]           [info]  done: exit 0, 3.2 min
21:34:30 [run] [summary]            [info]  finished in 4.3 min
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
2. `python -m scripts.datagen.geometry_ingest <folder> --data-root shared` (or `data_root: shared` in a config).
3. The parity reference must list the new airfoils. A master run whose machine check passed earlier in the
   same run rewrites it by itself; otherwise run `python -m scripts.datagen.machine_check --rewrite` on the
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
