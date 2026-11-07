"""
The machine check behind `machine_check.py`. It runs the parity test,
tests/integration/datagen/mesh/optimizer/test_machine_parity.py, and passes only when every test in it passed: a
skip counts as a failure. A pass is stamped in data/.machine_check.json with what was checked (the reference, the
c2d executable, the sampler and shape versions, the computer), so a mesh script can say when something has moved
since. It also rewrites the parity reference, keeping the old file when nothing but the writer's name would change.
"""

import datetime
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple, Optional

from src.datagen.geometry.common.constants import SAMPLER_VERSION, SHAPE_VERSION
from src.datagen.geometry.common.paths import DATA_ROOT, REPO_ROOT

PARITY_MODULE = "tests.integration.datagen.mesh.optimizer.test_machine_parity"
PARITY_TEST = REPO_ROOT / "tests" / "integration" / "datagen" / "mesh" / "optimizer" / "test_machine_parity.py"
REFERENCE = PARITY_TEST.with_name("machine_parity.json")
STAMP = DATA_ROOT / ".machine_check.json"
REWRITE = "python -m scripts.datagen.meshing.machine_check --rewrite"

_STEPS = {"test_this_machine_samples_like_the_reference": "step 1, sampling",
          "test_this_machine_meshes_like_the_reference": "step 2, c2d"}
_OUTCOMES = {"skipped": "skipped", "failure": "failed", "error": "error"}


class CheckResult(NamedTuple):
    passed: bool
    steps: list[tuple[str, str, str]]  # (step, passed | skipped | failed | error, why)


def Script_machine_check(test_path=PARITY_TEST) -> CheckResult:
    """
    Runs the parity test in a process of its own.

    Args:
        test_path (str | Path): The test module; another only in tests

    Returns:
        CheckResult: Passed only when it ran tests and every one passed
    """
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "parity.xml"
        proc = subprocess.run([sys.executable, "-m", "pytest", str(test_path), "-n", "0", "-q", "-p", "no:cacheprovider",
                               f"--junitxml={report}", "-o", "junit_family=xunit2"], cwd=REPO_ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not report.exists():
            return CheckResult(False, [("pytest", "error", (proc.stdout + proc.stderr).strip()[-2000:])])
        cases = list(ET.parse(report).getroot().iter("testcase"))
    steps = []
    for case in cases:
        outcome = next((c for c in case if c.tag in _OUTCOMES), None)
        why = "" if outcome is None else (outcome.get("message") or outcome.text or "").strip()
        steps.append((_STEPS.get(case.get("name"), case.get("name")),
                      "passed" if outcome is None else _OUTCOMES[outcome.tag], why))
    return CheckResult(bool(steps) and all(o == "passed" for _, o, _ in steps), steps)


def Script_reference_written_on() -> Optional[str]:
    try:
        return json.loads(REFERENCE.read_text(encoding="utf-8")).get("written_on")
    except (OSError, ValueError):
        return None


def Script_reference_problem() -> Optional[str]:
    """
    Why the parity reference cannot check this code and manifest, before anything runs.

    Returns:
        str | None: None when it is for this sampler and shape version and lists the shared manifest's airfoils
    """
    try:
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"no parity reference at {REFERENCE}: write it ({REWRITE})"
    if (ref["sampler_version"], ref["shape_version"]) != (SAMPLER_VERSION, SHAPE_VERSION):
        return (f"the parity reference is for {ref['sampler_version']}/{ref['shape_version']}, the code is at "
                f"{SAMPLER_VERSION}/{SHAPE_VERSION}. This is a release step, done by hand on the reference machine: "
                f"1. python -m scripts.datagen.geometry.ingest --data-root shared  2. {REWRITE}  3. commit "
                f"geometry.sqlite and machine_parity.json together")
    from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes  # the shared manifest's names
    stored, listed = {name for name, _ in Optim_load_shapes()}, set(ref["samples"])
    if stored != listed:
        return (f"the parity reference lists other airfoils than the shared manifest ({len(stored - listed)} not in "
                f"it, {len(listed - stored)} gone): rewrite it on the reference machine ({REWRITE}) and commit it "
                f"with the manifest")
    return None


def Script_rewrite_reference() -> str:
    """
    Rewrites the parity reference on this machine; the old file stays when only the writer's name would change.

    Returns:
        str: What happened, for the log

    Raises:
        RuntimeError: If the writer fails
    """
    old_text = REFERENCE.read_text(encoding="utf-8") if REFERENCE.exists() else None
    proc = subprocess.run([sys.executable, "-m", PARITY_MODULE], cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"the reference writer failed (exit {proc.returncode}):\n{proc.stderr.strip()}")
    new = json.loads(REFERENCE.read_text(encoding="utf-8"))
    old = json.loads(old_text) if old_text else {}
    if {k: v for k, v in old.items() if k != "written_on"} == {k: v for k, v in new.items() if k != "written_on"}:
        REFERENCE.write_text(old_text, encoding="utf-8", newline="")
        return "machine_parity.json unchanged: this machine writes it exactly as it is"
    added = len(set(new["samples"]) - set(old.get("samples", {})))
    gone = len(set(old.get("samples", {})) - set(new["samples"]))
    return (f"machine_parity.json rewritten on {new['written_on']}: {len(new['samples'])} airfoils ({added} new, "
            f"{gone} gone), {len(new.get('c2d', {}).get('meshes', {}))} meshes")


#region The stamp

def Script_write_stamp() -> None:
    """Stamps a passed check: what was checked, and when."""
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    state = {**_state(), "passed_at": datetime.datetime.now().isoformat(timespec="seconds")}
    STAMP.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def Script_stamp_problem() -> Optional[str]:
    """
    Why this machine's last passed check no longer vouches for it.

    Returns:
        str | None: The warning; None while nothing it checked has moved
    """
    advice = ": meshes made here may not hash like the stored runs (python -m scripts.datagen.meshing.machine_check, ~1 min)"
    try:
        stamp = json.loads(STAMP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "this machine has not passed machine_check" + advice
    now = _state()
    if now["mesher_sha256"] is None:
        return "no c2d executable is found here (tools.toml)" + advice
    if stamp.get("host") != now["host"]:
        reason = f"it passed on {stamp.get('host')}, not on this computer"
    elif stamp.get("mesher_sha256") != now["mesher_sha256"]:
        reason = f"its c2d changed ({str(stamp.get('mesher_sha256'))[:8]}... -> {now['mesher_sha256'][:8]}...)"
    elif stamp.get("reference_sha256") != now["reference_sha256"]:
        reason = "the parity reference changed"
    elif (stamp.get("sampler_version"), stamp.get("shape_version")) != (SAMPLER_VERSION, SHAPE_VERSION):
        reason = (f"the sampler or shape version changed ({stamp.get('sampler_version')}/{stamp.get('shape_version')} "
                  f"-> {SAMPLER_VERSION}/{SHAPE_VERSION})")
    else:
        return None
    return f"this machine has not passed machine_check since {reason}" + advice


def _state() -> dict:
    from src.datagen.meshing.optimizer.common.settings import Optim_mesher_sha256  # hashes the c2d in use
    try:
        mesher = Optim_mesher_sha256()
    except FileNotFoundError:
        mesher = None
    return {"host": socket.gethostname(), "mesher_sha256": mesher, "reference_sha256": _reference_digest(),
            "sampler_version": SAMPLER_VERSION, "shape_version": SHAPE_VERSION}


def _reference_digest() -> Optional[str]:
    """The reference's content, so line endings a checkout changes do not count."""
    try:
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return hashlib.sha256(json.dumps(ref, sort_keys=True).encode()).hexdigest()

#endregion
