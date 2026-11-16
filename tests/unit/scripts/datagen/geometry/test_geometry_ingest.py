"""
Tests for `scripts.datagen.geometry.ingest`, the ingest's command line, i.e.:

Step 1: Ensure the options and the CST settings reach the ingest, its data folder the run folder's when left out, its
digest printed and its suspect points warned
    - test_the_options_reach_the_ingest
Step 2: Ensure a mistake in what was asked stops it before the ingest runs, and the shared manifest takes only
constants.py's CST values
    - test_a_mistake_is_a_usage_error
    - test_the_shared_manifest_takes_only_the_constants_cst

The ingest itself is a stand-in; it is tested in the geometry tiers.
"""

import re
from types import SimpleNamespace

import pytest

from scripts.datagen.geometry import ingest as geometry_ingest
from src.datagen.constants import CST_FIT_TOL, CST_NOSE, CST_ORDER

_CONSTANTS = {"order": CST_ORDER, "nose": CST_NOSE, "fit_tol": CST_FIT_TOL}


@pytest.fixture
def ingest(monkeypatch):
    calls = []
    summary = SimpleNamespace(summary_path="raw/x/summary.txt")
    monkeypatch.setattr(geometry_ingest, "Geom_ingest_folder", lambda *args, **kw: calls.append((*args, kw)) or summary)
    monkeypatch.setattr(geometry_ingest, "Geom_format_digest", lambda s: "the digest")
    monkeypatch.setattr(geometry_ingest, "Geom_suspect_points", lambda s: ["a suspect point"])
    return calls


# region Step 1
def test_the_options_reach_the_ingest(ingest, tmp_path, data_root, capsys):
    here, data = tmp_path.resolve(), tmp_path.resolve() / "data_out"

    assert geometry_ingest.main([str(tmp_path), "--data-root", str(data), "--dry-run"]) == 0
    assert geometry_ingest.main(["--data-root", str(data), "--set", "cst_order=10", "--set", "cst_fit_tol=2e-3"]) == 0
    assert geometry_ingest.main([str(tmp_path)]) == 0
    left_out = ingest[2][1]
    assert left_out.name == "geometry_ingest" and left_out.parent.parent == data_root / "scratchpad"  # its run folder
    assert ingest == [(here, data, True, _CONSTANTS),
                      (None, data, False, {**_CONSTANTS, "order": 10, "fit_tol": 2e-3}),  # the stored airfoils only
                      (here, left_out, False, _CONSTANTS)]
    printed = capsys.readouterr().out
    assert "the digest" in printed and f"Every file's line: {data / 'raw'}" in printed
    assert re.search(r"\[cmd\] \[geometry_ingest\] +\[!\] +a suspect point", printed)  # a warning
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["nope"], "folder: no .dat folder at"),
    ([".", "--set", "cst_nose=pointy"], "cst_nose is one of separate, shared, lem, not 'pointy'"),
    ([".", "--set", "cst_order=0"], "cst_order is 1 or more, not 0"),
], ids=["no folder", "no such nose", "no such order"])
def test_a_mistake_is_a_usage_error(ingest, tmp_path, capsys, args, says):
    assert geometry_ingest.main([str(tmp_path / a) if a in (".", "nope") else a for a in args]) == 2

    assert says in capsys.readouterr().out and ingest == []


def test_the_shared_manifest_takes_only_the_constants_cst(ingest, tmp_path, data_root, capsys):
    assert geometry_ingest.main(["--data-root", "shared", "--set", "cst_order=10", "--set", "cst_nose=lem"]) == 2
    assert ("cst_order 10: the shared manifest is made with constants.py's CST values" in capsys.readouterr().out
            and ingest == [])  # constants.py's own nose given is no change

    assert geometry_ingest.main(["--data-root", "shared", "--dry-run", "--set", "cst_order=10"]) == 0  # writes nothing
    assert ingest[0][3]["order"] == 10
# endregion
