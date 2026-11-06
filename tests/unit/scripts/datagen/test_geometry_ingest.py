"""
Tests for `scripts.datagen.geometry_ingest`, the ingest's command line, i.e.:

Step 1: Ensure the options reach the ingest and its summary is printed
    - test_the_options_reach_the_ingest
Step 2: Ensure a mistake in what was asked stops it before the ingest runs
    - test_a_mistake_is_a_usage_error

The ingest itself is a stand-in; it is tested in the geometry tiers.
"""

from types import SimpleNamespace

import pytest

from scripts.datagen import geometry_ingest


@pytest.fixture
def ingest(monkeypatch):
    calls = []
    summary = SimpleNamespace(summary_path="raw/datagen/geometry/report/x/summary.txt")
    monkeypatch.setattr(geometry_ingest, "Geom_ingest_folder", lambda *args: calls.append(args) or summary)
    monkeypatch.setattr(geometry_ingest, "Geom_format_summary", lambda s: "the summary")
    return calls


# region Step 1
def test_the_options_reach_the_ingest(ingest, tmp_path, capsys):
    data = tmp_path / "data"

    assert geometry_ingest.main([str(tmp_path), "--data-root", str(data), "--dry-run"]) == 0
    assert geometry_ingest.main(["--data-root", str(data)]) == 0  # no folder: the stored airfoils only
    assert ingest == [(tmp_path, data, True), (None, data, False)]
    printed = capsys.readouterr().out
    assert "the summary" in printed and f"Summary written to {data / 'raw'}" in printed
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [(["nope", "--data-root", "d"], "nope is not a folder"),
                                        ([], "required: --data-root")], ids=["no folder", "no data root"])
def test_a_mistake_is_a_usage_error(ingest, capsys, args, says):
    with pytest.raises(SystemExit) as stopped:
        geometry_ingest.main(args)

    assert stopped.value.code == 2 and says in capsys.readouterr().err and ingest == []
# endregion
