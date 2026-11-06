import pytest

from scripts.datagen import engine, parity


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """The scripts' data/ in a temporary folder: the scratchpad, relative config paths and the stamp land there."""
    root = (tmp_path / "data").resolve()
    monkeypatch.setattr(engine, "DATA_ROOT", root)
    monkeypatch.setattr(parity, "STAMP", root / ".machine_check.json")
    return root
