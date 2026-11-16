import pytest

from scripts.datagen import engine
from scripts.datagen.meshing import parity


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch, airfoils_manifest):
    """
    The scripts' data/ in a temporary folder: the scratchpad, relative config paths and the stamp land there; the
    shared geometry manifest is the tests' airfoil one.
    """
    root = (tmp_path / "data").resolve()
    monkeypatch.setattr(engine, "DATA_ROOT", root)
    monkeypatch.setattr(engine, "SHARED_GEOMETRY_DB", airfoils_manifest)
    monkeypatch.setattr(parity, "STAMP", root / ".machine_check.json")
    return root
