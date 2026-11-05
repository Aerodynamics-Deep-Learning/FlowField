import pytest
from pathlib import Path

@pytest.fixture(scope="function")
def integration_root(tmp_path: Path) -> Path:
    """
    Provisions a unique temporary directory for integration test execution
    """
    root = tmp_path / "integration root"
    root.mkdir()
    
    return root