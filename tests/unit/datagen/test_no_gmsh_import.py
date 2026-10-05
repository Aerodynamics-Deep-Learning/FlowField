"""
Guards the tool boundary: gmsh is driven only as a separate executable, through the `.geo` scripts
`meshing/gmsh/geo.py` writes, so nothing in the repo may import its Python SDK, i.e.:

Step 1: Ensure no module under src/ or tests/ imports gmsh, by statement or by importer call
    - test_nothing_imports_the_gmsh_sdk

Relative imports are skipped, since `from .gmsh import ...` names FlowField's own meshing package.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_IMPORTER_CALLS = {"import_module", "__import__", "importorskip"}


def _imported_names(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        return [node.module or ""]
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
        func = getattr(node.func, "attr", getattr(node.func, "id", None))
        if func in _IMPORTER_CALLS and isinstance(node.args[0].value, str):
            return [node.args[0].value]
    return []


def _gmsh_imports(path: Path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    return [f"{path.relative_to(REPO)}:{node.lineno}" for node in ast.walk(tree)
            if any(name.split(".")[0] == "gmsh" for name in _imported_names(node))]


# region Step 1
def test_nothing_imports_the_gmsh_sdk():
    files = [p for top in ("src", "tests") for p in (REPO / top).rglob("*.py")]
    assert files, "scanned nothing"
    offenders = [hit for p in files for hit in _gmsh_imports(p)]
    assert offenders == [], f"gmsh SDK imported, drive gmsh through GeoScript instead: {offenders}"
# endregion
