"""
Where an interactive viewer reads what to show: `<test>.yaml` beside the test when there is one (gitignored, this
machine's own choice), else the committed `<test>.example.yaml`, which documents the format and is the default.
"""

from pathlib import Path

import yaml


def load_selection(test_file, *, example: bool = False) -> dict:
    """The viewer's selection; `example` reads the committed default whatever is beside it (the batch checks)."""
    test = Path(test_file)
    local, committed = test.with_suffix(".yaml"), test.with_suffix(".example.yaml")
    path = committed if example or not local.is_file() else local
    if not example:
        print(f"\n[viewer] selection from {path.name}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
