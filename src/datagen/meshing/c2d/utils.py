"""
c2d-local helpers that keep this package free of any `common` import.

`C2D_count_su2_nodes` is the cheap sister of `common.quality._analyze_su2`: the runner only needs a
node count to decide whether conversion produced a usable mesh, and pulling `_analyze_su2` in for
that both parsed the whole mesh (~16x slower) and closed an import cycle, since `common.quality`
imports `common.schemas`, which imports this package.
"""

import os


def C2D_count_su2_nodes(su2_path: str | None) -> int:
    """
    Reads the node count straight from a `.su2` file's `NPOIN=` header.

    Args:
        su2_path (str | None): Path to the `.su2` mesh, or None

    Returns:
        int: The declared node count, or 0 if the file is missing, unreadable, or has no `NPOIN=`
    """
    if not su2_path or not os.path.isfile(su2_path):
        return 0

    try:
        with open(su2_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):  # SU2 comment lines
                    continue
                tokens = line.split()
                if tokens[0].upper() == "NPOIN=":
                    return int(tokens[1])
    except (OSError, ValueError, IndexError):
        return 0

    return 0
