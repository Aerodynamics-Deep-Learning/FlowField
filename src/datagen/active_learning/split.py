"""
Held-out rows for a feasibility classifier, split by shape: every row of one geometry_id lands on one side,
so a p_conv shape seen at one condition can't be scored at another. Works on any row with `geometry_id`,
`settings_id`, `rules_version` and `label` (a `MeshRow`, later a p_conv row).

Groups are drawn per label, so the held-out side gets failures however rare: a group's label is its rows'
majority (exact for p_mesh, one row per shape), a tie counted as 0, the rarer outcome.
"""

from typing import NamedTuple, Sequence

import numpy as np


class RowSplit(NamedTuple):
    train: list
    test: list


def AL_check_one_setting(rows: Sequence) -> None:
    """
    Refuses rows that can't be learnt or scored together.

    Raises:
        ValueError: If there are no rows, or they span several settings_ids or rules versions
    """
    if not rows:
        raise ValueError("no rows")
    for field in ("settings_id", "rules_version"):
        found = sorted({getattr(r, field) for r in rows})
        if len(found) > 1:
            raise ValueError(f"rows must share one {field}, got {', '.join(found)}")


def AL_label_groups(rows: Sequence) -> dict[int, list[str]]:
    """The geometry_ids under each label, by their rows' majority label, each list sorted."""
    labels: dict[str, list[int]] = {}
    for r in rows:
        labels.setdefault(r.geometry_id, []).append(r.label)
    groups: dict[int, list[str]] = {0: [], 1: []}
    for geometry_id, found in sorted(labels.items()):
        groups[int(2 * sum(found) > len(found))].append(geometry_id)
    return groups


def AL_split_rows(rows: Sequence, test_fraction: float = 0.2, seed: int = 0) -> RowSplit:
    """
    Splits rows into training and held-out sides by geometry_id, each label's shapes in proportion.

    Args:
        rows (Sequence): The rows, one settings_id and rules version
        test_fraction (float): Share of each label's shapes held out, rounded
        seed (int): The draw; the same rows and seed give the same split

    Returns:
        RowSplit: The rows on each side, in their given order

    Raises:
        ValueError: For mixed or no rows, or a label whose shapes can't fill both sides
    """
    rows = list(rows)
    AL_check_one_setting(rows)
    rng = np.random.default_rng(seed)
    held: set[str] = set()
    for label, ids in AL_label_groups(rows).items():
        if not ids:
            raise ValueError(f"no rows labelled {label}: a split needs both labels")
        n_test = round(test_fraction * len(ids))
        if not 0 < n_test < len(ids):
            raise ValueError(f"label {label} has {len(ids)} shapes: {test_fraction:.0%} of them leaves a side "
                             f"without that label")
        held.update(rng.permutation(ids)[:n_test].tolist())
    return RowSplit(train=[r for r in rows if r.geometry_id not in held],
                    test=[r for r in rows if r.geometry_id in held])
