"""
Tests for `active_learning.split`: held-out rows by shape, i.e.:

Step 1: Ensure no shape is on both sides, both labels are on each, and a seed fixes the split
    - test_no_geometry_id_on_both_sides
    - test_both_labels_on_each_side
    - test_seed_fixes_the_split
Step 2: Ensure a split of rows that can't be compared, or can't hold both labels, is refused
    - test_mixed_rows_refused
    - test_too_few_of_a_label_refused
"""

import pytest

from src.datagen.active_learning.split import AL_split_rows
from tests.active_learning_rows import production, synthetic_rows

ROWS = synthetic_rows(60, seed=1)


def _ids(rows) -> list[str]:
    return [r.geometry_id for r in rows]


# region Step 1
def test_no_geometry_id_on_both_sides():
    # Two rows of one shape (as p_conv will have, one per condition) must land together
    twin = ROWS[0].model_copy(update={"attempt_id": "twin"})
    split = AL_split_rows(ROWS + [twin], test_fraction=0.25, seed=0)

    assert not set(_ids(split.train)) & set(_ids(split.test))
    assert sorted(_ids(split.train) + _ids(split.test)) == sorted(_ids(ROWS + [twin]))
    assert 0.15 < len(split.test) / (len(ROWS) + 1) < 0.35


def test_both_labels_on_each_side():
    split = AL_split_rows(ROWS, test_fraction=0.2, seed=3)

    assert {r.label for r in split.train} == {r.label for r in split.test} == {0, 1}


def test_seed_fixes_the_split():
    first, again, other = (AL_split_rows(ROWS, seed=s) for s in (0, 0, 1))

    assert _ids(first.test) == _ids(again.test)
    assert set(_ids(first.test)) != set(_ids(other.test))
# endregion


# region Step 2
def test_mixed_rows_refused():
    other_settings = synthetic_rows(10, seed=2, production_mesh=production(te_gap=0.003), prefix="h")
    with pytest.raises(ValueError, match="one settings_id"):
        AL_split_rows(ROWS + other_settings)

    other_rules = synthetic_rows(10, seed=2, rules_version="rules-r0", prefix="h")
    with pytest.raises(ValueError, match="one rules_version"):
        AL_split_rows(ROWS + other_rules)

    with pytest.raises(ValueError, match="no rows"):
        AL_split_rows([])


def test_too_few_of_a_label_refused():
    ones = [r for r in ROWS if r.label == 1]
    with pytest.raises(ValueError, match="no rows labelled 0"):
        AL_split_rows(ones)

    zeros = [r for r in ROWS if r.label == 0]
    with pytest.raises(ValueError, match="label 1 has 2 shapes"):  # 20 % of 2 rounds to 0: no 1 held out
        AL_split_rows(zeros + ones[:2], test_fraction=0.2)
# endregion
