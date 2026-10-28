"""
Tests for `meshing.optimizer.settings` and the constants it records, i.e.:

Step 1: Ensure the row holds every column under a stable id
    - test_row_holds_every_column_with_the_id_last
    - test_settings_id_is_the_hash_of_the_rest
    - test_a_rebuilt_exe_makes_new_settings
    - test_missing_exe_raises
Step 2: Ensure only the tunable columns can be overridden, each one moving the id
    - test_each_overridable_column_moves_the_id
    - test_overrides_cannot_hash_one_setting_two_ways
    - test_other_columns_cannot_be_overridden
Step 3: Ensure the constants agree with the config and schemas they describe
    - test_pinned_and_searched_cover_the_config_once
    - test_pinned_fields_keep_their_config_defaults
    - test_constants_match_what_they_claim
    - test_sampling_keeps_its_names
    - test_config_schema_fp_is_stable

No c2d needed: `find_tool` is pointed at a stand-in file.
"""

import hashlib
import json

import pytest

from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig, C2D_Topology
from src.datagen.meshing.common import constants as common
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer import constants as C
from src.datagen.meshing.optimizer import sampling
from src.datagen.meshing.optimizer.settings import (
    OVERRIDABLE, build_settings, config_schema_fp, mesher_sha256, pinned,
)
from src.datagen.meshing.optimizer.space import KNOBS
from src.datagen.schemas import Freestream

_COLUMNS = [
    "ladder_version", "backend", "topology", "design_re", "shape_version", "sampler_version",
    "mesher_sha256", "dx_max", "theta_max_deg", "precheck_margin", "te_gap", "cell_ceiling",
    "gate_skew_max", "gate_ortho_min", "gate_jac_min", "gate_ar_max", "gate_size_ratio_max",
    "gate_geo_dev_max", "golden_jmax", "jmax_bump_step", "jmax_bump_max", "rung2_budget", "rung3_budget",
    "max_bb_eval_factor", "nomad_seed", "nstar_json", "search_space_json", "pinned_json", "settings_id",
]


@pytest.fixture
def exe(tmp_path, monkeypatch):
    """A stand-in c2d executable that `find_tool` resolves."""
    path = tmp_path / "c2d.exe"
    path.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool",
                        lambda key: str(path) if key == "c2d" else None)
    return path


# region Step 1
def test_row_holds_every_column_with_the_id_last(exe):
    row = build_settings()

    assert list(row) == _COLUMNS  # created_at and label belong to the store, not the hash
    assert row["mesher_sha256"] == hashlib.sha256(b"not really c2d").hexdigest()
    assert json.loads(row["pinned_json"]) == pinned()


def test_settings_id_is_the_hash_of_the_rest(exe):
    row = build_settings()
    rest = {k: v for k, v in row.items() if k != "settings_id"}

    assert row["settings_id"] == hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()[:16]
    assert build_settings() == row


def test_a_rebuilt_exe_makes_new_settings(exe):
    before = build_settings()["settings_id"]
    exe.write_bytes(b"rebuilt c2d")

    assert build_settings()["settings_id"] != before


def test_missing_exe_raises(monkeypatch):
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: None)

    with pytest.raises(FileNotFoundError, match="tools.toml"):
        mesher_sha256()
# endregion


# region Step 2
@pytest.mark.parametrize("column", OVERRIDABLE)
def test_each_overridable_column_moves_the_id(exe, column):
    default = build_settings()
    new = default[column] + 1 if isinstance(default[column], int) else default[column] * 0.9
    row = build_settings(**{column: new})

    assert row[column] == new
    assert row["settings_id"] != default["settings_id"]


def test_overrides_cannot_hash_one_setting_two_ways(exe):
    # An int for a float column is the same setting; a float or bool for an int column is refused
    assert build_settings(gate_skew_max=1)["settings_id"] == build_settings(gate_skew_max=1.0)["settings_id"]
    with pytest.raises(TypeError, match="golden_jmax takes an int"):
        build_settings(golden_jmax=150.0)
    with pytest.raises(TypeError):
        build_settings(nomad_seed=True)


@pytest.mark.parametrize("column", ["dx_max", "design_re", "mesher_sha256", "search_space_json",
                                    "settings_id", "gate_skew"])
def test_other_columns_cannot_be_overridden(exe, column):
    # They describe the code or the exe, so an override would record settings the run did not use
    with pytest.raises(ValueError, match=f"cannot override {column}"):
        build_settings(**{column: 1})
# endregion


# region Step 3
def test_pinned_and_searched_cover_the_config_once():
    searched = {k.name for k in KNOBS}

    assert not searched & set(pinned())
    assert searched | set(pinned()) == set(C2D_MeshingConfig.model_fields)
    assert len(searched) + len(pinned()) == len(C2D_MeshingConfig.model_fields) == 23


def test_pinned_fields_keep_their_config_defaults():
    # Outside PINNED, a field is one c2d does not read on OGRD + HYPR, recorded at its default
    defaults = C2D_MeshingConfig()
    unread = {name: value for name, value in pinned().items() if name not in C.PINNED}

    assert set(unread) == {"nwke", "fdst", "fwkl", "fwki", "stp1", "stp2", "nrmt", "nrmb"}
    assert unread == {name: getattr(defaults, name) for name in unread}
    C2D_MeshingConfig(**C.PINNED, **C.SMOOTHING_DEFAULTS)  # the pinned values validate


def test_constants_match_what_they_claim():
    defaults = C2D_MeshingConfig()
    re_top = next(m.le for m in Freestream.model_fields["Re"].metadata if hasattr(m, "le"))

    assert C.SMOOTHING_DEFAULTS == {name: getattr(defaults, name) for name in C.SMOOTHING_DEFAULTS}
    assert {k: v for k, v in C.PINNED.items() if k != "topo"} == {
        k: getattr(defaults, k) for k in C.PINNED if k != "topo"}  # topo defaults to None, filled per run
    assert C.PINNED["topo"] == C2D_Topology.OGRD.value
    assert (C.BACKEND, C.TOPOLOGY) == (MeshBackend.C2D.value, MeshTopology.OGRD.value)
    assert C.DESIGN_RE == re_top
    assert (C.GATE_AR_MAX, C.GATE_SIZE_RATIO_MAX, C.GATE_GEO_DEV_MAX) == (
        common.ASPECT_RATIO_LIMIT, common.SIZE_RATIO_LIMIT, common.GEO_DEV_LIMIT)
    assert C.JMAX_MIN <= C.GOLDEN_JMAX <= C.JMAX_MAX


def test_sampling_keeps_its_names():
    assert (sampling.TE_GAP_DEFAULT, sampling.CELL_CEILING) == (C.TE_GAP_DEFAULT, C.CELL_CEILING)


def test_config_schema_fp_is_stable():
    fp = config_schema_fp()

    assert fp == config_schema_fp()
    assert len(fp) == 16 and int(fp, 16) >= 0
# endregion
