"""
Tests for `meshing.optimizer.common.settings` and the constants it records, i.e.:

Step 1: Ensure the row holds every column under a stable id
    - test_row_holds_every_column_with_the_id_last
    - test_settings_id_is_the_hash_of_the_rest
    - test_a_rebuilt_exe_makes_new_settings
    - test_missing_exe_raises
Step 2: Ensure only the tunable columns can be overridden, each one moving the id
    - test_each_overridable_column_moves_the_id
    - test_a_setting_that_breaks_the_search_is_refused
    - test_the_settings_a_run_sets_are_the_constants_overridable
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

from src.datagen import constants as D
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig, C2D_Topology
from src.datagen.meshing.common import constants as common
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.common import constants as C
from src.datagen.meshing.optimizer.common.settings import (
    OVERRIDABLE, STUDY_KEYS, Optim_build_settings, Optim_config_schema_fp, Optim_mesher_sha256, Optim_pinned,
    Optim_shape_version,
)
from src.datagen.meshing.optimizer.common.space import KNOBS
from src.datagen.meshing.optimizer.evaluate import sampling
from src.datagen.schemas import Freestream

_COLUMNS = [
    "ladder_version", "backend", "topology", "design_re", "shape_version", "sampler_version",
    "mesher_sha256", "dx_max", "theta_max_deg", "precheck_margin", "te_gap", "cell_ceiling",
    "gate_skew_max", "gate_ortho_min", "gate_jac_min", "gate_ar_max", "gate_size_ratio_max",
    "gate_geo_dev_max", "golden_jmax", "jmax_bump_step", "jmax_bump_max", "rung2_budget", "rung3_budget",
    "rung3_patience", "rung3_min_gain", "max_bb_eval_factor", "nomad_seed", "nstar_json", "search_space_json",
    "pinned_json", "smoothing_json", "settings_id",
]


@pytest.fixture
def exe(tmp_path, monkeypatch):
    """A stand-in c2d executable that `find_tool` resolves."""
    path = tmp_path / "c2d.exe"
    path.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool",
                        lambda key: str(path) if key == "c2d" else None)
    return path


# region Step 1
def test_row_holds_every_column_with_the_id_last(exe):
    row = Optim_build_settings()

    assert list(row) == _COLUMNS  # created_at and label belong to the store, not the hash
    assert row["mesher_sha256"] == hashlib.sha256(b"not really c2d").hexdigest()
    assert json.loads(row["pinned_json"]) == Optim_pinned()


def test_settings_id_is_the_hash_of_the_rest(exe):
    row = Optim_build_settings()
    rest = {k: v for k, v in row.items() if k != "settings_id"}

    assert row["settings_id"] == hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()[:16]
    assert Optim_build_settings() == row


def test_a_rebuilt_exe_makes_new_settings(exe):
    before = Optim_build_settings()["settings_id"]
    exe.write_bytes(b"rebuilt c2d")

    assert Optim_build_settings()["settings_id"] != before


def test_missing_exe_raises(monkeypatch):
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: None)

    with pytest.raises(FileNotFoundError, match="tools.toml"):
        Optim_mesher_sha256()
# endregion


# region Step 2
def _read(row, key):
    """Where a setting lands in the row: its own column, or the JSON one it folds into."""
    grid, pinned = json.loads(row["nstar_json"]), json.loads(row["pinned_json"])
    knobs = {k["name"]: k for k in json.loads(row["search_space_json"])}
    folded = {"nstar_le_grid": grid["le_grid"], "nstar_cw_grid": grid["cw_grid"], "y_plus": pinned["ypls"],
              "farfield_radius": pinned["radi"], "smoothing_defaults": json.loads(row["smoothing_json"]),
              "dx_mid_min": knobs["dx_mid"]["lo"], "ratio_min": knobs["le_ratio"]["lo"],
              "cluster_width_min": knobs["cluster_width"]["lo"], "cluster_width_max": knobs["cluster_width"]["hi"],
              "jmax_min": knobs["jmax"]["lo"], "jmax_max": knobs["jmax"]["hi"]}
    return folded[key] if key in folded else row[key]


@pytest.mark.parametrize("column", OVERRIDABLE)
def test_each_overridable_column_moves_the_id(exe, column):
    default = Optim_build_settings()
    if column in ("cst_order", "cst_nose"):  # both name the shapes studied, folding into shape_version
        new = {"cst_order": D.CST_ORDER + 1, "cst_nose": "separate"}[column]
        row = Optim_build_settings(**{column: new})
        assert row["shape_version"] == Optim_shape_version({column: new}) != default["shape_version"]
        assert row["settings_id"] != default["settings_id"]
        return
    old = _read(default, column)
    if column == "smoothing_defaults":  # those given, over the rest
        new, expected = {"alfa": 0.7}, {**old, "alfa": 0.7}
    else:
        new = expected = old + 1 if isinstance(old, int) else old * 0.9
    row = Optim_build_settings(**{column: new})

    assert _read(row, column) == expected
    assert row["settings_id"] != default["settings_id"]


def test_a_setting_that_breaks_the_search_is_refused(exe):
    assert json.loads(Optim_build_settings(dx_max=0.004)["search_space_json"])[0]["hi"] == 0.004  # dx_mid's top
    with pytest.raises(ValueError, match="empty search range for jmax"):
        Optim_build_settings(jmax_min=210)
    with pytest.raises(ValueError, match=r"alfa 3 not in \[0.5, 2\]"):
        Optim_build_settings(smoothing_defaults={"alfa": 3.0})
    with pytest.raises(ValueError, match="smoothing_defaults has no radi"):
        Optim_build_settings(smoothing_defaults={"radi": 3.0})
    with pytest.raises(ValueError, match="cst_nose is one of separate, shared, lem, not 'pointy'"):
        Optim_build_settings(cst_nose="pointy")


def test_the_settings_a_run_sets_are_the_constants_overridable(exe):
    # a config names each by its constant; one missing there could never be set
    assert sorted(key for key in STUDY_KEYS if key.upper() not in vars(D)) == []


def test_overrides_cannot_hash_one_setting_two_ways(exe):
    # An int for a float column is the same setting; a float or bool for an int column is refused
    same = Optim_build_settings(gate_skew_max=1.0)["settings_id"]
    assert Optim_build_settings(gate_skew_max=1)["settings_id"] == same
    with pytest.raises(TypeError, match="golden_jmax takes an int"):
        Optim_build_settings(golden_jmax=150.0)
    with pytest.raises(TypeError):
        Optim_build_settings(nomad_seed=True)


@pytest.mark.parametrize("column", ["precheck_margin", "sampler_version", "mesher_sha256", "search_space_json",
                                    "nstar_json", "settings_id", "gate_skew"])
def test_other_columns_cannot_be_overridden(exe, column):
    # They describe the code or the exe, so an override would record settings the run did not use
    with pytest.raises(ValueError, match=f"cannot override {column}"):
        Optim_build_settings(**{column: 1})
# endregion


# region Step 3
def test_pinned_and_searched_cover_the_config_once():
    searched = {k.name for k in KNOBS}

    assert not searched & set(Optim_pinned())
    assert searched | set(Optim_pinned()) == set(C2D_MeshingConfig.model_fields)
    assert len(searched) + len(Optim_pinned()) == len(C2D_MeshingConfig.model_fields) == 23


def test_pinned_fields_keep_their_config_defaults():
    # Outside PINNED, a field is one c2d does not read on OGRD + HYPR, recorded at its default
    defaults = C2D_MeshingConfig()
    unread = {name: value for name, value in Optim_pinned().items() if name not in C.PINNED}

    assert set(unread) == {"nwke", "fdst", "fwkl", "fwki", "stp1", "stp2", "nrmt", "nrmb"}
    assert unread == {name: getattr(defaults, name) for name in unread}
    C2D_MeshingConfig(**C.PINNED, **D.SMOOTHING_DEFAULTS)  # the pinned values validate


def test_constants_match_what_they_claim():
    defaults = C2D_MeshingConfig()
    re_top = next(m.le for m in Freestream.model_fields["Re"].metadata if hasattr(m, "le"))

    assert D.SMOOTHING_DEFAULTS == {name: getattr(defaults, name) for name in D.SMOOTHING_DEFAULTS}
    assert {k: v for k, v in C.PINNED.items() if k != "topo"} == {
        k: getattr(defaults, k) for k in C.PINNED if k != "topo"}  # topo defaults to None, filled per run
    assert C.PINNED["topo"] == C2D_Topology.OGRD.value
    assert (C.BACKEND, C.TOPOLOGY) == (MeshBackend.C2D.value, MeshTopology.OGRD.value)
    assert D.DESIGN_RE == re_top
    assert (D.GATE_AR_MAX, D.GATE_SIZE_RATIO_MAX, D.GATE_GEO_DEV_MAX) == (
        common.ASPECT_RATIO_LIMIT, common.SIZE_RATIO_LIMIT, common.GEO_DEV_LIMIT)
    assert D.JMAX_MIN <= D.GOLDEN_JMAX <= D.JMAX_MAX


def test_sampling_keeps_its_names():
    assert (sampling.TE_GAP, sampling.CELL_CEILING) == (D.TE_GAP, D.CELL_CEILING)


def test_config_schema_fp_is_stable():
    fp = Optim_config_schema_fp()

    assert fp == Optim_config_schema_fp()
    assert len(fp) == 16 and int(fp, 16) >= 0
# endregion
