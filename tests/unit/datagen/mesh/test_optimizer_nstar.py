"""
Tests for `meshing.optimizer.nstar`, the sampler-only n* search, i.e.:

Step 1: Ensure n* is the fewest points the sampler accepts on the grid
    - test_n_star_is_the_grids_fewest_accepted_points
    - test_ties_go_to_the_finest_nose_then_the_narrowest_cluster
Step 2: Ensure the grid comes from the settings and stays inside the search space
    - test_the_grid_comes_from_the_settings
    - test_n_stars_knobs_are_a_start_nomad_can_take
Step 3: Ensure what the search cannot do is reported, not hidden
    - test_no_accepted_point_gives_no_n_star
    - test_a_sampler_bug_is_raised

Step 1 samples the test shapes for real on a small grid; the rest use a stand-in sampler. The full grid
runs in the integration tier's ladder.
"""

import json
import math
from types import SimpleNamespace

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.geometry.common.schemas import Rejection
from src.datagen.geometry.sample.checks import SamplerAssertionError
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.constants import CLUSTER_WIDTH_MAX, CLUSTER_WIDTH_MIN, RATIO_MIN
from src.datagen.meshing.optimizer.nstar import n_star_search
from src.datagen.meshing.optimizer.sampling import build_study_contour
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.space import knobs_for, nomad_bounds, to_x
from tests.geometry_shapes import SHAPE_NAMES, stored_shape

SAMPLING = ("dx_mid", "le_ratio", "te_ratio", "cluster_width")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: str(exe))
    return build_settings()


def _sized(settings, le_grid, cw_grid):
    return {**settings, "nstar_json": json.dumps({"le_grid": le_grid, "cw_grid": cw_grid})}


class _Sampler:
    """A stand-in `Geom_sample`: `verdict(fields)` gives (rejection, n_required); every call kept."""

    def __init__(self, verdict):
        self.verdict, self.calls = verdict, []

    def __call__(self, contour, fields, max_points=None):
        self.calls.append((fields, max_points))
        rejection, n = self.verdict(fields)
        return SimpleNamespace(report=SimpleNamespace(rejection=rejection, n_required=n))


# region Step 1
@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_n_star_is_the_grids_fewest_accepted_points(settings, name):
    settings = _sized(settings, 8, 3)
    contour = build_study_contour(stored_shape(name), MeshBackend.C2D, MeshTopology.OGRD)
    result = n_star_search(contour, settings)

    # The grid, by hand: le_ratio log-spaced over [0.002, 1], cluster_width over its range
    les = [RATIO_MIN ** (1 - i / 7) for i in range(8)]
    cws = [CLUSTER_WIDTH_MIN + j * (CLUSTER_WIDTH_MAX - CLUSTER_WIDTH_MIN) / 2 for j in range(3)]
    accepted = []
    for le in les:
        for cw in cws:
            report = Geom_sample(contour, SamplingFields(dx_mid=DX_MAX, le_ratio=le, te_ratio=1.0, cluster_width=cw),
                                 max_points=settings["cell_ceiling"] // settings["golden_jmax"]).report
            if report.rejection is None:
                accepted.append(report.n_required)
    assert accepted and len(accepted) < 24  # the coarse noses are rejected on turn
    assert result.n == min(accepted) and result.evals == 24 and result.seconds > 0
    # Where every study shape lands: the coarsest dx, the narrowest cluster, te_ratio 1 (clamped)
    assert result.knobs["dx_mid"] == DX_MAX and result.knobs["te_ratio"] == 1.0
    assert result.knobs["cluster_width"] == CLUSTER_WIDTH_MIN
    again = Geom_sample(contour, SamplingFields(**result.knobs))
    assert again.accepted and again.report.n_points == result.n


def test_ties_go_to_the_finest_nose_then_the_narrowest_cluster(settings, monkeypatch):
    sampler = _Sampler(lambda f: (None, 400 if f.le_ratio > 0.01 and f.cluster_width > 0.1 else 450))
    monkeypatch.setattr("src.datagen.meshing.optimizer.nstar.Geom_sample", sampler)
    result = n_star_search(None, _sized(settings, 5, 4))

    tied = [f for f, _ in sampler.calls if f.le_ratio > 0.01 and f.cluster_width > 0.1]
    assert len(tied) > 1 and result.n == 400
    assert result.knobs == {k: getattr(tied[0], k) for k in SAMPLING}
    assert tied[0].le_ratio == min(f.le_ratio for f in tied)
    assert tied[0].cluster_width == min(f.cluster_width for f in tied if f.le_ratio == tied[0].le_ratio)
# endregion


# region Step 2
def test_the_grid_comes_from_the_settings(settings, monkeypatch):
    sampler = _Sampler(lambda f: (None, 400))
    monkeypatch.setattr("src.datagen.meshing.optimizer.nstar.Geom_sample", sampler)
    settings = build_settings(golden_jmax=150)
    result = n_star_search(None, settings)

    assert json.loads(settings["nstar_json"]) == {"le_grid": 40, "cw_grid": 30}
    assert result.evals == len(sampler.calls) == 1200
    les = sorted({f.le_ratio for f, _ in sampler.calls})
    cws = sorted({f.cluster_width for f, _ in sampler.calls})
    assert len(les) == 40 and len(cws) == 30
    assert (les[0], les[-1], cws[0], cws[-1]) == (RATIO_MIN, 1.0, CLUSTER_WIDTH_MIN, CLUSTER_WIDTH_MAX)
    steps = [math.log10(b / a) for a, b in zip(les, les[1:])]
    assert max(steps) - min(steps) < 1e-9  # evenly spaced in log10
    assert {(f.dx_mid, f.te_ratio) for f, _ in sampler.calls} == {(settings["dx_max"], 1.0)}
    assert {mp for _, mp in sampler.calls} == {300_000 // 150}  # the golden run's point ceiling


@pytest.mark.parametrize("corner", ["lowest", "highest"])
def test_n_stars_knobs_are_a_start_nomad_can_take(settings, monkeypatch, corner):
    # Only the grid's corner is accepted: its values sit on the space's bounds, which the log round trip
    # could step past, and a start outside them would crash NOMAD
    def verdict(f):
        if corner == "lowest":
            at = f.le_ratio < 0.0021 and f.cluster_width < 0.027
        else:
            at = f.le_ratio > 0.99 and f.cluster_width > 0.24
        return (None, 400) if at else (Rejection.TURN, 300)

    monkeypatch.setattr("src.datagen.meshing.optimizer.nstar.Geom_sample", _Sampler(verdict))
    result = n_star_search(None, settings)
    rung3 = knobs_for(3)
    lb, ub = nomad_bounds(rung3)
    x = to_x({**result.knobs, "jmax": 147, "asmt": 40, "epsi": 20.0, "funi": 0.02, "alfa": 1.0}, rung3)

    assert result.n == 400
    assert all(lo <= v <= hi for v, lo, hi in zip(x, lb, ub))
# endregion


# region Step 3
def test_no_accepted_point_gives_no_n_star(settings, monkeypatch):
    sampler = _Sampler(lambda f: (Rejection.POINTS, 9000))
    monkeypatch.setattr("src.datagen.meshing.optimizer.nstar.Geom_sample", sampler)
    result = n_star_search(None, _sized(settings, 4, 2))

    assert (result.n, result.knobs, result.evals) == (None, None, 8)


def test_a_sampler_bug_is_raised(settings, monkeypatch):
    def broken(contour, fields, max_points=None):
        raise SamplerAssertionError("stations out of order")

    monkeypatch.setattr("src.datagen.meshing.optimizer.nstar.Geom_sample", broken)
    with pytest.raises(SamplerAssertionError, match="stations out of order"):
        n_star_search(None, _sized(settings, 4, 2))
# endregion
