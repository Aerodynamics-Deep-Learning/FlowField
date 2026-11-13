"""
Tests for `active_learning.classifier`: p_mesh as a GP classifier on the 18 free values, i.e.:

Step 1: Ensure a fit on rows that can't make a model is refused, and a thin or edge-bound one warned of
    - test_one_label_refused
    - test_too_few_of_a_label_refused
    - test_other_settings_or_mixed_rules_refused
    - test_few_of_a_label_warned
    - test_hyperparameter_at_its_bound_warned
Step 2: Ensure the model keeps what it was fitted under, and refuses when the rules have moved on
    - test_fit_records_production_and_rules_version
    - test_stale_rules_version_refused
    - test_predict_before_fit_refused
Step 3: Ensure predictions: None for a rule-rejected design, p from the data, f_std growing away from it
    - test_rule_rejected_design_is_none
    - test_p_follows_the_boundary
    - test_f_std_grows_away_from_the_data
    - test_predict_many_matches_predict
Step 4: Ensure f_std is the GP's latent uncertainty, checked against an independent Laplace reference
    - test_sklearn_has_latent_mean_and_variance
    - test_latent_matches_an_independent_laplace
"""

import logging

import numpy as np
import pytest
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessClassifier

from src.datagen.active_learning.classifier import MeshClassifier, Prediction
from src.datagen.active_learning.design_vector import AL_free_to_cst
from src.datagen.active_learning.rules import RULES_VERSION
from tests.active_learning_rows import (
    BASE_X, KNOBS, PRODUCTION, design_at, production, synthetic_rows,
)
from tests.geometry_shapes import CUSP
from tests.unit.datagen.active_learning.test_al_rules import DROOPED, _cusp_with

ROWS = synthetic_rows(80, seed=0, flip=0.05)


@pytest.fixture(scope="module")
def fitted() -> MeshClassifier:
    return MeshClassifier(seed=0).fit(ROWS, PRODUCTION)


def _balanced(n_zero: int, n_one: int, rows=ROWS) -> list:
    return [r for r in rows if r.label == 0][:n_zero] + [r for r in rows if r.label == 1][:n_one]


# region Step 1
def test_one_label_refused():
    with pytest.raises(ValueError, match="only label 1"):
        MeshClassifier().fit(_balanced(0, 20), PRODUCTION)


def test_too_few_of_a_label_refused():
    with pytest.raises(ValueError, match="label 0 has 4 rows, need at least 5"):
        MeshClassifier().fit(_balanced(4, 20), PRODUCTION)


def test_other_settings_or_mixed_rules_refused():
    with pytest.raises(ValueError, match="settings_id"):
        MeshClassifier().fit(ROWS, production(te_gap=0.003))

    old = synthetic_rows(10, seed=5, rules_version="rules-r0", prefix="h")
    with pytest.raises(ValueError, match="one rules_version"):
        MeshClassifier().fit(ROWS + old, PRODUCTION)


def test_few_of_a_label_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="src.datagen.active_learning.classifier"):
        MeshClassifier().fit(_balanced(10, 20), PRODUCTION)
    assert "label 0 has 10 rows" in caplog.text and "label 1" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.datagen.active_learning.classifier"):
        MeshClassifier().fit(_balanced(20, 20), PRODUCTION)
    assert "rows" not in caplog.text


def test_hyperparameter_at_its_bound_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="src.datagen.active_learning.classifier"):
        MeshClassifier(length_scale_bounds=(50.0, 51.0)).fit(ROWS, PRODUCTION)  # far longer than the data wants

    assert "length_scale" in caplog.text and "lower bound 50" in caplog.text
# endregion


# region Step 2
def test_fit_records_production_and_rules_version(fitted):
    assert fitted.production == PRODUCTION and fitted.production.knobs == KNOBS
    assert fitted.rules_version == RULES_VERSION and fitted.n_train == len(ROWS)


def test_stale_rules_version_refused():
    clf = MeshClassifier().fit(ROWS, PRODUCTION)
    clf.rules_version = "rules-r0"  # fitted under rules since replaced

    with pytest.raises(ValueError, match="rules-r0.*current rules are rules-r1"):
        clf.predict(CUSP)


def test_predict_before_fit_refused():
    with pytest.raises(ValueError, match="not fitted"):
        MeshClassifier().predict(CUSP)
# endregion


# region Step 3
def test_rule_rejected_design_is_none(fitted):
    assert fitted.predict(_cusp_with(upper={4: -0.8})) is None  # the shape rules: crossed inside
    assert fitted.predict(DROOPED) is None                      # the mesh rules: its round TE cap
    assert isinstance(fitted.predict(CUSP), Prediction)


def test_p_follows_the_boundary(fitted):
    deep_yes, deep_no = (fitted.predict(AL_free_to_cst(design_at(s))) for s in (4.0, -4.0))

    assert deep_yes.p > 0.8 and deep_no.p < 0.2
    assert 0.0 < deep_no.p < deep_yes.p < 1.0


def test_f_std_grows_away_from_the_data(fitted):
    near = fitted.predict(AL_free_to_cst(np.asarray(ROWS[0].x)))
    away = BASE_X.copy()
    away[[5, 6, 13]] += 0.15  # 7.5 spreads off, in values the boundary ignores
    far = fitted.predict(AL_free_to_cst(away))

    assert far.f_std > 1.5 * near.f_std
    assert abs(far.p - 0.5) < abs(near.p - 0.5)  # the prior's 0.5, where nothing was seen


def test_predict_many_matches_predict(fitted):
    csts = [CUSP, DROOPED] + [AL_free_to_cst(design_at(s)) for s in (-2.0, 0.5, 3.0)]
    many = fitted.predict_many(csts)

    assert [m is None for m in many] == [False, True, False, False, False]
    for m, one in zip(many, map(fitted.predict, csts)):
        assert (m is None and one is None) or np.allclose(m, one, rtol=0, atol=1e-12)
# endregion


# region Step 4
def test_sklearn_has_latent_mean_and_variance():
    assert hasattr(GaussianProcessClassifier, "latent_mean_and_variance"), (
        "scikit-learn no longer has GaussianProcessClassifier.latent_mean_and_variance (public since 1.7); "
        "MeshClassifier's f_std reads it: pin scikit-learn or compute the Laplace variance in classifier.py")


def _laplace(K: np.ndarray, y: np.ndarray, k_star: np.ndarray, k_star_star: np.ndarray):
    """GPML Algorithms 3.1 and 3.2 for the logistic likelihood: latent mean and variance at the queries."""
    f = np.zeros(len(y))
    for _ in range(200):
        pi = expit(f)
        sw = np.sqrt(pi * (1 - pi))
        L = np.linalg.cholesky(np.eye(len(y)) + sw[:, None] * K * sw[None, :])
        b = pi * (1 - pi) * f + (y - pi)
        a = b - sw * np.linalg.solve(L.T, np.linalg.solve(L, sw * (K @ b)))
        f_new = K @ a
        if np.max(np.abs(f_new - f)) < 1e-13:
            f = f_new
            break
        f = f_new
    pi = expit(f)
    sw = np.sqrt(pi * (1 - pi))
    L = np.linalg.cholesky(np.eye(len(y)) + sw[:, None] * K * sw[None, :])
    v = np.linalg.solve(L, sw[:, None] * k_star)
    return k_star.T @ (y - pi), k_star_star - np.sum(v * v, axis=0)


def test_latent_matches_an_independent_laplace():
    rows = _balanced(6, 6)
    clf = MeshClassifier(optimize=False).fit(rows, PRODUCTION)  # the kernel as given: nothing fitted
    X = (np.stack([r.features() for r in rows]) - clf.x_mean_) / clf.x_std_
    queries = [design_at(s) for s in (-3.0, 0.0, 2.0)]
    Q = (np.stack(queries) - clf.x_mean_) / clf.x_std_
    k = clf.kernel_
    mean, var = _laplace(k(X), np.array([r.label for r in rows], float), k(X, Q), np.diag(k(Q)))

    got = clf.predict_features(np.stack(queries))
    np.testing.assert_allclose([g.f_mean for g in got], mean, rtol=0, atol=1e-6)
    np.testing.assert_allclose([g.f_std for g in got], np.sqrt(var), rtol=0, atol=1e-6)
    # p is the logistic averaged over the latent's normal: Gauss-Hermite stands in for sklearn's erf mix
    z, w = np.polynomial.hermite_e.hermegauss(80)
    ref_p = [np.sum(w * expit(m + s * z)) / np.sum(w) for m, s in zip(mean, np.sqrt(var))]
    np.testing.assert_allclose([g.p for g in got], ref_p, rtol=0, atol=2e-3)
# endregion
