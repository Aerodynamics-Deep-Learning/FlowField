"""
p_mesh: a Gaussian-process classifier of whether a design that passed the rules meshes under the production
settings. predict(cst) -> (p, f_std), or None for a design the rules reject.

    Prediction      p, and the GP's latent score: its mean and standard deviation f_std, the uncertainty
                    the sampler weighs (p alone can't tell "unsure" from "half the neighbours fail")
    _GPFeasibility  the model on a feature matrix, shared with a later p_conv classifier, whose rows add
                    the conditions (alpha, Re, M) to the design's free values
    MeshClassifier  p_mesh: the rules, then the model on the 18 free values

The GP: a learned amplitude times a Matern 5/2 with one shared length scale, on inputs standardised with
the training rows' mean and spread, fitted by Laplace's approximation (scikit-learn). The latent mean and
variance come from `latent_mean_and_variance`, public since scikit-learn 1.7.

Follow-up for the sampler: f_mean and f_std are on the latent scale, which the fitted amplitude stretches (a
refit can double both without changing p much). So an optimistic score is formed there, f_mean + k * f_std,
and only then turned into a probability (the logistic, averaged as `predict_proba` averages it); adding
f_std to p, or comparing f_std across refits, mixes scales.
"""

import logging
import warnings
from typing import NamedTuple, Optional, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from src.datagen.active_learning.design_vector import AL_cst_to_free, AL_free_to_cst
from src.datagen.active_learning.labels import ProductionMesh
from src.datagen.active_learning.rules import RULES_VERSION, AL_geometry_rules, AL_mesh_rules
from src.datagen.active_learning.split import AL_check_one_setting
from src.datagen.geometry.common.schemas import CSTParams

logger = logging.getLogger(__name__)

MIN_PER_LABEL = 5      # fewer rows of a label and there's no boundary to learn: refused
WARN_PER_LABEL = 20    # fewer and the boundary is a guess: warned
AMPLITUDE_BOUNDS = (1e-2, 1e2)
LENGTH_SCALE_BOUNDS = (1e-2, 1e2)  # in standardised units, so wide
BOUND_FACTOR = 1.05    # a hyperparameter within this factor of a bound is reported as stopped there
N_RESTARTS = 5


class Prediction(NamedTuple):
    p: float       # P(label 1), the latent averaged over its uncertainty
    f_mean: float  # the latent score's posterior mean
    f_std: float   # and standard deviation


class _GPFeasibility:
    """
    The GP on a feature matrix: fit from rows (`features()`, `label`, `settings_id`, `rules_version`),
    predict from features. Subclasses turn their inputs into features and run their rules.
    """

    def __init__(self, seed: int = 0, amplitude_bounds: tuple = AMPLITUDE_BOUNDS,
                 length_scale_bounds: tuple = LENGTH_SCALE_BOUNDS, optimize: bool = True):
        self.seed, self.optimize = seed, optimize
        self.amplitude_bounds, self.length_scale_bounds = amplitude_bounds, length_scale_bounds
        self._gpc: Optional[GaussianProcessClassifier] = None
        self.production: Optional[ProductionMesh] = None
        self.rules_version: Optional[str] = None
        self.n_train = 0

    def fit(self, rows: Sequence, production: ProductionMesh) -> "_GPFeasibility":
        """
        Fits the GP to rows of one production setting and rules version.

        Args:
            rows (Sequence): The training rows
            production (ProductionMesh): The settings they were labelled under, kept for predict's rules

        Returns:
            The classifier itself

        Raises:
            ValueError: For mixed rows, rows of other settings, one label, or fewer than MIN_PER_LABEL of either
        """
        rows = list(rows)
        AL_check_one_setting(rows)
        if rows[0].settings_id != production.settings_id:
            raise ValueError(f"rows are labelled under settings_id {rows[0].settings_id}, "
                             f"not the production {production.settings_id}")
        y = np.array([r.label for r in rows], dtype=int)
        counts = {label: int((y == label).sum()) for label in (0, 1)}
        if 0 in counts.values():
            raise ValueError(f"rows have only label {int(y[0])}: nothing to tell apart")
        for label, n in counts.items():
            if n < MIN_PER_LABEL:
                raise ValueError(f"label {label} has {n} rows, need at least {MIN_PER_LABEL}")
            if n < WARN_PER_LABEL:
                logger.warning(f"label {label} has {n} rows (under {WARN_PER_LABEL}): the boundary is a guess")

        X = np.stack([r.features() for r in rows])
        self.x_mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.x_std_ = np.where(std > 0, std, 1.0)
        kernel = (ConstantKernel(1.0, self.amplitude_bounds)
                  * Matern(length_scale=1.0, length_scale_bounds=self.length_scale_bounds, nu=2.5))
        gpc = GaussianProcessClassifier(kernel=kernel, optimizer="fmin_l_bfgs_b" if self.optimize else None,
                                        n_restarts_optimizer=N_RESTARTS if self.optimize else 0,
                                        random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # reported below, by name
            gpc.fit(self._standardise(X), y)
        self._gpc, self.production, self.rules_version, self.n_train = gpc, production, rows[0].rules_version, len(rows)
        if self.optimize:
            self._warn_at_bounds()
        return self

    @property
    def kernel_(self):
        """The fitted kernel (the given one when not optimised)."""
        self._check_fitted()
        return self._gpc.kernel_

    def predict_features(self, X: np.ndarray) -> list[Prediction]:
        """
        Predictions at feature rows, with no rules run: for rows already judged, as in k-fold.

        Args:
            X (np.ndarray): (n, n_features)

        Returns:
            list[Prediction]: One per row
        """
        self._check_fitted()
        Xs = self._standardise(np.atleast_2d(np.asarray(X, dtype=np.float64)))
        f_mean, f_var = self._gpc.latent_mean_and_variance(Xs)
        p = self._gpc.predict_proba(Xs)[:, 1]
        return [Prediction(p=float(a), f_mean=float(m), f_std=float(np.sqrt(max(v, 0.0))))
                for a, m, v in zip(p, f_mean, f_var)]

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) / self.x_std_

    def _check_fitted(self) -> None:
        if self._gpc is None:
            raise ValueError(f"{type(self).__name__} is not fitted")

    def _check_rules_current(self) -> None:
        self._check_fitted()
        if self.rules_version != RULES_VERSION:
            raise ValueError(f"fitted on rows that passed rules {self.rules_version}, the current rules are "
                             f"{RULES_VERSION}: refit on rows re-checked under them")

    def _warn_at_bounds(self) -> None:
        kernel = self._gpc.kernel_
        for hp, value in zip(kernel.hyperparameters, np.exp(kernel.theta)):
            lo, hi = np.asarray(hp.bounds, dtype=float).ravel()
            side = "lower" if value <= lo * BOUND_FACTOR else "upper" if value >= hi / BOUND_FACTOR else None
            if side is not None:
                kind = "amplitude" if "constant_value" in hp.name else "length_scale"
                logger.warning(f"{hp.name} = {value:.3g} reached its {side} bound {lo if side == 'lower' else hi:g}: "
                               f"{_AT_BOUND[kind, side]}")


# What a hyperparameter stopped at a bound says about the data
_AT_BOUND = {
    ("amplitude", "upper"): "the data asks for a sharper boundary than the cap allows, as a noise-free or "
                            "near-linear one does, where amplitude and length scale grow together and the data "
                            "can't fix them apart. p and its ranking are still usable; f_mean and f_std are on a "
                            "scale set by the cap, not by the data",
    ("amplitude", "lower"): "the latent is nearly flat: the model finds no signal, and p stays near the share of 1s",
    ("length_scale", "upper"): "the latent is nearly linear over the data: the boundary looks like a plane, or the "
                               "inputs barely matter; the length scale is set by the cap, not by the data",
    ("length_scale", "lower"): "the latent follows each row on its own: the labels look like noise at this "
                               "spacing, so p between rows is a guess",
}


class MeshClassifier(_GPFeasibility):
    """p_mesh: the rules, then the GP on a design's 18 free values."""

    def predict(self, cst: CSTParams) -> Optional[Prediction]:
        """
        p_mesh for one design.

        Args:
            cst (CSTParams): The design

        Returns:
            Prediction | None: None if the shape or mesh rules reject it, so p_mesh isn't defined there

        Raises:
            ValueError: If not fitted, fitted under other rules, or the CST isn't a design
        """
        return self.predict_many([cst])[0]

    def predict_many(self, csts: Sequence[CSTParams]) -> list[Optional[Prediction]]:
        """`predict` for many designs, in one GP call for those the rules pass."""
        self._check_rules_current()
        passed = [i for i, cst in enumerate(csts)
                  if AL_geometry_rules(cst).ok and AL_mesh_rules(cst, self.production).ok]
        out: list[Optional[Prediction]] = [None] * len(csts)
        if passed:
            found = self.predict_features(np.stack([AL_cst_to_free(csts[i]) for i in passed]))
            for i, prediction in zip(passed, found):
                out[i] = prediction
        return out

    def predict_rows(self, rows: Sequence) -> list[Optional[Prediction]]:
        """`predict_many` for rows' designs, the rules run again: for a held-out report."""
        return self.predict_many([AL_free_to_cst(r.x_array()) for r in rows])
