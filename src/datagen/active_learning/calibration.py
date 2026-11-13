"""
How far a feasibility classifier's p can be read as a probability, on rows it wasn't fitted on.

    AL_calibration_from_predictions  the report from (p, label) pairs, however they were held out
    AL_calibration_report            the report for a held-out set: the rules run again, then the model
    AL_grouped_kfold_predictions     out-of-fold predictions for every row, folds by shape and label, so a
                                     report can use all rows when failures are rare

The report: a reliability table (equal-width bins, empty ones left out), ECE = sum_b (n_b / N) |mean_p_b -
frac_b|, the Brier score against the base rate's (always predicting the held-out share of 1s), log loss,
and ROC AUC (None when only one label is held out).
"""

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

from src.datagen.active_learning.classifier import MeshClassifier, Prediction
from src.datagen.active_learning.labels import ProductionMesh
from src.datagen.active_learning.split import AL_check_one_setting, AL_label_groups

P_CLIP = 1e-12  # log loss of p = 0 or 1 on the wrong label is infinite: clipped


class ReliabilityBin(BaseModel):
    lo: float
    hi: float
    n: int
    mean_p: float
    frac_positive: float


class CalibrationReport(BaseModel):
    bins: list[ReliabilityBin]              # the non-empty ones, in order
    n: int                                  # rows scored
    n_rejected: int = 0                     # held-out rows the rules now reject: not scored
    ece: float
    brier: float
    base_rate_brier: float                  # p(1 - p) for the held-out share p of 1s
    brier_skill: Optional[float]            # 1 - brier / base_rate_brier; None if every label is the same
    log_loss: float
    roc_auc: Optional[float]                # None with one label held out

    def to_text(self) -> str:
        lines = [f"{'bin':>11} {'n':>5} {'mean p':>7} {'frac 1':>7}"]
        lines += [f"{b.lo:.2f}-{b.hi:.2f} {b.n:5d} {b.mean_p:7.3f} {b.frac_positive:7.3f}" for b in self.bins]
        skill = "n/a" if self.brier_skill is None else f"{self.brier_skill:.3f}"
        auc = "n/a" if self.roc_auc is None else f"{self.roc_auc:.3f}"
        lines += [f"n={self.n} (rejected by the rules: {self.n_rejected})  ECE={self.ece:.3f}  "
                  f"Brier={self.brier:.3f} (base rate {self.base_rate_brier:.3f}, skill {skill})  "
                  f"log loss={self.log_loss:.3f}  AUC={auc}"]
        return "\n".join(lines)


class KFoldPredictions(NamedTuple):
    predictions: list[Prediction]  # aligned with the rows given
    fold_of: dict[str, int]        # geometry_id -> the fold it was held out in


def AL_calibration_from_predictions(p: Sequence[float], labels: Sequence[int], n_bins: int = 10,
                                    n_rejected: int = 0) -> CalibrationReport:
    """
    The calibration report for held-out predictions.

    Args:
        p (Sequence[float]): P(label 1) per row
        labels (Sequence[int]): The rows' labels, 0 or 1
        n_bins (int): Equal-width bins over [0, 1], the last closed
        n_rejected (int): Rows left out before this, reported alongside

    Returns:
        CalibrationReport: The table and scores

    Raises:
        ValueError: If there are no predictions or the lengths differ
    """
    p, y = np.asarray(p, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError(f"need as many labels as predictions, at least one: got {len(p)} and {len(y)}")
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        inside = index == b
        if not inside.any():
            continue
        mean_p, frac = float(p[inside].mean()), float(y[inside].mean())
        bins.append(ReliabilityBin(lo=round(b / n_bins, 12), hi=round((b + 1) / n_bins, 12), n=int(inside.sum()),
                                   mean_p=mean_p, frac_positive=frac))
        ece += inside.sum() / len(p) * abs(mean_p - frac)

    brier = float(np.mean((p - y) ** 2))
    rate = float(y.mean())
    base = rate * (1 - rate)
    q = np.clip(p, P_CLIP, 1 - P_CLIP)
    log_loss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    one_label = rate in (0.0, 1.0)
    return CalibrationReport(bins=bins, n=len(p), n_rejected=n_rejected, ece=float(ece), brier=brier,
                             base_rate_brier=base, brier_skill=None if one_label else 1 - brier / base,
                             log_loss=log_loss, roc_auc=None if one_label else float(roc_auc_score(y, p)))


def AL_calibration_report(classifier: MeshClassifier, rows: Sequence, n_bins: int = 10) -> CalibrationReport:
    """
    The calibration report on held-out rows: their designs through the classifier's rules and model.

    Args:
        classifier (MeshClassifier): A fitted p_mesh
        rows (Sequence): Held-out rows of its settings and rules version
        n_bins (int): Bins of the reliability table

    Returns:
        CalibrationReport: For the rows the rules pass; the rest counted in `n_rejected`

    Raises:
        ValueError: For rows of another settings_id or rules version than the classifier's
    """
    rows = list(rows)
    AL_check_one_setting(rows)
    classifier._check_fitted()
    for field, want in (("settings_id", classifier.production.settings_id),
                        ("rules_version", classifier.rules_version)):
        if getattr(rows[0], field) != want:
            raise ValueError(f"held-out rows have {field} {getattr(rows[0], field)}, the classifier {want}")
    scored = [(pr.p, r.label) for pr, r in zip(classifier.predict_rows(rows), rows) if pr is not None]
    if not scored:
        raise ValueError(f"the rules reject all {len(rows)} held-out rows")
    p, y = zip(*scored)
    return AL_calibration_from_predictions(p, y, n_bins, n_rejected=len(rows) - len(scored))


def AL_grouped_kfold_predictions(rows: Sequence, production: ProductionMesh, k: int = 5, seed: int = 0,
                                 make_classifier: Callable = MeshClassifier) -> KFoldPredictions:
    """
    Out-of-fold predictions for every row: each fold's shapes are scored by a model fitted on the others.
    Folds are dealt per label, so each holds both labels when each label has at least k shapes.

    Args:
        rows (Sequence): The rows, one settings_id and rules version; they passed the rules when labelled
        production (ProductionMesh): Their settings
        k (int): Folds
        seed (int): The deal
        make_classifier (Callable): Makes an unfitted classifier, `MeshClassifier` unless testing

    Returns:
        KFoldPredictions: The predictions, aligned with `rows`, and each shape's fold

    Raises:
        ValueError: For mixed rows, k below 2, or a label with fewer than k shapes
    """
    rows = list(rows)
    AL_check_one_setting(rows)
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for label, ids in AL_label_groups(rows).items():
        if len(ids) < k:
            raise ValueError(f"k={k} folds, but label {label} has {len(ids)} shapes: a fold would lack it")
        for i, geometry_id in enumerate(rng.permutation(ids).tolist()):
            fold_of[geometry_id] = i % k

    predictions: list[Optional[Prediction]] = [None] * len(rows)
    for fold in range(k):
        held = [i for i, r in enumerate(rows) if fold_of[r.geometry_id] == fold]
        model = make_classifier().fit([r for r in rows if fold_of[r.geometry_id] != fold], production)
        for i, prediction in zip(held, model.predict_features(np.stack([rows[i].features() for i in held]))):
            predictions[i] = prediction
    return KFoldPredictions(predictions=predictions, fold_of=fold_of)
