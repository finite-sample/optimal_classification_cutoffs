"""High-level wrapper for threshold optimization."""

from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from .multiclass_coord import _assign_labels_shifted
from .optimizers import get_optimal_threshold
from .types import (
    AveragingMethodLiteral,
    ComparisonOperatorLiteral,
    EstimationModeLiteral,
    OptimizationMethodLiteral,
    SampleWeightLike,
    UtilityDict,
    UtilityMatrix,
)


class ThresholdOptimizer:
    """Optimizer for classification thresholds supporting both binary and multiclass.

    The class wraps threshold optimization functions and exposes a scikit-learn
    style ``fit``/``predict`` API. For multiclass, uses One-vs-Rest strategy.
    """

    def __init__(
        self,
        metric: str | None = None,
        verbose: bool = False,
        method: OptimizationMethodLiteral = "auto",
        comparison: ComparisonOperatorLiteral = ">",
        *,
        mode: EstimationModeLiteral = "empirical",
        utility: UtilityDict | None = None,
        utility_matrix: UtilityMatrix | None = None,
        minimize_cost: bool | None = None,
        beta: float = 1.0,
        class_weight: ArrayLike | None = None,
        average: AveragingMethodLiteral = "macro",
    ) -> None:
        """Create a new optimizer.

        Parameters
        ----------
        metric:
            Metric to optimize, e.g. ``"accuracy"``, ``"f1"``, ``"precision"``,
            ``"recall"``.
        verbose:
            If ``True``, print progress during threshold search.
        method:
            Optimization method:
            - ``"auto"``: Automatically selects best method (default)
            - ``"sort_scan"``: O(n log n) algorithm for piecewise metrics with
              vectorized implementation
            - ``"unique_scan"``: Evaluates all unique probabilities
            - ``"minimize"``: Uses ``scipy.optimize.minimize_scalar``
            - ``"gradient"``: Simple gradient ascent
            - ``"coord_ascent"``: Coordinate ascent for coupled multiclass
              optimization (single-label consistent)
        comparison:
            Comparison operator for thresholding: ">" (exclusive) or ">=" (inclusive).
        mode:
            Estimation regime to use:
            - ``"empirical"``: Use method parameter for empirical optimization (default)
            - ``"bayes"``: Return Bayes-optimal threshold/decisions under calibrated
              probabilities
              (requires utility or utility_matrix, ignores method and true_labs)
            - ``"expected"``: Use Dinkelbach method for expected F-beta optimization
              (supports sample weights and multiclass, binary/multilabel)
        utility:
            Optional utility specification for cost/benefit-aware optimization.
            Dict with keys "tp", "tn", "fp", "fn" specifying utilities/costs per
            outcome.
            For multiclass mode="bayes", can contain per-class vectors.
            Example: ``{"tp": 0, "tn": 0, "fp": -1, "fn": -5}`` for cost-sensitive.
        utility_matrix:
            Alternative to utility dict for multiclass Bayes decisions.
            Shape (D, K) array where D=decisions, K=classes.
            If provided with mode="bayes", returns class decisions rather than
            thresholds.
        minimize_cost:
            If True, interpret utility values as costs and minimize total cost. This
            automatically negates fp/fn values if they're positive.
        beta:
            F-beta parameter for expected mode (beta >= 0). beta=1 gives F1,
            beta < 1 emphasizes precision, beta > 1 emphasizes recall.
            Only used when mode="expected".
        class_weight:
            Optional per-class weights for weighted averaging in expected mode.
            Shape (K,) array. Only used when mode="expected" and average="weighted".
        average:
            Averaging strategy for multiclass:
            - "macro": per-class thresholds, unweighted mean metric
            - "weighted": per-class thresholds, class-weighted mean metric
            - "micro": single global threshold across all classes/instances
            - "none": per-class thresholds without averaging (same as macro)
        """
        if metric is None:
            metric = "accuracy"

        self.metric = metric
        self.verbose = verbose
        self.method = method
        self.comparison = comparison
        self.mode = mode
        self.utility = utility
        self.utility_matrix = utility_matrix
        self.minimize_cost = minimize_cost
        self.beta = beta
        self.class_weight = class_weight
        self.average = average
        self.threshold_: (
            float | np.ndarray[Any, Any] | dict[str, Any] | tuple[float, float] | None
        ) = None
        self.is_multiclass_: bool = False
        self.expected_score_: float | None = None
        self.n_classes_: int | None = None
        # Flag for utility_matrix with mode='bayes' - prediction uses U @ p -> argmax
        self._use_utility_matrix_in_predict = (
            mode == "bayes" and utility_matrix is not None
        )

    def fit(
        self,
        true_labs: ArrayLike,
        pred_prob: ArrayLike,
        sample_weight: SampleWeightLike = None,
    ) -> Self:
        """Estimate the optimal threshold(s) from labeled data.

        Parameters
        ----------
        true_labs:
            Array of true labels. For binary: (0, 1). For multiclass:
            (0, 1, 2, ..., n_classes-1).
        pred_prob:
            Predicted probabilities from a classifier. For binary: 1D array
            (n_samples,).
            For multiclass: 2D array (n_samples, n_classes).
        sample_weight:
            Optional array of sample weights for handling imbalanced datasets.

        Returns
        -------
        Self
            Fitted instance with ``threshold_`` attribute set.
        """
        pred_prob = np.asarray(pred_prob)

        # Validate probabilities for bayes/expected modes
        if self.mode in ("bayes", "expected"):
            self._require_probs01(pred_prob, "fit")

        # Check if multiclass (treat (n,1) as binary)
        self.is_multiclass_ = pred_prob.ndim == 2 and pred_prob.shape[1] > 1
        self.n_classes_ = pred_prob.shape[1] if self.is_multiclass_ else 2

        # Flatten (n,1) arrays to 1D for binary classification
        if pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
            pred_prob = pred_prob.ravel()

        use_general = (
            self.is_multiclass_
            or self.metric not in ["accuracy", "f1"]
            or sample_weight is not None
            or self.mode != "empirical"
            or self.utility is not None
            or self.utility_matrix is not None
            or self.minimize_cost is not None
        )
        if self.verbose:
            path = "general optimizer" if use_general else "simple binary optimizer"
            print(
                f"[ThresholdOptimizer] Using {path} "
                f"(mode={self.mode}, method={self.method}, metric={self.metric})"
            )
        if use_general:
            # Use the more general optimizer
            result = get_optimal_threshold(
                true_labs,
                pred_prob,
                self.metric,
                self.method,
                sample_weight,
                self.comparison,
                mode=self.mode,
                utility=self.utility,
                utility_matrix=self.utility_matrix,
                minimize_cost=self.minimize_cost,
                beta=self.beta,
                class_weight=self.class_weight,
                average=self.average,
            )

            if self._use_utility_matrix_in_predict:
                # Nothing to learn; we'll apply U at prediction time.
                self.threshold_ = "bayes/utility_matrix"  # type: ignore[assignment]
                self.expected_score_ = None
                return self

            # Normalize results
            if isinstance(result, tuple):
                # expected mode, binary: (threshold, score)
                self.threshold_, self.expected_score_ = result
            elif isinstance(result, dict):
                # expected mode, multiclass
                if "thresholds" in result:  # macro/weighted/none
                    self.threshold_ = cast(np.ndarray[Any, Any], result["thresholds"])
                    self.expected_score_ = float(
                        result.get("f_beta", result.get("score", np.nan))
                    )
                elif "threshold" in result:  # micro (single global threshold)
                    self.threshold_ = float(result["threshold"])
                    self.expected_score_ = float(
                        result.get("f_beta", result.get("score", np.nan))
                    )
                else:
                    raise RuntimeError("Unknown result dict format from optimizer.")
            else:
                self.threshold_ = result  # type: ignore[assignment]
        else:
            # Use standard optimizer for simple binary cases
            self.threshold_ = get_optimal_threshold(
                true_labs,
                pred_prob,
                self.metric,
                self.method,
                comparison=self.comparison,
            )  # type: ignore[assignment]

        return self

    def _require_probs01(self, arr: np.ndarray[Any, Any], where: str) -> None:
        """Validate that probabilities are in [0,1] for bayes/expected modes."""
        if np.any(~np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
            raise ValueError(
                f"{where}: pred_prob must be finite probabilities in [0,1]."
            )

    def predict(self, pred_prob: ArrayLike) -> np.ndarray[Any, Any]:
        """Convert probabilities to class predictions using the learned threshold(s).

        Parameters
        ----------
        pred_prob:
            Array of predicted probabilities to be thresholded.

        Returns
        -------
        np.ndarray[Any, Any]
            For binary: Integer array of predicted class labels (0, 1).
            For multiclass: Integer array of predicted class labels.
        """
        if self.threshold_ is None:
            raise RuntimeError("ThresholdOptimizer has not been fitted.")

        pred_prob = np.asarray(pred_prob)

        # Validate probabilities for bayes/expected modes
        if self.mode in ("bayes", "expected"):
            self._require_probs01(pred_prob, "predict")

        # Bayes with utility_matrix: predict directly via U @ p
        if self._use_utility_matrix_in_predict:
            U = np.asarray(self.utility_matrix, dtype=float)
            if pred_prob.ndim != 2 or pred_prob.shape[1] != U.shape[1]:
                raise ValueError(
                    "pred_prob shape must be (n_samples, K) matching utility_matrix "
                    "columns."
                )
            # decisions = argmax over rows of U @ p_i
            scores = pred_prob @ U.T  # shape (n_samples, D)
            return np.argmax(scores, axis=1)  # type: ignore[no-any-return]

        # Enforce same "multiclassness" as training
        is_multi_now = pred_prob.ndim == 2 and pred_prob.shape[1] > 1
        if is_multi_now != self.is_multiclass_:
            raise ValueError(
                "Prediction data dimensionality does not match fit: "
                f"is_multiclass_={self.is_multiclass_}, "
                f"got pred_prob.ndim={pred_prob.ndim} shape={pred_prob.shape}"
            )
        if (
            self.is_multiclass_
            and self.n_classes_ is not None
            and pred_prob.shape[1] != self.n_classes_
        ):
            raise ValueError(
                f"Expected {self.n_classes_} classes, got {pred_prob.shape[1]}."
            )

        # Flatten (n,1) arrays to 1D for binary classification
        if not self.is_multiclass_ and pred_prob.ndim == 2 and pred_prob.shape[1] == 1:
            pred_prob = pred_prob.ravel()

        if self.is_multiclass_:
            # Multiclass prediction strategy depends on optimization method
            if self.method == "coord_ascent":
                # Coordinate ascent uses argmax(P - tau) for single-label consistency
                return _assign_labels_shifted(
                    pred_prob, cast(np.ndarray[Any, Any], self.threshold_)
                )
            else:
                # One-vs-Rest prediction using per-class thresholds
                n_samples, n_classes = pred_prob.shape
                # Allow scalar (global) or per-class thresholds
                thr = self.threshold_
                if isinstance(thr, dict):
                    # Should not happen after normalization in fit, but guard:
                    thr = thr.get("thresholds", thr.get("threshold"))  # type: ignore[assignment]
                if isinstance(thr, (float, int, np.floating)):
                    tarr: float | np.ndarray[Any, Any] = float(thr)
                else:
                    tarr = np.asarray(thr)
                    if tarr.shape != (n_classes,):
                        raise ValueError(
                            f"Per-class threshold shape must be ({n_classes},), "
                            f"got {tarr.shape}."
                        )

                if self.comparison == ">":
                    binary_predictions = pred_prob > tarr
                else:  # ">="
                    binary_predictions = pred_prob >= tarr

                # For each sample, predict the class with highest probability among
                # those above threshold
                # If no classes above threshold, predict the class with highest
                # probability
                predictions = np.zeros(n_samples, dtype=int)

                for i in range(n_samples):
                    above_threshold = np.where(binary_predictions[i])[0]
                    if len(above_threshold) > 0:
                        # Among classes above threshold, pick the one with highest
                        # probability
                        predictions[i] = above_threshold[
                            np.argmax(pred_prob[i, above_threshold])
                        ]
                    else:
                        # No class above threshold, pick highest probability class
                        predictions[i] = np.argmax(pred_prob[i])

                return predictions
        else:
            # Binary prediction
            thr = float(self.threshold_)  # type: ignore[arg-type]
            out = (pred_prob > thr) if self.comparison == ">" else (pred_prob >= thr)
            # sklearn-style: return class labels not booleans
            return out.astype(int)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.
        """
        params = {
            "metric": self.metric,
            "verbose": self.verbose,
            "method": self.method,
            "comparison": self.comparison,
            "mode": self.mode,
            "utility": self.utility,
            "utility_matrix": self.utility_matrix,
            "minimize_cost": self.minimize_cost,
            "beta": self.beta,
            "class_weight": self.class_weight,
            "average": self.average,
        }
        return params

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        Self
            Estimator instance.
        """
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key!r}")
            setattr(self, key, value)

        # Update the utility matrix flag if relevant parameters changed
        self._use_utility_matrix_in_predict = (
            self.mode == "bayes" and self.utility_matrix is not None
        )

        return self
