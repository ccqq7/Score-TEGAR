from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_DECISION_THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 0.951, 0.05))


def decision_threshold_to_logit(threshold: float, space: str = "probability") -> float:
    """Validate a threshold and convert legacy probabilities without sigmoid saturation."""
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("decision threshold must be finite")
    if space == "logit":
        return threshold
    if space != "probability":
        raise ValueError(f"Unknown decision threshold space: {space!r}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"probability decision threshold must be in (0, 1), got {threshold}")
    return float(np.log(threshold) - np.log1p(-threshold))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def build_nested_label_protocol(
    outer_seen_labels: list[str],
    outer_unseen_labels: list[str],
    *,
    inner_seed: int,
    surrogate_fraction: float = 0.2,
    legacy_outer_unseen_validation: bool = False,
) -> dict[str, object]:
    """Build label sets for leakage-free nested GMLZSL model selection.

    Strict mode hides a deterministic subset of outer-seen labels from the
    training loss and uses only that subset as surrogate-unseen validation
    labels.  Outer-unseen labels remain forbidden until final test evaluation.
    """
    outer_seen = list(dict.fromkeys(str(label) for label in outer_seen_labels))
    outer_unseen = list(dict.fromkeys(str(label) for label in outer_unseen_labels))
    overlap = sorted(set(outer_seen) & set(outer_unseen))
    if overlap:
        raise ValueError(f"Outer seen/unseen labels overlap: {overlap}")
    if not outer_seen:
        raise ValueError("outer_seen_labels must not be empty")

    if legacy_outer_unseen_validation:
        return {
            "mode": "legacy_outer_unseen_validation",
            "outer_seen_labels": outer_seen,
            "outer_unseen_labels": outer_unseen,
            "inner_seen_labels": outer_seen,
            "surrogate_unseen_labels": outer_unseen,
            "training_hidden_labels": outer_unseen,
            "selection_seen_labels": outer_seen,
            "selection_unseen_labels": outer_unseen,
            "inner_seed": int(inner_seed),
            "surrogate_fraction": 0.0,
        }

    fraction = float(surrogate_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"surrogate_fraction must be in (0, 1), got {fraction}")
    if len(outer_seen) < 2:
        raise ValueError("Strict nested validation requires at least two outer-seen labels")

    surrogate_count = max(1, int(round(len(outer_seen) * fraction)))
    surrogate_count = min(surrogate_count, len(outer_seen) - 1)
    ordered_seen = sorted(outer_seen)
    rng = np.random.default_rng(int(inner_seed))
    surrogate_indices = set(rng.choice(len(ordered_seen), size=surrogate_count, replace=False).tolist())
    surrogate_unseen = [label for idx, label in enumerate(ordered_seen) if idx in surrogate_indices]
    surrogate_set = set(surrogate_unseen)
    inner_seen = [label for label in outer_seen if label not in surrogate_set]
    training_hidden = list(dict.fromkeys([*outer_unseen, *surrogate_unseen]))
    return {
        "mode": "strict_inner_surrogate",
        "outer_seen_labels": outer_seen,
        "outer_unseen_labels": outer_unseen,
        "inner_seen_labels": inner_seen,
        "surrogate_unseen_labels": surrogate_unseen,
        "training_hidden_labels": training_hidden,
        "selection_seen_labels": inner_seen,
        "selection_unseen_labels": surrogate_unseen,
        "inner_seed": int(inner_seed),
        "surrogate_fraction": fraction,
    }


def assert_no_forbidden_labels(
    evaluated_labels: list[str],
    forbidden_labels: list[str] | None,
    *,
    context: str,
) -> None:
    overlap = sorted(set(evaluated_labels) & set(forbidden_labels or []))
    if overlap:
        raise AssertionError(f"{context} attempted to evaluate forbidden outer-unseen labels: {overlap}")


def _classification_metrics_for_indices(
    logits: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
    *,
    threshold: float,
    threshold_space: str = "probability",
) -> dict[str, float]:
    logit_threshold = decision_threshold_to_logit(threshold, threshold_space)
    if not indices:
        return {"macro_F1": 0.0, "micro_F1": 0.0, "macro_recall": 0.0,
                "macro_precision": 0.0, "positive_prediction_rate": 0.0}

    truth = np.asarray(targets[:, indices] > 0.5, dtype=bool)
    # Float64 comparison avoids rounding the threshold back to float32. This
    # also preserves strict cutoffs just above a tied float32 score.
    predictions = np.asarray(logits[:, indices], dtype=np.float64) >= logit_threshold
    tp = np.logical_and(predictions, truth).sum(axis=0, dtype=np.float64)
    fp = np.logical_and(predictions, ~truth).sum(axis=0, dtype=np.float64)
    fn = np.logical_and(~predictions, truth).sum(axis=0, dtype=np.float64)

    f1_denominator = 2.0 * tp + fp + fn
    per_class_f1 = np.divide(
        2.0 * tp,
        f1_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=f1_denominator > 0,
    )
    recall_denominator = tp + fn
    valid_recall = recall_denominator > 0
    per_class_recall = np.divide(
        tp,
        recall_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=valid_recall,
    )
    total_tp = float(tp.sum())
    total_fp = float(fp.sum())
    total_fn = float(fn.sum())
    micro_denominator = 2.0 * total_tp + total_fp + total_fn
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return {
        "macro_F1": float(per_class_f1.mean()) if per_class_f1.size else 0.0,
        "micro_F1": float(2.0 * total_tp / micro_denominator) if micro_denominator > 0 else 0.0,
        "macro_recall": float(per_class_recall[valid_recall].mean()) if valid_recall.any() else 0.0,
        "macro_precision": float(precision.mean()) if precision.size else 0.0,
        "positive_prediction_rate": float(predictions.mean()) if predictions.size else 0.0,
    }


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    positive_count = int(y_true.sum())
    if positive_count == 0:
        return float("nan")

    ranking = np.argsort(-y_score)
    ranked_true = y_true[ranking]
    cumulative_hits = np.cumsum(ranked_true)
    precision = cumulative_hits / np.arange(1, len(ranked_true) + 1)
    ap = (precision * ranked_true).sum() / positive_count
    return float(ap)


def compute_per_class_ap(
    logits: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(label_names):
        ap = average_precision_binary(targets[:, idx], logits[:, idx])
        rows.append(
            {
                "label": label,
                "ap": ap,
                "positive_count": int(targets[:, idx].sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_map(per_class_ap: pd.DataFrame) -> float:
    values = per_class_ap["ap"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def summarize_frequency_groups(
    per_class_ap: pd.DataFrame,
    label_frequency: dict[str, int],
    rare_k: int = 10,
    freq_k: int = 10,
) -> dict[str, float | list[str]]:
    frequency_table = (
        pd.DataFrame({"label": list(label_frequency.keys()), "frequency": list(label_frequency.values())})
        .sort_values("frequency", ascending=True)
        .reset_index(drop=True)
    )
    rare_labels = frequency_table.head(rare_k)["label"].tolist()
    freq_labels = frequency_table.tail(freq_k)["label"].tolist()

    ap_lookup = dict(zip(per_class_ap["label"], per_class_ap["ap"]))
    rare_ap = np.asarray([ap_lookup[label] for label in rare_labels if label in ap_lookup], dtype=float)
    freq_ap = np.asarray([ap_lookup[label] for label in freq_labels if label in ap_lookup], dtype=float)
    rare_finite = rare_ap[np.isfinite(rare_ap)]
    freq_finite = freq_ap[np.isfinite(freq_ap)]

    return {
        "AP_rare": float(rare_finite.mean()) if rare_finite.size > 0 else float("nan"),
        "AP_freq": float(freq_finite.mean()) if freq_finite.size > 0 else float("nan"),
        "rare_labels": rare_labels,
        "freq_labels": freq_labels,
    }


def compute_zero_shot_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
    seen_labels: list[str] | None = None,
    unseen_labels: list[str] | None = None,
    decision_threshold: float = 0.5,
    forbidden_labels: list[str] | None = None,
    decision_threshold_space: str = "probability",
) -> dict[str, float | pd.DataFrame | list[str]]:
    decision_threshold_to_logit(decision_threshold, decision_threshold_space)
    unseen_labels = [label for label in (unseen_labels or []) if label in label_names]
    if seen_labels is None:
        seen_labels = [label for label in label_names if label not in unseen_labels]
    else:
        seen_labels = [label for label in seen_labels if label in label_names]

    assert_no_forbidden_labels(
        [*seen_labels, *unseen_labels],
        forbidden_labels,
        context="compute_zero_shot_metrics",
    )

    seen_indices = [label_names.index(label) for label in seen_labels]
    unseen_indices = [label_names.index(label) for label in unseen_labels]

    if seen_indices:
        seen_logits = logits[:, seen_indices]
        seen_targets = targets[:, seen_indices]
        seen_per_class = compute_per_class_ap(seen_logits, seen_targets, seen_labels)
        seen_ap_values = seen_per_class["ap"].dropna().to_numpy(dtype=float)
        seen_map = float(seen_ap_values.mean()) if seen_ap_values.size > 0 else 0.0
    else:
        seen_per_class = pd.DataFrame(columns=["label", "ap", "positive_count"])
        seen_map = 0.0

    if unseen_indices:
        unseen_logits = logits[:, unseen_indices]
        unseen_targets = targets[:, unseen_indices]
        unseen_per_class = compute_per_class_ap(unseen_logits, unseen_targets, unseen_labels)
        unseen_ap_values = unseen_per_class["ap"].dropna().to_numpy(dtype=float)
        unseen_map = float(unseen_ap_values.mean()) if unseen_ap_values.size > 0 else 0.0
    else:
        unseen_per_class = pd.DataFrame(columns=["label", "ap", "positive_count"])
        unseen_map = 0.0

    if (seen_map + unseen_map) <= 0.0:
        harmonic_mean = 0.0
    else:
        harmonic_mean = float((2.0 * seen_map * unseen_map) / (seen_map + unseen_map + 1e-8))

    overall_ap_values = np.concatenate(
        [
            seen_per_class["ap"].dropna().to_numpy(dtype=float),
            unseen_per_class["ap"].dropna().to_numpy(dtype=float),
        ]
    )
    overall_map = float(overall_ap_values.mean()) if overall_ap_values.size > 0 else 0.0

    seen_classification = _classification_metrics_for_indices(
        logits,
        targets,
        seen_indices,
        threshold=decision_threshold,
        threshold_space=decision_threshold_space,
    )
    unseen_classification = _classification_metrics_for_indices(
        logits,
        targets,
        unseen_indices,
        threshold=decision_threshold,
        threshold_space=decision_threshold_space,
    )
    seen_macro_f1 = float(seen_classification["macro_F1"])
    unseen_macro_f1 = float(unseen_classification["macro_F1"])
    if seen_macro_f1 + unseen_macro_f1 <= 0.0:
        f1_harmonic_mean = 0.0
    else:
        f1_harmonic_mean = float(
            2.0 * seen_macro_f1 * unseen_macro_f1 / (seen_macro_f1 + unseen_macro_f1 + 1e-8)
        )

    return {
        "seen_labels": seen_labels,
        "unseen_labels": unseen_labels,
        "seen_mAP": float(seen_map),
        "unseen_mAP": float(unseen_map),
        "harmonic_mean": harmonic_mean,
        "overall_mAP": overall_map,
        "decision_threshold": float(decision_threshold),
        "decision_threshold_space": decision_threshold_space,
        "seen_macro_F1": seen_macro_f1,
        "seen_micro_F1": float(seen_classification["micro_F1"]),
        "seen_macro_recall": float(seen_classification["macro_recall"]),
        "seen_macro_precision": float(seen_classification["macro_precision"]),
        "seen_positive_prediction_rate": float(seen_classification["positive_prediction_rate"]),
        "unseen_macro_F1": unseen_macro_f1,
        "unseen_micro_F1": float(unseen_classification["micro_F1"]),
        "unseen_macro_recall": float(unseen_classification["macro_recall"]),
        "unseen_macro_precision": float(unseen_classification["macro_precision"]),
        "unseen_positive_prediction_rate": float(unseen_classification["positive_prediction_rate"]),
        "f1_harmonic_mean": f1_harmonic_mean,
        "seen_per_class": seen_per_class,
        "unseen_per_class": unseen_per_class,
    }


def search_decision_threshold(
    logits: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
    *,
    seen_labels: list[str],
    unseen_labels: list[str],
    candidate_thresholds: tuple[float, ...] | list[float] | None = None,
    forbidden_labels: list[str] | None = None,
    strategy: str | None = None,
) -> tuple[float, dict[str, float | pd.DataFrame | list[str]]]:
    """Select a global inner-validation threshold, never using forbidden columns.

    By default, search every distinct raw score. Explicit legacy candidates
    retain probability-grid semantics unless ``strategy`` is explicitly set.
    """
    strategy = strategy or ("probability_grid" if candidate_thresholds is not None else "exact_logit")
    if strategy == "exact_logit":
        return _search_exact_logit_threshold(
            logits, targets, label_names, seen_labels=seen_labels,
            unseen_labels=unseen_labels, forbidden_labels=forbidden_labels,
        )
    if strategy != "probability_grid":
        raise ValueError(f"Unknown decision threshold search strategy: {strategy!r}")
    thresholds = tuple(DEFAULT_DECISION_THRESHOLDS if candidate_thresholds is None else candidate_thresholds)
    if not thresholds:
        raise ValueError("candidate_thresholds must not be empty")
    best_threshold = float(thresholds[0])
    best_metrics = compute_zero_shot_metrics(
        logits,
        targets,
        label_names,
        seen_labels=seen_labels,
        unseen_labels=unseen_labels,
        decision_threshold=best_threshold,
        forbidden_labels=forbidden_labels,
    )
    best_score = float(best_metrics["f1_harmonic_mean"])
    for threshold in thresholds[1:]:
        metrics = compute_zero_shot_metrics(
            logits,
            targets,
            label_names,
            seen_labels=seen_labels,
            unseen_labels=unseen_labels,
            decision_threshold=float(threshold),
            forbidden_labels=forbidden_labels,
        )
        score = float(metrics["f1_harmonic_mean"])
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _search_exact_logit_threshold(
    logits: np.ndarray,
    targets: np.ndarray,
    label_names: list[str],
    *,
    seen_labels: list[str],
    unseen_labels: list[str],
    forbidden_labels: list[str] | None,
) -> tuple[float, dict]:
    """Exact O(K log K) sweep, grouping equal scores before scoring a cutoff.

    Each sorted prediction changes only one class's F1. Accumulating these
    changes gives both macro-F1 curves without repeatedly sorting AP scores or
    materializing one prediction matrix per candidate. K is images * selected
    labels; outer-unseen columns are never consulted, including their scores.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.ndim != 2 or targets.shape != logits.shape or logits.shape[1] != len(label_names):
        raise ValueError("logits and targets must have shape (images, len(label_names))")
    if logits.shape[0] == 0:
        raise ValueError("Cannot select a decision threshold from empty validation data")
    labels = [*seen_labels, *unseen_labels]
    if not seen_labels or not unseen_labels:
        raise ValueError("Threshold selection requires nonempty seen and surrogate-unseen label sets")
    if len(set(label_names)) != len(label_names) or len(set(labels)) != len(labels):
        raise ValueError("Label names must be unique and selection sets must be disjoint")
    missing = set(labels) - set(label_names)
    if missing:
        raise ValueError(f"Unknown validation labels: {sorted(missing)}")
    assert_no_forbidden_labels(labels, forbidden_labels, context="search_decision_threshold")
    indices = [label_names.index(label) for label in labels]
    scores = np.asarray(logits[:, indices], dtype=np.float64)
    selected_targets = np.asarray(targets[:, indices], dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(selected_targets).all():
        raise ValueError("Selected validation scores and targets must be finite")
    truth = selected_targets > 0.5
    support = truth.sum(axis=0)
    classes = len(labels)
    seen_count = len(seen_labels)
    order = np.argsort(-scores.ravel(), kind="stable")
    sorted_scores = scores.ravel()[order]
    sorted_truth = truth.ravel()[order]
    sorted_classes = order % classes
    delta = np.zeros(len(order), dtype=np.float64)
    for col in range(classes):
        positions = np.flatnonzero(sorted_classes == col)
        tp = np.cumsum(sorted_truth[positions], dtype=np.float64)
        f1 = 2.0 * tp / (support[col] + np.arange(1, len(positions) + 1))
        delta[positions] = np.diff(f1, prepend=0.0) / (seen_count if col < seen_count else classes - seen_count)
    seen_curve = np.cumsum(np.where(sorted_classes < seen_count, delta, 0.0))
    unseen_curve = np.cumsum(np.where(sorted_classes >= seen_count, delta, 0.0))
    # A cutoff at s includes ALL scores equal to s, independent of image order.
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    seen_f1 = np.clip(seen_curve[ends], 0.0, 1.0)
    unseen_f1 = np.clip(unseen_curve[ends], 0.0, 1.0)
    hm = 2.0 * seen_f1 * unseen_f1 / (seen_f1 + unseen_f1 + 1e-8)
    best = float(hm.max())
    # Scores are descending. The last maximizer is the lower threshold,
    # consistent with the previous protocol's 1e-12 tie tolerance.
    chosen = int(np.flatnonzero(hm >= best - 1e-12)[-1])
    threshold = float(sorted_scores[ends[chosen]])
    metrics = compute_zero_shot_metrics(
        logits, targets, label_names, seen_labels=seen_labels, unseen_labels=unseen_labels,
        decision_threshold=threshold, decision_threshold_space="logit", forbidden_labels=forbidden_labels,
    )
    metrics["decision_threshold_strategy"] = "exact_logit"
    metrics["threshold_candidate_count"] = int(len(ends))
    return threshold, metrics
