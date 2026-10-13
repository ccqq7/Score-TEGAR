from __future__ import annotations

import unittest

import numpy as np

from utils.metrics import compute_zero_shot_metrics, search_decision_threshold
from sklearn.metrics import f1_score


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.log(probabilities / (1.0 - probabilities))


class ThresholdMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.label_names = ["seen", "surrogate"]
        self.targets = np.asarray(
            [[1, 0], [0, 1], [1, 0], [0, 1]],
            dtype=np.float32,
        )
        self.logits = _logit(
            np.asarray(
                [[0.8, 0.2], [0.4, 0.6], [0.9, 0.1], [0.3, 0.7]],
                dtype=np.float64,
            )
        )

    def test_reports_seen_and_unseen_f1_and_recall(self) -> None:
        metrics = compute_zero_shot_metrics(
            self.logits,
            self.targets,
            self.label_names,
            seen_labels=["seen"],
            unseen_labels=["surrogate"],
            decision_threshold=0.5,
        )
        for key in (
            "seen_macro_F1", "seen_micro_F1", "seen_macro_recall",
            "unseen_macro_F1", "unseen_micro_F1", "unseen_macro_recall",
            "f1_harmonic_mean",
        ):
            self.assertAlmostEqual(float(metrics[key]), 1.0, places=7)

    def test_threshold_is_selected_only_from_supplied_inner_labels(self) -> None:
        threshold, metrics = search_decision_threshold(
            self.logits,
            self.targets,
            self.label_names,
            seen_labels=["seen"],
            unseen_labels=["surrogate"],
            candidate_thresholds=[0.3, 0.5, 0.8],
        )
        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(float(metrics["f1_harmonic_mean"]), 1.0, places=7)

    def test_forbidden_outer_unseen_label_fails_fast(self) -> None:
        with self.assertRaises(AssertionError):
            compute_zero_shot_metrics(
                self.logits,
                self.targets,
                self.label_names,
                seen_labels=["seen"],
                unseen_labels=["surrogate"],
                forbidden_labels=["surrogate"],
            )

    def test_exact_search_handles_saturated_clip_scores(self) -> None:
        logits = np.array([[25., 20.], [20., 25.]])
        targets = np.eye(2)
        threshold, metrics = search_decision_threshold(
            logits, targets, self.label_names,
            seen_labels=["seen"], unseen_labels=["surrogate"],
        )
        self.assertEqual(threshold, 25.)
        self.assertEqual(metrics["decision_threshold_space"], "logit")
        self.assertAlmostEqual(metrics["unseen_macro_recall"], 1.)
        self.assertAlmostEqual(metrics["unseen_macro_precision"], 1.)
        self.assertAlmostEqual(metrics["unseen_positive_prediction_rate"], .5)

    def test_exact_sweep_matches_independent_brute_force(self) -> None:
        rng = np.random.default_rng(4909)
        names = ["s1", "s2", "u1", "u2"]
        for _ in range(30):
            # Discrete scores deliberately create many tied examples/labels.
            logits = rng.integers(-3, 4, size=(11, 4)).astype(float)
            targets = rng.integers(0, 2, size=logits.shape)
            expected_threshold = None
            expected_score = -1.
            for threshold in np.unique(logits):
                pred = logits >= threshold
                f1 = [f1_score(targets[:, i], pred[:, i], zero_division=0) for i in range(4)]
                s, u = np.mean(f1[:2]), np.mean(f1[2:])
                score = 2 * s * u / (s + u + 1e-8)
                if score > expected_score + 1e-12:
                    expected_threshold, expected_score = threshold, score
            threshold, metrics = search_decision_threshold(
                logits, targets, names, seen_labels=names[:2], unseen_labels=names[2:],
            )
            self.assertEqual(threshold, expected_threshold)
            self.assertAlmostEqual(metrics["f1_harmonic_mean"], expected_score, places=12)

    def test_exact_search_is_invariant_to_score_scale_and_shift(self) -> None:
        base, _ = search_decision_threshold(
            self.logits, self.targets, self.label_names,
            seen_labels=["seen"], unseen_labels=["surrogate"],
        )
        changed = self.logits * 100 + 80
        threshold, _ = search_decision_threshold(
            changed, self.targets, self.label_names,
            seen_labels=["seen"], unseen_labels=["surrogate"],
        )
        np.testing.assert_array_equal(self.logits >= base, changed >= threshold)

    def test_forbidden_scores_do_not_generate_candidates(self) -> None:
        labels = [*self.label_names, "outer_unseen"]
        logits = np.column_stack([self.logits, np.full(4, np.nan)])
        targets = np.column_stack([self.targets, np.full(4, np.nan)])
        t, _ = search_decision_threshold(
            logits, targets, labels, seen_labels=["seen"], unseen_labels=["surrogate"],
            forbidden_labels=["outer_unseen"],
        )
        logits[:, -1] = 1e9
        targets[:, -1] = 1
        t2, _ = search_decision_threshold(
            logits, targets, labels, seen_labels=["seen"], unseen_labels=["surrogate"],
            forbidden_labels=["outer_unseen"],
        )
        self.assertEqual(t, t2)
        with self.assertRaises(AssertionError):
            search_decision_threshold(logits, targets, labels, seen_labels=["seen"],
                                      unseen_labels=["outer_unseen"], forbidden_labels=["outer_unseen"])

    def test_empty_invalid_or_overlapping_validation_fails(self) -> None:
        cases = [
            (np.empty((0, 2)), np.empty((0, 2)), ["seen"], ["surrogate"]),
            (np.full((2, 2), np.nan), np.zeros((2, 2)), ["seen"], ["surrogate"]),
            (self.logits, self.targets, ["seen"], ["seen"]),
            (self.logits, self.targets, ["seen"], ["missing"]),
            (self.logits, self.targets, ["seen"], []),
        ]
        for logits, targets, seen, unseen in cases:
            with self.assertRaises(ValueError):
                search_decision_threshold(logits, targets, self.label_names,
                                          seen_labels=seen, unseen_labels=unseen)
        with self.assertRaises(ValueError):
            search_decision_threshold(self.logits, self.targets, self.label_names,
                                      seen_labels=["seen"], unseen_labels=["surrogate"], candidate_thresholds=[])

    def test_no_support_recall_is_zero_and_ties_keep_lower_threshold(self) -> None:
        logits = np.array([[3., 7.], [3., 7.]])
        t, metrics = search_decision_threshold(
            logits, np.zeros_like(logits), self.label_names,
            seen_labels=["seen"], unseen_labels=["surrogate"],
        )
        self.assertEqual(t, 3.)
        self.assertEqual(metrics["unseen_macro_recall"], 0.)
        self.assertEqual(metrics["f1_harmonic_mean"], 0.)


if __name__ == "__main__":
    unittest.main()
