from __future__ import annotations

import unittest

import torch

from train import (
    assert_strict_nonlogit_calibration_bias_zero,
    redact_hidden_targets,
    resolve_protocol_seeds,
    should_search_legacy_calibration_bias,
)
from utils.metrics import build_nested_label_protocol


class StrictProtocolTests(unittest.TestCase):
    def test_inner_surrogate_is_deterministic_and_outer_unseen_is_forbidden(self) -> None:
        outer_seen = [f"seen_{idx}" for idx in range(10)]
        outer_unseen = ["outer_u0", "outer_u1"]
        first = build_nested_label_protocol(
            outer_seen,
            outer_unseen,
            inner_seed=17,
            surrogate_fraction=0.2,
        )
        second = build_nested_label_protocol(
            outer_seen,
            outer_unseen,
            inner_seed=17,
            surrogate_fraction=0.2,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["surrogate_unseen_labels"]), 2)
        self.assertTrue(set(first["surrogate_unseen_labels"]).issubset(set(outer_seen)))
        self.assertFalse(set(first["selection_unseen_labels"]) & set(outer_unseen))
        self.assertEqual(
            set(first["training_hidden_labels"]),
            set(outer_unseen) | set(first["surrogate_unseen_labels"]),
        )

    def test_legacy_outer_unseen_validation_requires_explicit_switch(self) -> None:
        strict = build_nested_label_protocol(["a", "b", "c"], ["u"], inner_seed=3)
        legacy = build_nested_label_protocol(
            ["a", "b", "c"],
            ["u"],
            inner_seed=3,
            legacy_outer_unseen_validation=True,
        )
        self.assertEqual(strict["mode"], "strict_inner_surrogate")
        self.assertEqual(legacy["mode"], "legacy_outer_unseen_validation")
        self.assertEqual(legacy["selection_unseen_labels"], ["u"])

    def test_seed_zero_is_not_replaced_by_outer_seed(self) -> None:
        seeds = resolve_protocol_seeds(
            {"seed": 42},
            {
                "data": {"split_seed": 99},
                "training": {"training_seed": 0, "inner_split_seed": 0},
            },
        )
        self.assertEqual(seeds["outer_split_seed"], 42)
        self.assertEqual(seeds["training_seed"], 0)
        self.assertEqual(seeds["inner_split_seed"], 0)

    def test_hidden_annotations_are_erased_before_loss(self) -> None:
        targets = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        visible_mask = torch.tensor([1.0, 0.0, 1.0])
        redacted = redact_hidden_targets(targets, visible_mask)
        self.assertTrue(torch.equal(redacted[:, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(redacted[:, [0, 2]], targets[:, [0, 2]]))

    def test_strict_nonlogit_disables_legacy_calibration_bias(self) -> None:
        self.assertFalse(should_search_legacy_calibration_bias("zs_clip", False))
        self.assertTrue(should_search_legacy_calibration_bias("zs_clip", True))
        assert_strict_nonlogit_calibration_bias_zero("zs_clip", False, 0.0)
        with self.assertRaisesRegex(AssertionError, "calibration_bias=0"):
            assert_strict_nonlogit_calibration_bias_zero("zs_clip", False, 4.2)
        assert_strict_nonlogit_calibration_bias_zero("zs_clip", True, 4.2)


if __name__ == "__main__":
    unittest.main()
