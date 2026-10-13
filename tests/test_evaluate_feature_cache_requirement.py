from __future__ import annotations

import pytest
import torch

import evaluate
from evaluate import (
    assert_feature_cache_requirement,
    load_checkpoint_model_state,
    verify_locked_feature_cache,
)


def test_require_feature_cache_cli_flag(monkeypatch) -> None:
    digest = "a" * 64
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--require_feature_cache",
            "--strict_protocol",
            "--expected_feature_cache_sha256",
            digest,
        ],
    )
    args = evaluate.parse_args()
    assert args.require_feature_cache is True
    assert args.strict_protocol is True
    assert args.expected_feature_cache_sha256 == digest


def test_required_feature_cache_accepts_cached_pipeline() -> None:
    assert_feature_cache_requirement(
        require_feature_cache=True,
        use_feature_cache=True,
        mode="zs_clip",
    )


def test_required_feature_cache_rejects_raw_image_fallback() -> None:
    with pytest.raises(RuntimeError, match="--require_feature_cache"):
        assert_feature_cache_requirement(
            require_feature_cache=True,
            use_feature_cache=False,
            mode="zs_clip",
        )


def test_optional_feature_cache_keeps_legacy_raw_evaluation_available() -> None:
    assert_feature_cache_requirement(
        require_feature_cache=False,
        use_feature_cache=False,
        mode="baseline",
    )


def test_locked_cache_hash_is_checked_before_payload_loading(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "test.pt"
    cache.write_bytes(b"offline-feature-cache")
    expected = evaluate._sha256_file(cache)
    monkeypatch.setattr(
        evaluate,
        "resolve_evaluation_feature_cache_path",
        lambda _config, _split: cache,
    )

    assert verify_locked_feature_cache({}, split="test", expected_sha256=expected) == cache
    with pytest.raises(RuntimeError, match="SHA-256 mismatch before loading"):
        verify_locked_feature_cache({}, split="test", expected_sha256="0" * 64)


def test_strict_checkpoint_requires_exact_state_dict_while_legacy_remains_compatible() -> None:
    model = torch.nn.Linear(3, 2)
    payload = {"model_state": {**model.state_dict(), "legacy_extra": torch.ones(1)}}

    legacy_result = load_checkpoint_model_state(model, payload, strict_protocol=False)
    assert legacy_result.unexpected_keys == ["legacy_extra"]
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_checkpoint_model_state(model, payload, strict_protocol=True)
