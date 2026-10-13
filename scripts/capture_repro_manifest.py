from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
    "torchvision",
    "open-clip-torch",
]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the exact environment and immutable inputs for a strict run.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--protocol_config", required=True)
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--source_files", nargs="*", default=[])
    return parser.parse_args()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _git_revision(root: Path) -> dict[str, str | bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def capture_repro_manifest(
    *,
    output: str | Path,
    protocol_config: str | Path,
    artifacts: dict[str, str | Path],
    checkpoint: str | Path | None = None,
    source_files: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict:
    """Capture hashes by semantic role so a completed run can be audited later."""
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(checkpoint).resolve() if checkpoint is not None else None
    protocol_config_path = Path(protocol_config).resolve()
    artifact_paths = {role: Path(path).resolve() for role, path in artifacts.items()}
    source_paths = [Path(path).resolve() for path in source_files]
    paths = [protocol_config_path, *artifact_paths.values(), *source_paths]
    if checkpoint_path is not None:
        paths.append(checkpoint_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest inputs do not exist: {missing}")

    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "created_by": "scripts/capture_repro_manifest.py",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "packages": _package_versions(),
        "torch_runtime": {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else [],
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": _git_revision(root),
        "protocol_config": {
            "path": str(protocol_config_path),
            "size_bytes": protocol_config_path.stat().st_size,
            "sha256": sha256_file(protocol_config_path),
        },
        "checkpoint": (
            {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
            }
            if checkpoint_path is not None
            else None
        ),
        "artifacts": {
            role: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for role, path in artifact_paths.items()
        },
        "source_files": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        },
        "inputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths
        },
    }
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    positional_artifacts = {
        f"artifact_{index:03d}": path for index, path in enumerate(args.artifacts)
    }
    capture_repro_manifest(
        output=args.output,
        checkpoint=args.checkpoint,
        protocol_config=args.protocol_config,
        artifacts=positional_artifacts,
        source_files=args.source_files,
    )


if __name__ == "__main__":
    main()
