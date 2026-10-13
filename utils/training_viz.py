from __future__ import annotations

import json
import os
from pathlib import Path
import time
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def _atomic_write_path(target_path: Path, writer: Callable[[Path], None]) -> None:
    """Write a sibling artifact and atomically replace the public path.

    Windows indexers and antivirus scanners can briefly lock files that are
    rewritten every epoch. Retrying the final replacement avoids truncating a
    valid artifact and keeps training deterministic.
    """
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(
        f".{target_path.stem}.{os.getpid()}.{time.time_ns()}.tmp{target_path.suffix}"
    )
    try:
        writer(temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    max_replace_attempts = 20
    for attempt in range(max_replace_attempts):
        try:
            os.replace(temporary_path, target_path)
            return
        except PermissionError:
            if attempt + 1 == max_replace_attempts:
                raise
            time.sleep(min(0.05 * (2**attempt), 1.0))


class TrainingVisualizer:
    def __init__(
        self,
        output_dir: str | Path,
        enable_tensorboard: bool = True,
        run_name: str = "run",
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"
        self.latest_metrics_path = self.output_dir / "latest_metrics.json"
        self.curves_path = self.output_dir / "training_curves.png"
        self.history: list[dict] = []

        self.writer = None
        if enable_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard" / run_name))

    def write_run_info(self, payload: dict) -> None:
        run_info_path = self.output_dir / "run_info.json"
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        _atomic_write_path(run_info_path, lambda path: path.write_text(text, encoding="utf-8"))

    def log_batch(self, global_step: int, metrics: dict[str, float]) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if value is None:
                continue
            self.writer.add_scalar(f"batch/{key}", float(value), global_step)

    def log_epoch(self, metrics: dict[str, float | int | str | None]) -> None:
        self.history.append(dict(metrics))
        frame = pd.DataFrame(self.history)
        _atomic_write_path(
            self.metrics_path,
            lambda path: frame.to_csv(path, index=False, encoding="utf-8-sig"),
        )
        latest_metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
        _atomic_write_path(
            self.latest_metrics_path,
            lambda path: path.write_text(latest_metrics_text, encoding="utf-8"),
        )
        self._save_curves(frame)

        if self.writer is not None:
            epoch = int(metrics["epoch"])
            for key, value in metrics.items():
                if key == "epoch" or value is None:
                    continue
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/{key}", float(value), epoch)
            self.writer.flush()

    def _save_curves(self, frame: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.reshape(-1)

        axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", color="#1f77b4")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        val_metric_columns = [
            ("val_mAP", "overall", "#2ca02c"),
            ("val_seen_mAP", "seen", "#1f77b4"),
            ("val_unseen_mAP", "unseen", "#d62728"),
            ("val_harmonic_mean", "harmonic_mean", "#9467bd"),
        ]
        available_val_columns = [item for item in val_metric_columns if item[0] in frame.columns and frame[item[0]].notna().any()]
        if available_val_columns:
            for column_name, legend_name, color in available_val_columns:
                valid = frame.dropna(subset=[column_name])
                axes[1].plot(valid["epoch"], valid[column_name], marker="o", label=legend_name, color=color)
            axes[1].set_title("Validation Metrics")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].set_title("Validation Metrics")
            axes[1].text(0.5, 0.5, "No validation points yet", ha="center", va="center")
            axes[1].set_axis_off()

        axes[2].plot(frame["epoch"], frame["lr_main"], marker="o", label="main", color="#ff7f0e")
        if "lr_backbone" in frame.columns:
            axes[2].plot(frame["epoch"], frame["lr_backbone"], marker="o", label="backbone", color="#9467bd")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        axes[3].plot(frame["epoch"], frame["epoch_time_sec"], marker="o", color="#8c564b")
        axes[3].set_title("Epoch Time")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Seconds")
        axes[3].grid(True, alpha=0.3)

        try:
            fig.tight_layout()
            _atomic_write_path(
                self.curves_path,
                lambda path: fig.savefig(path, dpi=180, format="png"),
            )
        finally:
            plt.close(fig)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
