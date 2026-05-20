from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.infrastructure.artifact_manager import ArtifactManager


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def print_model_info(model_info: Mapping[str, Any]) -> None:
    print_section("Model (checkpoint)")
    keys = [
        "vocab_size",
        "embedding_dim",
        "hidden_size",
        "num_layers",
        "dropout_rate",
        "dense_hidden",
        "total_parameters",
        "trainable_parameters",
    ]
    for k in keys:
        if k in model_info:
            print(f"  {k}: {model_info[k]}")


def print_test_metrics(metrics: Mapping[str, Any]) -> None:
    print_section("Test metrics")
    print(
        f"  precision: {metrics.get('precision', 0):.4f}  "
        f"recall: {metrics.get('recall', 0):.4f}  "
        f"f1: {metrics.get('f1', 0):.4f}"
    )
    print(
        f"  samples: {metrics.get('num_samples', '—')}  "
        f"threshold: {metrics.get('threshold', '—')}"
    )


def print_confusion_matrix_text(metrics: Mapping[str, Any]) -> None:
    tp = int(metrics.get("tp", 0))
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    print_section("Confusion matrix (true rows, pred columns; Ham=0 Spam=1)")
    print(f"                 pred Ham    pred Spam")
    print(f"  true Ham (0)   {tn:6d}    {fp:6d}")
    print(f"  true Spam (1)  {fn:6d}    {tp:6d}")


def save_training_history_figure(
    history: Mapping[str, Any],
    out_path: Any,
    artifact_manager: ArtifactManager,
    title: str = "Training",
) -> None:
    train_losses = history.get("train_losses") or []
    val_losses = history.get("val_losses") or []
    val_acc = history.get("val_accuracies") or []
    if not train_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_losses, label="train loss", color="C0")
    ax.plot(epochs, val_losses, label="val loss", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(f"{title} — loss")
    if val_acc:
        ax2 = ax.twinx()
        ax2.plot(epochs, val_acc, label="val accuracy", color="C2", alpha=0.9)
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="best")
    else:
        ax.legend(loc="upper right")
    fig.tight_layout()
    artifact_manager.save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_figure(
    metrics: Mapping[str, Any],
    out_path: Any,
    artifact_manager: ArtifactManager,
    title: str = "Confusion matrix (test)",
) -> None:
    tp = float(metrics.get("tp", 0))
    tn = float(metrics.get("tn", 0))
    fp = float(metrics.get("fp", 0))
    fn = float(metrics.get("fn", 0))
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Ham (0)", "Pred Spam (1)"])
    ax.set_yticklabels(["True Ham (0)", "True Spam (1)"])
    fmt = int
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                fmt(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    artifact_manager.save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def save_learning_curve_metrics_figure(
    curve_data: list,
    out_path: Any,
    artifact_manager: ArtifactManager,
) -> None:
    if not curve_data:
        return
    sizes = [t[0] for t in curve_data]
    precs = [t[1] for t in curve_data]
    recs = [t[2] for t in curve_data]
    f1s = [t[3] for t in curve_data]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, precs, marker="o", label="precision")
    ax.plot(sizes, recs, marker="s", label="recall")
    ax.plot(sizes, f1s, marker="^", label="F1")
    ax.set_xlabel("Corpus size")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Learning curve — precision / recall / F1")
    fig.tight_layout()
    artifact_manager.save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def print_full_experiment_report(
    metrics: Mapping[str, Any],
    model_info: Optional[Mapping[str, Any]] = None,
) -> None:
    if model_info:
        print_model_info(model_info)
    print_test_metrics(metrics)
    print_confusion_matrix_text(metrics)
