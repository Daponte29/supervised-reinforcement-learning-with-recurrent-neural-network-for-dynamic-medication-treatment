from __future__ import annotations

import matplotlib.pyplot as plt


def plot_actor_critic_history(history: dict, show: bool = True):
    """Plot critic loss, actor loss, validation accuracy, and Jaccard curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["critic_loss"], marker="o", color="orange", label="Train Critic Loss")
    axes[0, 0].set(title="Critic Loss (Value Estimation Error)", xlabel="Epoch", ylabel="MSE Loss")

    axes[0, 1].plot(history["actor_loss"], marker="o", color="purple", label="Train Actor Loss")
    axes[0, 1].set(title="Actor Loss (Policy Optimization)", xlabel="Epoch", ylabel="Loss")

    axes[1, 0].plot(history["val_accuracy"], marker="o", color="green", label="Valid Accuracy")
    axes[1, 0].set(title="Validation Accuracy (Subset Match)", xlabel="Epoch", ylabel="Accuracy")

    axes[1, 1].plot(history["train_jaccard"], marker="o", color="red", label="Train Jaccard")
    axes[1, 1].plot(history["val_jaccard"], marker="o", color="blue", label="Valid Jaccard")
    axes[1, 1].set(title="Jaccard Score (IoU)", xlabel="Epoch", ylabel="Score (0-1)")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_basic_lstm_history(history: dict, show: bool = True):
    """Plot loss, validation accuracy, and Jaccard curves for the baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    axes[0].plot(history["train_loss"], marker="o", color="red", label="Train Loss")
    axes[0].plot(history["val_loss"], marker="o", color="blue", label="Valid Loss")
    axes[0].set(title="Training vs Validation Loss", xlabel="Epoch", ylabel="BCE Loss")

    axes[1].plot(history["val_accuracy"], marker="o", color="green", label="Valid Accuracy")
    axes[1].set(title="Validation Accuracy (Subset Match)", xlabel="Epoch", ylabel="Score")

    axes[2].plot(history["train_jaccard"], marker="o", color="orange", label="Train Jaccard")
    axes[2].plot(history["val_jaccard"], marker="o", color="purple", label="Valid Jaccard")
    axes[2].set(title="Jaccard Score (IoU)", xlabel="Epoch", ylabel="Score (0-1)")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
