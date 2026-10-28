import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DATASETS


def plot_confusion(model, X_test, y_test, disease, title):
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Risk", "Risk"])
    ax.set_yticklabels(["No Risk", "Risk"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fig.savefig(f"results/confusion_{disease}.png", dpi=150)
    plt.close(fig)


def plot_comparison(results_dir="results"):
    file_paths = [Path(results_dir) / f"cv_results_{d}.csv" for d in DATASETS]
    frames = []
    for path in file_paths:
        if path.exists():
            df = pd.read_csv(path)
            df["disease"] = path.stem.replace("cv_results_", "")
            frames.append(df)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(index="model", columns="disease", values="f1_mean", aggfunc="first")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean F1")
    ax.set_title("Mean F1 by disease and model")
    ax.legend(title="Disease")
    plt.tight_layout()
    fig.savefig(Path(results_dir) / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    plot_comparison()
    print("Comparison chart saved to results/comparison.png")


if __name__ == "__main__":
    main()
