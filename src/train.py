import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import DATASETS, N_FOLDS, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_dataset
from src.evaluate import plot_comparison
from src.models import get_models
from src.pipeline import build_pipeline


RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def cv_summary(pipeline, X_train, y_train):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
    )
    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        summary[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
        summary[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
    return summary


def save_confusion_matrix(model, X_test, y_test, disease):
    y_pred = model.predict(X_test)
    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(cm.values, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.iat[i, j]), va="center", ha="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Risk", "Risk"])
    ax.set_yticklabels(["No Risk", "Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{disease.upper()} Confusion Matrix")
    plt.tight_layout()
    fig.savefig(f"results/confusion_{disease}.png", dpi=150)
    plt.close(fig)


def get_k_for_encoding(X_train):
    encoded_count = build_pipeline(X_train, get_models()["Logistic Regression"], k="all").named_steps["preprocessor"].fit(X_train).get_feature_names_out().size
    return max(1, math.ceil(0.75 * encoded_count))


def train_for_disease(disease_name):
    X, y = load_dataset(disease_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model_map = get_models()
    cv_rows = []
    for model_name, model in model_map.items():
        pipeline = build_pipeline(X_train, model, k="all")
        metrics = cv_summary(pipeline, X_train, y_train)
        row = {"model": model_name, **metrics}
        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows)
    cv_df = cv_df.sort_values("f1_mean", ascending=False, ignore_index=True)
    cv_df.to_csv(f"results/cv_results_{disease_name}.csv", index=False)

    best_model_name = cv_df.iloc[0]["model"]
    best_model = model_map[best_model_name]
    best_f1 = float(cv_df.iloc[0]["f1_mean"])
    best_recall = float(cv_df.iloc[0]["recall_mean"])

    recall_alert = None
    for _, row in cv_df.iloc[1:].iterrows():
        if row["recall_mean"] - best_recall > 0.03 and (best_f1 - row["f1_mean"]) <= 0.02:
            recall_alert = row["model"]
            break

    selected_pipeline = build_pipeline(X_train, best_model, k="all")
    selected_pipeline.fit(X_train, y_train)

    k_candidate = max(1, math.ceil(0.75 * selected_pipeline.named_steps["preprocessor"].get_feature_names_out().size))
    second_pipeline = build_pipeline(X_train, best_model, k=k_candidate)
    second_summary = cv_summary(second_pipeline, X_train, y_train)
    second_f1 = second_summary["f1_mean"]
    if second_f1 > best_f1:
        best_model_name = f"{best_model_name} (k={k_candidate})"
        best_model = best_model
        best_f1 = second_f1
        selected_pipeline = second_pipeline

    selected_pipeline.fit(X_train, y_train)
    y_pred = selected_pipeline.predict(X_test)
    test_scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    save_confusion_matrix(selected_pipeline, X_test, y_test, disease_name)

    test_row = {
        "disease": disease_name,
        "model": best_model_name,
        **test_scores,
    }
    test_df = pd.DataFrame([test_row])
    if Path("results/test_results.csv").exists():
        prev = pd.read_csv("results/test_results.csv")
        test_df = pd.concat([prev, test_df], ignore_index=True).drop_duplicates(subset=["disease"], keep="last")
    test_df.to_csv("results/test_results.csv", index=False)

    meta = {
        "disease_label": DATASETS[disease_name]["label"],
        "winning_model_name": best_model_name,
        "expected_input_columns": list(X.columns),
        "column_types": {col: ("numeric" if pd.api.types.is_numeric_dtype(X[col]) else "categorical") for col in X.columns},
        "numeric_summary": {
            col: {
                "min": float(X[col].min()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                "median": float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                "max": float(X[col].max()) if pd.api.types.is_numeric_dtype(X[col]) else None,
            }
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        },
        "categorical_values": {
            col: sorted(X[col].dropna().astype(str).unique().tolist())
            for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        },
        "dataset_rows": int(len(X)),
        "dataset_columns": int(len(X.columns)),
        "test_scores": test_scores,
        "recall_check": recall_alert,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    with open(f"models/{disease_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(selected_pipeline, f"models/{disease_name}_model.pkl")

    print(f"\n{disease_name.upper()} | best model: {best_model_name} | F1: {best_f1:.4f} | recall: {test_scores['recall']:.4f}")
    if recall_alert:
        print(f"Recall alert: {recall_alert} had higher recall within 0.02 F1 of the best model.")
    print(cv_df[["model", "f1_mean", "recall_mean"]].to_string(index=False))
    return test_scores, best_model_name


if __name__ == "__main__":
    all_results = []
    for disease in DATASETS:
        result, model_name = train_for_disease(disease)
        all_results.append({"disease": disease, "model": model_name, **result})
    summary_df = pd.DataFrame(all_results)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    plot_comparison(results_dir="results")
