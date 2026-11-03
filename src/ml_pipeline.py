"""Reusable data and model functions for the tumor classification project."""

from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_NAMES = ["malignant", "benign"]


def load_data():
    """Return features, target, feature names and the sklearn dataset object."""
    dataset = load_breast_cancer()
    return dataset.data, dataset.target, dataset.feature_names, dataset


def build_model():
    """Create a preprocessing + classifier pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])


def train_and_evaluate():
    X, y, feature_names, dataset = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "features": int(X.shape[1]),
    }

    return {
        "model": model,
        "metrics": metrics,
        "report": classification_report(
            y_test, predictions, target_names=TARGET_NAMES
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "y_test": y_test,
        "probabilities": probabilities,
        "feature_names": feature_names,
        "dataset": dataset,
    }


def save_artifacts(result, model_path="models/tumor_classifier.joblib",
                   metrics_path="reports/metrics.json",
                   report_path="reports/classification_report.txt"):
    """Save trained model and human-readable evaluation outputs."""
    for path in (model_path, metrics_path, report_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["model"], model_path)
    Path(metrics_path).write_text(
        json.dumps(result["metrics"], indent=4), encoding="utf-8"
    )
    Path(report_path).write_text(result["report"], encoding="utf-8")
