"""Model utilities for training, saving, loading, and predicting.

This module reuses `build_pipeline` and the synthetic data generator from
`src.train` and provides helpers that the Streamlit app can call.
"""
import os
import joblib
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.train import build_pipeline, generate_synthetic_churn

import json


def infer_feature_types(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    numeric_features = [c for c in features if c.startswith("num_") or pd.api.types.is_numeric_dtype(df[c]) for c in [c]]
    # Fallback: treat object and category dtype as categorical
    categorical_features = [c for c in features if c not in numeric_features]
    return numeric_features, categorical_features


def train_model_from_df(df: pd.DataFrame, target: str = "churn", random_state: int = 42,
                        test_size: float = 0.2):
    """Train a pipeline on a provided DataFrame.

    Returns: (pipeline, metrics_dict, X_test, y_test)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    # Infer feature types
    # Use a simple heuristic: numeric dtypes are numeric features.
    features = [c for c in df.columns if c != target]
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in features if c not in numeric_features]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique())==2 else None
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return pipeline, metrics, X_test, y_test


def save_model(pipeline, models_dir: str = "models", prefix: str = "model") -> str:
    os.makedirs(models_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}-{timestamp}.pkl"
    path = os.path.join(models_dir, filename)
    joblib.dump(pipeline, path)

    # Save metadata to an index for a simple registry
    index_path = os.path.join(models_dir, "index.json")
    metadata = {
        "filename": filename,
        "path": path,
        "saved_at": timestamp,
        "n_features": getattr(pipeline, "n_features_in_", None),
    }
    try:
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                idx = json.load(f)
        else:
            idx = []
    except Exception:
        idx = []

    idx.append(metadata)
    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)

    return path


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_df(pipeline, df: pd.DataFrame):
    preds = pipeline.predict(df)
    proba = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline, "predict_proba") else None
    result = df.copy()
    result["pred_churn"] = preds
    if proba is not None:
        result["pred_churn_proba"] = np.round(proba, 4)
    return result


def list_saved_models(models_dir: str = "models"):
    index_path = os.path.join(models_dir, "index.json")
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def delete_saved_model(filename: str, models_dir: str = "models") -> bool:
    index_path = os.path.join(models_dir, "index.json")
    models_path = os.path.join(models_dir, filename)
    ok = False
    if os.path.exists(models_path):
        try:
            os.remove(models_path)
            ok = True
        except Exception:
            ok = False

    # Update index
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                idx = json.load(f)
            idx = [m for m in idx if m.get("filename") != filename]
            with open(index_path, "w") as f:
                json.dump(idx, f, indent=2)
        except Exception:
            pass

    return ok


def demo_synthetic(n_samples: int = 1000):
    return generate_synthetic_churn(n_samples=n_samples)
