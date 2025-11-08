"""Train a customer churn prediction model on synthetic data.

This script generates a synthetic dataset that resembles a churn problem with
numerical and categorical features, builds a preprocessing + RandomForest
pipeline, trains, evaluates, and saves the model to `models/model.pkl`.

Outputs printed: classification report, ROC AUC, and a small sample of predictions.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "model.pkl"))


def generate_synthetic_churn(n_samples=5000, random_state=42):
    """Generate a synthetic customer churn dataset.

    Returns a DataFrame with several numerical and categorical features and a
    binary target column named 'churn'.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.7, 0.3],
        class_sep=1.0,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])

    # Create some categorical features derived from numerical ones
    df["contract_type"] = pd.cut(df["num_0"], bins=3, labels=["month-to-month", "one-year", "two-year"]).astype(str)
    df["payment_method"] = pd.cut(df["num_1"], bins=4, labels=["electronic", "mail", "bank", "credit_card"]).astype(str)
    df["has_internet"] = (df["num_2"] > df["num_2"].median()).astype(int).astype(str)

    # Some engineered numeric features
    df["monthly_charges"] = (np.abs(df["num_3"]) * 50 + 20).round(2)
    df["tenure_months"] = (np.abs(df["num_4"]) * 12).abs().astype(int) + 1

    df["churn"] = y
    return df


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # Use sparse_output for compatibility with newer scikit-learn versions
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    return clf


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Generating synthetic data...")
    df = generate_synthetic_churn(n_samples=5000)

    target = "churn"
    features = [c for c in df.columns if c != target]

    # Choose numeric and categorical features explicitly
    numeric_features = [c for c in features if c.startswith("num_") or c in ["monthly_charges", "tenure_months"]]
    categorical_features = [c for c in features if c not in numeric_features]

    X = df[features]
    y = df[target]

    print(f"Features: {len(features)} (numeric: {len(numeric_features)}, categorical: {len(categorical_features)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        print("ROC AUC could not be computed")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    print(f"Saving model to {MODEL_PATH} ...")
    joblib.dump(pipeline, MODEL_PATH)

    # Show a small sample of predictions
    sample = X_test.iloc[:5].copy()
    sample_preds = pipeline.predict(sample)
    sample_probas = pipeline.predict_proba(sample)[:, 1]
    sample_result = sample.copy()
    sample_result["pred_churn"] = sample_preds
    sample_result["pred_churn_proba"] = np.round(sample_probas, 4)
    print("Sample predictions:")
    print(sample_result.head())


if __name__ == "__main__":
    main()
