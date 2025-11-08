"""Hyperparameter tuning helpers for the pipeline."""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

from src.train import build_pipeline


def tune_pipeline(df: pd.DataFrame, target: str, numeric_features, categorical_features,
                  param_distributions: Dict[str, Any], n_iter: int = 20, cv: int = 3, random_state: int = 42):
    """Tune the pipeline using RandomizedSearchCV.

    param_distributions: params keyed by pipeline step names, e.g.:
      { 'classifier__n_estimators': [50,100,200], 'classifier__max_depth': [None,10,20] }
    """
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    pipeline = build_pipeline(numeric_features, categorical_features)

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y)

    return search
