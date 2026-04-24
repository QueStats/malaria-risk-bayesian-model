"""Baseline logistic regression model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def fit_baseline_logistic(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Fit a standard logistic regression model using statsmodels."""
    X_with_intercept = sm.add_constant(X)
    model = sm.Logit(y, X_with_intercept)
    result = model.fit(disp=False)

    names = ["intercept"] + feature_names
    summary = pd.DataFrame(
        {
            "term": names,
            "coef": result.params,
            "std_error": result.bse,
            "p_value": result.pvalues,
        }
    )

    pred_prob = result.predict(X_with_intercept)
    metrics = {
        "auc": roc_auc_score(y, pred_prob),
        "accuracy": accuracy_score(y, pred_prob >= 0.5),
    }

    return {"model": result, "summary": summary, "metrics": metrics, "pred_prob": pred_prob}


def fit_train_test_logistic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Fit a train/test baseline model for a more honest benchmark."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_i = sm.add_constant(X_train)
    X_test_i = sm.add_constant(X_test)

    model = sm.Logit(y_train, X_train_i)
    result = model.fit(disp=False)
    pred_prob = result.predict(X_test_i)

    metrics = {
        "test_auc": roc_auc_score(y_test, pred_prob),
        "test_accuracy": accuracy_score(y_test, pred_prob >= 0.5),
    }

    names = ["intercept"] + feature_names
    summary = pd.DataFrame(
        {
            "term": names,
            "coef": result.params,
            "std_error": result.bse,
            "p_value": result.pvalues,
        }
    )

    return {"model": result, "summary": summary, "metrics": metrics, "pred_prob": pred_prob}
