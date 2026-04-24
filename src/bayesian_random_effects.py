"""Bayesian random-effects logistic regression using PyMC."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def fit_bayesian_random_effects(
    X: np.ndarray,
    y: np.ndarray,
    village: np.ndarray,
    feature_names: list[str],
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.90,
    random_seed: int = 42,
) -> dict:
    """Fit a Bayesian hierarchical logistic regression model."""
    n_villages = int(village.max()) + 1
    n_features = X.shape[1]

    coords = {
        "feature": feature_names,
        "village": np.arange(n_villages),
        "obs_id": np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("obs_id", "feature"))
        village_data = pm.Data("village", village, dims="obs_id")

        intercept = pm.Normal("intercept", mu=0.0, sigma=2.5)
        beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims="feature")
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, dims="village")
        alpha = pm.Deterministic("alpha", alpha_raw * sigma_alpha, dims="village")

        eta = intercept + pm.math.dot(X_data, beta) + alpha[village_data]
        pm.Bernoulli("malaria", logit_p=eta, observed=y, dims="obs_id")

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )

    coef_summary = az.summary(
        trace,
        var_names=["intercept", "beta", "sigma_alpha"],
        hdi_prob=0.95,
    )

    village_summary = az.summary(trace, var_names=["alpha"], hdi_prob=0.95)
    village_effects = pd.DataFrame(
        {
            "village_id": np.arange(n_villages),
            "mean_random_effect": trace.posterior["alpha"].mean(dim=("chain", "draw")).values,
            "prob_positive_effect": (trace.posterior["alpha"] > 0).mean(dim=("chain", "draw")).values,
        }
    )

    return {
        "model": model,
        "trace": trace,
        "coef_summary": coef_summary,
        "village_summary": village_summary,
        "village_effects": village_effects,
    }
