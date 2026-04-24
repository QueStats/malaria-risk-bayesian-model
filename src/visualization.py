"""Visualization functions for model outputs."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd


def plot_baseline_coefficients(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot coefficients from the baseline logistic regression model."""
    output_path = Path(output_path)
    plot_df = summary[summary["term"] != "intercept"].copy()

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["term"], plot_df["coef"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Log-odds coefficient")
    plt.title("Baseline Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gambia_village_map(
    df: pd.DataFrame,
    border_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot Gambia border and village locations, similar to the R map."""
    border_path = Path(border_path)
    output_path = Path(output_path)

    village_df = df[["x", "y", "village_id"]].drop_duplicates()
    borders = pd.read_csv(border_path)

    plt.figure(figsize=(10, 4))
    plt.plot(borders["x"], borders["y"], color="black", linewidth=1)
    plt.scatter(village_df["x"], village_df["y"], color="black", s=25)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gambia Village Locations")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_village_effect_maps(
    df: pd.DataFrame,
    border_path: str | Path,
    output_prefix: str | Path,
) -> None:
    """Create colored R-style maps for village random effects."""
    border_path = Path(border_path)
    output_prefix = Path(output_prefix)
    borders = pd.read_csv(border_path)

    # Map 1: posterior mean random effect
    plt.figure(figsize=(10, 4))
    plt.plot(borders["x"], borders["y"], color="black", linewidth=1)

    scatter = plt.scatter(
        df["x"],
        df["y"],
        c=df["mean_random_effect"],
        cmap="viridis",
        s=35,
    )

    plt.colorbar(scatter, label="Random effect mean")
    plt.title("Random effect mean")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_prefix.parent / f"{output_prefix.name}_mean.png", dpi=300)
    plt.close()

    # Map 2: posterior probability random effect is positive
    plt.figure(figsize=(10, 4))
    plt.plot(borders["x"], borders["y"], color="black", linewidth=1)

    scatter = plt.scatter(
        df["x"],
        df["y"],
        c=df["prob_positive_effect"],
        cmap="viridis",
        s=35,
        vmin=0,
        vmax=1,
    )

    plt.colorbar(scatter, label="P(random effect > 0)")
    plt.title("Probability the random effect is positive")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_prefix.parent / f"{output_prefix.name}_prob.png", dpi=300)
    plt.close()


def save_trace_plot(trace, output_path: str | Path) -> None:
    """Save trace plots for key Bayesian model parameters."""
    output_path = Path(output_path)
    az.plot_trace(trace, var_names=["intercept", "beta", "sigma_alpha"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()