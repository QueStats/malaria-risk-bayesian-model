"""Run the full malaria risk modeling workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baseline_logistic import fit_baseline_logistic, fit_train_test_logistic
from src.bayesian_random_effects import fit_bayesian_random_effects
from src.data_prep import load_gambia_data, make_model_matrices
from src.visualization import (
    plot_baseline_coefficients,
    plot_gambia_village_map,
    plot_village_effect_maps,
    save_trace_plot,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "gambia.csv"
BORDER_PATH = PROJECT_ROOT / "data" / "gambia_borders.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    df = load_gambia_data(DATA_PATH)

    plot_gambia_village_map(
        df,
        BORDER_PATH,
        FIGURES_DIR / "gambia_village_locations.png",
    )

    X, y, village, feature_names = make_model_matrices(df)

    baseline = fit_baseline_logistic(X, y, feature_names)
    baseline["summary"].to_csv(
        RESULTS_DIR / "baseline_logistic_coefficients.csv",
        index=False,
    )
    pd.DataFrame([baseline["metrics"]]).to_csv(
        RESULTS_DIR / "baseline_logistic_metrics.csv",
        index=False,
    )
    plot_baseline_coefficients(
        baseline["summary"],
        FIGURES_DIR / "baseline_logistic_coefficients.png",
    )

    test_baseline = fit_train_test_logistic(X, y, feature_names)
    pd.DataFrame([test_baseline["metrics"]]).to_csv(
        RESULTS_DIR / "train_test_baseline_metrics.csv",
        index=False,
    )

    bayes = fit_bayesian_random_effects(X, y, village, feature_names)
    bayes["coef_summary"].to_csv(
        RESULTS_DIR / "bayesian_coefficient_summary.csv",
    )
    bayes["village_effects"].to_csv(
        RESULTS_DIR / "village_random_effects.csv",
        index=False,
    )

    village_coords = df[["village_id", "x", "y"]].drop_duplicates()
    village_plot_df = village_coords.merge(
        bayes["village_effects"],
        on="village_id",
    )

    plot_village_effect_maps(
        village_plot_df,
        BORDER_PATH,
        FIGURES_DIR / "village_effects",
    )

    save_trace_plot(
        bayes["trace"],
        FIGURES_DIR / "bayesian_trace_plot.png",
    )

    print("Analysis complete.")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()