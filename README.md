# Bayesian Hierarchical Logistic Regression for Malaria Risk

This project models malaria infection risk in children using Bayesian hierarchical logistic regression. It quantifies individual risk factors while capturing unexplained variation across villages through random effects.

## Project Objective

Identify drivers of malaria infection and detect villages with elevated risk after controlling for observed covariates.

Key question:

Which factors influence malaria infection, and which villages exhibit excess risk beyond what observed variables explain?

## Data

The dataset contains observations on 2,035 children across 65 villages in The Gambia, with malaria infection recorded as a binary outcome.

Overall malaria prevalence is approximately 35.7%.

Each observation includes:
- Spatial coordinates (x, y) identifying village location
- Malaria infection status (pos ∈ {0,1})
- Demographic and environmental covariates:
  - Age (mean ≈ 1080 days)
  - Bed net usage (71.1% usage rate)
  - Treatment status (27.6% treated)
  - Vegetation index (mean ≈ 46.9)
  - Access to a health center (68.4%)

Villages are constructed by grouping observations with identical spatial coordinates, resulting in 65 distinct locations. The number of children per village ranges from 8 to 63, with an average of 31.3.

Substantial geographic variation exists in malaria risk:
- Village-level malaria rates range from 0.0 to 0.93
- Median village infection rate ≈ 0.37

Predictor relationships show clear patterns:
- Bed net use reduces malaria risk (48.5% → 30.5%)
- Treatment reduces malaria risk (39.3% → 26.4%)
- Health center access is associated with lower risk (41.0% → 33.3%)

These patterns, combined with large variation across villages, motivate the use of a hierarchical model to capture both individual-level effects and unobserved spatial heterogeneity.

## Methods

Two models are implemented and compared:

1. Baseline Logistic Regression
- Models infection probability using observed covariates
- Serves as a benchmark for predictive performance and interpretability

2. Bayesian Hierarchical Logistic Regression
- Adds village-level random intercepts
- Uses MCMC sampling to estimate posterior distributions
- Captures unobserved geographic heterogeneity in malaria risk

Model specification:

logit(P(Y_i = 1)) = X_i β + α_village[i]
α_j ~ Normal(0, σ_α)

Y_i: malaria infection status for child i  
X_i: covariates (age, bed net use, treatment status, vegetation index, health center access)  
α_j: village-level random effect  

## Key Outcomes

- Estimated malaria prevalence at 35.7% across 2,035 children in 65 villages  
- Identified substantial geographic variation, with village-level infection rates ranging from 0.0 to 0.93  
- Quantified protective effects of key interventions, including bed net use (48.5% → 30.5%) and treatment (39.3% → 26.4%)  
- Showed that baseline logistic regression fails to capture spatial heterogeneity, motivating the use of hierarchical modeling  
- Detected villages with elevated residual malaria risk after controlling for observed covariates  
- Produced full posterior distributions for all parameters, enabling uncertainty quantification and interpretable inference  

## Skills Demonstrated

- Bayesian modeling (PyMC)  
- Hierarchical / mixed-effects models  
- Logistic regression  
- MCMC sampling and diagnostics  
- Model comparison and evaluation  
- Statistical visualization and interpretation  

## Repository Structure

malaria-risk-bayesian-model/

├── data/
├── src/
│   ├── data_prep.py
│   ├── baseline_logistic.py
│   ├── bayesian_random_effects.py
│   ├── visualization.py
│   └── run_analysis.py
│
├── notebooks/
├── figures/
├── results/
│
├── requirements.txt
└── README.txt

## Outputs

- Logistic regression coefficients and performance metrics  
- Posterior summaries of Bayesian coefficients  
- Village-level random effect estimates  
- Predicted malaria probabilities  
- Model evaluation metrics (e.g., ROC-AUC)  
- Visualizations of uncertainty and geographic variation  

## How to Run

pip install -r requirements.txt

Place dataset in:
data/gambia.csv

Expected columns:
x, y, pos, age, netuse, treated, green, phc

Run:
python run_analysis.py