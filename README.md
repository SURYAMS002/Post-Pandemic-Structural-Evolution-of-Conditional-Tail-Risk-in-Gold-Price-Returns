# Post-Pandemic Structural Evolution of Conditional Tail Risk in Gold Price Returns

## A Regime-Based GARCH–EVT Framework

This project analyzes the structural evolution of gold price volatility and tail risk using a hybrid financial risk modeling framework that combines Markov Switching Models with Extreme Value Theory (EVT).

The study focuses on the SPDR Gold Shares ETF (GLD) daily closing prices from 2016 to 2026 to understand how conditional tail risk changed across the pre-pandemic, pandemic, and post-pandemic periods.

## Project Objective

Traditional volatility models such as EWMA and Historical Simulation often fail to capture structural regime changes and extreme downside risks during crisis periods.

This project aims to:

- Detect structural variance breaks in gold returns
- Classify economic regimes across different market conditions
- Compare baseline and regime-aware Value-at-Risk (VaR) models
- Analyze left-tail and right-tail asymmetry using EVT
- Build an interactive Streamlit dashboard for real-time scenario analysis

## Methodology

### Dataset

- Asset: GLD ETF (SPDR Gold Shares)
- Source: Yahoo Finance
- Period: 2016-01-29 to 2026-01-23
- Total Trading Days: 2,510

### Models Used

### Baseline Model

- EWMA Volatility Model (λ = 0.94)
- 250-Day Historical Simulation VaR

### Proposed Model

- Two-State Markov Switching Regression
- Peaks-Over-Threshold (POT)
- Generalized Pareto Distribution (GPD)
- EVT-Based Tail Risk Estimation

## Model Performance and Accuracy

### EWMA + Historical VaR

- Breach Rate: 1.194%
- Average Breach Severity: 0.006897
- Clustering Ratio: 0.037037
- Quantile Loss: 0.000315

### Markov Switching + EVT

- Breach Rate: 4.104%
- Average Breach Severity: 0.002722
- Clustering Ratio: 0.019417
- Quantile Loss: 0.000274

### Performance Improvement

- Quantile Loss improved by 13%
- Average Breach Severity reduced by nearly 60%
- Better responsiveness to sudden volatility regime shifts
- Improved tail-risk estimation during crisis periods

## Key Findings

- Two major structural breaks detected:
  - 2020-02-21 (COVID-19 pandemic onset)
  - 2025-04-04 (post-pandemic high-volatility regime)

- Three major volatility regimes identified:
  - Pre-Pandemic
  - Pandemic and Recovery
  - Post-Pandemic

- Post-pandemic variance is 3.3× higher than pre-pandemic variance

- Pandemic regime showed strong downside tail asymmetry:
  - Left Tail ξ = −0.578
  - Right Tail ξ = −0.004

## Streamlit Dashboard Features

The interactive dashboard includes:

- Regime Detection Visualization
- Competing Model Comparison
- Tail Evolution Analysis
- Counterfactual What-If Simulator
- Generated Result Audit Panel

The dashboard helps users simulate scenarios such as:

“What if the pandemic shock had not occurred?”

and compare actual vs counterfactual gold price trajectories.

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Plotly
- Streamlit
- Yahoo Finance API

## Project Type

Academic Research Project  
Risk and Fraud Analytics (MGT3013)

Integrated M.Tech – Computer Science and Business Analytics  
:contentReference[oaicite:0]{index=0}

## Research Contribution

This project contributes a regime-comparative empirical framework for post-pandemic gold tail risk analysis and provides practical insights for:

- Institutional Investors
- ETF Risk Managers
- Portfolio Hedging Teams
- Central Bank Reserve Managers

It demonstrates that static full-sample models can significantly underestimate current gold market risk, while regime-aware models provide stronger decision support for stress testing and VaR estimation.
