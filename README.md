# Explainable ML-Driven Portfolio Optimization — DSS Pipeline

Source code for the paper *“Explainable Machine Learning-Driven Portfolio
Optimization: A Comparative Analysis of Large-Cap and Mid-Cap U.S. Equity.”*

A modular Decision Support System (DSS) that integrates machine-learning return
forecasting (Random Forest, XGBoost, LSTM), Ledoit–Wolf-regularized Markowitz
optimization, and SHAP explainability, evaluated net of transaction costs across
U.S. large-cap and mid-cap equity universes (2015–2024).

## Quick start

```bash
pip install -r requirements.txt
# Edit config.py to choose the universe ("large" or "mid") and parameters
python run_all.py
```

Outputs are written to `outputs/` (large-cap) or `outputs_mid/` (mid-cap):
tables as `*.csv`, figures in `figures/`.

## Configuration

All experiment settings live in `config.py`: ticker universes, date range,
walk-forward split, forecast horizon, risk aversion (λ), max weight, and
transaction cost. Set `UNIVERSE = "large"` or `"mid"` and the matching
`OUTPUT_DIR` before running.

## Pipeline

| Script | Purpose |
|---|---|
| `01_data.py` | Download & clean adjusted prices (Yahoo Finance / yfinance) |
| `02_features.py` | Engineer 25 price-based technical indicators |
| `03_train_models.py` | Walk-forward training of Linear/RF/XGBoost forecasters |
| `03b_lstm.py` | Optional LSTM forecaster (needs TensorFlow) |
| `04_backtest.py` | Portfolio backtest: Equal-Weight, MVO, Risk Parity, ML-DSS |
| `05_shap.py` | SHAP feature-attribution analysis for XGBoost |
| `06_figures.py` | Generate all figures (300 DPI, print-ready fonts) |
| `07_robustness.py` | Sensitivity to λ, transaction cost, and max-weight cap |

## Reviewer-response additions

The following scripts were added to address the peer-review comments; they run on
the forecasts already saved in `outputs/predictions/` (no retraining required):

| Script | Purpose | Output |
|---|---|---|
| `08_turnover_aware.py` | Transaction-cost-aware optimizer with an explicit **L1 turnover penalty** `γ` added to the mean–variance objective (`γ = 0` recovers the turnover-blind baseline). Sweeps `γ` for XGBoost and Random Forest. | `turnover_aware_table.csv` |
| `09_significance.py` | **Jobson–Korkie** Sharpe-difference test with the **Memmel (2003)** correction on the backtested return series. | `sharpe_significance.csv` |
| `10_rebalance_sensitivity.py` | **Rebalancing-frequency** sensitivity (monthly vs. quarterly) at several cost levels. | `rebalance_sensitivity.csv` |

Run them per universe by setting `config.py` (or overriding `OUTPUT_DIR`), e.g.:

```bash
python scripts/08_turnover_aware.py
python scripts/09_significance.py
python scripts/10_rebalance_sensitivity.py
```

### Key findings from the added analyses (large-cap)

- Adding the L1 turnover penalty raises net Sharpe (XGBoost 0.06 → 0.18,
  Random Forest 0.33 → 0.44) by cutting turnover.
- Quarterly (vs. monthly) rebalancing halves turnover (≈405% → 219%) and raises
  the XGBoost net Sharpe from 0.06 to 0.52.
- The underperformance of turnover-blind ML strategies vs. equal-weight is
  statistically significant (XGBoost p < 0.001; Random Forest p = 0.043).

These reinforce the paper’s central conclusion: the binding constraint on
economic performance is portfolio turnover / implementation cost, not forecast
accuracy.

## Data & reproducibility

Historical prices are downloaded from Yahoo Finance via `yfinance` (≥ 0.2.28) and
are freely accessible subject to Yahoo Finance terms of use. Random seeds are
fixed where applicable to support reproducibility.

## Note on repository name

This repository was created under the author’s previous surname (“Mautova”) prior
to a legal name change. The author’s current name is **Amangeldikyzy**; both refer
to the same individual.
