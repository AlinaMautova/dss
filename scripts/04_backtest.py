"""
scripts/04_backtest.py
----------------------
Walk-forward portfolio backtesting comparing:
  1. Equal-Weight (1/N)
  2. Classical MVO (historical inputs)
  3. Risk Parity
  4. ML-DSS (Random Forest)
  5. ML-DSS (XGBoost)
  6. ML-DSS (LSTM)  – if lstm_predictions.pkl exists

Uses Markowitz mean-variance optimisation with constraints.

OUTPUTS:
  outputs/portfolio_returns.csv   – daily portfolio returns per strategy
  outputs/portfolio_weights.pkl   – weight series per strategy
  outputs/performance_table.csv   – Table 3 in the article
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (OUTPUT_DIR, INITIAL_TRAIN_YEARS, REBALANCE_FREQ,
                    RISK_AVERSION, MAX_WEIGHT, TRANSACTION_COST)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

returns   = pd.read_csv(f"{OUTPUT_DIR}/returns.csv", index_col=0, parse_dates=True)
TICKERS   = returns.columns.tolist()
N         = len(TICKERS)

with open(f"{OUTPUT_DIR}/predictions/tree_predictions.pkl", "rb") as f:
    tree_preds = pickle.load(f)

lstm_path = f"{OUTPUT_DIR}/predictions/lstm_predictions.pkl"
LSTM_AVAILABLE = os.path.exists(lstm_path)
if LSTM_AVAILABLE:
    with open(lstm_path, "rb") as f:
        lstm_preds_raw = pickle.load(f)

# ── Rebalancing dates ────────────────────────────────────────
split_start  = returns.index[0] + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
rb_dates     = pd.date_range(split_start, returns.index[-1], freq=REBALANCE_FREQ)

# ── Optimisation helpers ─────────────────────────────────────
def mvo_weights(mu, sigma, lam=RISK_AVERSION, w_max=MAX_WEIGHT):
    """Mean-variance optimisation with budget, long-only, and max-weight constraints."""
    n = len(mu)
    def neg_utility(w):
        return -(w @ mu - 0.5 * lam * w @ sigma @ w)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, w_max)] * n
    w0 = np.ones(n) / n
    res = minimize(neg_utility, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    if res.success:
        w = np.clip(res.x, 0, w_max)
        return w / w.sum()
    return w0   # fallback: equal weight

def risk_parity_weights(sigma):
    n = sigma.shape[0]
    def risk_budget_obj(w):
        port_vol = np.sqrt(w @ sigma @ w)
        mrc = sigma @ w / port_vol
        rc  = w * mrc
        return np.sum((rc - rc.mean()) ** 2)
    w0 = np.ones(n) / n
    bounds = [(0.01, 0.5)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(risk_budget_obj, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    if res.success:
        w = np.clip(res.x, 0, 0.5)
        return w / w.sum()
    return w0

def get_ml_expected_returns(pred_dict, rb_date, strategy_name):
    """Extract forward expected returns from prediction dict."""
    mu = np.zeros(N)
    for j, ticker in enumerate(TICKERS):
        key = (rb_date.strftime("%Y-%m-%d"), ticker)
        if key in pred_dict:
            preds = pred_dict[key]["pred"]
            mu[j] = np.mean(preds) if preds else 0.0
    return mu

# ── Backtesting loop ─────────────────────────────────────────
strategies = {
    "EqualWeight":  [],
    "ClassicalMVO": [],
    "RiskParity":   [],
    "RF":           [],
    "XGBoost":      [],
}
if LSTM_AVAILABLE:
    strategies["LSTM"] = []

weight_history = {k: {} for k in strategies}
prev_weights   = {k: np.ones(N) / N for k in strategies}

daily_rets = {}
for i, rb_date in enumerate(rb_dates[:-1]):
    next_rb = rb_dates[i + 1]

    # Get period return data up to rb_date (training window)
    train_rets = returns[returns.index <= rb_date]

    # Estimate covariance (Ledoit-Wolf shrinkage)
    if len(train_rets) > N + 20:
        lw = LedoitWolf().fit(train_rets.values)
        sigma = lw.covariance_
    else:
        sigma = np.diag(train_rets.var().values)

    # Historical mean returns (annualised)
    mu_hist = train_rets.mean().values * 252

    # ── Compute weights for each strategy ───────────────────
    w_eq = np.ones(N) / N
    w_mvo = mvo_weights(mu_hist, sigma)
    w_rp  = risk_parity_weights(sigma)
    w_rf  = mvo_weights(
        get_ml_expected_returns(tree_preds["RandomForest"], rb_date, "RF"),
        sigma
    )
    w_xgb = mvo_weights(
        get_ml_expected_returns(tree_preds["XGBoost"], rb_date, "XGB"),
        sigma
    )

    new_weights = {
        "EqualWeight":  w_eq,
        "ClassicalMVO": w_mvo,
        "RiskParity":   w_rp,
        "RF":           w_rf,
        "XGBoost":      w_xgb,
    }
    if LSTM_AVAILABLE:
        new_weights["LSTM"] = mvo_weights(
            get_ml_expected_returns(lstm_preds_raw, rb_date, "LSTM"), sigma
        )

    for name, w in new_weights.items():
        weight_history[name][rb_date] = w

    # ── Apply weights to next-period daily returns ───────────
    period_rets = returns[(returns.index > rb_date) & (returns.index <= next_rb)]

    for name, w in new_weights.items():
        # Transaction cost: proportional to turnover
        turnover = np.sum(np.abs(w - prev_weights[name])) / 2
        tc_cost  = TRANSACTION_COST * turnover

        port_ret = period_rets.values @ w
        port_ret[0] -= tc_cost   # deduct TC at first day of period

        for date, r in zip(period_rets.index, port_ret):
            daily_rets.setdefault(date, {})[name] = r

        prev_weights[name] = w.copy()

port_df = pd.DataFrame(daily_rets).T.sort_index()
port_df.to_csv(f"{OUTPUT_DIR}/portfolio_returns.csv")

with open(f"{OUTPUT_DIR}/portfolio_weights.pkl", "wb") as f:
    pickle.dump(weight_history, f)

# ── Performance metrics ──────────────────────────────────────
rf_rate = 0.04 / 252   # approximate daily risk-free rate

perf_rows = []
for strat in port_df.columns:
    r = port_df[strat].dropna()
    ann_ret  = r.mean() * 252
    ann_vol  = r.std()  * np.sqrt(252)
    excess   = r - rf_rate
    sharpe   = excess.mean() / r.std() * np.sqrt(252)

    # Max drawdown
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    mdd  = dd.min()

    # Annual turnover (approximate)
    wh = weight_history[strat]
    dates = sorted(wh.keys())
    turnovers = []
    for j in range(1, len(dates)):
        turnovers.append(np.sum(np.abs(wh[dates[j]] - wh[dates[j-1]])) / 2)
    ann_turnover = np.mean(turnovers) * 12 if turnovers else 0

    perf_rows.append({
        "Strategy":               strat,
        "Ann. Return (%)":        round(ann_ret * 100, 2),
        "Ann. Volatility (%)":    round(ann_vol * 100, 2),
        "Sharpe Ratio":           round(sharpe, 3),
        "Max Drawdown (%)":       round(mdd * 100, 2),
        "Ann. Turnover (%)":      round(ann_turnover * 100, 1),
    })

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(f"{OUTPUT_DIR}/performance_table.csv", index=False)

print("\n── PERFORMANCE TABLE (→ Table 3 in article) ──")
print(perf_df.to_string(index=False))
print(f"\n  Saved to {OUTPUT_DIR}/performance_table.csv")
print("  *** INSERT these values into Table 3 of the article ***")
