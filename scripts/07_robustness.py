"""
scripts/07_robustness.py
------------------------
Sensitivity analysis of ML-DSS (XGBoost) Sharpe ratio
to: risk aversion lambda, transaction cost, and max weight constraint.

OUTPUT:
  outputs/robustness_table.csv  (→ Table 4 in article)
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

returns = pd.read_csv(f"{OUTPUT_DIR}/returns.csv", index_col=0, parse_dates=True)
TICKERS = returns.columns.tolist()
N       = len(TICKERS)

with open(f"{OUTPUT_DIR}/predictions/tree_predictions.pkl", "rb") as f:
    tree_preds = pickle.load(f)

split_start = returns.index[0] + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
rb_dates    = pd.date_range(split_start, returns.index[-1], freq=REBALANCE_FREQ)

def mvo_weights_params(mu, sigma, lam, w_max):
    n = len(mu)
    def neg_u(w): return -(w @ mu - 0.5 * lam * w @ sigma @ w)
    res = minimize(neg_u, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, w_max)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"ftol":1e-9, "maxiter":500})
    if res.success:
        w = np.clip(res.x, 0, w_max)
        return w / w.sum()
    return np.ones(n)/n

def run_backtest(lam, tc, w_max):
    prev_w = np.ones(N)/N
    daily  = {}
    for i, rb_date in enumerate(rb_dates[:-1]):
        next_rb = rb_dates[i+1]
        train_r = returns[returns.index <= rb_date]
        if len(train_r) > N + 20:
            lw = LedoitWolf().fit(train_r.values)
            sigma = lw.covariance_
        else:
            sigma = np.diag(train_r.var().values)
        mu = np.zeros(N)
        for j, ticker in enumerate(TICKERS):
            key = (rb_date.strftime("%Y-%m-%d"), ticker)
            if key in tree_preds["XGBoost"]:
                preds = tree_preds["XGBoost"][key]["pred"]
                mu[j] = np.mean(preds) if preds else 0.0
        w = mvo_weights_params(mu, sigma, lam, w_max)
        period = returns[(returns.index > rb_date) & (returns.index <= next_rb)]
        turnover = np.sum(np.abs(w - prev_w)) / 2
        port_ret = period.values @ w
        port_ret[0] -= tc * turnover
        for date, r in zip(period.index, port_ret):
            daily[date] = r
        prev_w = w.copy()
    r = pd.Series(daily).dropna()
    rf = 0.04 / 252
    sharpe = (r - rf).mean() / r.std() * np.sqrt(252)
    return round(sharpe, 3)

print("Running robustness analysis (may take a few minutes)...")

# Base values
LAM_BASE = RISK_AVERSION
TC_BASE  = TRANSACTION_COST
WM_BASE  = MAX_WEIGHT

rows = []

# ── Risk aversion sensitivity ───────────────────────────────
for lam in [0.5, 1.0, 2.0, 5.0, 10.0]:
    s = run_backtest(lam, TC_BASE, WM_BASE)
    rows.append({"Parameter": f"Risk aversion (λ={lam})", "Sharpe Ratio": s})
    print(f"  λ={lam}: Sharpe={s}")

# ── Transaction cost sensitivity ────────────────────────────
for tc in [0.0, 0.0005, 0.001, 0.002, 0.005]:
    s = run_backtest(LAM_BASE, tc, WM_BASE)
    rows.append({"Parameter": f"Transaction cost ({int(tc*10000)} bps)", "Sharpe Ratio": s})
    print(f"  TC={tc}: Sharpe={s}")

# ── Max weight sensitivity ───────────────────────────────────
for wm in [0.10, 0.15, 0.20, 0.30, 0.50]:
    s = run_backtest(LAM_BASE, TC_BASE, wm)
    rows.append({"Parameter": f"Max weight ({int(wm*100)}%)", "Sharpe Ratio": s})
    print(f"  MaxW={wm}: Sharpe={s}")

rob_df = pd.DataFrame(rows)
rob_df.to_csv(f"{OUTPUT_DIR}/robustness_table.csv", index=False)
print(f"\nRobustness table saved to {OUTPUT_DIR}/robustness_table.csv")
print("  *** INSERT selected rows into Table 4 of the article ***")
print(rob_df.to_string(index=False))
