"""
scripts/08_turnover_aware.py
----------------------------
Transaction-cost-aware (turnover-penalized) Markowitz optimization.

Addresses the core reviewer request: incorporate an explicit L1 turnover /
transaction-cost penalty directly into the mean-variance objective, rather
than evaluating an ML forecasting pipeline whose optimizer ignores turnover.

Objective solved at each rebalancing date:
    max_w   w'mu  -  (lambda/2) w'Sigma w  -  gamma * ||w - w_prev||_1
    s.t.    sum(w) = 1,  0 <= w_i <= w_max

gamma = 0 recovers the original (turnover-blind) ML-DSS used in the paper.
gamma > 0 penalizes trading away from the current holdings, trading off
forecast-driven repositioning against realized transaction costs.

OUTPUT:
  outputs/turnover_aware_table.csv        (large-cap)
  outputs_mid/turnover_aware_table.csv    (mid-cap, if run with UNIVERSE=mid)
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


def tc_aware_weights(mu, sigma, w_prev, lam, gamma, w_max):
    """Mean-variance weights with an explicit L1 turnover penalty gamma*||w-w_prev||_1."""
    n = len(mu)

    def neg_utility(w):
        turnover = np.sum(np.abs(w - w_prev))
        return -(w @ mu - 0.5 * lam * w @ sigma @ w - gamma * turnover)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, w_max)] * n
    res = minimize(neg_utility, w_prev.copy(), method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    if res.success:
        w = np.clip(res.x, 0, w_max)
        return w / w.sum()
    return w_prev.copy()


def get_mu(model, rb_date):
    mu = np.zeros(N)
    for j, ticker in enumerate(TICKERS):
        key = (rb_date.strftime("%Y-%m-%d"), ticker)
        if key in tree_preds[model]:
            preds = tree_preds[model][key]["pred"]
            mu[j] = np.mean(preds) if preds else 0.0
    return mu


def backtest(model, gamma, lam=RISK_AVERSION, tc=TRANSACTION_COST, w_max=MAX_WEIGHT):
    prev_w   = np.ones(N) / N
    daily    = {}
    turnovers = []
    for i, rb_date in enumerate(rb_dates[:-1]):
        next_rb = rb_dates[i + 1]
        train_r = returns[returns.index <= rb_date]
        if len(train_r) > N + 20:
            sigma = LedoitWolf().fit(train_r.values).covariance_
        else:
            sigma = np.diag(train_r.var().values)
        mu = get_mu(model, rb_date)
        w  = tc_aware_weights(mu, sigma, prev_w, lam, gamma, w_max)

        turnover = np.sum(np.abs(w - prev_w)) / 2
        turnovers.append(turnover)
        period = returns[(returns.index > rb_date) & (returns.index <= next_rb)]
        port_ret = period.values @ w
        port_ret[0] -= tc * turnover
        for date, r in zip(period.index, port_ret):
            daily[date] = r
        prev_w = w.copy()

    r = pd.Series(daily).dropna()
    rf = 0.04 / 252
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = (r - rf).mean() / r.std() * np.sqrt(252)
    cum = (1 + r).cumprod(); dd = (cum - cum.cummax()) / cum.cummax()
    mdd = dd.min()
    ann_turnover = np.mean(turnovers) * 12
    return {
        "Ann. Return (%)":     round(ann_ret * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Net Sharpe":          round(sharpe, 3),
        "Max Drawdown (%)":    round(mdd * 100, 2),
        "Ann. Turnover (%)":   round(ann_turnover * 100, 1),
    }


print("Transaction-cost-aware optimization sweep (may take a few minutes)...")
rows = []
for model in ["XGBoost", "RandomForest"]:
    for gamma in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        stats = backtest(model, gamma)
        row = {"Model": model, "gamma": gamma, **stats}
        rows.append(row)
        print(f"  {model:12s} gamma={gamma:<7} "
              f"Sharpe={stats['Net Sharpe']:<7} "
              f"Turnover={stats['Ann. Turnover (%)']:<7} "
              f"Ret={stats['Ann. Return (%)']}")

df = pd.DataFrame(rows)
out = f"{OUTPUT_DIR}/turnover_aware_table.csv"
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(df.to_string(index=False))
