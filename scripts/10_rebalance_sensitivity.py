"""
scripts/10_rebalance_sensitivity.py
-----------------------------------
Rebalancing-frequency sensitivity of the ML-DSS (XGBoost) strategy,
addressing the reviewer request to test additional rebalancing frequencies
(monthly vs. quarterly) alongside transaction-cost assumptions.

Quarter-end rebalancing dates are a subset of the month-end dates at which
the walk-forward forecasts were produced, so the existing predictions are
reused without look-ahead. For each (frequency, cost) pair we report net
Sharpe and annualized turnover.

OUTPUT:
  outputs/rebalance_sensitivity.csv
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (OUTPUT_DIR, INITIAL_TRAIN_YEARS,
                    RISK_AVERSION, MAX_WEIGHT)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

returns = pd.read_csv(f"{OUTPUT_DIR}/returns.csv", index_col=0, parse_dates=True)
TICKERS = returns.columns.tolist()
N       = len(TICKERS)

with open(f"{OUTPUT_DIR}/predictions/tree_predictions.pkl", "rb") as f:
    tree_preds = pickle.load(f)


def mvo(mu, sigma, lam=RISK_AVERSION, w_max=MAX_WEIGHT):
    n = len(mu)
    res = minimize(lambda w: -(w @ mu - 0.5 * lam * w @ sigma @ w),
                   np.ones(n)/n, method="SLSQP", bounds=[(0, w_max)]*n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                   options={"ftol": 1e-9, "maxiter": 800})
    w = np.clip(res.x, 0, w_max) if res.success else np.ones(n)/n
    return w / w.sum()


def backtest(freq, tc):
    split_start = returns.index[0] + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
    rb_dates = pd.date_range(split_start, returns.index[-1], freq=freq)
    prev_w = np.ones(N)/N
    daily, turns, matched = {}, [], 0
    for i, rb in enumerate(rb_dates[:-1]):
        nxt = rb_dates[i+1]
        tr = returns[returns.index <= rb]
        sigma = LedoitWolf().fit(tr.values).covariance_ if len(tr) > N+20 else np.diag(tr.var().values)
        mu = np.zeros(N)
        for j, t in enumerate(TICKERS):
            key = (rb.strftime("%Y-%m-%d"), t)
            if key in tree_preds["XGBoost"]:
                p = tree_preds["XGBoost"][key]["pred"]; mu[j] = np.mean(p) if p else 0.0
                matched += 1
        w = mvo(mu, sigma)
        per = returns[(returns.index > rb) & (returns.index <= nxt)]
        turn = np.sum(np.abs(w - prev_w))/2; turns.append(turn)
        pr = per.values @ w
        if len(pr): pr[0] -= tc * turn
        for d, r in zip(per.index, pr): daily[d] = r
        prev_w = w.copy()
    r = pd.Series(daily).dropna()
    sharpe = (r - 0.04/252).mean() / r.std() * np.sqrt(252)
    periods_per_year = {"W-FRI": 52, "ME": 12, "QE": 4}[freq]
    ann_turn = np.mean(turns) * periods_per_year
    return round(sharpe, 3), round(ann_turn*100, 1), matched


rows = []
for freq, label in [("ME", "Monthly"), ("QE", "Quarterly")]:
    for tc in [0.0005, 0.001, 0.002]:
        s, tv, matched = backtest(freq, tc)
        rows.append({"Rebalancing": label, "Cost (bps)": int(tc*10000),
                     "Net Sharpe": s, "Ann. Turnover (%)": tv})
        print(f"  {label:10s} {int(tc*10000):>3} bps  Sharpe={s:<7} Turnover={tv}  (matched preds={matched})")

df = pd.DataFrame(rows)
out = f"{OUTPUT_DIR}/rebalance_sensitivity.csv"
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(df.to_string(index=False))
