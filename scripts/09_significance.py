"""
scripts/09_significance.py
--------------------------
Statistical significance testing of Sharpe-ratio differences between
strategies, addressing the reviewer request for formal inference
(Jobson-Korkie test with the Memmel (2003) correction).

For two return series with Sharpe ratios SR_i, SR_j, the Memmel-corrected
Jobson-Korkie statistic is asymptotically N(0,1) under H0: SR_i = SR_j:

    z = (SR_i - SR_j) / sqrt(theta)
    theta = (1/T) * ( 2 - 2*rho
                      + 0.5*(SR_i^2 + SR_j^2) - SR_i*SR_j*(rho^2 + 1) )

where rho is the correlation of the two return series and T the sample size.
Sharpe ratios here are computed on excess daily returns (no annualization,
since the test is scale-invariant to a common factor).

OUTPUT:
  outputs/sharpe_significance.csv
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR

import numpy as np
import pandas as pd
from scipy import stats

port = pd.read_csv(f"{OUTPUT_DIR}/portfolio_returns.csv", index_col=0, parse_dates=True)
rf_daily = 0.04 / 252


def sharpe(x):
    x = x.dropna()
    return (x - rf_daily).mean() / x.std()


def jk_memmel(ri, rj):
    """Return (SR_i, SR_j, z, two-sided p) for H0: SR_i == SR_j."""
    df = pd.concat([ri, rj], axis=1).dropna()
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    T = len(df)
    sr_i, sr_j = sharpe(a), sharpe(b)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (1.0 / T) * (2 - 2 * rho
                         + 0.5 * (sr_i**2 + sr_j**2)
                         - sr_i * sr_j * (rho**2 + 1))
    z = (sr_i - sr_j) / np.sqrt(theta)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return sr_i, sr_j, z, p


# Annualization factor for reporting Sharpe in familiar units
ANN = np.sqrt(252)

pairs = [
    ("XGBoost",      "EqualWeight"),
    ("RF",           "EqualWeight"),
    ("LSTM",         "EqualWeight"),
    ("ClassicalMVO", "EqualWeight"),
    ("XGBoost",      "ClassicalMVO"),
    ("EqualWeight",  "RiskParity"),
]

rows = []
for i, j in pairs:
    if i not in port.columns or j not in port.columns:
        continue
    sr_i, sr_j, z, p = jk_memmel(port[i], port[j])
    rows.append({
        "Comparison":        f"{i} vs {j}",
        f"Sharpe (ann.) A":  round(sr_i * ANN, 3),
        f"Sharpe (ann.) B":  round(sr_j * ANN, 3),
        "z-stat":            round(z, 3),
        "p-value":           round(p, 4),
        "Significant @5%":   "Yes" if p < 0.05 else "No",
    })

df = pd.DataFrame(rows)
out = f"{OUTPUT_DIR}/sharpe_significance.csv"
df.to_csv(out, index=False)
print("Jobson-Korkie / Memmel Sharpe-difference tests")
print(df.to_string(index=False))
print(f"\nSaved to {out}")
