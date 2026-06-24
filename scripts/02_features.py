"""
scripts/02_features.py
----------------------
Constructs technical and macroeconomic features.
All features are constructed with look-ahead-bias-free logic
(using only past data at each point).

INPUT:  outputs/prices_clean.csv, outputs/returns.csv
OUTPUT: outputs/features.pkl  (dict of DataFrames, one per asset)
        outputs/feature_list.txt
"""

import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR

import pandas as pd
import numpy as np

prices  = pd.read_csv(f"{OUTPUT_DIR}/prices_clean.csv", index_col=0, parse_dates=True)
returns = pd.read_csv(f"{OUTPUT_DIR}/returns.csv",      index_col=0, parse_dates=True)

TICKERS = prices.columns.tolist()

def make_features(price_series, return_series):
    """Build feature matrix for a single asset."""
    df = pd.DataFrame(index=price_series.index)

    r = return_series  # log returns

    # ── Lagged returns ──────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag{lag}"] = r.shift(lag)

    # ── Cumulative returns ──────────────────────────────────
    for w in [5, 10, 20, 60]:
        df[f"cum_ret_{w}d"] = r.shift(1).rolling(w).sum()

    # ── Simple moving averages (price) ──────────────────────
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = price_series.shift(1).rolling(w).mean()

    # ── Price relative to SMA (z-score style) ───────────────
    for w in [20, 50]:
        sma = price_series.shift(1).rolling(w).mean()
        df[f"price_vs_sma{w}"] = (price_series.shift(1) - sma) / sma

    # ── Rolling volatility ──────────────────────────────────
    for w in [10, 20, 60]:
        df[f"vol_{w}d"] = r.shift(1).rolling(w).std() * np.sqrt(252)

    # ── RSI (14-day) ─────────────────────────────────────────
    delta = price_series.diff().shift(1)
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ── MACD ─────────────────────────────────────────────────
    ema12 = price_series.shift(1).ewm(span=12, adjust=False).mean()
    ema26 = price_series.shift(1).ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Band width ─────────────────────────────────
    sma20 = price_series.shift(1).rolling(20).mean()
    std20 = price_series.shift(1).rolling(20).std()
    df["bb_width"] = (2 * std20) / (sma20 + 1e-9)
    df["bb_pos"]   = (price_series.shift(1) - sma20) / (2 * std20 + 1e-9)

    # ── Target variable: 1-month-ahead log return ────────────
    df["target"] = r.rolling(21).sum().shift(-21)   # forward-looking, exclude from X!

    return df.dropna()


print("Engineering features for all assets...")
features = {}
for ticker in TICKERS:
    features[ticker] = make_features(prices[ticker], returns[ticker])
    print(f"  {ticker}: {features[ticker].shape}")

# Save
with open(f"{OUTPUT_DIR}/features.pkl", "wb") as f:
    pickle.dump(features, f)

# Save feature list (excluding target)
feat_cols = [c for c in list(features[TICKERS[0]].columns) if c != "target"]
with open(f"{OUTPUT_DIR}/feature_list.txt", "w") as f:
    f.write("\n".join(feat_cols))

print(f"\n  Total features: {len(feat_cols)}")
print(f"  Feature list saved to {OUTPUT_DIR}/feature_list.txt")
print(f"  Features saved to {OUTPUT_DIR}/features.pkl")
