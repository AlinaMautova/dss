"""
scripts/01_data.py
------------------
Downloads historical price data and macroeconomic features,
applies preprocessing, and saves cleaned data.

OUTPUT: outputs/prices_clean.csv, outputs/returns.csv
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TICKERS, START_DATE, END_DATE, OUTPUT_DIR

import pandas as pd
import numpy as np
import yfinance as yf

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Download adjusted close prices ──────────────────────
print("Downloading price data...")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)["Close"]
print(f"  Downloaded: {raw.shape[0]} days x {raw.shape[1]} assets")

# ── 2. Handle missing values ────────────────────────────────
# Forward-fill up to 2 consecutive NaN (non-trading days)
prices = raw.ffill(limit=2)

# Drop assets with >5% missing after fill
missing_frac = prices.isna().mean()
dropped = missing_frac[missing_frac > 0.05].index.tolist()
if dropped:
    print(f"  Dropping assets with >5% missing: {dropped}")
prices = prices.drop(columns=dropped)

# Drop remaining rows with any NaN (start-up period)
prices = prices.dropna()
print(f"  Clean price matrix: {prices.shape}")

# ── 3. Compute log returns ───────────────────────────────────
returns = np.log(prices / prices.shift(1)).dropna()

# ── 4. Save ─────────────────────────────────────────────────
prices.to_csv(f"{OUTPUT_DIR}/prices_clean.csv")
returns.to_csv(f"{OUTPUT_DIR}/returns.csv")
print(f"  Saved: {OUTPUT_DIR}/prices_clean.csv")
print(f"  Saved: {OUTPUT_DIR}/returns.csv")

# ── 5. Summary statistics (for article Table / narrative) ───
print("\n── Return summary statistics ──")
print(returns.describe().round(6))

# Annualised stats
ann_ret  = returns.mean() * 252
ann_vol  = returns.std()  * np.sqrt(252)
print("\nAnnualised mean return (log):")
print(ann_ret.round(4))
print("\nAnnualised volatility:")
print(ann_vol.round(4))
