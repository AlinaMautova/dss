"""
scripts/05_shap.py
------------------
SHAP interpretability analysis for XGBoost model.

OUTPUTS:
  outputs/figures/fig08_shap_beeswarm.png
  outputs/figures/fig09_shap_waterfall_<ticker>.png
  outputs/shap_importance.csv
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

with open(f"{OUTPUT_DIR}/features.pkl", "rb") as f:
    features = pickle.load(f)

TICKERS  = list(features.keys())
FEAT_COLS = [c for c in features[TICKERS[0]].columns if c != "target"]

# ── Train a full XGBoost on all data for SHAP (demonstration model) ──
print("Training XGBoost on full dataset for SHAP analysis...")
all_X, all_y = [], []
for ticker in TICKERS:
    df = features[ticker].dropna()
    all_X.append(df[FEAT_COLS].values)
    all_y.append(df["target"].values)

X = np.vstack(all_X)
y = np.concatenate(all_y)

model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=1.0,
                          random_state=42, verbosity=0)
model.fit(X, y)

# ── SHAP values ──────────────────────────────────────────────
print("Computing SHAP values (this may take a few minutes)...")
explainer = shap.TreeExplainer(model)

# Use a sample to keep computation manageable
sample_idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
X_sample   = X[sample_idx]
shap_vals  = explainer.shap_values(X_sample)

# ── Figure 8: Beeswarm plot ──────────────────────────────────
print("Generating Figure 8: SHAP beeswarm plot...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_sample, feature_names=FEAT_COLS,
                  show=False, max_display=20)
plt.title("SHAP Feature Importance – XGBoost (Global)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig08_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig08_shap_beeswarm.png")
print("  *** INSERT AS FIGURE 8 in article ***")

# ── Figure 9: Waterfall for a single observation ────────────
# Use the first ticker, most recent available date
sample_ticker = TICKERS[0]
df_t   = features[sample_ticker].dropna()
x_last = df_t[FEAT_COLS].values[-1:].astype(float)
sv_one = explainer.shap_values(x_last)

print(f"\nGenerating Figure 9: SHAP waterfall for {sample_ticker} (most recent date)...")
exp = shap.Explanation(
    values=sv_one[0],
    base_values=explainer.expected_value,
    data=x_last[0],
    feature_names=FEAT_COLS
)
plt.figure(figsize=(10, 7))
shap.plots.waterfall(exp, max_display=15, show=False)
plt.title(f"SHAP Waterfall – {sample_ticker} ({df_t.index[-1].date()})", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig09_shap_waterfall_{sample_ticker}.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig09_shap_waterfall_{sample_ticker}.png")
print("  *** INSERT AS FIGURE 9 in article ***")

# ── Save global importance ────────────────────────────────────
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": FEAT_COLS,
    "Mean |SHAP|": mean_abs_shap
}).sort_values("Mean |SHAP|", ascending=False)

importance_df.to_csv(f"{OUTPUT_DIR}/shap_importance.csv", index=False)
print("\nTop 10 features by global SHAP importance:")
print(importance_df.head(10).to_string(index=False))
print(f"\n  Saved: {OUTPUT_DIR}/shap_importance.csv")
