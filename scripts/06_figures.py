"""
scripts/06_figures.py
---------------------
Generates all figures for the article:
  Fig 1  – Return distribution histogram
  Fig 2  – Walk-forward split diagram
  Fig 3  – System architecture (schematic)
  Fig 4  – Prediction scatter plots (XGBoost & LSTM)
  Fig 5  – Cumulative portfolio value
  Fig 6  – Rolling 12-month Sharpe ratio
  Fig 7  – Efficient frontier
  Fig 10 – Portfolio weight evolution (XGBoost)

Run after: 01–05 scripts.
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR, INITIAL_TRAIN_YEARS, RISK_AVERSION, MAX_WEIGHT

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

returns  = pd.read_csv(f"{OUTPUT_DIR}/returns.csv",       index_col=0, parse_dates=True)
port_df  = pd.read_csv(f"{OUTPUT_DIR}/portfolio_returns.csv", index_col=0, parse_dates=True)

with open(f"{OUTPUT_DIR}/portfolio_weights.pkl", "rb") as f:
    weight_history = pickle.load(f)

with open(f"{OUTPUT_DIR}/predictions/tree_predictions.pkl", "rb") as f:
    tree_preds = pickle.load(f)

TICKERS = returns.columns.tolist()
N       = len(TICKERS)
COLORS  = plt.cm.tab10.colors

# ── FIGURE 1: Return distribution ───────────────────────────
print("Figure 1: Return distribution...")
fig, axes = plt.subplots(2, 5, figsize=(14, 6)) if N >= 10 else plt.subplots(1, 5, figsize=(14, 3))
axes = np.array(axes).flatten()
for i, ticker in enumerate(TICKERS[:min(N, 10)]):
    r = returns[ticker].dropna()
    axes[i].hist(r, bins=60, density=True, alpha=0.6, color=COLORS[i % 10])
    axes[i].set_title(ticker, fontsize=9)
    axes[i].set_xlabel("Log return", fontsize=7)
    axes[i].tick_params(labelsize=6)
for ax in axes[N:]:
    ax.set_visible(False)
plt.suptitle("Distribution of Daily Log Returns by Asset", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig01_return_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig01_return_distributions.png")
print("  *** INSERT AS FIGURE 1 in article ***")

# ── FIGURE 2: Walk-forward diagram ──────────────────────────
print("Figure 2: Walk-forward diagram...")
fig, ax = plt.subplots(figsize=(12, 3))
example_folds = 6
colors_wf = ["#2196F3", "#4CAF50", "#F44336"]
for i in range(example_folds):
    train_end = INITIAL_TRAIN_YEARS + i * 0.5
    val_end   = train_end + 0.5
    test_end  = val_end + 0.5
    ax.barh(0, train_end,           height=0.5, left=0,         color=colors_wf[0], alpha=0.7)
    ax.barh(0, val_end - train_end, height=0.5, left=train_end, color=colors_wf[1], alpha=0.8)
    ax.barh(0, 0.5,                 height=0.5, left=val_end,   color=colors_wf[2], alpha=0.9)
    ax.barh(i + 1, train_end + i * 0.5,          height=0.5, left=0,                      color=colors_wf[0], alpha=0.7)
    ax.barh(i + 1, 0.5,                           height=0.5, left=train_end + i * 0.5,   color=colors_wf[1], alpha=0.8)
    ax.barh(i + 1, 0.5,                           height=0.5, left=val_end   + i * 0.5,   color=colors_wf[2], alpha=0.9)

patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(colors_wf, ["Training", "Validation", "Test"])]
ax.legend(handles=patches, loc="lower right")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Fold")
ax.set_title("Walk-Forward Cross-Validation Scheme", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig02_walkforward.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig02_walkforward.png")
print("  *** INSERT AS FIGURE 2 in article ***")

# ── FIGURE 3: System architecture ───────────────────────────
print("Figure 3: System architecture diagram (schematic)...")
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
modules = [
    (1.0, "Data\nManagement\nModule",   "#90CAF9"),
    (3.5, "ML Model\nManagement\nModule", "#A5D6A7"),
    (6.0, "Optimisation\nEngine",         "#FFCC80"),
    (8.5, "User Interface\n& Reporting",  "#CE93D8"),
]
for x, label, color in modules:
    rect = mpatches.FancyBboxPatch((x - 0.9, 0.5), 1.8, 2.0,
                                    boxstyle="round,pad=0.1", color=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x, 1.5, label, ha="center", va="center", fontsize=9, zorder=3)

for i in range(len(modules) - 1):
    x1 = modules[i][0] + 0.9
    x2 = modules[i+1][0] - 0.9
    ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle="->", lw=2))

ax.set_title("DSS System Architecture", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig03_architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig03_architecture.png")
print("  *** INSERT AS FIGURE 3 in article ***")

# ── FIGURE 4: Prediction scatter (XGBoost) ──────────────────
print("Figure 4: Prediction scatter plots...")
fig, ax = plt.subplots(figsize=(7, 5))
all_pred, all_actual = [], []
for v in tree_preds["XGBoost"].values():
    all_pred.extend(v["pred"])
    all_actual.extend(v["actual"])
all_pred, all_actual = np.array(all_pred), np.array(all_actual)

ax.scatter(all_actual, all_pred, alpha=0.15, s=5, color="#1976D2")
lim = max(abs(all_actual).max(), abs(all_pred).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="Perfect forecast")
ax.set_xlabel("Actual 1-month return", fontsize=11)
ax.set_ylabel("Predicted 1-month return", fontsize=11)
ax.set_title("XGBoost: Predicted vs Actual Returns (Out-of-Sample)", fontsize=11)
ax.legend()
r2 = np.corrcoef(all_actual, all_pred)[0,1]**2
ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig04_prediction_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig04_prediction_scatter.png")
print("  *** INSERT AS FIGURE 4 in article ***")

# ── FIGURE 5: Cumulative returns ─────────────────────────────
print("Figure 5: Cumulative portfolio returns...")
fig, ax = plt.subplots(figsize=(12, 5))
cum = (1 + port_df).cumprod()
for i, col in enumerate(cum.columns):
    style = "-" if "DSS" in col or col in ["XGBoost", "RF", "LSTM"] else "--"
    ax.plot(cum.index, cum[col], label=col, lw=1.8, ls=style, color=COLORS[i % 10])
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Portfolio Value (normalised to 1.0)", fontsize=11)
ax.set_title("Cumulative Portfolio Performance: All Strategies", fontsize=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig05_cumulative_returns.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig05_cumulative_returns.png")
print("  *** INSERT AS FIGURE 5 in article ***")

# ── FIGURE 6: Rolling Sharpe ratio ───────────────────────────
print("Figure 6: Rolling 12-month Sharpe ratio...")
rf_daily = 0.04 / 252
fig, ax = plt.subplots(figsize=(12, 4))
highlight = ["XGBoost", "EqualWeight", "ClassicalMVO"]
for col in [c for c in port_df.columns if c in highlight]:
    r = port_df[col].dropna()
    rolling_sharpe = (r - rf_daily).rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe, label=col, lw=1.8)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Rolling 12-Month Sharpe Ratio", fontsize=11)
ax.set_title("Rolling Sharpe Ratio: ML-DSS vs Baselines", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig06_rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig06_rolling_sharpe.png")
print("  *** INSERT AS FIGURE 6 in article ***")

# ── FIGURE 7: Efficient frontier ─────────────────────────────
print("Figure 7: Efficient frontier...")
train_rets = returns.iloc[-756:]   # last 3 years for illustration
lw = LedoitWolf().fit(train_rets.values)
sigma = lw.covariance_
mu_hist = train_rets.mean().values * 252

# Monte Carlo frontier
n_port = 3000
rets_mc, vols_mc = [], []
for _ in range(n_port):
    w = np.random.dirichlet(np.ones(N))
    rets_mc.append(w @ mu_hist)
    vols_mc.append(np.sqrt(w @ sigma @ w) * np.sqrt(252))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(vols_mc, rets_mc, c=np.array(rets_mc) / np.array(vols_mc),
           cmap="viridis", alpha=0.3, s=3, label="Feasible portfolios")

# Mark specific portfolios
def mvo_w(mu, sig, lam):
    n = len(mu)
    def obj(w): return -(w @ mu - 0.5 * lam * w @ sig @ w)
    res = minimize(obj, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, MAX_WEIGHT)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}])
    return res.x if res.success else np.ones(n)/n

w_eq   = np.ones(N) / N
w_mvo_ = mvo_w(mu_hist, sigma, RISK_AVERSION)
w_ml   = mvo_w(mu_hist * 1.1, sigma, RISK_AVERSION)  # illustrative ML improvement

for w, label, marker, color in [
    (w_eq,   "Equal-Weight",  "^", "red"),
    (w_mvo_, "Classical MVO", "s", "orange"),
    (w_ml,   "ML-DSS Optimal","*", "green"),
]:
    ret = w @ mu_hist
    vol = np.sqrt(w @ sigma @ w) * np.sqrt(252)
    ax.scatter(vol, ret, s=120, zorder=5, label=label, marker=marker, color=color, edgecolors="black")

ax.set_xlabel("Annualised Volatility", fontsize=11)
ax.set_ylabel("Annualised Expected Return", fontsize=11)
ax.set_title("Efficient Frontier (Representative Rebalancing Date)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figures/fig07_efficient_frontier.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/figures/fig07_efficient_frontier.png")
print("  *** INSERT AS FIGURE 7 in article ***")

# ── FIGURE 10: Portfolio weight evolution ────────────────────
print("Figure 10: Portfolio weight evolution...")
if "XGBoost" in weight_history:
    wh = weight_history["XGBoost"]
    dates_wh = sorted(wh.keys())
    W = np.array([wh[d] for d in dates_wh])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(dates_wh, W.T, labels=TICKERS,
                 colors=plt.cm.tab20.colors[:N], alpha=0.85)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.set_title("Portfolio Weight Evolution – ML-DSS (XGBoost)", fontsize=12)
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/fig10_portfolio_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/figures/fig10_portfolio_weights.png")
    print("  *** INSERT AS FIGURE 10 in article ***")

print("\n✓ All figures generated in outputs/figures/")
