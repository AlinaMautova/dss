"""
scripts/03_train_models.py
--------------------------
Walk-forward training and evaluation of:
  - Linear Regression (baseline)
  - Random Forest
  - XGBoost
  - LSTM
  - Feed-Forward Neural Network

OUTPUTS:
  outputs/predictions/  – per-model, per-fold predictions
  outputs/model_metrics.csv  – Table 2 in the article
"""

import os, sys, pickle, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (OUTPUT_DIR, INITIAL_TRAIN_YEARS, VALIDATION_MONTHS,
                    REBALANCE_FREQ, FORECAST_HORIZON, LSTM_LOOKBACK)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)

with open(f"{OUTPUT_DIR}/features.pkl", "rb") as f:
    features = pickle.load(f)

TICKERS  = list(features.keys())
FEAT_COLS = [c for c in features[TICKERS[0]].columns if c != "target"]

# ── Walk-forward split dates ────────────────────────────────
all_dates = sorted(features[TICKERS[0]].index)
split_start = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
rebalance_dates = pd.date_range(split_start, all_dates[-1], freq=REBALANCE_FREQ)

print(f"Walk-forward folds: {len(rebalance_dates)}")

# ── Model definitions ────────────────────────────────────────
def get_models():
    return {
        "LinearRegression": Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=300, max_depth=5,
                                                  min_samples_leaf=10,
                                                  n_jobs=-1, random_state=42),
        "XGBoost":          xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                              learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              reg_alpha=0.1, reg_lambda=1.0,
                                              random_state=42, verbosity=0),
    }

# ── Walk-forward loop (tree-based models) ───────────────────
all_preds = {m: {} for m in ["LinearRegression", "RandomForest", "XGBoost"]}

for i, rb_date in enumerate(rebalance_dates):
    # Training data: everything up to rb_date - validation_months
    train_end = rb_date - pd.DateOffset(months=VALIDATION_MONTHS)
    test_end  = rb_date + pd.DateOffset(months=1)

    for ticker in TICKERS:
        df = features[ticker]
        train_df = df[df.index <= train_end]
        test_df  = df[(df.index > rb_date) & (df.index <= test_end)]

        if len(train_df) < 252 or len(test_df) == 0:
            continue

        X_train = train_df[FEAT_COLS].values
        y_train = train_df["target"].values
        X_test  = test_df[FEAT_COLS].values
        y_test  = test_df["target"].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        models = get_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            key = (rb_date.strftime("%Y-%m-%d"), ticker)
            if key not in all_preds[name]:
                all_preds[name][key] = {"pred": [], "actual": []}
            all_preds[name][key]["pred"].extend(preds.tolist())
            all_preds[name][key]["actual"].extend(y_test.tolist())

    if (i + 1) % 12 == 0:
        print(f"  Processed fold {i+1}/{len(rebalance_dates)}")

# ── LSTM (simplified version – replace with full model below) ──
print("\nNote: LSTM requires TensorFlow. Run with: python scripts/03b_lstm.py")
print("Saving tree-model predictions...")

with open(f"{OUTPUT_DIR}/predictions/tree_predictions.pkl", "wb") as f:
    pickle.dump(all_preds, f)

# ── Compute metrics ──────────────────────────────────────────
metrics_rows = []
for model_name, preds_dict in all_preds.items():
    all_pred   = []
    all_actual = []
    for v in preds_dict.values():
        all_pred.extend(v["pred"])
        all_actual.extend(v["actual"])
    all_pred   = np.array(all_pred)
    all_actual = np.array(all_actual)

    mse = mean_squared_error(all_actual, all_pred)
    mae = mean_absolute_error(all_actual, all_pred)
    dir_acc = np.mean(np.sign(all_pred) == np.sign(all_actual)) * 100

    metrics_rows.append({
        "Model":                    model_name,
        "MSE (×10⁻⁴)":             round(mse * 1e4, 3),
        "MAE (×10⁻²)":             round(mae * 1e2, 3),
        "Directional Accuracy (%)": round(dir_acc, 1),
    })
    print(f"  {model_name}: MSE={mse:.6f}  MAE={mae:.6f}  DirAcc={dir_acc:.1f}%")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)
print(f"\n  Table 2 data saved to {OUTPUT_DIR}/model_metrics.csv")
print("  *** INSERT these values into Table 2 of the article ***")
