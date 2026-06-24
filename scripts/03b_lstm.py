"""
scripts/03b_lstm.py
-------------------
LSTM forecasting model (TensorFlow/Keras).

Run separately after 03_train_models.py.
Requires: pip install tensorflow

OUTPUT: outputs/predictions/lstm_predictions.pkl
        updates outputs/model_metrics.csv
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (OUTPUT_DIR, INITIAL_TRAIN_YEARS, VALIDATION_MONTHS,
                    REBALANCE_FREQ, LSTM_LOOKBACK)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)

with open(f"{OUTPUT_DIR}/features.pkl", "rb") as f:
    features = pickle.load(f)

TICKERS  = list(features.keys())
FEAT_COLS = [c for c in features[TICKERS[0]].columns if c != "target"]
SEQ_LEN   = LSTM_LOOKBACK

def build_lstm(n_features, n_units=64):
    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),
        LSTM(n_units, return_sequences=True),
        Dropout(0.2),
        LSTM(n_units // 2),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

all_dates   = sorted(features[TICKERS[0]].index)
split_start = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
rb_dates    = pd.date_range(split_start, all_dates[-1], freq=REBALANCE_FREQ)

lstm_preds = {}
es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)

for i, rb_date in enumerate(rb_dates):
    train_end = rb_date - pd.DateOffset(months=VALIDATION_MONTHS)
    test_end  = rb_date + pd.DateOffset(months=1)

    for ticker in TICKERS:
        df       = features[ticker]
        train_df = df[df.index <= train_end]
        test_df  = df[(df.index > rb_date) & (df.index <= test_end)]

        if len(train_df) < SEQ_LEN + 50 or len(test_df) == 0:
            continue

        scaler  = StandardScaler()
        X_all   = scaler.fit_transform(train_df[FEAT_COLS].values)
        y_all   = train_df["target"].values

        X_seq, y_seq = make_sequences(X_all, y_all, SEQ_LEN)

        val_split = int(len(X_seq) * 0.85)
        X_tr, y_tr = X_seq[:val_split], y_seq[:val_split]
        X_val, y_val = X_seq[val_split:], y_seq[val_split:]

        model = build_lstm(len(FEAT_COLS))
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=50, batch_size=32, callbacks=[es], verbose=0)

        # Predict on test
        X_test_raw = test_df[FEAT_COLS].values
        # Build sequences using last SEQ_LEN rows of train + test
        combined   = pd.concat([train_df.tail(SEQ_LEN), test_df])
        X_comb     = scaler.transform(combined[FEAT_COLS].values)
        X_comb_seq, _ = make_sequences(X_comb, np.zeros(len(X_comb)), SEQ_LEN)
        preds = model.predict(X_comb_seq[-len(test_df):], verbose=0).flatten()

        key = (rb_date.strftime("%Y-%m-%d"), ticker)
        lstm_preds[key] = {
            "pred": preds.tolist(),
            "actual": test_df["target"].values.tolist()
        }

    if (i + 1) % 6 == 0:
        print(f"  LSTM fold {i+1}/{len(rb_dates)}")

with open(f"{OUTPUT_DIR}/predictions/lstm_predictions.pkl", "wb") as f:
    pickle.dump(lstm_preds, f)

# Metrics
all_pred, all_actual = [], []
for v in lstm_preds.values():
    all_pred.extend(v["pred"])
    all_actual.extend(v["actual"])
all_pred, all_actual = np.array(all_pred), np.array(all_actual)

mse     = mean_squared_error(all_actual, all_pred)
mae     = mean_absolute_error(all_actual, all_pred)
dir_acc = np.mean(np.sign(all_pred) == np.sign(all_actual)) * 100
print(f"\nLSTM  MSE={mse:.6f}  MAE={mae:.6f}  DirAcc={dir_acc:.1f}%")
print("  *** INSERT LSTM row into Table 2 of the article ***")

# Update metrics CSV
metrics_df = pd.read_csv(f"{OUTPUT_DIR}/model_metrics.csv")
lstm_row = pd.DataFrame([{
    "Model": "LSTM",
    "MSE (×10⁻⁴)": round(mse * 1e4, 3),
    "MAE (×10⁻²)": round(mae * 1e2, 3),
    "Directional Accuracy (%)": round(dir_acc, 1),
}])
metrics_df = pd.concat([metrics_df, lstm_row], ignore_index=True)
metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)
print(f"  Updated {OUTPUT_DIR}/model_metrics.csv")
