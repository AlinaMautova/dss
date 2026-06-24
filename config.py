# ============================================================
#  config.py  –  Edit this file before running any script
# ============================================================

# --- Asset universe ---
# List of ticker symbols (Yahoo Finance format)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "UNH",
    "V", "XOM", "JNJ", "PG", "HD"
]

# --- Date range ---
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# --- Walk-forward parameters ---
INITIAL_TRAIN_YEARS = 3      # years for initial training window
VALIDATION_MONTHS   = 6      # validation window per fold
REBALANCE_FREQ      = "ME"   # "ME" = month-end, "QE" = quarter-end

# --- Forecasting ---
FORECAST_HORIZON    = 21     # trading days ahead to predict
LSTM_LOOKBACK       = 60     # days of history for LSTM input

# --- Optimisation ---
RISK_AVERSION       = 2.0    # lambda in mean-variance objective
MAX_WEIGHT          = 0.20   # maximum single-asset weight (20%)
TRANSACTION_COST    = 0.001  # 10 bps per unit turnover

# --- Output directory ---
OUTPUT_DIR = "outputs"
