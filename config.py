# ============================================================
#  config.py  –  Edit this file before running any script
# ============================================================

# --- Asset universe ---
# List of ticker symbols (Yahoo Finance format)
LARGE_CAP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "UNH",
    "V", "XOM", "JNJ", "PG", "HD"
]

MIDCAP_TICKERS = [
    "EMN",   # Eastman Chemical
    "FSLR",  # First Solar
    "RPM",
    "ALGN",
    "AOS",
    "CMA",
    "MAS",
    "WTRG",
    "BJ",
    "DAR",
    "LEA",
    "NDSN",
    "OGE",
    "PFGC",
    "RLI",
    "SAIC",
    "TAP",
    "WHR",
    "ZBRA",
    "CHDN",
    "DTM",
    "GATX",
    "HLI",
    "ITT",
    "JBL",
    "LKQ",
    "MAT",
    "NOV",
    "PNR",
    "R",
    "SITE",
    "SWK",
    "TFX",
    "UHS",
    "WCC"
]

ALL_TICKERS = LARGE_CAP_TICKERS + MIDCAP_TICKERS

UNIVERSE = "mid"  # large | mid

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
OUTPUT_DIR = "outputs_mid"

if UNIVERSE == "large":
    TICKERS = LARGE_CAP_TICKERS
elif UNIVERSE == "mid":
    TICKERS = MIDCAP_TICKERS
else:
    raise ValueError("UNIVERSE must be 'large' or 'mid'")