"""
run_all.py  –  DSS Portfolio ML Pipeline
=========================================
Run from the project root:
    pip install -r requirements.txt
    python run_all.py

Edit config.py first to set your tickers, date range, and parameters.
"""

import subprocess, sys

steps = [
    ("1. Data download & preprocessing",  "python scripts/01_data.py"),
    ("2. Feature engineering",            "python scripts/02_features.py"),
    ("3. Train & evaluate ML models",     "python scripts/03_train_models.py"),
    ("3b. LSTM model (optional)",         "python scripts/03b_lstm.py"),
    ("4. Portfolio backtesting",          "python scripts/04_backtest.py"),
    ("5. SHAP interpretability",          "python scripts/05_shap.py"),
    ("6. Generate all figures",           "python scripts/06_figures.py"),
    ("7. Robustness checks",              "python scripts/07_robustness.py"),
]

skip_on_error = {"3b. LSTM model (optional)"}   # LSTM is optional (needs TensorFlow)

for label, cmd in steps:
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        if label in skip_on_error:
            print(f"  Skipping optional step: {label}")
        else:
            print(f"\nERROR in step: {label}")
            sys.exit(1)

print("\n" + "="*60)
print("✓  All pipeline steps completed.")
print("   Figures  →  outputs/figures/")
print("   Tables   →  outputs/*.csv")
print("="*60)
