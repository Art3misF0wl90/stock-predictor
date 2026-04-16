import os

TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "NVDA"]

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

SEQUENCE_LENGTH = 20
FORWARD_DAYS    = 1

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

RANDOM_SEED = 42

MACRO_TICKERS = {
    "vix":      "^VIX",
    "treasury": "^TNX",
    "dollar":   "DX-Y.NYB",
}

SENTIMENT_LOOKBACK_DAYS = 3

# Holding period analysis
# How many trading days to track returns after each signal
HOLDING_PERIODS = [5, 21, 63, 126]
# 5   = 1 week
# 21  = 1 month
# 63  = 1 quarter
# 126 = 6 months