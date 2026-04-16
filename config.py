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