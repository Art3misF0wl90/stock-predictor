import os

TICKERS = [
    "AAPL", "MSFT", "TSLA", "JPM", "NVDA",  # original
    "GOOGL", "AMZN", "META", "SPY", "AMD",
    "GME",   # new
]

from datetime import date

START_DATE = "2015-01-01"
END_DATE   = str(date.today())   # always uses today's date

FORWARD_DAYS      = 21
FORWARD_DAYS_LIST = [1, 21, 63]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

RANDOM_SEED = 42

MACRO_TICKERS = {
    "vix":      "^VIX",
    "treasury": "^TNX",
    "dollar":   "DX-Y.NYB",
}

SENTIMENT_LOOKBACK_DAYS = 3
HOLDING_PERIODS = [5, 21, 63, 126]