import os
from datetime import date

# -- Watchlist

TICKERS = [
    "AAPL", "MSFT", "TSLA", "JPM", "NVDA"   #original 5
    "GOOGL", "AMZN", "META", "SPY", "AMD",  #Added
    "GME"
]

#TIckers that have unreliable or missing earnings date from yfinance.
# These are excluded from earnings feature engineering during training
# and prediction, to avoid data leakage and errors.
TICKERS_NO_EARNINGS = ["TSLA", "SPY"]

#---- Date range

START_DATE = "2015-01-01"
END_DATE   = str(date.today())  #always fetch up to current date

#---- Model training

# Forward prediction horizons (in trading days) tried during model selection
# The best-oerforming horizon per ticker is saved to models/<ticker>_config.pkl
FORWARD_DAYS = 21                   #default horizon used in combine model
FORWARD_DAYS_LIST = [1, 21, 63]     #horizons evaluated during train_classical.py


# Train/val/test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15

#TEST_RATIO  = 0.15  #not used directly, calculated as 1 - TRAIN_RATIO - VAL_RATIO

#LSTM sequence length (in trading days(how many days the model sees at once))
SEQUENCE_LENGTH = 20

RANDOM_SEED = 42

#--- Signal filters
# These thresholds are applied in predict.py and backtest.py to reduce
# false positives and improve overall performance. Any change here affects both live signals and 
# backtest results, so they should be chosen carefully based on validation performance.


# Minimum prob_up to act on a raw model signal
# Below this, the signal is too close to 50/50 to be useful
MIN_CONFIDENCE = 0.55

# Per-ticker overrides for MIN_CONFIDENCE, based on validation performance. 
# NVDA and GOOGL have historically lower AUC, so we accept lower confidence for them.
TICKER_MIN_CONFIDENCE = {
    "AAPL":  0.55,
    "MSFT":  0.55,
    "TSLA":  0.55,
    "JPM":   0.55,
    "NVDA":  0.50,
    "GOOGL": 0.50,
    "AMZN":  0.52,
    "META":  0.52,
    "SPY":   0.55,
    "AMD":   0.55,
}

# VIX above this level signals a market crisis. Model assumptions about
# return distributions may break down in high-volatility environments
MAX_VIX = 30.0

# Minimum backtest win rate for a signal to be labeled "BUY" (not WEAK BUY)
MIN_WIN_RATE = 0.70

# RSI above this is condidered overbought - avoid chasing momentum
RSI_OVERBOUGHT = 70.0

# Earnings blackout window in days. Within this many days of an earnings 
# announcement, we avoid acting on signals due to increased volatility and unpredictability.
EARNINGS_BLACKOUT_DAYS = 3

# Consecutive losing signals before the model pauses.
# Indicates the model is out of synch with current market conditions
MAX_CONSECUTIVE_LOSSES = 3

# --- Watchlist quality thresholds (used in analyze.py)
# Minimum quality score to allow a ticker to be added to the watchlist.
MIN_WATCHLIST_SCORE = 40

# Minimum years of price history required for training
MIN_HISTORY_YEARS = 2

# Minimum average daily dollar volume. Below this, slippage is too large.
MIN_DAILY_VOLUME = 500_000   # shares
 
# Penny stock floor — models trained below this price are unreliable.
MIN_STOCK_PRICE = 5.0
 
# Annualized volatility above this is considered extreme — harder to predict.
MAX_ANNUALIZED_VOL = 1.5
 
# Minimum fraction of expected trading days that must have data.
# Gaps below this suggest delisted periods or data quality issues.
MIN_DATA_COVERAGE = 0.85
 
# ── Options flow thresholds (used in options_flow.py) ────────────────────────
 
# PCR below this is considered bullish (calls dominating).
PCR_BULLISH_THRESHOLD = 0.7
 
# PCR above this is considered bearish (heavy put hedging).
PCR_BEARISH_THRESHOLD = 1.2
 
# Volume/OI ratio above this flags a contract as unusual.
# A contract with 3x more volume than open interest suggests fresh directional bets.
UNUSUAL_VOLUME_RATIO = 2.0
 
# ── Data pipeline ─────────────────────────────────────────────────────────────
 
MACRO_TICKERS = {
    "vix":      "^VIX",
    "treasury": "^TNX",
    "dollar":   "DX-Y.NYB",
}
 
# How many days of news to roll-average sentiment over.
# Higher = smoother but slower to respond to news; lower = noisier.
SENTIMENT_LOOKBACK_DAYS = 3
 
# How stale the macro cache can be before auto-refreshing (in days).
MACRO_CACHE_MAX_AGE_DAYS = 7
 
# Backtest holding periods to evaluate (in trading days).
HOLDING_PERIODS = [5, 21, 63, 126]
 
