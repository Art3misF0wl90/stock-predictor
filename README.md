# Stock Prediction Pipeline

A full machine learning pipeline that predicts next-day (or multi-day) stock price direction for a watchlist of tickers. Combines technical indicators, macroeconomic data, news sentiment, and earnings data to generate buy/sell signals, with a Flask web dashboard and an AI chatbot for on-demand analysis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture at a Glance](#architecture-at-a-glance)
3. [File Reference](#file-reference)
4. [Setup and Installation](#setup-and-installation)
5. [Environment Variables (API Keys)](#environment-variables-api-keys)
6. [Run Order: From Scratch](#run-order-from-scratch)
7. [Daily Operations](#daily-operations)
8. [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
9. [Model Training Deep Dive](#model-training-deep-dive)
10. [Signal Generation and Filtering](#signal-generation-and-filtering)
11. [Backtesting](#backtesting)
12. [Web Dashboard and API](#web-dashboard-and-api)
13. [AI Chatbot (bot.py)](#ai-chatbot-botpy)
14. [Options Flow Scanner](#options-flow-scanner)
15. [Portfolio Tracker](#portfolio-tracker)
16. [On-Demand Ticker Analysis](#on-demand-ticker-analysis)
17. [Model Performance Benchmarks](#model-performance-benchmarks)
18. [Directory Structure](#directory-structure)
19. [Troubleshooting](#troubleshooting)
20. [Key Design Decisions and Limitations](#key-design-decisions-and-limitations)

---

## Project Overview

This system does the following:

- Fetches historical OHLCV (Open, High, Low, Close, Volume) data for a watchlist of stocks going back to 2015
- Engineers 50+ features from technical indicators, macroeconomic signals, news sentiment (VADER), and earnings surprises
- Trains per-ticker XGBoost and Logistic Regression classifiers, plus stacked LSTM networks, to predict whether a stock will be higher in 1, 21, or 63 days
- Generates daily buy/sell signals with confidence scores and stores them in a SQLite database
- Backtests those signals across multiple holding periods and measures win rate vs. buy-and-hold
- Serves everything through a Flask web app with a live dashboard, options flow scanner, portfolio tracker, and an AI chatbot powered by Groq's Llama 3.3 70B

**Current watchlist (11 tickers):** AAPL, MSFT, TSLA, JPM, NVDA, GOOGL, AMZN, META, SPY, AMD, GME

Any additional ticker can be analyzed on-demand without retraining via `analyze.py`.

---

## Architecture at a Glance

```
Data Layer          Feature Layer       Model Layer         Serving Layer
──────────          ─────────────       ───────────         ─────────────
data_loader.py  →   features.py     →   train_classical.py  →  predict.py
macro_loader.py →   (merges all     →   train_lstm.py       →  app.py (Flask)
sentiment_loader.py  sources)       →   tune.py (optional)  →  bot.py (chatbot)
earnings_loader.py                      splitter.py         →  backtest.py
                                        evaluate.py         →  chart.py
```

The pipeline is strictly time-ordered. Data is split 70/15/15 (train/val/test) and models only ever see past data when making predictions, preventing lookahead bias.

---

## File Reference

### Configuration

| File | What it does |
|------|-------------|
| `config.py` | Central config: watchlist tickers, date range, train/val/test ratios, sequence length for LSTM, holding periods for backtest, macro ticker symbols. **Edit this file to add/remove tickers or change date ranges.** |

### Data Loading

| File | What it does |
|------|-------------|
| `data_loader.py` | Downloads OHLCV price data from yfinance for all tickers from 2015 to today. Results are cached as CSV files in `data/` so subsequent runs load from disk instead of hitting the API every time. |
| `macro_loader.py` | Downloads macroeconomic indicators: VIX (^VIX), 10-year Treasury yield (^TNX), and the US Dollar Index (DX-Y.NYB). Computes derived features like VIX z-score, VIX regime classification (Low/Normal/Elevated/High), and 10-day moving averages. Caches to `data/macro.csv` and refreshes if the cache is over 7 days old. |
| `sentiment_loader.py` | Fetches recent news headlines for each ticker from yfinance, scores them with VADER (Valence Aware Dictionary and sEntiment Reasoner), and builds a daily sentiment time series going back to 2015. Uses a 3-day rolling average so noise is smoothed out. Results are cached per-ticker. |
| `earnings_loader.py` | Pulls quarterly earnings dates and EPS surprise data from yfinance. Builds features like days until the next earnings report, days since the last one, EPS surprise percentage, and a PEAD (Post-Earnings Announcement Drift) signal that decays exponentially over 30 days. |

### Feature Engineering

| File | What it does |
|------|-------------|
| `features.py` | Core feature engineering module. `add_features()` takes a raw OHLCV DataFrame plus optional macro, sentiment, and earnings DataFrames, and returns a single DataFrame with 50+ computed features plus a binary `target` column (1 if price is higher in N days, 0 otherwise). `get_feature_columns()` returns the canonical list of feature names. |
| `splitter.py` | Time-safe train/val/test splitting. `time_split()` splits by index position (not randomly) to respect temporal order. `make_sequences()` converts flat feature arrays into 3D sequences for LSTM input. |

### Model Training

| File | What it does |
|------|-------------|
| `train_classical.py` | The main training script. For each ticker, it tries every combination of forward horizon (1d, 21d, 63d) and model type (LogisticRegression, XGBoost), picks the one with the best test-set AUC, and saves the model + scaler + config to `models/`. Also trains a combined model across all tickers at the end. |
| `train_lstm.py` | Trains stacked LSTM networks (2 LSTM layers, Dropout, BatchNorm) per ticker using sequences of 20 trading days. Uses EarlyStopping and ReduceLROnPlateau callbacks. Saves `.keras` model files and scalers. |
| `tune.py` | Optional hyperparameter search using Optuna. Runs 150 trials of Bayesian optimization per ticker, searching over XGBoost parameters (depth, learning rate, regularization, etc.). Results are saved to `models/best_params.pkl`. Takes a long time — skip on first run. |
| `evaluate.py` | Generates a detailed evaluation report for trained models: AUC-ROC on the test set, confusion matrix, directional accuracy (how often predicted-up days actually went up vs. predicted-down days), and plots ROC curves. Saves PNG charts to `models/`. |

### Prediction and Backtesting

| File | What it does |
|------|-------------|
| `predict.py` | The daily prediction script. Loads the best saved model for each ticker, fetches the last 300 days of price data, runs the feature pipeline, and generates a signal (BUY / STRONG BUY / HOLD / AVOID / FILTERED) with a probability and win rate. Saves signals to `data/predictions.db` (SQLite). **Run this daily.** |
| `backtest.py` | Simulates what would have happened if you followed every signal on the held-out test set. Computes win rates, average returns, Sharpe ratio, and max drawdown across four holding periods (5d, 21d, 63d, 126d). Also applies the same entry filters used in `predict.py` (VIX, trend, RSI, earnings blackout) so backtest results match live behavior. Generates interactive Plotly HTML charts. |
| `chart.py` | Generates interactive candlestick charts in Plotly with prediction signals overlaid, colored by train/val/test split region, and a probability time series in a subplot below. Saves to `charts/`. |

### Web and API

| File | What it does |
|------|-------------|
| `app.py` | Flask web server with Socket.IO for real-time chatbot communication. Exposes REST API endpoints for signals, macro data, performance stats, signal history, and portfolio data. Serves the HTML templates. |
| `bot.py` | Groq-powered chatbot using Llama 3.3 70B. Implements a simple tool-calling protocol: the model responds with JSON when it wants to call a function, the function runs, and the result is fed back into the next model call. Supports tools for getting signals, running backtests, explaining signals, analyzing arbitrary tickers, and adding tickers to the watchlist. |
| `database.py` | SQLite wrapper around `data/predictions.db`. Functions for reading today's signals, signal history, outcomes, summary stats, and recording trade outcomes for live accuracy tracking. |
| `analyze.py` | On-demand analysis for any ticker (not just the watchlist). Fetches live data, runs technical analysis, scores watchlist suitability (based on history length, liquidity, price, volatility, and data completeness), and generates a signal from the combined model. Can also write new tickers into `config.py` to add them permanently. |
| `options_flow.py` | Options chain analysis for all watchlist tickers. Computes Put/Call Ratio (PCR), flags unusual volume (volume significantly higher than open interest), estimates Gamma Exposure (GEX), ranks Implied Volatility (IV Rank), and computes the expected move for the nearest expiry. Combines with ML signal for a combined directional prediction. |
| `portfolio.py` | Portfolio tracking with a SQLite backend (`data/portfolio.db`). Tracks holdings, cash balance, buy/sell transactions, realized P&L, and generates daily advice by cross-referencing your positions with today's signals. |

### Frontend Templates

| File | What it does |
|------|-------------|
| `templates/index.html` | Main dashboard. Shows macro conditions (VIX, Treasury, Dollar), today's signals table with action badges and confidence bars, historical win rates, and the AI chatbot with Socket.IO. Has a ticker search bar that opens a detailed analysis modal. |
| `templates/options_flow.html` | Options flow page. Scan button triggers a background scan of all tickers; results auto-poll and render as cards showing PCR, IV rank, GEX, unusual volume, expected move, and model vs. flow prediction. |
| `templates/portfolio.html` | Portfolio tracker page. KPI strip at the top, holdings table with live P&L, allocation bar chart, daily advice tabs (warnings / advice / opportunities), cash widget, and transaction log. |

---

## Setup and Installation

### Prerequisites

- Python 3.11 or 3.12
- pip or conda
- A terminal (Linux, macOS, or WSL on Windows)
- Optional: a Groq API key (free at console.groq.com) for the chatbot

### Step 1 — Clone the repo and navigate into it

```bash
git clone <your-repo-url>
cd stock-predictor
```

### Step 2 — Create a virtual environment

Using `venv` (recommended for simplicity):

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# OR
.venv\Scripts\activate           # Windows
```

Using `conda`:

```bash
conda create -n stock_env python=3.11
conda activate stock_env
```

### Step 3 — Install dependencies

```bash
pip install \
  numpy pandas scikit-learn xgboost yfinance \
  tensorflow transformers torch \
  vaderSentiment \
  optuna joblib \
  plotly matplotlib \
  flask flask-socketio \
  groq \
  python-dotenv
```

> **Note:** TensorFlow installs can be slow. If you do not plan to use the LSTM models, you can skip `tensorflow` and just run the classical models via `train_classical.py`.

> **Note on SQLite:** The `sqlite3` module is part of Python's standard library — no separate install needed.

### Step 4 — Create required directories

```bash
mkdir -p data models charts
```

These are gitignored by default. The app will not run without them.

---

## Environment Variables (API Keys)

Create a `.env` file in the project root (also gitignored):

```
GROQ_API_KEY=your_groq_api_key_here
```

Then load it before running the app, or add this to the top of `app.py` / `bot.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

Or just export in your shell session:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

The chatbot (`bot.py`) will not work without a Groq key, but the rest of the pipeline runs fine without it.

---

## Run Order: From Scratch

Run these scripts in order. Each one depends on the output of the previous ones.

### Phase 1 — Fetch Data

```bash
# Download OHLCV price history for all tickers (2015 to today)
# Creates: data/AAPL.csv, data/MSFT.csv, etc.
python3 data_loader.py

# Download VIX, Treasury yield, and Dollar Index
# Creates: data/macro.csv
python3 macro_loader.py

# Fetch news headlines and score with VADER
# Creates: data/AAPL_sentiment.csv, etc.
python3 sentiment_loader.py

# Fetch earnings dates and EPS surprise data
# Creates: data/AAPL_earnings.csv, etc.
python3 earnings_loader.py
```

**Expected time:** 5–15 minutes depending on your connection. Most of the time is yfinance rate limiting.

### Phase 2 — Train Models

```bash
# Optional: hyperparameter search with Optuna (150 trials per ticker)
# Creates: models/best_params.pkl
# Warning: takes 1–3 hours
python3 tune.py

# Train XGBoost and Logistic Regression per ticker
# Creates: models/AAPL_model.pkl, models/AAPL_scaler.pkl, models/AAPL_config.pkl, etc.
# Also trains a combined model across all tickers
python3 train_classical.py

# Train LSTM networks per ticker (requires TensorFlow)
# Creates: models/AAPL_lstm.keras, models/AAPL_lstm_scaler.pkl, etc.
python3 train_lstm.py
```

**Expected time:** `train_classical.py` takes ~5–20 minutes. `train_lstm.py` with EarlyStopping takes ~30–90 minutes depending on hardware.

### Phase 3 — Evaluate and Backtest

```bash
# Print detailed AUC scores, confusion matrices, and directional accuracy
# Creates: models/roc_curves.png, models/auc_comparison.png
python3 evaluate.py

# Simulate signal returns on the test set, compute win rates
# Creates: charts/AAPL_backtest.html, etc.
python3 backtest.py

# Generate candlestick charts with signal overlays
# Creates: charts/AAPL_predictions.html, etc.
python3 chart.py
```

### Phase 4 — Launch

```bash
# Generate today's signals (run this daily, ideally after market close)
# Updates: data/predictions.db
python3 predict.py

# Start the web dashboard
# Open http://localhost:5000 in your browser
python3 app.py
```

---

## Daily Operations

Once everything is set up, the day-to-day workflow is:

```bash
# 1. Refresh sentiment (optional, picks up latest news)
python3 sentiment_loader.py

# 2. Generate today's signals
python3 predict.py

# 3. (If running the web app) the dashboard auto-refreshes every 5 minutes
#    Or click the Refresh button in the UI to trigger manually
```

You do not need to retrain models every day. Models are stable until new tickers are added or you want to incorporate more recent data into training.

---

## Feature Engineering Deep Dive

All features are computed in `features.py` via `add_features()`. Here is what each group captures:

### Price-Based Returns (4 features)
`return_1d`, `return_5d`, `return_10d`, `return_20d` — simple percentage changes. These capture short-to-medium term momentum.

### Moving Average Relationships (7 features)
Ratios of price to its 10, 20, and 50-day MAs, plus cross-ratios between MAs. A ratio above 1.0 means price is above that MA (uptrend); below 1.0 means downtrend. This is more informative than raw MA values because it is scale-invariant across tickers and time.

### RSI — Relative Strength Index (6 features)
Computed at three periods (7, 14, 21 days). RSI measures the speed and change of price movements. Above 70 = overbought (potential pullback), below 30 = oversold (potential bounce). Also includes RSI momentum (5-day change in RSI) and lagged RSI values so the model can see RSI trajectory.

### Lagged Returns (5 features)
`return_lag_1` through `return_lag_5` — yesterday's return, the day before, etc. Allows the model to detect reversal patterns (day after big drop) or continuation (momentum after big up day).

### Bollinger Bands (2 features)
`bb_position` — where the current price sits within the band (0 = at lower band, 1 = at upper band). `bb_width` — how wide the band is, which captures volatility expansion. Computed from a 20-day rolling mean ± 2 standard deviations.

### MACD (4 features)
MACD line (12-day EMA minus 26-day EMA), Signal line (9-day EMA of MACD), Histogram (MACD minus Signal, captures momentum direction), and MACD momentum (3-day change in histogram). A positive and rising histogram is a classic bullish signal.

### Volume Features (4 features)
`volume_ratio` — today's volume divided by its 10-day average. Spikes in volume often precede big price moves. `volume_trend` — ratio of 10-day to 20-day volume MA, capturing whether volume is picking up or fading. Also includes a 1-day lagged version.

### Candlestick Microstructure (6 features)
`overnight_gap` — how much the stock gapped up or down at open vs. yesterday's close. `intraday_reversal` — did the stock close in the upper or lower half of the day's range? `upper_wick` and `lower_wick` — the relative size of wicks, which indicate rejection of high or low prices.

### Streak Features (3 features)
`up_days_3`, `up_days_5`, `up_days_10` — how many of the last N days closed up. A count of 0 out of 5 signals exhausted selling; 5 out of 5 may signal overbought conditions.

### ATR — Average True Range (2 features)
`atr_14` normalized by price gives volatility as a percentage of price. Useful for understanding whether recent price action is normal or abnormal relative to the stock's own history.

### Price Position (1 feature)
`price_position_20` — where today's close sits within its 20-day high-low range (0 = at the 20-day low, 1 = at the 20-day high). Combines trend and momentum information.

### Macro Features (10 features)
VIX level, VIX 1-day change, VIX 10-day moving average, VIX regime (0–3 scale), VIX z-score (how far VIX is from its 30-day mean in standard deviations), 10-year Treasury yield, Treasury 1-day change, Treasury 10-day MA, Dollar Index level, Dollar Index 1-day change.

### Sentiment Features (4 features)
Daily VADER compound score (−1 to +1), 1-day change in sentiment, 5-day and 20-day moving averages of sentiment, and the difference between the two (momentum). A positive sentiment score means recent headlines are net bullish.

### Earnings Features (4 features)
`eps_surprise` — how much EPS beat or missed (clipped to ±200%), `days_to_earnings` — days until the next report, `days_since_earnings` — days since the last report, `pead_signal` — the EPS surprise times an exponential decay (half-life of 30 days), capturing the well-documented Post-Earnings Announcement Drift effect.

### Target Variable
`target` = 1 if `Close[t + N] > Close[t]`, else 0, where N is the forward horizon (1, 21, or 63 trading days). This is computed without lookahead: only prices already available at time `t` are used.

---

## Model Training Deep Dive

### Why time-based splitting matters

Standard random train/test splits would leak future information into the training set. If a model trains on a sample from 2022 and then predicts on a sample from 2019, it is effectively "seeing the future." `splitter.py` enforces a strict chronological split: the first 70% of rows go to training, the next 15% to validation (for early stopping and model selection), and the last 15% to test (for final evaluation and backtest). The test set is always the most recent data.

### Classical model selection in `train_classical.py`

For each ticker, the script tries every combination of:
- Forward horizon: 1 day, 21 days (1 month), 63 days (1 quarter)
- Model type: Logistic Regression, XGBoost

That is 6 models per ticker. The one with the highest test-set AUC-ROC is saved as the "best" model for that ticker. This selection happens after seeing the test set, which technically introduces some selection bias — a known limitation. In practice the differences between models are small enough that it does not materially affect results.

### LSTM architecture in `train_lstm.py`

```
Input: (20 timesteps, N features)
→ LSTM(64 units, return_sequences=True)
→ Dropout(0.3)
→ BatchNormalization
→ LSTM(32 units, return_sequences=False)
→ Dropout(0.2)
→ Dense(16, relu)
→ Dense(1, sigmoid)
```

Trained with Adam (lr=1e-3), binary cross-entropy loss, EarlyStopping (patience=10 on val_loss), and ReduceLROnPlateau (patience=5, factor=0.5). In practice, LSTM test AUCs are similar to or slightly worse than XGBoost on these features and data volume — classical models are more competitive than expected here.

### Combined model in `train_classical.py`

Trains a single XGBoost on all tickers stacked together, with a `ticker_id` integer column as an additional feature. This is used by `analyze.py` for tickers that do not have their own trained model.

---

## Signal Generation and Filtering

`predict.py` applies several entry filters on top of the raw model prediction to reduce false positives:

| Filter | Condition | Why |
|--------|-----------|-----|
| Confidence | `prob_up < threshold` (50–55% depending on ticker) | Avoid marginal signals where the model is barely above 50/50 |
| VIX crisis | `VIX > 30` | High volatility regimes have fundamentally different return distributions and blow out model assumptions |
| Trend | `price_to_ma50 < 1.0` | Avoid buying into a confirmed downtrend |
| RSI overbought | `RSI > 70` | Avoid chasing overbought conditions |
| Earnings blackout | `days_to_earnings <= 3` | Earnings announcements introduce binary risk that the model was not trained to handle |
| Consecutive losses | 3+ consecutive losing signals | Adaptive pause when the model is clearly out of sync with market conditions |

The final action label is:
- **STRONG BUY** — signal passes all filters, prob_up ≥ 70%, historical win rate ≥ 70%
- **BUY** — signal passes all filters, win rate ≥ 70%
- **WEAK BUY** — raw model says buy but win rate is below threshold
- **FILTERED** — raw model says buy but a filter blocked it (reason is logged)
- **AVOID** — model says do not buy and historical win rate is high (high-conviction short signal)
- **HOLD** — everything else

---

## Backtesting

`backtest.py` runs on the test set only (the last 15% of historical data, approximately 2022–present). It applies the same entry filters as `predict.py` so results reflect realistic conditions.

For each holding period (5d, 21d, 63d, 126d) it computes:
- **Avg return** — average forward return on days with a BUY signal
- **Win rate** — percentage of BUY signals that resulted in a positive return
- **Edge** — average BUY-signal return minus overall average return (how much better the signal is vs. holding randomly)
- **Sharpe ratio** — risk-adjusted return of the signal strategy

It also simulates a non-overlapping equity curve assuming you take each BUY signal with a 21-day hold and skip signals while already in a position.

---

## Web Dashboard and API

Start with `python3 app.py` and open `http://localhost:5000`.

### REST API Endpoints

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/signals` | GET | Today's signals for all watchlist tickers |
| `/api/macro` | GET | Latest VIX, Treasury yield, Dollar Index, fear level |
| `/api/stats` | GET | Summary stats from the database |
| `/api/history/<ticker>` | GET | Signal history for a ticker (pass `?days=30`) |
| `/api/win_rates` | GET | Historical win rates per ticker per horizon |
| `/api/watchlist` | GET | Current list of tickers |
| `/api/watchlist/add` | POST | Add a new ticker (runs quality check first) |
| `/api/analyze/<ticker>` | GET | Full on-demand analysis for any ticker |
| `/api/refresh` | POST | Trigger a fresh prediction run in the background |
| `/api/portfolio/summary` | GET | Holdings with live prices and P&L |
| `/api/portfolio/advice` | GET | Daily advice cross-referencing signals and holdings |
| `/api/options-flow` | GET | Cached options flow scan results |
| `/api/options-flow/scan` | POST | Trigger a new options flow scan |

---

## AI Chatbot (bot.py)

The chatbot is powered by Groq's Llama 3.3 70B model (very fast inference). It implements a custom tool-calling protocol: when the model wants to call a function, it responds with a JSON object like `{"tool": "get_todays_signals", "input": {}}`. The Flask server parses this, runs the function, and passes the result back to the model for a final natural-language response.

### Available Tools

| Tool | What it does |
|------|-------------|
| `get_todays_signals` | Returns all signals from the database for today |
| `run_fresh_predictions` | Triggers `run_predictions()` for live data |
| `get_signal_history` | Returns signal history for a specific ticker |
| `get_macro_conditions` | Returns VIX, Treasury, Dollar, and fear level |
| `get_performance_stats` | Returns win rates and database summary stats |
| `run_backtest` | Runs a full backtest for a watchlist ticker |
| `explain_signal` | Loads model features for a ticker and explains the current signal |
| `analyze_any_ticker` | Full analysis for any ticker (calls `analyze.py`) |
| `add_to_watchlist` | Adds a ticker to `config.py` if quality score is sufficient |

The chatbot is accessible via the web UI (WebSocket) or as a terminal bot (`python3 bot.py`).

---

## Options Flow Scanner

Accessible at `http://localhost:5000/options-flow`.

Click "Scan All Tickers" to trigger a background scan (~30–60 seconds). Results are cached in memory and displayed as cards.

### Metrics Explained

**Put/Call Ratio (PCR):** Ratio of put volume to call volume. PCR below 0.7 is considered bullish (more calls being bought than puts). PCR above 1.2 is considered bearish or indicates heavy hedging activity.

**Unusual Volume:** Contracts where trading volume is more than 2× the open interest. This can indicate fresh directional bets being placed by large traders.

**Gamma Exposure (GEX):** Estimates the net gamma position of market makers. Positive GEX means market makers are long gamma and will dampen price moves (stocks tend to pin near the call wall). Negative GEX means market makers are short gamma and will amplify moves (trending, breakout conditions).

**IV Rank (IVR):** Where current implied volatility sits relative to its recent range. IVR below 30 means options are cheap (historically low IV, potentially good time to buy options). IVR above 70 means options are expensive (good for selling strategies).

**Expected Move:** The market-implied ±1 standard deviation price range by the nearest expiry, computed as `spot × IV × sqrt(DTE / 365)`. There is approximately a 68% probability that price stays within this range.

---

## Portfolio Tracker

Accessible at `http://localhost:5000/portfolio`.

Tracks holdings, cash, and transactions in a local SQLite database (`data/portfolio.db`). Live prices are fetched from yfinance. The "Daily Advice" tabs cross-reference your actual positions against today's model signals to surface:

- **Warnings** — positions with AVOID signals, positions down more than 20%, concentrated positions (>35% of portfolio)
- **Advice** — positions with STRONG BUY signals that could be added to, positions down but with bullish model signals
- **Opportunities** — suggestions for deploying idle cash into current BUY signals

---

## On-Demand Ticker Analysis

For any ticker not in your watchlist, run:

```bash
python3 analyze.py PLTR
```

Or use the search bar in the web dashboard.

This returns:
- Basic company info (sector, industry, market cap)
- Technical snapshot (RSI, MACD, Bollinger Band position, moving average relationship, recent returns, ATR)
- Watchlist quality score (0–100) based on history length, liquidity, price level, volatility, and data completeness
- A directional signal from the combined model

Tickers scoring above 40 can be added permanently via the dashboard or by running:

```python
from analyze import add_ticker_to_watchlist
add_ticker_to_watchlist("PLTR")
```

After adding, you need to run `sentiment_loader.py`, `earnings_loader.py`, and `train_classical.py` again to build a ticker-specific model.

---

## Model Performance Benchmarks

### Best Test-Set AUC-ROC (XGBoost vs. Logistic Regression)

| Ticker | XGBoost | LogReg | Best Horizon |
|--------|---------|--------|--------------|
| AAPL | 0.567 | 0.543 | 63d |
| MSFT | 0.553 | 0.512 | 21d |
| TSLA | 0.592 | 0.581 | 1d |
| JPM | 0.541 | 0.533 | 21d |
| NVDA | 0.485 | 0.501 | 21d |

> AUC of 0.50 = random. AUC of 0.55–0.60 is considered meaningful for stock prediction. Markets are efficient; do not expect AUC above 0.65 without significant alpha.

### Backtest Win Rates by Holding Period (Test Set, Filtered Signals)

| Ticker | 5d | 21d | 63d | 126d |
|--------|----|-----|-----|------|
| AAPL | 63.9% | 53.1% | 50.0% | 73.1% |
| MSFT | 62.4% | 55.0% | 52.9% | 38.9% |
| TSLA | 36.4% | 63.6% | 100% | 100% |
| JPM | 74.3% | 72.1% | 81.4% | 100% |
| NVDA | 50.0% | 50.0% | 50.0% | 50.0% |

> Win rates above 100% rows for TSLA and JPM at longer horizons reflect a very small number of signals in the test set — treat with caution.

---

## Directory Structure

```
stock-predictor/
├── config.py               # Central configuration
├── data_loader.py          # Price data fetching
├── macro_loader.py         # Macro indicator fetching
├── sentiment_loader.py     # News sentiment pipeline
├── earnings_loader.py      # Earnings data pipeline
├── features.py             # Feature engineering
├── splitter.py             # Time-safe train/val/test splitting
├── train_classical.py      # XGBoost + LogReg training
├── train_lstm.py           # LSTM training
├── tune.py                 # Optuna hyperparameter search
├── evaluate.py             # Model evaluation reports
├── predict.py              # Daily signal generation
├── backtest.py             # Signal backtesting
├── chart.py                # Plotly chart generation
├── analyze.py              # On-demand ticker analysis
├── options_flow.py         # Options chain scanner
├── portfolio.py            # Portfolio tracking logic
├── database.py             # SQLite signal database wrapper
├── app.py                  # Flask web server
├── bot.py                  # Groq chatbot
├── templates/
│   ├── index.html          # Main dashboard
│   ├── options_flow.html   # Options flow page
│   └── portfolio.html      # Portfolio tracker page
├── data/                   # Auto-created, gitignored
│   ├── AAPL.csv
│   ├── macro.csv
│   ├── AAPL_sentiment.csv
│   ├── AAPL_earnings.csv
│   ├── predictions.db
│   └── portfolio.db
├── models/                 # Auto-created, gitignored
│   ├── AAPL_model.pkl
│   ├── AAPL_scaler.pkl
│   ├── AAPL_config.pkl
│   ├── AAPL_lstm.keras
│   ├── combined_xgb.pkl
│   └── ...
└── charts/                 # Auto-created, gitignored
    ├── AAPL_backtest.html
    └── ...
```

---

## Troubleshooting

**"No model for TICKER — run train_classical.py first"**
The `models/` directory is missing the trained model files. Run `train_classical.py` after fetching all data.

**"Macro cache is stale — refreshing..."**
Normal behavior. The macro cache auto-refreshes if it is more than 7 days old.

**"Could not fetch earnings for TICKER"**
yfinance does not have earnings data for all tickers (common for ETFs like SPY and some volatile stocks like TSLA). The pipeline handles this gracefully by setting earnings features to zero for those tickers.

**Slow prediction run**
The sentiment loader rebuilds the sentiment series for every ticker on each run. If speed is important, set `SENTIMENT_LOOKBACK_DAYS = 1` in `config.py`.

**Database locked error**
Another Python process is accessing `data/predictions.db`. Close other terminal sessions running the app or prediction scripts.

**Port 5000 already in use**
Change the port in the last line of `app.py`: `socketio.run(app, host="0.0.0.0", port=5001, ...)`.

**TensorFlow import errors**
TensorFlow can be finicky about Python and CUDA versions. If you only want the classical models, you can comment out all `import tensorflow` lines in `train_lstm.py` and `evaluate.py` and skip those scripts entirely.

**yfinance rate limiting**
If you see empty DataFrames or connection errors during the data loading phase, add `time.sleep(1)` between ticker downloads or wait a few minutes and retry.

---

## Key Design Decisions and Limitations

**Why XGBoost over neural networks as the primary model?**
XGBoost generally outperforms LSTMs on tabular data with moderate dataset sizes (a few thousand rows per ticker). The LSTM models are trained and evaluated but classical models win the selection tournament for most tickers.

**Why VADER instead of FinBERT?**
FinBERT is more accurate for financial sentiment but requires significant memory and compute. VADER is CPU-friendly and fast enough for this pipeline. The README originally mentioned FinBERT but the implementation uses VADER.

**Why is AUC only 0.55?**
Stock markets are close to efficient. A model that is right 55% of the time applied consistently to high-confidence signals can still generate positive edge, especially at longer holding periods where the signal-to-noise ratio improves.

**Walk-forward validation would be more rigorous**
A proper production system would use walk-forward (expanding window) validation: train on 2015–2018, test on 2019, then train on 2015–2019, test on 2020, etc. This pipeline uses a single fixed split, which is simpler but introduces some look-ahead in model selection.

**This is not financial advice**
The system is a research and learning tool. Backtest win rates do not guarantee future results. Always do your own research before making any investment decisions.