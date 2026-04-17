# Stock Prediction Pipeline

A comprehensive machine learning pipeline for predicting next-day stock price direction using XGBoost and LSTM models. Combines technical analysis, macroeconomic indicators, news sentiment, and earnings data to generate trading signals for a watchlist of 10+ stocks with a Flask-based web interface and AI chatbot.

## Overview

This system trains and deploys ensemble models to predict stock price direction:
- **Base Models**: XGBoost and Logistic Regression (technical/macro/sentiment features)
- **Deep Learning**: Stacked LSTM networks for time-series patterns
- **Signal Generation**: Real-time predictions with confidence scoring and portfolio backtesting
- **Web Interface**: Flask app with live signals, performance metrics, and AI chatbot
- **Live Chatbot**: Groq Llama-based assistant that can analyze any ticker on demand

## Supported Tickers

**Core Watchlist** (10): AAPL, MSFT, TSLA, JPM, NVDA, GOOGL, AMZN, META, SPY, AMD, GME

Can analyze any additional ticker on demand via `analyze_any_ticker()`.

## Features

**27 Total Features** engineered from:

- **Technical Indicators** (13): RSI, MACD, Bollinger Bands, 20/50-day moving averages, returns, volume, ATR, CCI, ADX, OBV
- **Macroeconomic** (6): VIX level + regime + z-score, 10-year Treasury yield, USD index (DXY), ES 100 (S&P 100)
- **Sentiment** (4): FinBERT-scored news sentiment over 3-day lookback window
- **Earnings** (4, optional): EPS surprise, revenue surprise, days since last earnings, earnings coming this week
- **Price Action** (2): Log returns (1d, 5d)

## Core Python Files

| File | Purpose |
|------|---------|
| **config.py** | Central configuration: tickers, date ranges, thresholds, feature windows |
| **data_loader.py** | Fetches OHLCV data from yfinance for all tickers (2015-present), caches as CSV |
| **macro_loader.py** | Fetches macro indicators: VIX, 10Y Treasury yield, USD index; caches daily |
| **sentiment_loader.py** | Fetches news headlines, scores with FinBERT, aggregates 3-day sentiment rolling window |
| **earnings_loader.py** | Loads quarterly earnings dates and EPS/revenue surprises from yfinance |
| **features.py** | Core feature engineering: computes all 27 features from raw OHLCV + macro/sentiment/earnings data |

## Model Training Files

| File | Purpose |
|------|---------|
| **splitter.py** | Time-safe train/validation/test split (70/15/15) respecting temporal order to prevent lookahead bias |
| **train_classical.py** | Trains Logistic Regression and XGBoost classifiers per ticker, saves models + scaler + config |
| **train_lstm.py** | Trains stacked LSTM networks per ticker with 20/50-day lookback windows, uses Keras/TensorFlow |
| **tune.py** | Optuna-based hyperparameter tuning for XGBoost (150 trials per ticker, optimizes AUC on validation set) |
| **evaluate.py** | Generates detailed evaluation reports: AUC-ROC, directional accuracy, confusion matrices, holding period analysis |

## Prediction & Analysis Files

| File | Purpose |
|------|---------|
| **predict.py** | Runs daily predictions on watchlist tickers; loads latest data, applies feature pipeline, queries models, generates buy/sell signals with probability scores, stores in SQLite database |
| **backtest.py** | Backtests signal returns over holding periods (1d, 21d, 63d, 126d); computes win rates per ticker; applies VIX/trend/RSI/earnings filters; generates HTML charts showing equity curves and holding period analysis |
| **analyze.py** | On-demand analysis tool for any ticker (in or out of watchlist): fetches price data, runs technical analysis, scores watchlist suitability (liquidity, volatility, history), applies combined model, recommends whether to add permanently |

## Web & API Files

| File | Purpose |
|------|---------|
| **app.py** | Flask web server with SocketIO; routes for live signals API, macro data API, performance stats, signal history; serves [templates/index.html](templates/index.html) frontend |
| **bot.py** | Groq Llama 3.3 70B chatbot integrated with Flask; system prompt has tools: calls `run_predictions()`, `get_todays_signals()`, `analyze_any_ticker()`, `add_to_watchlist()`; can explain signals and analyze arbitrary tickers |
| **database.py** | SQLite wrapper (data/predictions.db): `get_todays_signals()`, `get_signal_history()`, `get_signal_accuracy()`, `get_actionable_signals()`, `get_recent_performance()`; tracks signal outcomes |

## Visualization & Charting

| File | Purpose |
|------|---------|
| **chart.py** | Generates interactive candlestick charts with technical overlays (moving averages, Bollinger Bands, RSI) using Plotly; saves per-ticker HTML charts in [charts/](charts/) directory |

## Directory Structure

```
/data/                          # Raw data and database
├── {TICKER}.csv               # Historical OHLCV data
├── {TICKER}_earnings.csv      # Earnings dates and surprises
├── {TICKER}_sentiment.csv     # Aggregated sentiment scores
├── macro.csv                  # Macro indicator cache (VIX, Treasury, DXY)
└── predictions.db             # SQLite: signals, outcomes, accuracy tracking

/models/                        # Trained model artifacts
├── {TICKER}_model.pkl         # XGBoost classifier
├── {TICKER}_scaler.pkl        # StandardScaler for features
├── {TICKER}_config.pkl        # Model config (feature columns)
├── {TICKER}_lstm.keras        # LSTM model (TensorFlow/Keras)
├── combined_xgb.pkl           # Combined ensemble model (for analyze.py)
└── combined_scaler.pkl        # Scaler for combined model

/charts/                        # Generated Plotly HTML charts
├── {TICKER}_backtest.html     # Equity curves from backtest results
├── {TICKER}_predictions.html  # Price + signals + technicals
└── {TICKER}_holding_periods.html  # Win rate by holding period

/templates/                     # Flask frontend templates
└── index.html                 # Real-time dashboard with WebSocket updates

/notebooks/                     # Jupyter notebooks (experimental analysis)
```

## Setup & Installation

### Prerequisites
- Python 3.11+
- conda or venv

### Environment Setup
```bash
conda create -n stock_env python=3.11
conda activate stock_env
pip install \
    numpy pandas scikit-learn xgboost yfinance tensorflow transformers torch \
    optuna joblib plotly matplotlib flask flask-socketio groq sqlite3
```

### API Keys (optional)
```bash
# For news sentiment (Massive/Polygon API)
export MASSIVE_API_KEY="your_massive_api_key_here"

# For chatbot (Groq)
export GROQ_API_KEY="your_groq_api_key_here"
```

## Run Order (From Scratch)

### Phase 1: Data Loading
```bash
# Fetch OHLCV data from yfinance (2015-present)
python3 data_loader.py

# Fetch macro indicators (VIX, 10Y Treasury, USD index)
python3 macro_loader.py

# Fetch and score news sentiment via FinBERT
python3 sentiment_loader.py

# Load earnings dates and surprises
python3 earnings_loader.py
```

### Phase 2: Model Training
```bash
# Hyperparameter tuning with Optuna (optional, takes time)
python3 tune.py

# Train XGBoost + Logistic Regression per ticker
python3 train_classical.py

# Train LSTM networks per ticker
python3 train_lstm.py
```

### Phase 3: Evaluation & Backtesting
```bash
# Generate detailed performance reports
python3 evaluate.py

# Backtest signals over multiple holding periods (1d, 21d, 63d, 126d)
python3 backtest.py

# Generate interactive price + signal charts
python3 chart.py
```

### Phase 4: Daily Operations
```bash
# Generate today's buy/sell signals (run daily)
python3 predict.py

# Launch web interface (with live dashboard + chatbot)
python3 app.py
# Navigate to http://localhost:5000
```

### On-Demand Analysis
```bash
# Analyze any ticker (in or out of watchlist)
python3 analyze.py
```

## Model Performance

### Best Test-Set Results (AUC-ROC)
| Ticker | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| AAPL   | 0.567   | 0.543               |
| MSFT   | 0.553   | 0.512               |
| TSLA   | 0.592   | 0.581               |
| JPM    | 0.541   | 0.533               |
| NVDA   | 0.485   | 0.501               |

### Backtest Win Rates by Holding Period
| Ticker | 1-Day | 21-Day | 63-Day | 126-Day |
|--------|-------|--------|--------|---------|
| AAPL   | 63.9% | 53.1%  | 50.0%  | 73.1%   |
| MSFT   | 62.4% | 55.0%  | 52.9%  | 38.9%   |
| TSLA   | 36.4% | 63.6%  | 100%   | 100%    |
| JPM    | 74.3% | 72.1%  | 81.4%  | 100%    |
| NVDA   | 50.0% | 50.0%  | 50.0%  | 50.0%   |

## Feature Engineering Details

### Technical Features (computed per ticker)
- **Momentum**: RSI(14), MACD, Bollinger Band %B, CCI, ADX
- **Trend**: 20-day and 50-day moving averages, log returns
- **Volatility**: Bollinger Band width, ATR
- **Volume**: OBV (On-Balance Volume)

### Macro Features (shared across tickers)
- **Volatility Index**: VIX level, regime classification (low/normal/elevated/high), z-score
- **Rates**: 10-year Treasury yield
- **Currency**: USD Index (DXY), S&P 100 futures (ES)

### Sentiment Features (3-day rolling average)
FinBERT scores news headlines as bullish/neutral/bearish, aggregated into rolling window.

### Earnings Features (per ticker, except TSLA/SPY)
- EPS surprise (%) clipped to [-200%, +200%]
- Revenue surprise (%) clipped to [-200%, +200%]
- Days since last earnings release
- Boolean: earnings report coming this week

## Signal Generation

Predictions from XGBoost/LSTM models combined to generate actionable signals:

**BUY signals** generated when:
- Model probability of up-move ≥ ticker-specific threshold (50-55%)
- VIX < 30 (optional macro filter)
- Technical filters pass (optional): 20-day uptrend, RSI not overbought, earnings not conflicting

**Confidence Scoring**: Probability of up-move from 0-100%

**Filtering**: Per-ticker win rates by holding period recommend optimal position duration

## Web Dashboard & API

### Flask Routes
- `GET /` → Real-time dashboard (HTML)
- `GET /api/signals` → Today's buy/sell signals (JSON)
- `GET /api/macro` → Latest macro data + fear gauge
- `GET /api/performance` → Win rates, accuracy stats per ticker
- `WebSocket /chat` → Groq-powered chatbot for analysis

### Chatbot Features
- Explain any signal in the database
- Analyze any ticker on demand (fetch fresh data, run technical + ML analysis)
- Recommend watchlist additions
- Query historical signal accuracy
- Discuss model methodology and limitations

## Key Assumptions & Limitations

1. **Time-safe splits**: Train/val/test splits respect temporal order (no lookahead bias)
2. **Feature stationarity**: All features normalized per ticker with StandardScaler
3. **Forward bias mitigation**: Only use features computable at market open (no future earnings/sentiment)
4. **Macro features**: Shared across tickers, updated daily
5. **Earnings data**: Optional per ticker; TSLA/SPY excluded due to data gaps
6. **LSTM architecture**: 20 and 50-day lookback windows with stacked layers
7. **Signal filters**: Macro, technical, and earnings filters can be toggled per backtest

## Development Notes

- **Sentiment source**: News headlines via Massive/Polygon API, scored with FinBERT
- **Macro data source**: yfinance (^VIX, ^TNX, DX-Y.NYB)
- **Price data source**: yfinance OHLCV
- **Earnings data source**: yfinance earnings_dates API
- **Model serialization**: joblib (XGBoost/scalers), Keras (LSTM)
- **Feature caching**: None (recomputed each run; set SENTIMENT_LOOKBACK_DAYS in config.py to balance recency vs speed)
- **Database**: SQLite for signal history and outcome tracking

## Troubleshooting

**Missing data**: Check `data/` directory; rerun `data_loader.py`, `macro_loader.py`, `sentiment_loader.py`

**Model not found**: Ensure `models/` directory exists and models are trained; run `train_classical.py` and `train_lstm.py`

**API errors**: Check environment variables (`MASSIVE_API_KEY`, `GROQ_API_KEY`) and API rate limits

**Database locked**: Close other Python processes accessing `data/predictions.db`

**Slow predictions**: Sentiment scoring can be slow; set SENTIMENT_LOOKBACK_DAYS to 1 in config.py for speed