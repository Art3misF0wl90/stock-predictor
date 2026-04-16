# Stock Prediction Pipeline

ML pipeline predicting next-day stock price direction using XGBoost and LSTM.

## Tickers
AAPL, MSFT, TSLA, JPM, NVDA

## Features (27 total)
- Technical: RSI, MACD, Bollinger Bands, moving averages, returns, volume
- Macro: VIX (+ regime + zscore), treasury yield, dollar index
- Sentiment: FinBERT-scored news headlines via Massive/Polygon API

## Files
| File | Purpose |
|------|---------|
| `config.py` | Central settings |
| `data_loader.py` | OHLCV data via yfinance |
| `macro_loader.py` | VIX, treasury, dollar |
| `sentiment_loader.py` | News sentiment via FinBERT |
| `features.py` | Feature engineering |
| `splitter.py` | Time-safe train/val/test split |
| `train_classical.py` | Logistic Regression + XGBoost |
| `train_lstm.py` | Stacked LSTM |
| `evaluate.py` | AUC + directional accuracy report |
| `chart.py` | Interactive candlestick charts |

## Setup
```bash
conda create -n stock_env python=3.11
conda activate stock_env
pip install numpy pandas scikit-learn xgboost yfinance tensorflow \
            transformers torch polygon-api-client joblib plotly matplotlib
export MASSIVE_API_KEY="your_key_here"
```

## Run order
```bash
python3 data_loader.py
python3 macro_loader.py
python3 sentiment_loader.py
python3 train_classical.py
python3 train_lstm.py
python3 evaluate.py
python3 chart.py
```

## Best results (Test AUC)
| Model | AAPL | MSFT | TSLA | JPM | NVDA |
|-------|------|------|------|-----|------|
| XGBoost | 0.547 | 0.533 | 0.592 | 0.531 | 0.485 |
| LSTM | 0.541 | 0.503 | 0.544 | 0.513 | 0.510 |