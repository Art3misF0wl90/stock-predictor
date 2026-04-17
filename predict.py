import os
import joblib
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date
import yfinance as yf

from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from earnings_loader import build_earnings_features
from features import add_features, get_feature_columns
from config import TICKERS

MIN_WIN_RATE_THRESHOLD = 0.70

TICKER_WIN_RATES = {
    "AAPL": {"1d": 0.686, "21d": 0.686, "63d": 0.789},
    "MSFT": {"1d": 0.848, "21d": 0.957, "63d": 0.943},
    "TSLA": {"1d": 0.552, "21d": 0.552, "63d": 0.464},
    "JPM":  {"1d": 0.619, "21d": 0.952, "63d": 1.000},
    "NVDA": {"1d": 0.658, "21d": 1.000, "63d": 1.000},
}

TICKERS_NO_EARNINGS = ["TSLA"]

def setup_database():
    conn = sqlite3.connect("data/predictions.db")
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            signal      INTEGER NOT NULL,
            prob_up     REAL NOT NULL,
            horizon     TEXT NOT NULL,
            win_rate    REAL NOT NULL,
            action      TEXT NOT NULL,
            close_price REAL,
            model_name  TEXT,
            fwd_days    INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id   INTEGER,
            ticker      TEXT,
            entry_date  TEXT,
            entry_price REAL,
            exit_date   TEXT,
            exit_price  REAL,
            return_pct  REAL,
            was_correct INTEGER,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    """)
    conn.commit()
    conn.close()

def fetch_latest_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="300d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)
    df.dropna(inplace=True)
    return df

def get_today_signal(ticker, macro_df, sentiment, earnings=None):
    model_path  = os.path.join("models", f"{ticker}_model.pkl")
    scaler_path = os.path.join("models", f"{ticker}_scaler.pkl")
    config_path = os.path.join("models", f"{ticker}_config.pkl")

    if not os.path.exists(model_path):
        print(f"  No model for {ticker} — run train_classical.py first")
        return None

    model      = joblib.load(model_path)
    scaler     = joblib.load(scaler_path)
    cfg        = joblib.load(config_path)
    fwd_days   = cfg["fwd_days"]
    model_name = cfg["model_name"]
    feat_cols  = cfg["feat_cols"]

    df = fetch_latest_data(ticker)
    if df.empty:
        return None

    sentiment_series = sentiment.get(ticker) if sentiment else None

    earnings_df = (
        None if ticker in TICKERS_NO_EARNINGS
        else build_earnings_features(ticker, df)
    )

    df_feat = add_features(df,
                           macro_df=macro_df,
                           sentiment_series=sentiment_series,
                           earnings_df=earnings_df,
                           forward_days=fwd_days,
                           predict_mode=True)

    if df_feat.empty:
        print(f"  Warning: empty features for {ticker}")
        return None

    feat_cols = [c for c in feat_cols if c in df_feat.columns]

    if not feat_cols:
        print(f"  Warning: no matching feature columns for {ticker}")
        return None

    X_today  = df_feat[feat_cols].iloc[[-1]].values
    X_scaled = scaler.transform(X_today)
    prob_up  = float(model.predict_proba(X_scaled)[0][1])
    signal   = int(prob_up >= 0.5)

    horizon_map = {1: "1d", 21: "21d", 63: "63d", 126: "126d"}
    horizon     = horizon_map.get(fwd_days, f"{fwd_days}d")
    win_rate    = float(TICKER_WIN_RATES.get(ticker, {}).get(horizon, 0.5))

    if signal == 1 and win_rate >= MIN_WIN_RATE_THRESHOLD:
        action = "STRONG BUY" if prob_up >= 0.7 else "BUY"
    elif signal == 1:
        action = "WEAK BUY"
    elif signal == 0 and win_rate >= MIN_WIN_RATE_THRESHOLD:
        action = "AVOID"
    else:
        action = "HOLD"

    return {
        "ticker":      ticker,
        "date":        str(date.today()),
        "signal":      signal,
        "prob_up":     float(round(prob_up, 4)),
        "horizon":     horizon,
        "win_rate":    win_rate,
        "action":      action,
        "close_price": float(df_feat["Close"].iloc[-1]),
        "model_name":  model_name,
        "fwd_days":    fwd_days,
    }

def save_signals(signals):
    conn  = sqlite3.connect("data/predictions.db")
    c     = conn.cursor()
    today = str(date.today())
    c.execute("DELETE FROM signals WHERE date = ?", (today,))
    for s in signals:
        if s is None:
            continue
        c.execute("""
            INSERT INTO signals
            (date, ticker, signal, prob_up, horizon,
             win_rate, action, close_price, model_name, fwd_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s["date"], s["ticker"], s["signal"], s["prob_up"],
            s["horizon"], s["win_rate"], s["action"],
            s["close_price"], s["model_name"], s["fwd_days"]
        ))
    conn.commit()
    conn.close()

def print_signal_report(signals):
    today    = datetime.now().strftime("%Y-%m-%d")
    valid    = [s for s in signals if s is not None]

    print(f"\n{'═'*65}")
    print(f"  DAILY SIGNALS — {today}")
    print(f"  Models evaluated: {len(valid)}/{len(signals)}")
    print(f"{'═'*65}")
    print(f"  {'Ticker':<8} {'Action':<12} {'Confidence':>11} "
          f"{'Win rate':>10} {'Horizon':>8} {'Price':>8} {'Model':>8}")
    print(f"  {'─'*62}")

    priority = {"STRONG BUY": 0, "BUY": 1, "WEAK BUY": 2,
                "HOLD": 3, "AVOID": 4}

    signals_sorted = sorted(
        valid,
        key=lambda x: (priority.get(x["action"], 5), -x["prob_up"])
    )

    for s in signals_sorted:
        print(f"  {s['ticker']:<8} {s['action']:<12} "
              f"{s['prob_up']:>11.1%} "
              f"{s['win_rate']:>10.1%} "
              f"{s['horizon']:>8} "
              f"${s['close_price']:>7.2f} "
              f"{s['model_name']:>8}")

    print(f"\n  Actionable (win rate >= {MIN_WIN_RATE_THRESHOLD:.0%}):")
    actionable = [s for s in signals_sorted
                  if s["action"] in ("STRONG BUY", "BUY")]

    if actionable:
        for s in actionable:
            print(f"  → {s['ticker']}: {s['action']} "
                  f"| hold {s['horizon']} "
                  f"| {s['win_rate']:.0%} historical win rate "
                  f"| ${s['close_price']:.2f} "
                  f"| confidence {s['prob_up']:.1%}")
    else:
        print("  → No high-confidence buy signals today")
        print(f"\n  All signals:")
        for s in signals_sorted:
            print(f"     {s['ticker']}: {s['action']} "
                  f"(prob_up={s['prob_up']:.1%}, "
                  f"signal={s['signal']})")

    print(f"{'═'*65}\n")

def run_predictions():
    print("Loading data for prediction...")
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()

    setup_database()

    signals = []
    for ticker in TICKERS:
        print(f"  Generating signal for {ticker}...")
        signal = get_today_signal(ticker, macro_df, sentiment)
        signals.append(signal)

    save_signals(signals)
    print_signal_report(signals)
    return signals

if __name__ == "__main__":
    run_predictions()
