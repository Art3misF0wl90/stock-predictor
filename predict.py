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
# Per-ticker minimum confidence — some tickers need lower threshold
TICKER_MIN_PROB = {
    "AAPL":  0.55,
    "MSFT":  0.55,
    "TSLA":  0.55,
    "JPM":   0.55,
    "NVDA":  0.50,  # lower — model is very conservative on NVDA
    "GOOGL": 0.50,  # lower — model is very conservative on GOOGL
    "AMZN":  0.52,
    "META":  0.52,
    "SPY":   0.55,
    "AMD":   0.55,
}
MIN_PROB               = 0.55
MAX_VIX                = 30
TICKERS_NO_EARNINGS    = ["TSLA", "SPY"]

TICKER_WIN_RATES = {
    "AAPL":  {"1d": 0.639, "21d": 0.531, "63d": 0.500, "126d": 0.731},
    "MSFT":  {"1d": 0.624, "21d": 0.550, "63d": 0.529, "126d": 0.389},
    "TSLA":  {"1d": 0.364, "21d": 0.636, "63d": 1.000, "126d": 1.000},
    "JPM":   {"1d": 0.743, "21d": 0.721, "63d": 0.814, "126d": 1.000},
    "NVDA":  {"1d": 0.500, "21d": 0.500, "63d": 0.500, "126d": 0.500},
    "GOOGL": {"1d": 0.500, "21d": 0.500, "63d": 0.500, "126d": 0.500},
    "AMZN":  {"1d": 1.000, "21d": 1.000, "63d": 1.000, "126d": 1.000},
    "META":  {"1d": 1.000, "21d": 1.000, "63d": 1.000, "126d": 0.000},
    "SPY":   {"1d": 0.750, "21d": 0.500, "63d": 0.500, "126d": 0.500},
    "AMD":   {"1d": 0.633, "21d": 0.400, "63d": 0.900, "126d": 0.967},
}

TICKER_BEST_HORIZON = {
    "AAPL": "126d",
    "MSFT": "126d",
    "TSLA": "126d",
    "JPM":  "126d",
    "NVDA": "126d",
}

def setup_database():
    conn = sqlite3.connect("data/predictions.db")
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            signal        INTEGER NOT NULL,
            prob_up       REAL NOT NULL,
            horizon       TEXT NOT NULL,
            win_rate      REAL NOT NULL,
            action        TEXT NOT NULL,
            close_price   REAL,
            model_name    TEXT,
            fwd_days      INTEGER,
            filter_reason TEXT,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
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

def apply_entry_filters(row, prob_up: float, ticker: str = "") -> tuple:
    """
    Applies entry filters to a buy signal.
    Returns (passes_filter, filter_reason).
    """
# Per-ticker confidence filter
    ticker_min = TICKER_MIN_PROB.get(ticker, MIN_PROB)
    if prob_up < ticker_min:
        return False, f"low_confidence ({prob_up:.1%} < {ticker_min:.0%})"

    # VIX crisis filter
    if "vix" in row.index:
        vix = row.get("vix", 0)
        if not pd.isna(vix) and float(vix) > MAX_VIX:
            return False, f"high_vix ({float(vix):.1f} > {MAX_VIX})"

    # Trend filter — price below 50-day MA
    if "price_to_ma50" in row.index:
        ptma = row.get("price_to_ma50", 1.0)
        if not pd.isna(ptma) and float(ptma) < 1.0:
            return False, f"below_50ma ({float(ptma):.3f})"

    # RSI overbought filter
    if "rsi_14" in row.index:
        rsi = row.get("rsi_14", 50)
        if not pd.isna(rsi) and float(rsi) > 70:
            return False, f"overbought_rsi ({float(rsi):.1f})"

    # Earnings blackout filter
    if "days_to_earnings" in row.index:
        dte = row.get("days_to_earnings", 90)
        if not pd.isna(dte) and 0 < float(dte) <= 3:
            return False, f"earnings_in_{int(dte)}d"

    return True, None

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

    # Use earnings only if the saved model was trained with them
    has_earnings = any(
        "eps" in c or "pead" in c or "earnings" in c
        for c in feat_cols
    )
    earnings_df = (
        None if not has_earnings
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
    raw_signal = int(prob_up >= 0.5)

    horizon_map = {1: "1d", 21: "21d", 63: "63d", 126: "126d"}
    horizon     = horizon_map.get(fwd_days, f"{fwd_days}d")
    win_rate    = float(TICKER_WIN_RATES.get(ticker, {}).get(horizon, 0.5))

    # Apply entry filters
    latest        = df_feat.iloc[-1]
    filter_reason = None
    final_signal  = raw_signal

    if raw_signal == 1:
        passes, filter_reason = apply_entry_filters(latest, prob_up, ticker)
        if not passes:
            final_signal = 0

    # Determine action
    if final_signal == 1 and win_rate >= MIN_WIN_RATE_THRESHOLD:
        action = "STRONG BUY" if prob_up >= 0.7 else "BUY"
    elif raw_signal == 1 and final_signal == 0:
        action = f"FILTERED ({filter_reason})"
    elif raw_signal == 1:
        action = "WEAK BUY"
    elif raw_signal == 0 and win_rate >= MIN_WIN_RATE_THRESHOLD:
        action = "AVOID"
    else:
        action = "HOLD"

    return {
        "ticker":        ticker,
        "date":          str(date.today()),
        "signal":        final_signal,
        "raw_signal":    raw_signal,
        "prob_up":       float(round(prob_up, 4)),
        "horizon":       horizon,
        "win_rate":      win_rate,
        "action":        action,
        "close_price":   float(df_feat["Close"].iloc[-1]),
        "model_name":    model_name,
        "fwd_days":      fwd_days,
        "filter_reason": filter_reason,
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
             win_rate, action, close_price, model_name,
             fwd_days, filter_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s["date"], s["ticker"], s["signal"], s["prob_up"],
            s["horizon"], s["win_rate"], s["action"],
            s["close_price"], s["model_name"], s["fwd_days"],
            s.get("filter_reason"),
        ))
    conn.commit()
    conn.close()

def print_signal_report(signals):
    today = datetime.now().strftime("%Y-%m-%d")
    valid = [s for s in signals if s is not None]

    print(f"\n{'═'*70}")
    print(f"  DAILY SIGNALS — {today}")
    print(f"  Models evaluated: {len(valid)}/{len(signals)}")
    print(f"{'═'*70}")
    print(f"  {'Ticker':<8} {'Action':<22} {'Confidence':>11} "
          f"{'Win rate':>10} {'Horizon':>8} {'Price':>8}")
    print(f"  {'─'*68}")

    priority = {"STRONG BUY": 0, "BUY": 1, "WEAK BUY": 2,
                "HOLD": 3, "AVOID": 4}

    signals_sorted = sorted(
        valid,
        key=lambda x: (priority.get(x["action"].split("(")[0].strip(), 3),
                       -x["prob_up"])
    )

    for s in signals_sorted:
        print(f"  {s['ticker']:<8} {s['action']:<22} "
              f"{s['prob_up']:>11.1%} "
              f"{s['win_rate']:>10.1%} "
              f"{s['horizon']:>8} "
              f"${s['close_price']:>7.2f}")

    print(f"\n  Actionable (win rate >= {MIN_WIN_RATE_THRESHOLD:.0%}, "
          f"confidence >= {MIN_PROB:.0%}):")
    actionable = [s for s in signals_sorted
                  if s["action"] in ("STRONG BUY", "BUY")]

    if actionable:
        for s in actionable:
            print(f"  → {s['ticker']}: {s['action']} "
                  f"| hold {s['horizon']} "
                  f"| {s['win_rate']:.0%} win rate "
                  f"| ${s['close_price']:.2f} "
                  f"| confidence {s['prob_up']:.1%}")
    else:
        print("  → No high-confidence buy signals today")
        print(f"\n  All signals:")
        for s in signals_sorted:
            fr = f" — filtered: {s['filter_reason']}" \
                 if s.get("filter_reason") else ""
            print(f"     {s['ticker']}: {s['action']} "
                  f"(prob={s['prob_up']:.1%}){fr}")

    print(f"{'═'*70}\n")

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