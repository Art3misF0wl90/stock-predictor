# app/ml/predict.py
#
# Daily signal generation — the runtime prediction pipeline.
#
# run_predictions() is the main entry point.  It loads each ticker's trained
# model, fetches the last 300 days of price data, runs the full feature
# pipeline, applies entry filters, classifies the signal, and writes results
# to the SQLite database.
#
# Public API (imported by bot.py, backtest.py, options_flow.py, app routes):
#   run_predictions()      — generate + save signals for all watchlist tickers
#   get_today_signal()     — generate a signal dict for one ticker
#   apply_entry_filters()  — apply VIX / RSI / trend / earnings blackout gates
#   fetch_latest_data()    — download the last 300 days of OHLCV for one ticker
#   TICKER_WIN_RATES       — dict of historical backtest win rates per ticker

import os
import sqlite3
from datetime import datetime, date

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from app.data import fetch_macro, load_all_sentiment, build_earnings_features
from app.ml.features import add_features, get_feature_columns
from config import (
    TICKERS,
    MIN_CONFIDENCE,
    TICKER_MIN_CONFIDENCE,
    MAX_VIX,
    MIN_WIN_RATE,
    RSI_OVERBOUGHT,
    EARNINGS_BLACKOUT_DAYS,
    MAX_CONSECUTIVE_LOSSES,
    TICKERS_NO_EARNINGS,
)

# ---------------------------------------------------------------------------
# Historical backtest win rates (hard-coded from the last full backtest run).
# These are runtime constants, not configuration — they belong here rather
# than in config.py because they are computed by backtest.py and would
# normally be read from the database once enough outcome data accumulates.
# ---------------------------------------------------------------------------

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

_DB_PATH = "data/predictions.db"


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def setup_database() -> None:
    """Create the signals and outcomes tables if they do not already exist."""
    conn = sqlite3.connect(_DB_PATH)
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


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_latest_data(ticker: str) -> pd.DataFrame:
    """Download the last 300 trading days of OHLCV data for one ticker."""
    df = yf.download(ticker, period="300d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Entry filters
# ---------------------------------------------------------------------------

def apply_entry_filters(
    row: pd.Series,
    prob_up: float,
    ticker: str = "",
) -> tuple[bool, str | None]:
    """
    Apply entry quality gates to a raw buy signal.

    Gates (in order of check):
      1. Per-ticker confidence floor — signal too close to 50/50
      2. VIX crisis          — model assumptions break down above MAX_VIX
      3. Trend filter        — avoid buying into a confirmed downtrend
      4. RSI overbought      — avoid chasing extended momentum
      5. Earnings blackout   — binary event risk near earnings dates

    Returns (passes: bool, reason: str | None).
    reason is None when the signal passes all filters.
    """
    ticker_min = TICKER_MIN_CONFIDENCE.get(ticker, MIN_CONFIDENCE)
    if prob_up < ticker_min:
        return False, f"low_confidence ({prob_up:.1%} < {ticker_min:.0%})"

    if "vix" in row.index:
        vix = row.get("vix", 0)
        if not pd.isna(vix) and float(vix) > MAX_VIX:
            return False, f"high_vix ({float(vix):.1f} > {MAX_VIX})"

    if "price_to_ma50" in row.index:
        ptma = row.get("price_to_ma50", 1.0)
        if not pd.isna(ptma) and float(ptma) < 1.0:
            return False, f"below_50ma ({float(ptma):.3f})"

    if "rsi_14" in row.index:
        rsi = row.get("rsi_14", 50)
        if not pd.isna(rsi) and float(rsi) > RSI_OVERBOUGHT:
            return False, f"overbought_rsi ({float(rsi):.1f})"

    if "days_to_earnings" in row.index:
        dte = row.get("days_to_earnings", 90)
        if not pd.isna(dte) and 0 < float(dte) <= EARNINGS_BLACKOUT_DAYS:
            return False, f"earnings_in_{int(dte)}d"

    return True, None


# ---------------------------------------------------------------------------
# Per-ticker signal generation
# ---------------------------------------------------------------------------

def get_today_signal(
    ticker: str,
    macro_df: pd.DataFrame,
    sentiment: dict,
) -> dict | None:
    """
    Load the trained model for one ticker and generate today's signal.

    Returns a signal dict or None if the model file is missing or the
    feature pipeline fails to produce a valid row.
    """
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

    has_earnings = any(
        "eps" in c or "pead" in c or "earnings" in c
        for c in feat_cols
    )
    earnings_df = (
        None if not has_earnings
        else build_earnings_features(ticker, df)
    )

    df_feat = add_features(
        df,
        macro_df=macro_df,
        sentiment_series=sentiment_series,
        earnings_df=earnings_df,
        forward_days=fwd_days,
        predict_mode=True,
    )

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
    horizon  = horizon_map.get(fwd_days, f"{fwd_days}d")
    win_rate = float(TICKER_WIN_RATES.get(ticker, {}).get(horizon, 0.5))

    latest        = df_feat.iloc[-1]
    filter_reason = None
    final_signal  = raw_signal

    if raw_signal == 1:
        passes, filter_reason = apply_entry_filters(latest, prob_up, ticker)
        if not passes:
            final_signal = 0

    # ── Action label ───────────────────────────────────────────────────────
    if final_signal == 1 and win_rate >= MIN_WIN_RATE:
        action = "STRONG BUY" if prob_up >= 0.7 else "BUY"
    elif raw_signal == 1 and final_signal == 0:
        action = f"FILTERED ({filter_reason})"
    elif raw_signal == 1:
        action = "WEAK BUY"
    elif raw_signal == 0 and win_rate >= MIN_WIN_RATE:
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


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------

def save_signals(signals: list[dict]) -> None:
    """Replace today's signals in the database with the newly generated ones."""
    conn  = sqlite3.connect(_DB_PATH)
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


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_signal_report(signals: list[dict]) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    valid = [s for s in signals if s is not None]

    print(f"\n{'═'*70}")
    print(f"  DAILY SIGNALS — {today}")
    print(f"  Models evaluated: {len(valid)}/{len(signals)}")
    print(f"{'═'*70}")
    print(
        f"  {'Ticker':<8} {'Action':<22} {'Confidence':>11} "
        f"{'Win rate':>10} {'Horizon':>8} {'Price':>8}"
    )
    print(f"  {'─'*68}")

    priority = {"STRONG BUY": 0, "BUY": 1, "WEAK BUY": 2, "HOLD": 3, "AVOID": 4}
    signals_sorted = sorted(
        valid,
        key=lambda x: (priority.get(x["action"].split("(")[0].strip(), 3), -x["prob_up"]),
    )

    for s in signals_sorted:
        print(
            f"  {s['ticker']:<8} {s['action']:<22} "
            f"{s['prob_up']:>11.1%} "
            f"{s['win_rate']:>10.1%} "
            f"{s['horizon']:>8} "
            f"${s['close_price']:>7.2f}"
        )

    print(f"\n  Actionable (win rate >= {MIN_WIN_RATE:.0%}, confidence >= {MIN_CONFIDENCE:.0%}):")
    actionable = [s for s in signals_sorted if s["action"] in ("STRONG BUY", "BUY")]

    if actionable:
        for s in actionable:
            print(
                f"  → {s['ticker']}: {s['action']} "
                f"| hold {s['horizon']} "
                f"| {s['win_rate']:.0%} win rate "
                f"| ${s['close_price']:.2f} "
                f"| confidence {s['prob_up']:.1%}"
            )
    else:
        print("  → No high-confidence buy signals today")
        for s in signals_sorted:
            fr = f" — filtered: {s['filter_reason']}" if s.get("filter_reason") else ""
            print(f"     {s['ticker']}: {s['action']} (prob={s['prob_up']:.1%}){fr}")

    print(f"{'═'*70}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_predictions() -> list[dict]:
    """
    Generate and persist today's signals for all watchlist tickers.

    Called by run_predictions() in the web routes and by the CLI.
    """
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
