# app/services/database.py
#
# SQLite wrapper for the predictions database (data/predictions.db).
#
# All reads go through here so the rest of the codebase never touches
# raw SQL.  The signals table is written by predict.py; the outcomes
# table is written by record_outcome() when a trade is closed.
#
# Table schemas are created by predict.setup_database() on first run.

import sqlite3
from datetime import date, timedelta

import pandas as pd

_DB_PATH = "data/predictions.db"


def get_todays_signals() -> pd.DataFrame:
    """Return all signals generated today, sorted by win_rate descending."""
    conn = sqlite3.connect(_DB_PATH)
    df   = pd.read_sql(
        "SELECT * FROM signals WHERE date = ? ORDER BY win_rate DESC",
        conn, params=(str(date.today()),),
    )
    conn.close()
    return df


def get_signal_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """Return signal history for one ticker over the last N days."""
    since = str(date.today() - timedelta(days=days))
    conn  = sqlite3.connect(_DB_PATH)
    df    = pd.read_sql(
        """SELECT * FROM signals
           WHERE ticker = ? AND date >= ?
           ORDER BY date DESC""",
        conn, params=(ticker, since),
    )
    conn.close()
    return df


def get_actionable_signals() -> pd.DataFrame:
    """Return today's BUY and STRONG BUY signals, sorted by prob_up."""
    conn = sqlite3.connect(_DB_PATH)
    df   = pd.read_sql(
        """SELECT * FROM signals
           WHERE date = ?
             AND action IN ('BUY', 'STRONG BUY')
           ORDER BY prob_up DESC""",
        conn, params=(str(date.today()),),
    )
    conn.close()
    return df


def get_signal_accuracy(ticker: str = None) -> pd.DataFrame:
    """
    Return win rate and average return from the outcomes table.

    Pass ticker=None to get results for all tickers grouped.
    """
    conn  = sqlite3.connect(_DB_PATH)
    query = """
        SELECT ticker,
               COUNT(*)            AS total_signals,
               SUM(was_correct)    AS correct,
               AVG(return_pct)     AS avg_return,
               AVG(was_correct)    AS win_rate
        FROM outcomes
    """
    if ticker:
        query += " WHERE ticker = ?"
        df = pd.read_sql(query, conn, params=(ticker,))
    else:
        query += " GROUP BY ticker ORDER BY win_rate DESC"
        df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_recent_performance() -> pd.DataFrame:
    """
    Return the last 50 signals joined with their outcomes (if recorded).
    Used by the web dashboard and bot to show live accuracy.
    """
    conn = sqlite3.connect(_DB_PATH)
    df   = pd.read_sql("""
        SELECT s.ticker, s.date, s.action, s.prob_up,
               s.horizon, s.close_price,
               o.return_pct, o.was_correct, o.exit_date
        FROM signals s
        LEFT JOIN outcomes o ON s.id = o.signal_id
        ORDER BY s.date DESC
        LIMIT 50
    """, conn)
    conn.close()
    return df


def record_outcome(
    signal_id: int,
    exit_date: str,
    exit_price: float,
    entry_price: float,
) -> None:
    """
    Record the result of a trade once the holding period is complete.

    Computes return_pct and was_correct from entry/exit prices and
    inserts a row into the outcomes table so accuracy can be tracked
    over time.
    """
    return_pct  = (exit_price - entry_price) / entry_price
    was_correct = int(return_pct > 0)
    conn = sqlite3.connect(_DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO outcomes
        (signal_id, exit_date, exit_price, return_pct, was_correct)
        VALUES (?, ?, ?, ?, ?)
    """, (signal_id, exit_date, exit_price, return_pct, was_correct))
    conn.commit()
    conn.close()


def get_summary_stats() -> dict:
    """
    Return high-level database statistics for the web dashboard.

    Includes total signals, total buy signals, outcomes recorded,
    live win rate, and average return on winning trades.
    """
    conn = sqlite3.connect(_DB_PATH)
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM signals")
    total_signals = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM signals WHERE action IN ('BUY','STRONG BUY')")
    total_buy = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM outcomes WHERE was_correct = 1")
    correct = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM outcomes")
    total_outcomes = c.fetchone()[0]

    c.execute("SELECT AVG(return_pct) FROM outcomes WHERE was_correct = 1")
    avg_win = c.fetchone()[0] or 0

    conn.close()
    return {
        "total_signals":  total_signals,
        "total_buy":      total_buy,
        "total_outcomes": total_outcomes,
        "correct":        correct,
        "win_rate":       correct / total_outcomes if total_outcomes > 0 else 0,
        "avg_win_return": avg_win,
    }


# ---------------------------------------------------------------------------
# Run standalone to inspect the database
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Signal history:")
    df = get_todays_signals()
    if df.empty:
        print("  No signals yet — run predict.py first")
    else:
        print(df[["ticker", "action", "prob_up", "win_rate", "horizon", "close_price"]].to_string())

    print("\nSummary stats:")
    for k, v in get_summary_stats().items():
        print(f"  {k}: {v}")
