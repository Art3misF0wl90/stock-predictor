# app/routes/api.py
#
# Blueprint for the core prediction API.
# Covers: signals, macro conditions, model stats, signal history,
# win rates, watchlist management, per-ticker analysis, and the
# background prediction refresh trigger.

import threading
from datetime import date

from flask import Blueprint, jsonify, request

from app.ml import run_predictions, TICKER_WIN_RATES
from app.services import (
    get_todays_signals,
    get_signal_history,
    get_summary_stats,
    analyze_ticker,
    add_ticker_to_watchlist,
)
from app.data import fetch_macro

api_bp = Blueprint("api", __name__)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@api_bp.route("/api/signals")
def api_signals():
    df = get_todays_signals()
    if df.empty:
        return jsonify({"date": str(date.today()), "signals": []})
    return jsonify({"date": str(date.today()), "signals": df.to_dict("records")})


@api_bp.route("/api/history/<ticker>")
def api_history(ticker):
    days = int(request.args.get("days", 90))
    df = get_signal_history(ticker.upper(), days)
    if df.empty:
        return jsonify({"ticker": ticker, "history": []})
    return jsonify({
        "ticker": ticker,
        "history": df[["date", "action", "prob_up", "win_rate", "close_price"]].to_dict("records"),
    })


@api_bp.route("/api/refresh", methods=["POST"])
def api_refresh():
    t = threading.Thread(target=run_predictions, daemon=True)
    t.start()
    return jsonify({"status": "running", "message": "Predictions refreshing in background..."})


@api_bp.route("/api/win_rates")
def api_win_rates():
    return jsonify(TICKER_WIN_RATES)


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------

@api_bp.route("/api/macro")
def api_macro():
    try:
        macro_df = fetch_macro()
        latest = macro_df.iloc[-1]
        prev   = macro_df.iloc[-2]
        vix    = float(latest["vix"])

        if vix < 15:
            fear = "Low"
        elif vix < 20:
            fear = "Normal"
        elif vix < 30:
            fear = "Elevated"
        else:
            fear = "High"

        return jsonify({
            "vix":          round(vix, 2),
            "vix_change":   round(vix - float(prev["vix"]), 2),
            "fear_level":   fear,
            "treasury_10y": round(float(latest["treasury"]), 3),
            "dollar_index": round(float(latest["dollar"]), 2),
            "date":         str(macro_df.index[-1].date()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@api_bp.route("/api/stats")
def api_stats():
    return jsonify(get_summary_stats())


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@api_bp.route("/api/watchlist")
def api_watchlist():
    from config import TICKERS
    return jsonify({"tickers": TICKERS})


@api_bp.route("/api/watchlist/add", methods=["POST"])
def api_add_to_watchlist():
    data   = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    quality = analyze_ticker(ticker)
    score   = quality.get("quality", {}).get("score", 0)

    if score < 40:
        return jsonify({
            "status":  "rejected",
            "reason":  "Quality score too low",
            "score":   score,
            "verdict": quality.get("quality", {}).get("verdict"),
        }), 400

    return jsonify(add_ticker_to_watchlist(ticker))


# ---------------------------------------------------------------------------
# Per-ticker analysis (on-demand, any ticker)
# ---------------------------------------------------------------------------

@api_bp.route("/api/analyze/<ticker>")
def api_analyze(ticker):
    return jsonify(analyze_ticker(ticker.upper()))