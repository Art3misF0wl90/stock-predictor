import os
import threading
from datetime import date, datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit

from predict import run_predictions, TICKER_WIN_RATES
from database import (get_todays_signals, get_signal_history,
                      get_summary_stats, get_recent_performance)
from macro_loader import fetch_macro
from bot import chat

app = Flask(__name__)
app.config["SECRET_KEY"] = "stock_predictor_secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# Global conversation history per session
conversation_histories = {}

# Cache for options flow scan results
_options_cache: dict = {"results": None, "timestamp": None, "scanning": False}
 
# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/signals")
def api_signals():
    df = get_todays_signals()
    if df.empty:
        return jsonify({"date": str(date.today()), "signals": []})
    signals = df.to_dict("records")
    return jsonify({"date": str(date.today()), "signals": signals})

@app.route("/api/macro")
def api_macro():
    try:
        macro_df = fetch_macro()
        latest   = macro_df.iloc[-1]
        prev     = macro_df.iloc[-2]
        vix      = float(latest["vix"])

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

@app.route("/api/stats")
def api_stats():
    stats = get_summary_stats()
    return jsonify(stats)

@app.route("/api/history/<ticker>")
def api_history(ticker):
    days = int(request.args.get("days", 90))
    df   = get_signal_history(ticker.upper(), days)
    if df.empty:
        return jsonify({"ticker": ticker, "history": []})
    return jsonify({
        "ticker":  ticker,
        "history": df[["date","action","prob_up",
                        "win_rate","close_price"]].to_dict("records")
    })

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    def run_in_background():
        run_predictions()
    thread = threading.Thread(target=run_in_background)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "running",
                    "message": "Predictions refreshing in background..."})

@app.route("/api/win_rates")
def api_win_rates():
    return jsonify(TICKER_WIN_RATES)

@app.route("/api/watchlist")
def api_watchlist():
    from config import TICKERS
    return jsonify({"tickers": TICKERS})

@app.route("/api/watchlist/add", methods=["POST"])
def api_add_to_watchlist():
    data   = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    from analyze import analyze_ticker, add_ticker_to_watchlist
    quality = analyze_ticker(ticker)
    score   = quality.get("quality", {}).get("score", 0)

    if score < 40:
        return jsonify({
            "status":  "rejected",
            "reason":  "Quality score too low",
            "score":   score,
            "verdict": quality.get("quality", {}).get("verdict"),
        }), 400

    result = add_ticker_to_watchlist(ticker)
    return jsonify(result)

@app.route("/api/analyze/<ticker>")
def api_analyze(ticker):
    from analyze import analyze_ticker
    result = analyze_ticker(ticker.upper())
    return jsonify(result)

# ── Options Flow ───────────────────────────────────────────────────────────────
 
@app.route("/options-flow")
def options_flow_page():
    return render_template("options_flow.html")

@app.route("/portfolio")
def portfolio_page():
    return render_template("portfolio.html")

# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.route("/api/portfolio/summary")
def api_portfolio_summary():
    from portfolio import get_portfolio_summary
    return jsonify(get_portfolio_summary())

@app.route("/api/portfolio/advice")
def api_portfolio_advice():
    from portfolio import get_portfolio_advice
    from database import get_todays_signals
    signals_df = get_todays_signals()
    signals = signals_df.to_dict("records") if not signals_df.empty else []
    return jsonify(get_portfolio_advice(signals))

@app.route("/api/portfolio/holdings", methods=["GET"])
def api_portfolio_holdings():
    from portfolio import get_holdings
    return jsonify(get_holdings())

@app.route("/api/portfolio/holdings", methods=["POST"])
def api_portfolio_add_holding():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    shares = float(data.get("shares", 0))
    avg_cost = float(data.get("avg_cost", 0))
    notes = data.get("notes", "")
    if not ticker or shares <= 0 or avg_cost <= 0:
        return jsonify({"error": "ticker, shares, and avg_cost required"}), 400
    from portfolio import upsert_holding
    return jsonify(upsert_holding(ticker, shares, avg_cost, notes))

@app.route("/api/portfolio/holdings/<ticker>", methods=["DELETE"])
def api_portfolio_remove_holding(ticker):
    from portfolio import remove_holding
    return jsonify(remove_holding(ticker.upper()))

@app.route("/api/portfolio/buy", methods=["POST"])
def api_portfolio_buy():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    shares = float(data.get("shares", 0))
    price = float(data.get("price", 0))
    deduct = data.get("deduct_cash", True)
    if not ticker or shares <= 0 or price <= 0:
        return jsonify({"error": "ticker, shares, price required"}), 400
    from portfolio import buy_shares
    return jsonify(buy_shares(ticker, shares, price, deduct))

@app.route("/api/portfolio/sell", methods=["POST"])
def api_portfolio_sell():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    shares = float(data.get("shares", 0))
    price = float(data.get("price", 0))
    add_cash = data.get("add_to_cash", True)
    if not ticker or shares <= 0 or price <= 0:
        return jsonify({"error": "ticker, shares, price required"}), 400
    from portfolio import sell_shares
    return jsonify(sell_shares(ticker, shares, price, add_cash))

@app.route("/api/portfolio/cash", methods=["GET"])
def api_portfolio_cash():
    from portfolio import get_cash
    return jsonify({"cash": get_cash()})

@app.route("/api/portfolio/cash", methods=["POST"])
def api_portfolio_set_cash():
    data = request.get_json()
    amount = data.get("amount")
    note = data.get("note", "")
    if amount is None:
        return jsonify({"error": "amount required"}), 400
    from portfolio import set_cash
    return jsonify(set_cash(float(amount), note))

@app.route("/api/portfolio/cash/adjust", methods=["POST"])
def api_portfolio_adjust_cash():
    data = request.get_json()
    delta = data.get("delta")
    note = data.get("note", "")
    if delta is None:
        return jsonify({"error": "delta required"}), 400
    from portfolio import adjust_cash
    return jsonify(adjust_cash(float(delta), note))

@app.route("/api/portfolio/transactions")
def api_portfolio_transactions():
    limit = int(request.args.get("limit", 50))
    from portfolio import get_transactions
    return jsonify(get_transactions(limit))
 
@app.route("/api/options-flow")
def api_options_flow():
    return jsonify({
        "scanning":  _options_cache["scanning"],
        "has_data":  _options_cache["results"] is not None,
        "results":   _options_cache["results"] or {},
        "timestamp": _options_cache["timestamp"],
    })
 
@app.route("/api/options-flow/scan", methods=["POST"])
def api_options_flow_scan():
    if _options_cache["scanning"]:
        return jsonify({"status": "already_scanning"})
 
    def _scan():
        _options_cache["scanning"] = True
        try:
            from options_flow import scan_all
            from config import TICKERS
            _options_cache["results"]   = scan_all(TICKERS)
            _options_cache["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Options flow scan error: {e}")
        finally:
            _options_cache["scanning"] = False
 
    threading.Thread(target=_scan, daemon=True).start()
    return jsonify({"status": "started"})
 
@app.route("/api/options-flow/chart")
def api_options_flow_chart():
    chart_path = os.path.join("charts", "options_flow_dashboard.html")
    if _options_cache.get("results"):
        from options_flow import generate_options_dashboard
        generate_options_dashboard(results=_options_cache["results"])
    if not os.path.exists(chart_path):
        return "No chart available — run a scan first.", 404
    with open(chart_path, "r", encoding="utf-8") as f:
        html = f.read()
    return Response(html, content_type="text/html")
 

# ── WebSocket for bot chat ─────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    conversation_histories[session_id] = []
    emit("connected", {"message": "Connected to Stock Prediction Bot"})

@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id in conversation_histories:
        del conversation_histories[session_id]

@socketio.on("message")
def handle_message(data):
    session_id = request.sid
    user_msg   = data.get("message", "").strip()

    if not user_msg:
        return

    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]

    emit("thinking", {"status": "thinking"})

    try:
        response, updated_history = chat(user_msg, history)
        conversation_histories[session_id] = updated_history
        emit("response", {
            "message":   response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        emit("response", {
            "message":   f"Error: {str(e)}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  Stock Predictor Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("═"*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
