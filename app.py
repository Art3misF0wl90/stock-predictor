import os
import json
import threading
from datetime import date, datetime
from flask import Flask, render_template, request, jsonify
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
