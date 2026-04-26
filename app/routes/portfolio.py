# app/routes/portfolio.py
#
# Blueprint for the portfolio tracker API.
# All routes live under /api/portfolio/* and mirror what was
# previously inlined in app.py.

from flask import Blueprint, jsonify, request, render_template

portfolio_bp = Blueprint("portfolio", __name__)


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@portfolio_bp.route("/portfolio")
def portfolio_page():
    return render_template("portfolio.html")


# ---------------------------------------------------------------------------
# Summary & advice
# ---------------------------------------------------------------------------

@portfolio_bp.route("/api/portfolio/summary")
def api_portfolio_summary():
    from app.services.portfolio import get_portfolio_summary
    return jsonify(get_portfolio_summary())


@portfolio_bp.route("/api/portfolio/advice")
def api_portfolio_advice():
    from app.services.portfolio import get_portfolio_advice
    from database import get_todays_signals
    signals_df = get_todays_signals()
    signals = signals_df.to_dict("records") if not signals_df.empty else []
    return jsonify(get_portfolio_advice(signals))


# ---------------------------------------------------------------------------
# Holdings — read / write / delete
# ---------------------------------------------------------------------------

@portfolio_bp.route("/api/portfolio/holdings", methods=["GET"])
def api_portfolio_holdings():
    from app.services.portfolio import get_holdings
    return jsonify(get_holdings())


@portfolio_bp.route("/api/portfolio/holdings", methods=["POST"])
def api_portfolio_add_holding():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    shares = float(data.get("shares", 0))
    avg_cost = float(data.get("avg_cost", 0))
    notes = data.get("notes", "")

    if not ticker or shares <= 0 or avg_cost <= 0:
        return jsonify({"error": "ticker, shares, and avg_cost required"}), 400

    from app.services.portfolio import upsert_holding
    return jsonify(upsert_holding(ticker, shares, avg_cost, notes))


@portfolio_bp.route("/api/portfolio/holdings/<ticker>", methods=["DELETE"])
def api_portfolio_remove_holding(ticker):
    from app.services.portfolio import remove_holding
    return jsonify(remove_holding(ticker.upper()))


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@portfolio_bp.route("/api/portfolio/buy", methods=["POST"])
def api_portfolio_buy():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    shares = float(data.get("shares", 0))
    price = float(data.get("price", 0))
    deduct = data.get("deduct_cash", True)

    if not ticker or shares <= 0 or price <= 0:
        return jsonify({"error": "ticker, shares, price required"}), 400

    from app.services.portfolio import buy_shares
    return jsonify(buy_shares(ticker, shares, price, deduct))


@portfolio_bp.route("/api/portfolio/sell", methods=["POST"])
def api_portfolio_sell():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    shares = float(data.get("shares", 0))
    price = float(data.get("price", 0))
    add_cash = data.get("add_to_cash", True)

    if not ticker or shares <= 0 or price <= 0:
        return jsonify({"error": "ticker, shares, price required"}), 400

    from app.services.portfolio import sell_shares
    return jsonify(sell_shares(ticker, shares, price, add_cash))


# ---------------------------------------------------------------------------
# Cash management
# ---------------------------------------------------------------------------

@portfolio_bp.route("/api/portfolio/cash", methods=["GET"])
def api_portfolio_cash():
    from app.services.portfolio import get_cash
    return jsonify({"cash": get_cash()})


@portfolio_bp.route("/api/portfolio/cash", methods=["POST"])
def api_portfolio_set_cash():
    data = request.get_json()
    amount = data.get("amount")
    note = data.get("note", "")
    if amount is None:
        return jsonify({"error": "amount required"}), 400
    from app.services.portfolio import set_cash
    return jsonify(set_cash(float(amount), note))


@portfolio_bp.route("/api/portfolio/cash/adjust", methods=["POST"])
def api_portfolio_adjust_cash():
    data = request.get_json()
    delta = data.get("delta")
    note = data.get("note", "")
    if delta is None:
        return jsonify({"error": "delta required"}), 400
    from app.services.portfolio import adjust_cash
    return jsonify(adjust_cash(float(delta), note))


# ---------------------------------------------------------------------------
# Transaction log
# ---------------------------------------------------------------------------

@portfolio_bp.route("/api/portfolio/transactions")
def api_portfolio_transactions():
    limit = int(request.args.get("limit", 50))
    from app.services.portfolio import get_transactions
    return jsonify(get_transactions(limit))