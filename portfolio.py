import sqlite3
import json
from datetime import date, datetime
from typing import Optional
import yfinance as yf
import pandas as pd
 
DB_PATH = "data/portfolio.db"
 
 
def init_portfolio_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
 
    c.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL UNIQUE,
            shares      REAL NOT NULL DEFAULT 0,
            avg_cost    REAL NOT NULL DEFAULT 0,
            notes       TEXT,
            added_at    TEXT NOT NULL
        )
    """)
 
    c.execute("""
        CREATE TABLE IF NOT EXISTS cash (
            id          INTEGER PRIMARY KEY,
            balance     REAL NOT NULL DEFAULT 0,
            updated_at  TEXT NOT NULL
        )
    """)
 
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            action      TEXT NOT NULL,
            shares      REAL,
            price       REAL,
            amount      REAL,
            note        TEXT,
            ts          TEXT NOT NULL
        )
    """)
 
    # seed cash row
    c.execute("INSERT OR IGNORE INTO cash (id, balance, updated_at) VALUES (1, 0, ?)",
              (str(datetime.now()),))
 
    conn.commit()
    conn.close()
 
 
# ── Cash ──────────────────────────────────────────────────────────────────────
 
def get_cash() -> float:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT balance FROM cash WHERE id = 1")
    row = c.fetchone()
    conn.close()
    return float(row[0]) if row else 0.0
 
 
def set_cash(amount: float, note: str = "") -> dict:
    old = get_cash()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE cash SET balance = ?, updated_at = ? WHERE id = 1",
              (amount, str(datetime.now())))
    c.execute("""INSERT INTO transactions (ticker, action, amount, note, ts)
                 VALUES (NULL, 'CASH_SET', ?, ?, ?)""",
              (amount, note or f"Set cash to ${amount:,.2f}", str(datetime.now())))
    conn.commit()
    conn.close()
    return {"old_balance": old, "new_balance": amount}
 
 
def adjust_cash(delta: float, note: str = "") -> dict:
    old = get_cash()
    new = old + delta
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE cash SET balance = ?, updated_at = ? WHERE id = 1",
              (new, str(datetime.now())))
    action = "CASH_DEPOSIT" if delta >= 0 else "CASH_WITHDRAW"
    c.execute("""INSERT INTO transactions (ticker, action, amount, note, ts)
                 VALUES (NULL, ?, ?, ?, ?)""",
              (action, abs(delta), note, str(datetime.now())))
    conn.commit()
    conn.close()
    return {"old_balance": old, "new_balance": new, "delta": delta}
 
 
# ── Holdings ──────────────────────────────────────────────────────────────────
 
def get_holdings() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM holdings ORDER BY ticker")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
 
 
def upsert_holding(ticker: str, shares: float, avg_cost: float,
                   notes: str = "") -> dict:
    ticker = ticker.upper()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO holdings (ticker, shares, avg_cost, notes, added_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            shares   = excluded.shares,
            avg_cost = excluded.avg_cost,
            notes    = excluded.notes
    """, (ticker, shares, avg_cost, notes, str(datetime.now())))
    c.execute("""INSERT INTO transactions (ticker, action, shares, price, note, ts)
                 VALUES (?, 'SET_POSITION', ?, ?, ?, ?)""",
              (ticker, shares, avg_cost, notes, str(datetime.now())))
    conn.commit()
    conn.close()
    return {"ticker": ticker, "shares": shares, "avg_cost": avg_cost}
 
 
def remove_holding(ticker: str) -> dict:
    ticker = ticker.upper()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
    deleted = c.rowcount
    if deleted:
        c.execute("""INSERT INTO transactions (ticker, action, note, ts)
                     VALUES (?, 'REMOVE', 'Position removed', ?)""",
                  (ticker, str(datetime.now())))
    conn.commit()
    conn.close()
    return {"ticker": ticker, "removed": deleted > 0}
 
 
def buy_shares(ticker: str, shares: float, price: float,
               deduct_cash: bool = True) -> dict:
    ticker = ticker.upper()
    cost = shares * price
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("SELECT shares, avg_cost FROM holdings WHERE ticker = ?", (ticker,))
        row = c.fetchone()
        if row:
            old_shares, old_cost = float(row[0]), float(row[1])
            new_shares = old_shares + shares
            new_avg = (old_shares * old_cost + cost) / new_shares
        else:
            new_shares = shares
            new_avg = price

        c.execute("""
            INSERT INTO holdings (ticker, shares, avg_cost, notes, added_at)
            VALUES (?, ?, ?, '', ?)
            ON CONFLICT(ticker) DO UPDATE SET
                shares   = excluded.shares,
                avg_cost = excluded.avg_cost
        """, (ticker, new_shares, new_avg, str(datetime.now())))

        c.execute("""INSERT INTO transactions (ticker, action, shares, price, amount, note, ts)
                     VALUES (?, 'BUY', ?, ?, ?, ?, ?)""",
                  (ticker, shares, price, cost, f"Bought {shares} @ ${price:.2f}", str(datetime.now())))

        conn.commit()
    finally:
        conn.close()
    return {"ticker": ticker, "shares_bought": shares, "new_total": new_shares,
            "new_avg_cost": round(new_avg, 4), "total_cost": round(cost, 2)}
 
 
def sell_shares(ticker: str, shares: float, price: float,
                add_to_cash: bool = True) -> dict:
    ticker = ticker.upper()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("SELECT shares, avg_cost FROM holdings WHERE ticker = ?", (ticker,))
        row = c.fetchone()
        if not row:
            return {"error": f"No position in {ticker}"}

        old_shares, avg_cost = float(row[0]), float(row[1])
        if shares > old_shares:
            return {"error": f"Only {old_shares} shares held, cannot sell {shares}"}

        proceeds = shares * price
        realized_pnl = (price - avg_cost) * shares
        new_shares = old_shares - shares

        if new_shares < 1e-6:
            c.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))
        else:
            c.execute("UPDATE holdings SET shares = ? WHERE ticker = ?",
                      (new_shares, ticker))

        c.execute("""INSERT INTO transactions (ticker, action, shares, price, amount, note, ts)
                     VALUES (?, 'SELL', ?, ?, ?, ?, ?)""",
                  (ticker, shares, price, proceeds,
                   f"Sold {shares} @ ${price:.2f}, PnL ${realized_pnl:+.2f}", str(datetime.now())))
        conn.commit()
    finally:
        conn.close()

    if add_to_cash:
        adjust_cash(proceeds, f"Sell {shares} {ticker} @ ${price:.2f}")

    return {"ticker": ticker, "shares_sold": shares, "shares_remaining": new_shares,
            "proceeds": round(proceeds, 2), "realized_pnl": round(realized_pnl, 2)}
 
 
# ── Live prices ───────────────────────────────────────────────────────────────
 
def fetch_live_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period="2d", progress=False, auto_adjust=True)
        prices = {}
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
            for t in tickers:
                if t in close.columns:
                    prices[t] = float(close[t].dropna().iloc[-1])
        else:
            if not raw.empty:
                prices[tickers[0]] = float(raw["Close"].dropna().iloc[-1])
        return prices
    except Exception:
        return {}
 
 
# ── Portfolio summary ─────────────────────────────────────────────────────────
 
def get_portfolio_summary() -> dict:
    init_portfolio_db()
    holdings = get_holdings()
    cash = get_cash()
 
    if not holdings:
        return {
            "holdings": [],
            "cash": cash,
            "total_invested": 0,
            "total_market_value": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "total_value": cash,
            "prices": {},
        }
 
    tickers = [h["ticker"] for h in holdings]
    prices = fetch_live_prices(tickers)
 
    total_invested = 0.0
    total_market_value = 0.0
    enriched = []
 
    for h in holdings:
        t = h["ticker"]
        price = prices.get(t)
        cost_basis = h["shares"] * h["avg_cost"]
        market_value = h["shares"] * price if price else None
        pnl = (market_value - cost_basis) if market_value is not None else None
        pnl_pct = (pnl / cost_basis * 100) if (pnl is not None and cost_basis > 0) else None
 
        total_invested += cost_basis
        if market_value is not None:
            total_market_value += market_value
 
        enriched.append({
            **h,
            "current_price": round(price, 4) if price else None,
            "cost_basis": round(cost_basis, 2),
            "market_value": round(market_value, 2) if market_value is not None else None,
            "pnl": round(pnl, 2) if pnl is not None else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        })
 
    total_pnl = total_market_value - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
 
    return {
        "holdings": enriched,
        "cash": round(cash, 2),
        "total_invested": round(total_invested, 2),
        "total_market_value": round(total_market_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "total_value": round(total_market_value + cash, 2),
        "prices": {k: round(v, 4) for k, v in prices.items()},
    }
 
 
# ── AI advice ─────────────────────────────────────────────────────────────────
 
def get_portfolio_advice(signals: list[dict]) -> dict:
    """
    Cross-references holdings against today's signals and cash position
    to generate actionable daily advice.
    """
    init_portfolio_db()
    summary = get_portfolio_summary()
    holdings = {h["ticker"]: h for h in summary["holdings"]}
    cash = summary["cash"]
    total_value = summary["total_value"]
 
    signal_map = {s["ticker"]: s for s in signals} if signals else {}
 
    advice = []
    warnings = []
    opportunities = []
 
    # ── Check existing positions against signals ──
    for ticker, h in holdings.items():
        sig = signal_map.get(ticker, {})
        action = sig.get("action", "")
        prob_up = sig.get("prob_up", 0.5)
        pnl_pct = h.get("pnl_pct") or 0
        price = h.get("current_price")
        avg_cost = h["avg_cost"]
 
        if action in ("AVOID",):
            warnings.append({
                "ticker": ticker,
                "type": "SELL_SIGNAL",
                "severity": "high",
                "message": f"{ticker}: Model says AVOID — you hold {h['shares']} shares "
                           f"(P&L: {pnl_pct:+.1f}%). Consider reducing or exiting.",
                "action": "Consider selling",
            })
        elif action == "STRONG BUY" and h["shares"] > 0:
            advice.append({
                "ticker": ticker,
                "type": "ADD_TO_WINNER",
                "severity": "low",
                "message": f"{ticker}: STRONG BUY signal — current position up {pnl_pct:+.1f}%. "
                           f"Model confidence {prob_up*100:.0f}%. May consider adding.",
                "action": "Optional add",
            })
        elif action in ("BUY", "WEAK BUY") and pnl_pct < -10:
            advice.append({
                "ticker": ticker,
                "type": "DIP_BUY",
                "severity": "medium",
                "message": f"{ticker}: Down {pnl_pct:.1f}% from cost, but model says {action}. "
                           f"Potential averaging opportunity.",
                "action": "Consider averaging down",
            })
 
        # stop-loss check
        if pnl_pct < -20:
            warnings.append({
                "ticker": ticker,
                "type": "STOP_LOSS",
                "severity": "high",
                "message": f"{ticker}: Down {pnl_pct:.1f}% — below 20% drawdown threshold. "
                           f"Review position.",
                "action": "Review stop-loss",
            })
 
    # ── Cash deployment suggestions ──
    if cash > 500:
        strong_buys = [s for s in signals if s.get("action") == "STRONG BUY"
                       and s["ticker"] not in holdings]
        buys = [s for s in signals if s.get("action") == "BUY"
                and s["ticker"] not in holdings]
 
        targets = strong_buys[:3] + buys[:2]
        if targets and total_value > 0:
            cash_pct = cash / total_value * 100
            suggest_deploy = min(cash * 0.5, cash - 500)  # keep $500 reserve
 
            if suggest_deploy > 100:
                per_pos = suggest_deploy / max(len(targets), 1)
                for sig in targets:
                    price = summary["prices"].get(sig["ticker"])
                    shares_est = int(per_pos / price) if price and price > 0 else None
                    opportunities.append({
                        "ticker": sig["ticker"],
                        "type": "CASH_DEPLOY",
                        "severity": "medium",
                        "message": f"Deploy ~${per_pos:,.0f} into {sig['ticker']} "
                                   f"({sig['action']}, {sig.get('prob_up',0)*100:.0f}% prob up"
                                   + (f", ~{shares_est} shares @ ${price:.2f}" if shares_est else "")
                                   + ")",
                        "action": "Buy",
                        "suggested_amount": round(per_pos, 2),
                        "estimated_shares": shares_est,
                    })
            if cash_pct > 40:
                advice.append({
                    "ticker": None,
                    "type": "HIGH_CASH",
                    "severity": "medium",
                    "message": f"Cash is {cash_pct:.0f}% of portfolio (${cash:,.0f}). "
                               f"Consider deploying into current BUY signals.",
                    "action": "Deploy cash",
                })
 
    # ── Concentration check ──
    if total_value > 0:
        for ticker, h in holdings.items():
            mv = h.get("market_value") or 0
            weight = mv / total_value * 100
            if weight > 35:
                warnings.append({
                    "ticker": ticker,
                    "type": "CONCENTRATION",
                    "severity": "medium",
                    "message": f"{ticker} is {weight:.0f}% of portfolio — concentrated position.",
                    "action": "Consider trimming",
                })
 
    return {
        "date": str(date.today()),
        "warnings": warnings,
        "advice": advice,
        "opportunities": opportunities,
        "cash": cash,
        "total_value": total_value,
        "summary": summary,
    }
 
 
def get_transactions(limit: int = 50) -> list[dict]:
    init_portfolio_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM transactions ORDER BY ts DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
 
 
init_portfolio_db()