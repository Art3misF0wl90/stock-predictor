import os
import json
from groq import Groq
from datetime import date, datetime

from predict import run_predictions, fetch_latest_data, TICKER_WIN_RATES
from database import (get_todays_signals, get_signal_history,
                      get_summary_stats)
from macro_loader import fetch_macro
from sentiment_loader import load_all_sentiment
from backtest import get_signal_returns
from data_loader import load_all_tickers
from earnings_loader import load_all_earnings, build_earnings_features
from features import add_features
from config import TICKERS

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a stock prediction assistant with access to a machine
learning prediction system trained on AAPL, MSFT, TSLA, JPM, and NVDA.

The models use technical indicators, macroeconomic data, news sentiment scored
by FinBERT, and earnings surprise data. They were trained on 2015-2024 data.

You can analyze ANY stock ticker on demand using analyze_any_ticker — not just
the 10 in the watchlist. When a user asks about a ticker you don't recognize or
one not in the watchlist, immediately call analyze_any_ticker to get fresh data.

For watchlist tickers (AAPL, MSFT, TSLA, JPM, NVDA, GOOGL, AMZN, META, SPY, AMD)
use the existing tools like get_todays_signals and explain_signal.

For any other ticker, use analyze_any_ticker which:
- Fetches live price data
- Runs technical analysis
- Scores watchlist suitability (history, liquidity, volatility)
- Uses the combined model for a directional signal
- Recommends whether to add it permanently

If the user asks to add a ticker permanently, use add_to_watchlist which will
check quality and modify config.py if suitable.

Key model facts:
- MSFT LogReg 21d: 95.7% historical win rate at 1 quarter hold
- JPM XGBoost 1d: 100% win rate at 6-month hold in backtest
- NVDA LogReg 21d: rare signals but very high confidence when fired
- TSLA XGBoost 21d: volatile, use shorter hold periods
- AAPL LogReg 63d: 78.9% win rate at 1 quarter

You have tools available. To call a tool respond ONLY with valid JSON in this exact format:
{"tool": "tool_name", "input": {"param": "value"}}

Available tools:
- get_todays_signals: Get today's buy/sell signals for all tickers
- run_fresh_predictions: Fetch latest data and run fresh predictions
- get_signal_history: {"ticker": "AAPL", "days": 30}
- get_macro_conditions: Get VIX, treasury yield, dollar index
- get_performance_stats: Get historical win rates and stats
- run_backtest: {"ticker": "MSFT"}
- explain_signal: {"ticker": "NVDA"}
- analyze_any_ticker: {"ticker": "PLTR"}
- add_to_watchlist: {"ticker": "PLTR"}

If you need to call a tool, respond with ONLY the JSON above, nothing else.
If you do NOT need a tool, respond normally in plain English.

Always remind users these are model predictions, not financial advice.

When a user asks what to buy, what signals look like, or anything requiring
data — call the appropriate tool IMMEDIATELY without asking for permission
or explaining what you are about to do. Just call it.

For watchlist tickers (AAPL, MSFT, TSLA, JPM, NVDA, GOOGL, AMZN, META, SPY, AMD)
use get_todays_signals and explain_signal.
For ANY other ticker the user asks about, IMMEDIATELY call analyze_any_ticker.
If the user asks to add a ticker permanently to the watchlist, call add_to_watchlist."""

TOOLS_THAT_NEED_INPUT = ["get_signal_history", "run_backtest", "explain_signal"]

def tool_get_todays_signals():
    df = get_todays_signals()
    if df.empty:
        return {"status": "no_signals",
                "message": "No signals in database. Run predictions first.",
                "signals": []}
    signals = df[["ticker","action","prob_up","win_rate",
                   "horizon","close_price","model_name"]].to_dict("records")
    return {
        "date":    str(date.today()),
        "signals": signals,
        "summary": {
            "buy_signals":   len([s for s in signals if "BUY" in s["action"]]),
            "avoid_signals": len([s for s in signals if s["action"] == "AVOID"]),
            "hold_signals":  len([s for s in signals if s["action"] == "HOLD"]),
        }
    }

def tool_run_fresh_predictions():
    print("  [Bot] Running fresh predictions...")
    signals = run_predictions()
    valid   = [s for s in signals if s]
    return {"date": str(date.today()), "signals": valid, "count": len(valid)}

def tool_get_signal_history(ticker: str, days: int = 30):
    df = get_signal_history(ticker.upper(), days)
    if df.empty:
        return {"ticker": ticker, "history": [],
                "message": "No history found. Run predict.py daily to build history."}
    return {
        "ticker":        ticker,
        "days":          days,
        "history":       df[["date","action","prob_up","win_rate",
                              "close_price"]].to_dict("records"),
        "total_signals": len(df),
        "buy_signals":   len(df[df["action"].isin(["BUY","STRONG BUY"])]),
    }

def tool_get_macro_conditions():
    macro_df = fetch_macro()
    latest   = macro_df.iloc[-1]
    prev     = macro_df.iloc[-2]
    vix      = float(latest["vix"])
    vix_prev = float(prev["vix"])
    tsy      = float(latest["treasury"])
    dollar   = float(latest["dollar"])

    if vix < 15:
        fear = "Low (calm market)"
    elif vix < 20:
        fear = "Normal"
    elif vix < 30:
        fear = "Elevated (caution)"
    else:
        fear = "High (fear/crisis mode)"

    return {
        "date":           str(macro_df.index[-1].date()),
        "vix":            round(vix, 2),
        "vix_change":     round(vix - vix_prev, 2),
        "fear_level":     fear,
        "treasury_10y":   round(tsy, 3),
        "dollar_index":   round(dollar, 2),
    }

def tool_get_performance_stats():
    stats = get_summary_stats()
    return {
        "database_stats":           stats,
        "historical_win_rates":     TICKER_WIN_RATES,
    }

def tool_run_backtest(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return {"error": f"{ticker} not in watchlist. Available: {TICKERS}"}
    print(f"  [Bot] Running backtest for {ticker}...")
    all_data  = load_all_tickers()
    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()
    earnings  = load_all_earnings(all_data)
    signals_df = get_signal_returns(
        ticker, all_data[ticker],
        macro_df=macro_df, sentiment=sentiment, earnings=earnings)
    if signals_df is None or signals_df.empty:
        return {"error": "Could not generate backtest signals"}
    buy     = signals_df[signals_df["signal"] == 1]
    results = {}
    for days in [5, 21, 63, 126]:
        col = f"return_{days}d"
        if col in buy.columns:
            ret = buy[col].dropna()
            if not ret.empty:
                results[f"{days}d"] = {
                    "avg_return": round(float(ret.mean()), 4),
                    "win_rate":   round(float((ret > 0).mean()), 4),
                    "signals":    len(ret),
                }
    return {
        "ticker":          ticker,
        "test_period":     f"{signals_df['date'].min().date()} to {signals_df['date'].max().date()}",
        "buy_signals":     len(buy),
        "holding_periods": results,
    }

def tool_explain_signal(ticker: str):
    ticker = ticker.upper()
    import joblib
    config_path = os.path.join("models", f"{ticker}_config.pkl")
    if not os.path.exists(config_path):
        return {"error": f"No model found for {ticker}"}

    cfg        = joblib.load(config_path)
    fwd_days   = cfg["fwd_days"]
    model_name = cfg["model_name"]
    feat_cols  = cfg["feat_cols"]

    macro_df  = fetch_macro()
    sentiment = load_all_sentiment()
    df        = fetch_latest_data(ticker)
    sent      = sentiment.get(ticker)

    # Use earnings only if the saved model was trained with them
    has_earnings = any(
        "eps" in c or "pead" in c or "earnings" in c
        for c in feat_cols
    )
    from earnings_loader import build_earnings_features
    earn = build_earnings_features(ticker, df) if has_earnings else None

    df_feat = add_features(df, macro_df=macro_df,
                           sentiment_series=sent,
                           earnings_df=earn,
                           forward_days=fwd_days,
                           predict_mode=True)

    if df_feat.empty:
        return {"error": "Could not generate features"}

    latest = df_feat.iloc[-1]
    key_features = {}
    for col in ["rsi_14", "macd_hist", "bb_position", "return_5d",
                "return_20d", "vix", "sentiment", "pead_signal",
                "overnight_gap", "price_position_20", "up_days_5"]:
        if col in latest.index:
            key_features[col] = round(float(latest[col]), 4)

    scaler    = joblib.load(os.path.join("models", f"{ticker}_scaler.pkl"))
    model     = joblib.load(os.path.join("models", f"{ticker}_model.pkl"))
    feat_cols_present = [c for c in feat_cols if c in df_feat.columns]
    X         = df_feat[feat_cols_present].iloc[[-1]].values
    prob_up   = float(model.predict_proba(scaler.transform(X))[0][1])

    return {
        "ticker":       ticker,
        "model":        model_name,
        "horizon":      f"{fwd_days}d",
        "prob_up":      round(prob_up, 4),
        "signal":       "BUY" if prob_up >= 0.5 else "SELL",
        "key_features": key_features,
        "interpretation": {
            "rsi_14":            "RSI > 70 = overbought, < 30 = oversold",
            "bb_position":       "0 = at lower band, 1 = at upper band",
            "macd_hist":         "positive = bullish momentum, negative = bearish",
            "vix":               "higher VIX = more market fear",
            "sentiment":         "positive = bullish news, negative = bearish news",
            "pead_signal":       "positive = recent earnings beat drifting up",
            "price_position_20": "0 = at 20-day low, 1 = at 20-day high",
            "overnight_gap":     "positive = gapped up overnight",
            "up_days_5":         "how many of last 5 days closed up",
        }
    }

def tool_analyze_any_ticker(ticker: str) -> str:
    from analyze import analyze_ticker
    print(f"  [Bot] Analyzing {ticker.upper()}...")
    result = analyze_ticker(ticker)
    return json.dumps(result, default=str)

def tool_add_to_watchlist(ticker: str) -> str:
    from analyze import add_ticker_to_watchlist, analyze_ticker
    ticker = ticker.upper()

    # Run quality check first
    quality_result = analyze_ticker(ticker)
    quality_score  = quality_result.get("quality", {}).get("score", 0)

    if quality_score < 40:
        return json.dumps({
            "status":  "rejected",
            "reason":  "Quality score too low for reliable predictions",
            "quality": quality_result.get("quality", {}),
        })

    result = add_ticker_to_watchlist(ticker)
    return json.dumps(result, default=str)

def tool_analyze_any_ticker(ticker: str) -> str:
    from analyze import analyze_ticker
    print(f"  [Bot] Analyzing {ticker.upper()}...")
    result = analyze_ticker(ticker)
    return json.dumps(result, default=str)

def tool_add_to_watchlist(ticker: str) -> str:
    from analyze import add_ticker_to_watchlist, analyze_ticker
    ticker = ticker.upper()
    quality_result = analyze_ticker(ticker)
    quality_score  = quality_result.get("quality", {}).get("score", 0)
    if quality_score < 40:
        return json.dumps({
            "status":  "rejected",
            "reason":  "Quality score too low for reliable predictions",
            "quality": quality_result.get("quality", {}),
        })
    result = add_ticker_to_watchlist(ticker)
    return json.dumps(result, default=str)

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "get_todays_signals":
            result = tool_get_todays_signals()
        elif tool_name == "run_fresh_predictions":
            result = tool_run_fresh_predictions()
        elif tool_name == "get_signal_history":
            result = tool_get_signal_history(
                tool_input.get("ticker", "AAPL"),
                tool_input.get("days", 30))
        elif tool_name == "get_macro_conditions":
            result = tool_get_macro_conditions()
        elif tool_name == "get_performance_stats":
            result = tool_get_performance_stats()
        elif tool_name == "run_backtest":
            result = tool_run_backtest(tool_input.get("ticker", "AAPL"))
        elif tool_name == "explain_signal":
            result = tool_explain_signal(tool_input.get("ticker", "AAPL"))
        elif tool_name == "analyze_any_ticker":
            result = tool_analyze_any_ticker(tool_input.get("ticker", ""))
        elif tool_name == "add_to_watchlist":
            result = tool_add_to_watchlist(tool_input.get("ticker", ""))
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, default=str)

def chat(user_message: str, history: list) -> tuple:
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )
    reply = response.choices[0].message.content.strip()

    # Check if model wants to call a tool
    tool_called = False
    if reply.startswith("{"):
        try:
            parsed     = json.loads(reply)
            tool_name  = parsed.get("tool", "")
            tool_input = parsed.get("input", {})
            if tool_name:
                tool_called = True
                print(f"  [Bot] Calling tool: {tool_name}")
                result = dispatch_tool(tool_name, tool_input)

                history.append({"role": "assistant", "content": reply})
                history.append({"role": "user",
                                "content": f"Tool result for {tool_name}: {result}\n\nNow give a clear, helpful response to the user based on this data."})

                messages2  = [{"role": "system", "content": SYSTEM_PROMPT}] + history
                response2  = client.chat.completions.create(
                    model=MODEL,
                    messages=messages2,
                    max_tokens=2048,
                    temperature=0.3,
                )
                final = response2.choices[0].message.content.strip()
                history.append({"role": "assistant", "content": final})
                return final, history
        except json.JSONDecodeError:
            pass

    if not tool_called:
        history.append({"role": "assistant", "content": reply})

    return reply, history

def run_terminal_bot():
    print("\n" + "═"*60)
    print("  Stock Prediction Bot — powered by Groq + Llama 3.1 70B")
    print("  Type 'quit' to exit | 'help' for example questions")
    print("═"*60 + "\n")

    history = []

    examples = [
        "What should I buy today?",
        "What's the current market fear level?",
        "Explain why you're bearish on NVDA",
        "Run a backtest on MSFT",
        "Show me AAPL signal history",
        "What are the win rates for each ticker?",
        "Should I be worried about the market right now?",
        "Refresh predictions with latest data",
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "help":
            print("\nExample questions:")
            for ex in examples:
                print(f"  - {ex}")
            continue

        print("\nBot: ", end="", flush=True)
        try:
            response, history = chat(user_input, history)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_terminal_bot()
