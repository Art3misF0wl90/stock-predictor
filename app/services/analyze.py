# app/services/analyze.py
#
# On-demand analysis for any ticker — including those not in the watchlist.
#
# analyze_ticker() is the main entry point.  It fetches live data, computes
# technical indicators, scores watchlist suitability, and runs the combined
# model for a directional signal.  It is called by:
#   - The /api/analyze/<ticker> route
#   - The bot's analyze_any_ticker tool
#   - add_ticker_to_watchlist() (quality gate)
#
# add_ticker_to_watchlist() appends a new ticker to config.py and returns
# the list of follow-up scripts the user needs to run.

import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from app.data import fetch_macro
from app.ml.features import add_features, get_feature_columns
from config import TICKERS

_COMBINED_MODEL_PATH  = os.path.join("models", "combined_xgb.pkl")
_COMBINED_SCALER_PATH = os.path.join("models", "combined_scaler.pkl")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_ticker_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV for any ticker symbol (not limited to the watchlist)."""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Adj Close" in df.columns:
            df.drop(columns=["Adj Close"], inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  Could not fetch {ticker}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Watchlist quality assessment
# ---------------------------------------------------------------------------

# Tickers known to be driven by social media rather than fundamentals.
_MEME_STOCKS = {"GME", "AMC", "BBBY", "KOSS", "NOK", "BB", "HOOD"}


def assess_watchlist_quality(ticker: str, df: pd.DataFrame) -> dict:
    """
    Score a ticker's suitability for full model training (0–100).

    Checks: history length, average daily volume, price floor,
    annualised volatility, and data coverage (gap detection).

    Returns a dict with score, verdict, per-check values, and a list
    of human-readable issues found.
    """
    issues  = []
    score   = 100
    verdict = "SUITABLE"

    # ── History ──────────────────────────────────────────────────────────
    years = len(df) / 252
    if years < 2:
        issues.append(f"Only {years:.1f} years of data (need 3+)")
        score -= 40
        verdict = "NOT SUITABLE"
    elif years < 3:
        issues.append(f"Only {years:.1f} years of data (3+ preferred)")
        score -= 15

    # ── Liquidity ─────────────────────────────────────────────────────────
    avg_volume = df["Volume"].tail(60).mean()
    if avg_volume < 500_000:
        issues.append(f"Low liquidity: avg {avg_volume/1e6:.1f}M daily volume")
        score -= 30
        verdict = "NOT SUITABLE"
    elif avg_volume < 2_000_000:
        issues.append(f"Moderate liquidity: {avg_volume/1e6:.1f}M daily volume")
        score -= 10

    # ── Price floor ───────────────────────────────────────────────────────
    last_price = float(df["Close"].iloc[-1])
    if last_price < 5:
        issues.append(f"Penny stock territory (${last_price:.2f})")
        score -= 30
        verdict = "NOT SUITABLE"
    elif last_price < 10:
        issues.append(f"Low price stock (${last_price:.2f}) — higher noise")
        score -= 10

    # ── Volatility ────────────────────────────────────────────────────────
    annualized_vol = df["Close"].pct_change().dropna().std() * np.sqrt(252)
    if annualized_vol > 1.5:
        issues.append(f"Extreme volatility: {annualized_vol:.0%} annualized")
        score -= 20
    elif annualized_vol > 0.8:
        issues.append(f"High volatility: {annualized_vol:.0%} annualized")
        score -= 5

    # ── Data coverage ─────────────────────────────────────────────────────
    business_days = pd.bdate_range(df.index[0], df.index[-1])
    coverage      = len(df) / len(business_days)
    if coverage < 0.85:
        issues.append(f"Data gaps: only {coverage:.0%} of trading days present")
        score -= 20

    # ── Already in watchlist ──────────────────────────────────────────────
    if ticker.upper() in TICKERS:
        issues.append("Already in main watchlist — use full model instead")
        verdict = "IN WATCHLIST"

    # ── Verdict label ─────────────────────────────────────────────────────
    if score >= 80 and verdict == "SUITABLE":
        verdict = "EXCELLENT — recommend adding"
    elif score >= 60 and verdict == "SUITABLE":
        verdict = "GOOD — worth adding"
    elif score >= 40 and verdict == "SUITABLE":
        verdict = "MARGINAL — monitor first"

    # ── Meme stock penalty ────────────────────────────────────────────────
    if ticker.upper() in _MEME_STOCKS:
        issues.append("Meme stock — price driven by social media, not fundamentals")
        score -= 20
        if verdict not in ("NOT SUITABLE", "IN WATCHLIST"):
            verdict = "MARGINAL — monitor first"

    return {
        "score":            score,
        "verdict":          verdict,
        "years_of_data":    round(years, 1),
        "avg_daily_volume": round(avg_volume / 1e6, 2),
        "last_price":       round(last_price, 2),
        "annualized_vol":   round(annualized_vol, 4),
        "data_coverage":    round(coverage, 3),
        "issues":           issues,
    }


# ---------------------------------------------------------------------------
# Technical snapshot
# ---------------------------------------------------------------------------

def get_technical_snapshot(df: pd.DataFrame) -> dict:
    """
    Compute key technical indicators for the most recent bar.

    Returns a flat dict suitable for JSON serialisation and
    display in the web dashboard modal.
    """
    close = df["Close"]

    # RSI (14)
    delta    = close.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rsi      = float((100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1])

    # MACD histogram
    ema12   = close.ewm(span=12).mean()
    ema26   = close.ewm(span=26).mean()
    macd_h  = float(((ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()).iloc[-1])

    # Bollinger Band position (0 = lower band, 1 = upper band)
    ma20    = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    bb_pos  = float(
        ((close - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)).iloc[-1]
    )

    # Moving averages
    ma50    = float(close.rolling(50).mean().iloc[-1])
    ma200   = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    price   = float(close.iloc[-1])

    # Returns
    ret_5d  = float(close.pct_change(5).iloc[-1])
    ret_20d = float(close.pct_change(20).iloc[-1])
    ret_63d = float(close.pct_change(63).iloc[-1]) if len(close) >= 63 else None

    # ATR (14)
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - close.shift(1)).abs()
    lc  = (df["Low"]  - close.shift(1)).abs()
    atr = float(pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1])

    if rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"

    return {
        "price":        round(price, 2),
        "rsi_14":       round(rsi, 2),
        "rsi_signal":   rsi_signal,
        "macd_hist":    round(macd_h, 4),
        "bb_position":  round(bb_pos, 4),
        "ma50":         round(ma50, 2),
        "ma200":        round(ma200, 2) if ma200 else None,
        "above_50ma":   price > ma50,
        "above_200ma":  (price > ma200) if ma200 else None,
        "return_5d":    round(ret_5d, 4),
        "return_20d":   round(ret_20d, 4),
        "return_63d":   round(ret_63d, 4) if ret_63d else None,
        "atr":          round(atr, 2),
        "atr_pct":      round(atr / price, 4),
    }


# ---------------------------------------------------------------------------
# Combined model signal
# ---------------------------------------------------------------------------

def get_combined_model_signal(
    ticker: str,
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> dict:
    """
    Run the combined (all-ticker) XGBoost model on the latest bar.

    Used for tickers that don't have a dedicated trained model.
    Sentiment and earnings are zeroed out since we don't have cached
    data for arbitrary tickers.
    """
    if not os.path.exists(_COMBINED_MODEL_PATH):
        return {"error": "Combined model not found — run train_classical.py"}

    try:
        model  = joblib.load(_COMBINED_MODEL_PATH)
        scaler = joblib.load(_COMBINED_SCALER_PATH)

        # Provide a zero sentiment series so the feature pipeline
        # produces the right number of columns.
        sentiment_series = pd.Series(0.0, index=df.index, name="sentiment")

        df_feat = add_features(
            df.copy(),
            macro_df=macro_df,
            sentiment_series=sentiment_series,
            earnings_df=None,
            predict_mode=True,
        )

        if df_feat.empty:
            return {"error": "Could not generate features"}

        # Ensure earnings columns exist (zeroed) so column count matches.
        from earnings_loader import get_earnings_feature_columns
        for col in get_earnings_feature_columns():
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        feat_cols = get_feature_columns(
            include_macro=True,
            include_sentiment=True,
            include_earnings=True,
        )
        feat_cols = [c for c in feat_cols if c in df_feat.columns]

        # ticker_id=0 is a safe out-of-distribution default for unknown tickers
        df_feat["ticker_id"]   = 0
        feat_cols_final        = feat_cols + ["ticker_id"]
        feat_cols_final        = [c for c in feat_cols_final if c in df_feat.columns]

        X       = df_feat[feat_cols_final].iloc[[-1]].values
        prob_up = float(model.predict_proba(scaler.transform(X))[0][1])

        return {
            "prob_up":    round(prob_up, 4),
            "signal":     "BUY"   if prob_up >= 0.55 else
                          "WATCH" if prob_up >= 0.45 else "SELL",
            "confidence": round(abs(prob_up - 0.5) * 2, 4),
            "note":       "Combined model signal — no ticker-specific training yet",
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str) -> dict:
    """
    Run the full on-demand analysis pipeline for any ticker symbol.

    Returns a dict containing company info, technicals, watchlist quality
    assessment, combined model signal, and a list of key observations.
    Safe to call from the web route or the bot — returns error key on failure.
    """
    ticker = ticker.upper().strip()
    print(f"  Analyzing {ticker}...")

    df = _fetch_ticker_data(ticker, period="5y")
    if df.empty or len(df) < 60:
        return {
            "ticker": ticker,
            "error":  f"Insufficient data for {ticker} — may be invalid symbol",
        }

    # Company metadata
    try:
        info         = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)
        sector       = info.get("sector", "Unknown")
        industry     = info.get("industry", "Unknown")
        market_cap   = info.get("marketCap", 0)
        mc_str = (
            f"${market_cap/1e9:.1f}B" if market_cap > 1e9
            else f"${market_cap/1e6:.0f}M" if market_cap > 1e6
            else "Unknown"
        )
    except Exception:
        company_name = ticker
        sector = industry = "Unknown"
        mc_str = "Unknown"

    macro_df   = fetch_macro()
    technicals = get_technical_snapshot(df)
    quality    = assess_watchlist_quality(ticker, df)
    signal     = get_combined_model_signal(ticker, df, macro_df)

    # ── Human-readable observations ────────────────────────────────────────
    summary = []
    if technicals["rsi_14"] > 70:
        summary.append("RSI overbought — potential pullback")
    elif technicals["rsi_14"] < 30:
        summary.append("RSI oversold — potential bounce")

    if technicals["macd_hist"] > 0:
        summary.append("MACD positive — bullish momentum")
    else:
        summary.append("MACD negative — bearish momentum")

    if technicals["bb_position"] > 0.8:
        summary.append("Near upper Bollinger Band — extended")
    elif technicals["bb_position"] < 0.2:
        summary.append("Near lower Bollinger Band — oversold")

    if not technicals["above_50ma"]:
        summary.append("Below 50-day MA — in downtrend")

    return {
        "ticker":         ticker,
        "company":        company_name,
        "sector":         sector,
        "industry":       industry,
        "market_cap":     mc_str,
        "date":           str(date.today()),
        "technicals":     technicals,
        "quality":        quality,
        "model_signal":   signal,
        "trend":          "Uptrend" if technicals["above_50ma"] else "Downtrend",
        "summary":        summary,
        "in_watchlist":   ticker in TICKERS,
        "recommendation": quality["verdict"],
    }


# ---------------------------------------------------------------------------
# Watchlist management
# ---------------------------------------------------------------------------

def add_ticker_to_watchlist(ticker: str) -> dict:
    """
    Append ticker to the TICKERS list in config.py.

    Does a simple string replacement to insert the new ticker after the
    last existing one.  Returns a status dict with next_steps instructions
    when successful.
    """
    ticker = ticker.upper().strip()

    if ticker in TICKERS:
        return {
            "status":  "already_exists",
            "message": f"{ticker} is already in the watchlist",
        }

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.py")
    config_path = os.path.normpath(config_path)

    with open(config_path, "r") as f:
        content = f.read()

    if f'"{ticker}"' in content or f"'{ticker}'" in content:
        return {
            "status":  "already_exists",
            "message": f"{ticker} already appears in config.py",
        }

    last_ticker = TICKERS[-1]
    old_str     = f'"{last_ticker}",'
    new_str     = f'"{last_ticker}",\n    "{ticker}",'

    if old_str not in content:
        return {
            "status":  "error",
            "message": "Could not modify config.py — edit manually",
        }

    content = content.replace(old_str, new_str, 1)
    with open(config_path, "w") as f:
        f.write(content)

    return {
        "status":  "added",
        "ticker":  ticker,
        "message": f"{ticker} added to watchlist",
        "next_steps": [
            "Run: python3 sentiment_loader.py (fetch news for new ticker)",
            "Run: python3 earnings_loader.py  (fetch earnings data)",
            "Run: python3 train_classical.py  (train ticker-specific model)",
            "Run: python3 backtest.py         (evaluate new model)",
            "Run: python3 predict.py          (generate first signal)",
        ],
    }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "PLTR"
    result = analyze_ticker(ticker)
    print(json.dumps(result, indent=2, default=str))
