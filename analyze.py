# analyze.py
# On-demand analysis for any ticker not in the main watchlist.
# Uses the combined model for quick signals, plus technical analysis
# and a watchlist quality assessment.

import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime

from macro_loader import fetch_macro
from features import add_features, get_feature_columns
from config import TICKERS

COMBINED_MODEL_PATH  = os.path.join("models", "combined_xgb.pkl")
COMBINED_SCALER_PATH = os.path.join("models", "combined_scaler.pkl")

def fetch_ticker_data(ticker: str,
                      period: str = "5y") -> pd.DataFrame:
    """Fetches OHLCV data for any ticker."""
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

def assess_watchlist_quality(ticker: str,
                              df: pd.DataFrame) -> dict:
    """
    Evaluates whether a ticker is suitable for full training.
    Checks history length, liquidity, volatility regime,
    and data completeness.
    """
    issues   = []
    score    = 100
    verdict  = "SUITABLE"

    # History check — need at least 3 years for meaningful training
    years = len(df) / 252
    if years < 2:
        issues.append(f"Only {years:.1f} years of data (need 3+)")
        score  -= 40
        verdict = "NOT SUITABLE"
    elif years < 3:
        issues.append(f"Only {years:.1f} years of data (3+ preferred)")
        score -= 15

    # Liquidity check — avg daily volume
    avg_volume = df["Volume"].tail(60).mean()
    if avg_volume < 500_000:
        issues.append(f"Low liquidity: avg {avg_volume/1e6:.1f}M daily volume")
        score  -= 30
        verdict = "NOT SUITABLE"
    elif avg_volume < 2_000_000:
        issues.append(f"Moderate liquidity: {avg_volume/1e6:.1f}M daily volume")
        score -= 10

    # Price check — penny stocks are unpredictable
    last_price = float(df["Close"].iloc[-1])
    if last_price < 5:
        issues.append(f"Penny stock territory (${last_price:.2f})")
        score  -= 30
        verdict = "NOT SUITABLE"
    elif last_price < 10:
        issues.append(f"Low price stock (${last_price:.2f}) — higher noise")
        score -= 10

    # Volatility check — extreme volatility = harder to predict
    daily_returns = df["Close"].pct_change().dropna()
    annualized_vol = daily_returns.std() * np.sqrt(252)
    if annualized_vol > 1.5:
        issues.append(f"Extreme volatility: {annualized_vol:.0%} annualized")
        score -= 20
    elif annualized_vol > 0.8:
        issues.append(f"High volatility: {annualized_vol:.0%} annualized")
        score -= 5

    # Gap check — missing trading days
    business_days = pd.bdate_range(df.index[0], df.index[-1])
    coverage      = len(df) / len(business_days)
    if coverage < 0.85:
        issues.append(f"Data gaps: only {coverage:.0%} of trading days present")
        score -= 20

    # Already in watchlist
    if ticker.upper() in TICKERS:
        issues.append("Already in main watchlist — use full model instead")
        verdict = "IN WATCHLIST"

    if score >= 80 and verdict == "SUITABLE":
        verdict = "EXCELLENT — recommend adding"
    elif score >= 60 and verdict == "SUITABLE":
        verdict = "GOOD — worth adding"
    elif score >= 40 and verdict == "SUITABLE":
        verdict = "MARGINAL — monitor first"

    # Meme stock check
    MEME_STOCKS = ["GME", "AMC", "BBBY", "KOSS", "NOK", "BB", "HOOD"]
    if ticker.upper() in MEME_STOCKS:
        issues.append("Meme stock — price driven by social media, not fundamentals")
        score  -= 20
        verdict = "MARGINAL — monitor first" if verdict != "NOT SUITABLE" else verdict
        
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

def get_technical_snapshot(df: pd.DataFrame) -> dict:
    """Returns key technical indicators for the latest bar."""
    close = df["Close"]

    # RSI
    delta    = close.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / (loss + 1e-9)
    rsi      = float((100 - 100 / (1 + rs)).iloc[-1])

    # MACD
    ema12    = close.ewm(span=12).mean()
    ema26    = close.ewm(span=26).mean()
    macd     = ema12 - ema26
    signal   = macd.ewm(span=9).mean()
    macd_h   = float((macd - signal).iloc[-1])

    # Bollinger Bands
    ma20     = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_pos   = float(
        ((close - (ma20 - 2*std20)) /
         (4*std20 + 1e-9)).iloc[-1]
    )

    # Moving averages
    ma50     = float(close.rolling(50).mean().iloc[-1])
    ma200    = float(close.rolling(200).mean().iloc[-1]) \
               if len(close) >= 200 else None
    price    = float(close.iloc[-1])

    # Returns
    ret_5d   = float(close.pct_change(5).iloc[-1])
    ret_20d  = float(close.pct_change(20).iloc[-1])
    ret_63d  = float(close.pct_change(63).iloc[-1]) \
               if len(close) >= 63 else None

    # ATR
    hl       = df["High"] - df["Low"]
    hc       = (df["High"] - close.shift(1)).abs()
    lc       = (df["Low"]  - close.shift(1)).abs()
    atr      = float(
        pd.concat([hl, hc, lc], axis=1)
        .max(axis=1).rolling(14).mean().iloc[-1]
    )

    # Trend
    above_50ma  = price > ma50
    above_200ma = (price > ma200) if ma200 else None

    # RSI interpretation
    if rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"

    return {
        "price":          round(price, 2),
        "rsi_14":         round(rsi, 2),
        "rsi_signal":     rsi_signal,
        "macd_hist":      round(macd_h, 4),
        "bb_position":    round(bb_pos, 4),
        "ma50":           round(ma50, 2),
        "ma200":          round(ma200, 2) if ma200 else None,
        "above_50ma":     above_50ma,
        "above_200ma":    above_200ma,
        "return_5d":      round(ret_5d, 4),
        "return_20d":     round(ret_20d, 4),
        "return_63d":     round(ret_63d, 4) if ret_63d else None,
        "atr":            round(atr, 2),
        "atr_pct":        round(atr / price, 4),
    }

def get_combined_model_signal(ticker: str,
                               df: pd.DataFrame,
                               macro_df: pd.DataFrame) -> dict:
    if not os.path.exists(COMBINED_MODEL_PATH):
        return {"error": "Combined model not found — run train_classical.py"}

    try:
        model  = joblib.load(COMBINED_MODEL_PATH)
        scaler = joblib.load(COMBINED_SCALER_PATH)

        # Build features with sentiment and earnings as zeros
        # Unknown tickers don't have these but we need the columns
        # to match the combined model's 57 + ticker_id = 58 features
        sentiment_series = pd.Series(
            0.0,
            index=df.index,
            name="sentiment"
        )

        df_feat = add_features(df.copy(),
                               macro_df=macro_df,
                               sentiment_series=sentiment_series,
                               earnings_df=None,
                               predict_mode=True)

        if df_feat.empty:
            return {"error": "Could not generate features"}

        # Add zero earnings columns manually since earnings_df is None
        from earnings_loader import get_earnings_feature_columns
        for col in get_earnings_feature_columns():
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        # Get all 57 features + ticker_id
        feat_cols = get_feature_columns(
            include_macro=True,
            include_sentiment=True,
            include_earnings=True,
        )
        feat_cols = [c for c in feat_cols if c in df_feat.columns]

        # ticker_id = -1 for unknown tickers
        # Use 0 as a safer value since -1 is out of training distribution
        df_feat["ticker_id"] = 0
        feat_cols_final = feat_cols + ["ticker_id"]
        feat_cols_final = [c for c in feat_cols_final if c in df_feat.columns]

        X       = df_feat[feat_cols_final].iloc[[-1]].values
        X_s     = scaler.transform(X)
        prob_up = float(model.predict_proba(X_s)[0][1])

        return {
            "prob_up":    round(prob_up, 4),
            "signal":     "BUY"   if prob_up >= 0.55 else
                          "WATCH" if prob_up >= 0.45 else "SELL",
            "confidence": round(abs(prob_up - 0.5) * 2, 4),
            "note":       "Combined model signal — no ticker-specific training yet",
        }
    except Exception as e:
        return {"error": str(e)}
def analyze_ticker(ticker: str) -> dict:
    """
    Full analysis pipeline for any ticker.
    Returns technicals, quality assessment, and combined model signal.
    """
    ticker = ticker.upper().strip()
    print(f"  Analyzing {ticker}...")

    # Fetch data
    df = fetch_ticker_data(ticker, period="5y")
    if df.empty or len(df) < 60:
        return {
            "ticker": ticker,
            "error":  f"Insufficient data for {ticker} — may be invalid symbol",
        }

    # Get company info
    try:
        info         = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)
        sector       = info.get("sector", "Unknown")
        industry     = info.get("industry", "Unknown")
        market_cap   = info.get("marketCap", 0)
        mc_str       = (f"${market_cap/1e9:.1f}B" if market_cap > 1e9
                        else f"${market_cap/1e6:.0f}M" if market_cap > 1e6
                        else "Unknown")
    except Exception:
        company_name = ticker
        sector       = "Unknown"
        industry     = "Unknown"
        mc_str       = "Unknown"

    macro_df   = fetch_macro()
    technicals = get_technical_snapshot(df)
    quality    = assess_watchlist_quality(ticker, df)
    signal     = get_combined_model_signal(ticker, df, macro_df)

    # Build human-readable summary
    trend = ("Uptrend" if technicals["above_50ma"] else "Downtrend")
    rsi_s = technicals["rsi_signal"]

    summary_points = []
    if technicals["rsi_14"] > 70:
        summary_points.append("RSI overbought — potential pullback")
    elif technicals["rsi_14"] < 30:
        summary_points.append("RSI oversold — potential bounce")

    if technicals["macd_hist"] > 0:
        summary_points.append("MACD positive — bullish momentum")
    else:
        summary_points.append("MACD negative — bearish momentum")

    if technicals["bb_position"] > 0.8:
        summary_points.append("Near upper Bollinger Band — extended")
    elif technicals["bb_position"] < 0.2:
        summary_points.append("Near lower Bollinger Band — oversold")

    if not technicals["above_50ma"]:
        summary_points.append("Below 50-day MA — in downtrend")

    return {
        "ticker":       ticker,
        "company":      company_name,
        "sector":       sector,
        "industry":     industry,
        "market_cap":   mc_str,
        "date":         str(date.today()),
        "technicals":   technicals,
        "quality":      quality,
        "model_signal": signal,
        "trend":        trend,
        "summary":      summary_points,
        "in_watchlist": ticker in TICKERS,
        "recommendation": quality["verdict"],
    }

def add_ticker_to_watchlist(ticker: str) -> dict:
    """
    Adds a ticker to config.py TICKERS list and triggers retraining.
    Returns instructions for what to run next.
    """
    ticker = ticker.upper().strip()

    if ticker in TICKERS:
        return {"status": "already_exists",
                "message": f"{ticker} is already in the watchlist"}

    # Read current config
    config_path = os.path.join(
        os.path.dirname(__file__), "config.py")
    with open(config_path, "r") as f:
        content = f.read()

    # Find the TICKERS list and add the new ticker
    if f'"{ticker}"' in content or f"'{ticker}'" in content:
        return {"status": "already_exists",
                "message": f"{ticker} already appears in config.py"}

    # Insert before the closing bracket of TICKERS list
    old = f'"{TICKERS[-1]}",   # new\n]'
    new_line = f'"{TICKERS[-1]}",   # new\n    "{ticker}",\n]'

    # Simpler approach — find last ticker and add after
    last_ticker = TICKERS[-1]
    old_str     = f'"{last_ticker}",'
    new_str     = f'"{last_ticker}",\n    "{ticker}",'

    if old_str in content:
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
            ]
        }
    else:
        return {
            "status":  "error",
            "message": "Could not modify config.py — edit manually",
        }

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "PLTR"
    result = analyze_ticker(ticker)
    print(json.dumps(result, indent=2, default=str))
