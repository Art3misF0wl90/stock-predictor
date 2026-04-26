# app/ml/features.py
#
# Feature engineering for the stock prediction pipeline.
#
# add_features() is the single entry point: it takes a raw OHLCV DataFrame
# and optional macro / sentiment / earnings DataFrames and returns a new
# DataFrame with 50+ computed feature columns plus a binary `target` column.
#
# get_feature_columns() returns the canonical ordered list of feature names
# that models expect, parameterised by which optional feature groups are
# included.  This list must stay in sync with what add_features() produces.
#
# Feature groups (see README for full descriptions):
#   Price returns       — 1d / 5d / 10d / 20d pct changes
#   Moving averages     — price-to-MA ratios and MA cross-ratios
#   RSI                 — 7 / 14 / 21-day, momentum, lagged values
#   Lagged returns      — t-1 through t-10
#   Bollinger Bands     — position within band, band width
#   MACD                — line, signal, histogram, momentum
#   Volume              — ratio to 10d MA, trend, lagged ratio
#   Candlestick         — overnight gap, intraday reversal, upper/lower wicks
#   Streak              — count of up-closes in last 3 / 5 / 10 days
#   ATR                 — 14-day average true range, normalised by price
#   Price position      — where close sits in its 20-day high-low range
#   Macro               — VIX, treasury, dollar features (optional)
#   Sentiment           — VADER scores and momentum (optional)
#   Earnings            — EPS surprise, PEAD signal, days-to/since (optional)

import pandas as pd
import numpy as np

from config import FORWARD_DAYS


def add_features(
    df: pd.DataFrame,
    macro_df: pd.DataFrame = None,
    sentiment_series: pd.Series = None,
    earnings_df: pd.DataFrame = None,
    forward_days: int = None,
    predict_mode: bool = False,
) -> pd.DataFrame:
    """
    Compute all features and attach them to a copy of df.

    Parameters
    ----------
    df              Raw OHLCV DataFrame indexed by date.
    macro_df        Output of fetch_macro(); joined on date index.
    sentiment_series  Daily VADER compound score Series; joined on date index.
    earnings_df     Output of build_earnings_features(); columns merged in.
    forward_days    Horizon for the target variable (default: FORWARD_DAYS).
    predict_mode    When True, skip target computation and only drop rows
                    where feature columns are NaN (not the whole row).
                    Use this when generating live signals so the last row
                    is always kept.
    """
    df = df.copy()
    fwd = forward_days if forward_days is not None else FORWARD_DAYS

    # ── Price-based returns ────────────────────────────────────────────────
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    # ── Moving averages and price-to-MA ratios ─────────────────────────────
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()

    # Ratios > 1.0 = price above MA (uptrend); < 1.0 = below (downtrend).
    # Scale-invariant: works across all tickers and time periods.
    df["price_to_ma10"] = df["Close"] / df["ma_10"]
    df["price_to_ma20"] = df["Close"] / df["ma_20"]
    df["price_to_ma50"] = df["Close"] / df["ma_50"]
    df["ma_cross_10_50"] = df["ma_10"] / df["ma_50"]
    df["ma_cross_10_20"] = df["ma_10"] / df["ma_20"]

    # ── RSI — Relative Strength Index ─────────────────────────────────────
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    for period in [7, 21]:
        ag   = gain.rolling(period).mean()
        al   = loss.rolling(period).mean()
        rs_p = ag / (al + 1e-9)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs_p))

    # Momentum and lagged RSI let the model see RSI trajectory.
    df["rsi_momentum"] = df["rsi_14"] - df["rsi_14"].shift(5)
    df["rsi_lag_1"]    = df["rsi_14"].shift(1)
    df["rsi_lag_3"]    = df["rsi_14"].shift(3)

    # ── Lagged returns ─────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std  = df["Close"].rolling(20).std()
    bb_upper     = rolling_mean + 2 * rolling_std
    bb_lower     = rolling_mean - 2 * rolling_std

    # 0 = at lower band, 1 = at upper band
    df["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["bb_width"]    = (bb_upper - bb_lower) / (rolling_mean + 1e-9)

    # ── MACD ──────────────────────────────────────────────────────────────
    ema_12            = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26            = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    # 3-day change in histogram captures acceleration/deceleration of momentum
    df["macd_momentum"] = df["macd_hist"] - df["macd_hist"].shift(3)

    # ── Volume ────────────────────────────────────────────────────────────
    df["volume_ma10"]  = df["Volume"].rolling(10).mean()
    df["volume_ma20"]  = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ma10"] + 1e-9)
    df["volume_lag_1"] = df["volume_ratio"].shift(1)
    df["volume_trend"] = df["volume_ma10"] / (df["volume_ma20"] + 1e-9)

    # ── Candlestick microstructure ─────────────────────────────────────────
    df["open_close_diff"] = df["Open"] - df["Close"]
    df["high_low_diff"]   = df["High"] - df["Low"]

    # Positive = gapped up overnight; negative = gapped down
    df["overnight_gap"] = (
        (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-9)
    )
    # +1 = closed at high (bullish); -1 = closed at low (bearish)
    df["intraday_reversal"] = (
        (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    )
    # Upper wick = rejection of high; lower wick = rejection of low
    df["upper_wick"] = (
        (df["High"] - df[["Open", "Close"]].max(axis=1))
        / (df["High"] - df["Low"] + 1e-9)
    )
    df["lower_wick"] = (
        (df[["Open", "Close"]].min(axis=1) - df["Low"])
        / (df["High"] - df["Low"] + 1e-9)
    )

    # ── Streak features ───────────────────────────────────────────────────
    for window in [3, 5, 10]:
        df[f"up_days_{window}"] = sum(
            [(df["Close"].shift(i) > df["Close"].shift(i + 1)).astype(int)
             for i in range(window)]
        )

    # ── ATR — Average True Range ───────────────────────────────────────────
    hl        = df["High"] - df["Low"]
    hc        = (df["High"] - df["Close"].shift(1)).abs()
    lc        = (df["Low"]  - df["Close"].shift(1)).abs()
    tr        = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"]         = tr.rolling(14).mean()
    df["atr_normalized"] = df["atr_14"] / (df["Close"] + 1e-9)

    # ── Price position within 20-day range ────────────────────────────────
    rolling_high_20 = df["High"].rolling(20).max()
    rolling_low_20  = df["Low"].rolling(20).min()
    df["price_position_20"] = (
        (df["Close"] - rolling_low_20)
        / (rolling_high_20 - rolling_low_20 + 1e-9)
    )

    # ── Optional: Macro features ───────────────────────────────────────────
    if macro_df is not None:
        from app.data import get_macro_feature_columns
        macro_cols    = get_macro_feature_columns()
        macro_aligned = macro_df[macro_cols].copy()
        # Strip timezone info so the join index aligns correctly
        if macro_aligned.index.tzinfo is not None:
            macro_aligned.index = macro_aligned.index.tz_localize(None)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(macro_aligned, how="left")
        # Forward-fill weekends/holidays, then back-fill start-of-series gaps
        df[macro_cols] = df[macro_cols].ffill().bfill()

    # ── Optional: Sentiment features ──────────────────────────────────────
    if sentiment_series is not None:
        s = sentiment_series.copy()
        if s.index.tzinfo is not None:
            s.index = s.index.tz_localize(None)
        df = df.join(s.rename("sentiment"), how="left")
        df["sentiment"]          = df["sentiment"].ffill().fillna(0.0)
        df["sentiment_change"]   = df["sentiment"].diff(1)
        df["sentiment_ma5"]      = df["sentiment"].rolling(5).mean()
        df["sentiment_ma20"]     = df["sentiment"].rolling(20).mean()
        # Positive = sentiment accelerating; negative = fading
        df["sentiment_momentum"] = df["sentiment_ma5"] - df["sentiment_ma20"]

    # ── Optional: Earnings features ───────────────────────────────────────
    if earnings_df is not None:
        from app.data import get_earnings_feature_columns
        for col in get_earnings_feature_columns():
            if col in earnings_df.columns:
                df[col] = earnings_df[col].values

    # ── Target variable ────────────────────────────────────────────────────
    if not predict_mode:
        # 1 if price is higher in fwd trading days, 0 otherwise.
        # shift(-fwd) looks forward without touching any future features —
        # the feature columns are all computed from past data only.
        df["target"] = (df["Close"].shift(-fwd) > df["Close"]).astype(int)
        df.dropna(inplace=True)
    else:
        # In predict mode keep all rows but drop any where a feature is NaN.
        # This preserves the most recent row (today) which shift(-fwd) would
        # otherwise remove.
        feature_cols = [c for c in df.columns if c != "target"]
        df = df.dropna(subset=feature_cols)

    return df


def get_feature_columns(
    include_macro: bool = True,
    include_sentiment: bool = True,
    include_earnings: bool = True,
) -> list[str]:
    """
    Return the canonical ordered list of feature column names.

    This list must match exactly what add_features() produces.
    Models are trained and loaded using this list to select columns,
    so order and spelling are significant.
    """
    base = [
        # Returns
        "return_1d", "return_5d", "return_10d", "return_20d",
        # Lagged returns
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
        # Lagged RSI
        "rsi_lag_1", "rsi_lag_3",
        # Moving average ratios
        "price_to_ma10", "price_to_ma20", "price_to_ma50",
        "ma_cross_10_50", "ma_cross_10_20",
        # RSI
        "rsi_14", "rsi_7", "rsi_21", "rsi_momentum",
        # Bollinger Bands
        "bb_position", "bb_width",
        # MACD
        "macd", "macd_signal", "macd_hist", "macd_momentum",
        # Volume
        "volume_ratio", "volume_trend", "volume_lag_1",
        # Candlestick
        "open_close_diff", "high_low_diff",
        "overnight_gap", "intraday_reversal",
        "upper_wick", "lower_wick",
        # Streak
        "up_days_3", "up_days_5", "up_days_10",
        # ATR
        "atr_normalized",
        # Price position
        "price_position_20",
    ]

    if include_macro:
        base += [
            "vix", "vix_change", "vix_ma10", "vix_regime", "vix_zscore",
            "treasury", "treasury_change", "treasury_ma10",
            "dollar", "dollar_change",
        ]

    if include_sentiment:
        base += [
            "sentiment", "sentiment_change",
            "sentiment_ma5", "sentiment_momentum",
        ]

    if include_earnings:
        base += [
            "eps_surprise",
            "days_to_earnings", "days_since_earnings",
            "pead_signal",
        ]

    return base
