import pandas as pd
import numpy as np
from config import FORWARD_DAYS

def add_features(df: pd.DataFrame,
                 macro_df: pd.DataFrame = None,
                 sentiment_series: pd.Series = None,
                 earnings_df: pd.DataFrame = None,
                 forward_days: int = None,
                 predict_mode: bool = False) -> pd.DataFrame:
    df  = df.copy()
    fwd = forward_days if forward_days is not None else FORWARD_DAYS

    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()

    df["price_to_ma10"] = df["Close"] / df["ma_10"]
    df["price_to_ma20"] = df["Close"] / df["ma_20"]
    df["price_to_ma50"] = df["Close"] / df["ma_50"]

    df["ma_cross_10_50"] = df["ma_10"] / df["ma_50"]
    df["ma_cross_10_20"] = df["ma_10"] / df["ma_20"]

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

    df["rsi_momentum"] = df["rsi_14"] - df["rsi_14"].shift(5)
    df["rsi_lag_1"]    = df["rsi_14"].shift(1)
    df["rsi_lag_3"]    = df["rsi_14"].shift(3)

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    rolling_mean      = df["Close"].rolling(20).mean()
    rolling_std       = df["Close"].rolling(20).std()
    bb_upper          = rolling_mean + 2 * rolling_std
    bb_lower          = rolling_mean - 2 * rolling_std
    df["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["bb_width"]    = (bb_upper - bb_lower) / (rolling_mean + 1e-9)

    ema_12              = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26              = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]          = ema_12 - ema_26
    df["macd_signal"]   = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]     = df["macd"] - df["macd_signal"]
    df["macd_momentum"] = df["macd_hist"] - df["macd_hist"].shift(3)

    df["volume_ma10"]  = df["Volume"].rolling(10).mean()
    df["volume_ma20"]  = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ma10"] + 1e-9)
    df["volume_lag_1"] = df["volume_ratio"].shift(1)
    df["volume_trend"] = df["volume_ma10"] / (df["volume_ma20"] + 1e-9)

    df["open_close_diff"] = df["Open"] - df["Close"]
    df["high_low_diff"]   = df["High"] - df["Low"]

    df["overnight_gap"] = (
        (df["Open"] - df["Close"].shift(1)) /
        (df["Close"].shift(1) + 1e-9)
    )
    df["intraday_reversal"] = (
        (df["Close"] - df["Open"]) /
        (df["High"] - df["Low"] + 1e-9)
    )
    df["upper_wick"] = (
        (df["High"] - df[["Open", "Close"]].max(axis=1)) /
        (df["High"] - df["Low"] + 1e-9)
    )
    df["lower_wick"] = (
        (df[["Open", "Close"]].min(axis=1) - df["Low"]) /
        (df["High"] - df["Low"] + 1e-9)
    )

    for window in [3, 5, 10]:
        df[f"up_days_{window}"] = sum(
            [(df["Close"].shift(i) > df["Close"].shift(i + 1)).astype(int)
             for i in range(window)]
        )

    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"]         = tr.rolling(14).mean()
    df["atr_normalized"] = df["atr_14"] / (df["Close"] + 1e-9)

    rolling_high_20 = df["High"].rolling(20).max()
    rolling_low_20  = df["Low"].rolling(20).min()
    df["price_position_20"] = (
        (df["Close"] - rolling_low_20) /
        (rolling_high_20 - rolling_low_20 + 1e-9)
    )

    if macro_df is not None:
        from macro_loader import get_macro_feature_columns
        macro_cols    = get_macro_feature_columns()
        macro_aligned = macro_df[macro_cols].copy()
        if macro_aligned.index.tzinfo is not None:
            macro_aligned.index = macro_aligned.index.tz_localize(None)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(macro_aligned, how="left")
        df[macro_cols] = df[macro_cols].ffill().bfill()

    if sentiment_series is not None:
        s = sentiment_series.copy()
        if s.index.tzinfo is not None:
            s.index = s.index.tz_localize(None)
        df = df.join(s.rename("sentiment"), how="left")
        df["sentiment"]          = df["sentiment"].ffill().fillna(0.0)
        df["sentiment_change"]   = df["sentiment"].diff(1)
        df["sentiment_ma5"]      = df["sentiment"].rolling(5).mean()
        df["sentiment_ma20"]     = df["sentiment"].rolling(20).mean()
        df["sentiment_momentum"] = df["sentiment_ma5"] - df["sentiment_ma20"]

    if earnings_df is not None:
        from earnings_loader import get_earnings_feature_columns
        earn_cols = get_earnings_feature_columns()
        for col in earn_cols:
            if col in earnings_df.columns:
                df[col] = earnings_df[col].values

    if not predict_mode:
        df["target"] = (df["Close"].shift(-fwd) > df["Close"]).astype(int)
        df.dropna(inplace=True)
    else:
        feature_cols = [c for c in df.columns if c != "target"]
        df = df.dropna(subset=feature_cols)

    return df

def get_feature_columns(include_macro=True,
                        include_sentiment=True,
                        include_earnings=True) -> list:
    base = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
        "rsi_lag_1", "rsi_lag_3",
        "price_to_ma10", "price_to_ma20", "price_to_ma50",
        "ma_cross_10_50", "ma_cross_10_20",
        "rsi_14", "rsi_7", "rsi_21", "rsi_momentum",
        "bb_position", "bb_width",
        "macd", "macd_signal", "macd_hist", "macd_momentum",
        "volume_ratio", "volume_trend", "volume_lag_1",
        "open_close_diff", "high_low_diff",
        "overnight_gap", "intraday_reversal",
        "upper_wick", "lower_wick",
        "up_days_3", "up_days_5", "up_days_10",
        "atr_normalized",
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
