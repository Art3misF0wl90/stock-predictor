import pandas as pd
import numpy as np
from config import FORWARD_DAYS

def add_features(df: pd.DataFrame,
                 macro_df: pd.DataFrame = None,
                 sentiment_series: pd.Series = None,
                 earnings_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Moving averages
    df["ma_10"]         = df["Close"].rolling(10).mean()
    df["ma_50"]         = df["Close"].rolling(50).mean()
    df["price_to_ma10"] = df["Close"] / df["ma_10"]
    df["price_to_ma50"] = df["Close"] / df["ma_50"]
    df["ma_cross"]      = df["ma_10"] / df["ma_50"]

    # RSI
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean      = df["Close"].rolling(20).mean()
    rolling_std       = df["Close"].rolling(20).std()
    bb_upper          = rolling_mean + 2 * rolling_std
    bb_lower          = rolling_mean - 2 * rolling_std
    df["bb_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # MACD
    ema_12            = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26            = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Volume
    df["volume_ma10"]  = df["Volume"].rolling(10).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ma10"] + 1e-9)

    # Intraday
    df["open_close_diff"] = df["Open"] - df["Close"]
    df["high_low_diff"]   = df["High"] - df["Low"]

    # Macro
    if macro_df is not None:
        from macro_loader import get_macro_feature_columns
        macro_cols = get_macro_feature_columns()
        # Ensure both indexes are timezone-naive before joining
        macro_aligned = macro_df[macro_cols].copy()
        if macro_aligned.index.tzinfo is not None:
            macro_aligned.index = macro_aligned.index.tz_localize(None)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(macro_aligned, how="left")
        df[macro_cols] = df[macro_cols].ffill()

    # Sentiment
    if sentiment_series is not None:
        s = sentiment_series.copy()
        # Strip timezone if present
        if s.index.tzinfo is not None:
            s.index = s.index.tz_localize(None)
        df = df.join(s.rename("sentiment"), how="left")
        df["sentiment"]        = df["sentiment"].ffill().fillna(0.0)
        df["sentiment_change"] = df["sentiment"].diff(1)
        df["sentiment_ma5"]    = df["sentiment"].rolling(5).mean()

    # Earnings
    if earnings_df is not None:
        from earnings_loader import get_earnings_feature_columns
        earn_cols = get_earnings_feature_columns()
        for col in earn_cols:
            if col in earnings_df.columns:
                df[col] = earnings_df[col].values

    # Target
    df["target"] = (df["Close"].shift(-FORWARD_DAYS) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df

def get_feature_columns(include_macro=True,
                        include_sentiment=True,
                        include_earnings=True) -> list:
    base = [
        "return_1d", "return_5d", "return_10d",
        "price_to_ma10", "price_to_ma50", "ma_cross",
        "rsi_14",
        "bb_position",
        "macd", "macd_signal", "macd_hist",
        "volume_ratio",
        "open_close_diff", "high_low_diff",
    ]
    if include_macro:
        base += [
            "vix", "vix_change", "vix_ma10", "vix_regime", "vix_zscore",
            "treasury", "treasury_change", "treasury_ma10",
            "dollar", "dollar_change",
        ]
    if include_sentiment:
        base += ["sentiment", "sentiment_change", "sentiment_ma5"]
    if include_earnings:
        base += [
            "eps_surprise", "rev_surprise",
            "days_to_earnings", "days_since_earnings",
            "pead_signal",
        ]
    return base