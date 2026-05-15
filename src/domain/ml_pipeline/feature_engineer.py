"""
FeatureEngineer — computes the feature matrix from OHLCV and macro data.

FeatureEngineer is a pure computation component. It reads from MarketCache,
computes features, and returns a DataFrame. It never writes to the database
or to disk — that is the caller's responsibility.

Two modes:

    Train mode (feat_cols=None):
        Computes all available features for the ticker and its selected
        macro indicators. Returns the feature matrix AND the ordered list
        of column names (feat_cols). The Trainer writes feat_cols into
        TickerConfig and saves it to config.pkl.

    Inference mode (feat_cols=list):
        Receives feat_cols from config.pkl. Computes the same features,
        then selects and reorders columns to match feat_cols exactly.
        If the result does not match, FeatureCountMismatchError is raised.
        This enforces the architectural invariant: inference always uses
        the exact same feature set the model was trained on.

Feature groups:
    1. Price features   — returns, log returns, price ratios
    2. Volume features  — volume change, volume z-score
    3. Momentum         — RSI, MACD, MACD signal, MACD histogram
    4. Volatility       — rolling std, ATR, Bollinger Band width
    5. Trend            — SMA crossovers, EMA crossovers, ADX
    6. Macro features   — lagged returns for each selected macro indicator
    7. Sentiment        — aggregated weighted sentiment score (if available)

Target variable:
    The target column ("target") is always computed in train mode and
    never returned in inference mode. It represents the forward return
    direction — 1 (positive) or 0 (negative) — over fwd_days trading days.

Depends on:
    MarketCache         — source of OHLCV and macro data
    SentimentRepository — source of sentiment scores
    FeatureEngineeringError     — raised on computation failure
    FeatureCountMismatchError   — raised on inference column mismatch

Exposes:
    build_train_matrix(symbol, macro_symbols, fwd_days)
        → (feature_df, feat_cols, target_series)

    build_inference_matrix(symbol, macro_symbols, feat_cols)
        → feature_df (one row, columns matching feat_cols exactly)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.market_cache import MarketCache
from src.utils.exceptions import (
    FeatureCountMismatchError,
    FeatureEngineeringError,
)

if TYPE_CHECKING:
    from src.data.repositories.sentiment_repository import SentimentRepository


class FeatureEngineer:
    """
    Computes the feature matrix from OHLCV and macro data.

    One instance is created at startup and shared across all components
    that need feature computation.

    Usage — train mode:
        engineer = FeatureEngineer(cache, sentiment_repo)
        feature_df, feat_cols, target = engineer.build_train_matrix(
            symbol="AAPL",
            macro_symbols=["^VIX", "DX-Y.NYB", "GLD"],
            fwd_days=5,
        )

    Usage — inference mode:
        feature_df = engineer.build_inference_matrix(
            symbol="AAPL",
            macro_symbols=["^VIX", "DX-Y.NYB", "GLD"],
            feat_cols=config.feat_cols,   # from config.pkl
        )
    """

    def __init__(
        self,
        cache: MarketCache,
        sentiment_repo: "SentimentRepository",
    ) -> None:
        """
        Args:
            cache: MarketCache instance for reading OHLCV and macro data.
            sentiment_repo: SentimentRepository for reading sentiment scores.
                            Sentiment is optional — if no records exist for
                            a ticker, the sentiment feature is set to 0.0.
        """
        self._cache = cache
        self._sentiment_repo = sentiment_repo

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_train_matrix(
        self,
        symbol: str,
        macro_symbols: list[str],
        fwd_days: int,
    ) -> tuple[pd.DataFrame, list[str], pd.Series]:
        """
        Compute the full feature matrix for training.

        Builds all feature groups, attaches macro features, attaches
        sentiment, drops NaN rows, computes the target variable, then
        splits into (features, feat_cols, target).

        feat_cols is the ordered list of feature column names. The Trainer
        must write this into TickerConfig — it is the anchor that prevents
        feature count mismatches at inference time.

        Args:
            symbol: The ticker to build features for.
            macro_symbols: List of macro indicator symbols selected for
                           this ticker by MacroCorrelationAnalyzer.
            fwd_days: Forward horizon in trading days. The target variable
                      is 1 if the close price is higher in fwd_days days,
                      0 otherwise.

        Returns:
            Tuple of:
                feature_df  — DataFrame of features, NaN rows dropped.
                              Index is DatetimeIndex (trading dates).
                feat_cols   — Ordered list of column names in feature_df.
                              Must be written to TickerConfig by Trainer.
                target      — Series of 0/1 labels aligned with feature_df.

        Raises:
            FeatureEngineeringError: If feature computation fails for any
                reason, or if the result is empty after dropping NaN rows.
        """
        try:
            df = self._load_ohlcv(symbol)
            df = self._add_price_features(df)
            df = self._add_volume_features(df)
            df = self._add_momentum_features(df)
            df = self._add_volatility_features(df)
            df = self._add_trend_features(df)
            df = self._add_macro_features(df, macro_symbols)
            df = self._add_sentiment_feature(df, symbol)
            df = self._add_target(df, fwd_days)

            # Drop rows with any NaN. Rolling window features produce NaN
            # for the first N rows — these cannot be used for training.
            df = df.dropna()

            if df.empty:
                raise FeatureEngineeringError(
                    f"Feature matrix for {symbol} is empty after dropping NaN "
                    f"rows. The ticker may not have enough history for the "
                    f"configured window sizes."
                )

            # Separate features from target
            target = df["target"].astype(int)
            feature_df = df.drop(columns=["target"])
            feat_cols = list(feature_df.columns)

            return feature_df, feat_cols, target

        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(
                f"Feature engineering failed for {symbol}: {exc}",
                cause=exc,
            )

    def build_inference_matrix(
        self,
        symbol: str,
        macro_symbols: list[str],
        feat_cols: list[str],
    ) -> pd.DataFrame:
        """
        Compute the feature matrix for inference — one row, exact columns.

        Builds all feature groups, then selects and reorders columns to
        match feat_cols from config.pkl exactly. The last row of the
        resulting DataFrame is returned — that is the "current" feature
        vector representing today's market state.

        Args:
            symbol: The ticker to build features for.
            macro_symbols: Macro symbols from config.pkl (config.macro_symbols).
                           Must match what was used at train time.
            feat_cols: The ordered column list from config.pkl.
                       This is the ground truth — inference output must
                       match it exactly.

        Returns:
            A single-row DataFrame with columns matching feat_cols in
            exactly the same order. Index is the most recent trading date.

        Raises:
            FeatureEngineeringError: If feature computation fails.
            FeatureCountMismatchError: If the computed feature matrix does
                not contain all columns listed in feat_cols, or if it
                produces extra columns not in feat_cols.
        """
        try:
            df = self._load_ohlcv(symbol)
            df = self._add_price_features(df)
            df = self._add_volume_features(df)
            df = self._add_momentum_features(df)
            df = self._add_volatility_features(df)
            df = self._add_trend_features(df)
            df = self._add_macro_features(df, macro_symbols)
            df = self._add_sentiment_feature(df, symbol)

            # Drop rows with NaN — keep only complete rows
            df = df.dropna()

            if df.empty:
                raise FeatureEngineeringError(
                    f"Feature matrix for {symbol} is empty after dropping NaN "
                    f"rows during inference. The cache may need updating."
                )

            # Verify all expected columns are present
            computed_cols = set(df.columns)
            expected_cols = set(feat_cols)

            missing_cols = expected_cols - computed_cols
            extra_cols = computed_cols - expected_cols

            if missing_cols or extra_cols:
                raise FeatureCountMismatchError(
                    f"Feature column mismatch for {symbol}. "
                    f"Missing from computed: {sorted(missing_cols)}. "
                    f"Extra in computed (not in config): {sorted(extra_cols)}. "
                    f"This usually means a macro indicator was added or removed "
                    f"from the database after this ticker was last trained. "
                    f"Retrain {symbol} to fix this.",
                    expected=len(feat_cols),
                    actual=len(computed_cols),
                )

            # Select and reorder to exactly match feat_cols from config
            result = df[feat_cols]

            # Return only the last row — current market state
            return result.iloc[[-1]]

        except (FeatureEngineeringError, FeatureCountMismatchError):
            raise
        except Exception as exc:
            raise FeatureEngineeringError(
                f"Inference feature engineering failed for {symbol}: {exc}",
                cause=exc,
            )

    # ------------------------------------------------------------------
    # Feature group builders — each adds columns to df and returns it
    # ------------------------------------------------------------------

    def _load_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Load the OHLCV DataFrame from cache.

        Raises:
            FeatureEngineeringError: If the cache read fails.
        """
        try:
            return self._cache.read(symbol)
        except FileNotFoundError as exc:
            raise FeatureEngineeringError(
                f"No cached OHLCV data for {symbol}. "
                f"DataLoader must run before FeatureEngineer.",
                cause=exc,
            )

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-derived features.

        Features added:
            return_1d   — 1-day return: (close - prev_close) / prev_close
            return_5d   — 5-day return
            return_20d  — 20-day return
            log_ret_1d  — natural log of (close / prev_close)
            hl_ratio    — (high - low) / close, daily range as fraction of close
            co_ratio    — (close - open) / open, intraday direction strength
        """
        df = df.copy()
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
        df["co_ratio"] = (df["close"] - df["open"]) / df["open"]
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-derived features.

        Features added:
            vol_change  — 1-day volume change ratio
            vol_zscore  — z-score of volume over 20-day rolling window.
                          Values > 2 indicate unusually high volume (breakout
                          signal). Values < -2 indicate unusually low volume.
        """
        df = df.copy()
        df["vol_change"] = df["volume"].pct_change(1)
        rolling_mean = df["volume"].rolling(20).mean()
        rolling_std = df["volume"].rolling(20).std()
        df["vol_zscore"] = (df["volume"] - rolling_mean) / rolling_std
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicator features.

        Features added:
            rsi_14      — Relative Strength Index, 14-period.
                          Values > 70 suggest overbought.
                          Values < 30 suggest oversold.
            macd        — MACD line: EMA(12) - EMA(26)
            macd_signal — Signal line: EMA(9) of MACD
            macd_hist   — Histogram: macd - macd_signal

        RSI formula:
            RS = avg_gain / avg_loss over the window
            RSI = 100 - (100 / (1 + RS))
        """
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicator features.

        Features added:
            vol_20d     — 20-day rolling standard deviation of daily returns.
                          Annualized (multiplied by sqrt(252)).
            atr_14      — Average True Range, 14-period.
                          Measures daily price movement magnitude.
            bb_width    — Bollinger Band width: (upper - lower) / middle.
                          Higher values indicate more volatility.

        ATR formula:
            True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            ATR = EMA(14) of True Range
        """
        df = df.copy()

        # 20-day annualized volatility
        daily_ret = df["close"].pct_change()
        df["vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)

        # ATR
        prev_close = df["close"].shift(1)
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hpc": (df["high"] - prev_close).abs(),
            "lpc": (df["low"] - prev_close).abs(),
        }).max(axis=1)
        df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        df["bb_width"] = (upper - lower) / sma_20

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicator features.

        Features added:
            sma_cross   — SMA(20) / SMA(50) ratio.
                          > 1.0 means short-term average above long-term (uptrend).
                          < 1.0 means downtrend.
            ema_cross   — EMA(12) / EMA(26) ratio. Same interpretation.
            price_vs_sma50 — close / SMA(50). How far price is from its 50-day avg.
            adx_14      — Average Directional Index, 14-period.
                          Measures trend strength regardless of direction.
                          > 25 indicates a strong trend.
                          < 20 indicates a weak or ranging market.
        """
        df = df.copy()

        # SMA and EMA crossovers
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["sma_cross"] = sma_20 / sma_50.replace(0, np.nan)
        df["ema_cross"] = ema_12 / ema_26.replace(0, np.nan)
        df["price_vs_sma50"] = df["close"] / sma_50.replace(0, np.nan)

        # ADX — measures trend strength
        # +DM: today's high - yesterday's high (if positive, else 0)
        # -DM: yesterday's low - today's low (if positive, else 0)
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()

        plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0.0)
        minus_dm = (-low_diff).where((-low_diff > 0) & (-low_diff > high_diff), 0.0)

        prev_close = df["close"].shift(1)
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hpc": (df["high"] - prev_close).abs(),
            "lpc": (df["low"] - prev_close).abs(),
        }).max(axis=1)

        atr14 = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)

        dx_denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / dx_denom
        df["adx_14"] = dx.ewm(span=14, adjust=False).mean()

        return df

    def _add_macro_features(
        self,
        df: pd.DataFrame,
        macro_symbols: list[str],
    ) -> pd.DataFrame:
        """
        Add lagged return features for each selected macro indicator.

        For each macro symbol, computes the 1-day return of that indicator
        and aligns it to the ticker's trading dates via a left join on the
        date index. Missing macro values (market holidays, data gaps) are
        forward-filled then backfilled.

        Feature name pattern: macro_{symbol}_ret1
            e.g. "macro_^VIX_ret1", "macro_GLD_ret1"

        The "macro_" prefix and "_ret1" suffix make macro features easy
        to identify in the feat_cols list.

        Args:
            df: The ticker's feature DataFrame so far.
            macro_symbols: List of macro indicator symbols to attach.

        Returns:
            df with one new column per macro symbol.
        """
        df = df.copy()

        for macro_sym in macro_symbols:
            col_name = f"macro_{macro_sym}_ret1"

            if not self._cache.exists(macro_sym):
                # Macro data not cached — fill with zeros rather than
                # crashing. MacroCorrelationAnalyzer should have fetched
                # this, but if it's missing we degrade gracefully.
                df[col_name] = 0.0
                continue

            try:
                macro_df = self._cache.read(macro_sym)
                macro_ret = macro_df["close"].pct_change(1).rename(col_name)

                # Left join: keep all rows from ticker df, attach macro returns
                # where dates match. Missing dates get NaN, then filled.
                df = df.join(macro_ret, how="left")
                df[col_name] = df[col_name].ffill().bfill()

            except Exception:
                # If anything goes wrong reading this macro, fill with zeros
                # rather than failing the entire feature computation.
                df[col_name] = 0.0

        return df

    def _add_sentiment_feature(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Add a weighted aggregate sentiment score as a feature.

        Queries the last 30 days of sentiment records for this ticker
        and computes: sum(raw_score * source_weight) / sum(source_weight)

        If no sentiment records exist, the feature is set to 0.0 for all
        rows. Sentiment is a weak signal for many tickers — its absence
        does not make the feature matrix invalid.

        Feature added:
            sentiment_score — weighted aggregate sentiment, range [-1, 1].
                              Positive values indicate bullish sentiment.
                              Negative values indicate bearish sentiment.

        Args:
            df: The ticker's feature DataFrame so far.
            symbol: The ticker symbol to fetch sentiment for.

        Returns:
            df with one new column: "sentiment_score".
        """
        df = df.copy()

        try:
            records = self._sentiment_repo.get_by_ticker(symbol, days=30)

            if not records:
                df["sentiment_score"] = 0.0
                return df

            total_weight = sum(r.source_weight for r in records)
            if total_weight == 0:
                df["sentiment_score"] = 0.0
                return df

            weighted_score = sum(
                r.raw_score * r.source_weight for r in records
            ) / total_weight

            df["sentiment_score"] = weighted_score

        except Exception:
            # Sentiment read failure — degrade gracefully to 0.0
            df["sentiment_score"] = 0.0

        return df

    def _add_target(self, df: pd.DataFrame, fwd_days: int) -> pd.DataFrame:
        """
        Compute the binary classification target variable.

        Target definition:
            1 — close price in fwd_days trading days is higher than today
            0 — close price in fwd_days trading days is equal or lower

        Uses shift(-fwd_days) to look forward. This produces NaN for the
        last fwd_days rows — those rows are dropped by the caller's
        dropna() call.

        Args:
            df: The feature DataFrame.
            fwd_days: How many trading days forward to look.

        Returns:
            df with a new "target" column (float — dropna handles it better
            than int when NaN rows exist).
        """
        df = df.copy()
        future_close = df["close"].shift(-fwd_days)
        df["target"] = (future_close > df["close"]).astype(float)
        # Set last fwd_days rows to NaN so dropna removes them
        df.loc[df.index[-fwd_days:], "target"] = np.nan
        return df