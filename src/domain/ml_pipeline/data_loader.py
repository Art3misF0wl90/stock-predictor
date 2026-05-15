"""
DataLoader — fetches OHLCV market data from yfinance and writes to MarketCache.

DataLoader is the only component in the system that calls yfinance directly.
Everything else that needs market data reads from MarketCache. This isolates
the external API dependency to one place — if yfinance changes its API or
you swap it for a different data source, only this file changes.

Two fetch modes:
    Full fetch:
        Downloads the complete available history for a ticker.
        Used during ticker onboarding (first time a ticker is added).
        Overwrites any existing cache entry.

    Incremental fetch:
        Downloads only rows newer than the last cached date.
        Used by the daily scheduled job to keep the cache current.
        Appends new rows to the existing cache rather than re-fetching
        everything. Much faster for the daily update cycle.

Data contract:
    DataLoader produces a DataFrame with these columns exactly:
        open, high, low, close, volume
    With a DatetimeIndex named "date".
    This contract is enforced by _normalize() before writing to cache.

Validation:
    DataLoader validates that the fetched DataFrame meets the minimum
    row threshold from SYSTEM_CONFIG before writing. If not enough data
    is available, InsufficientDataError is raised. If the fetch itself
    fails (network error, invalid symbol), DataFetchError is raised.

Depends on:
    MarketCache         — writes validated data here
    ConfigRepository    — reads min_rows threshold from SYSTEM_CONFIG
    DataFetchError      — raised on yfinance failure
    InsufficientDataError — raised when data exists but is too sparse

Exposes:
    fetch_full(symbol)          — full history fetch + cache write
    fetch_incremental(symbol)   — append-only update fetch
    fetch_macro(macro_symbol)   — fetch a macro indicator (same logic)
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from src.data.market_cache import MarketCache
from src.data.repositories.config_repository import ConfigRepository
from src.utils.exceptions import DataFetchError, InsufficientDataError
from src.utils.types import ConfigCategory


# Columns yfinance returns that we keep.
# yfinance also returns "Dividends" and "Stock Splits" — we drop those.
_KEEP_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Renamed to match our internal schema (lowercase).
_RENAME_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

# How long to wait between yfinance calls when fetching multiple tickers.
# yfinance will rate-limit aggressive callers — a small sleep avoids this.
_INTER_FETCH_SLEEP_SECONDS = 0.5


class DataLoader:
    """
    Fetches OHLCV data from yfinance and writes it to MarketCache.

    One DataLoader instance is created at startup and shared across all
    components that need to trigger a data fetch.

    Usage:
        loader = DataLoader(cache=market_cache, config_repo=config_repo)

        # During onboarding:
        loader.fetch_full("AAPL")

        # During daily update:
        loader.fetch_incremental("AAPL")
    """

    def __init__(
        self,
        cache: MarketCache,
        config_repo: ConfigRepository,
    ) -> None:
        """
        Args:
            cache: The MarketCache instance to write fetched data to.
            config_repo: ConfigRepository for reading SYSTEM_CONFIG values.
                         Used to get the min_rows threshold and fetch window.
        """
        self._cache = cache
        self._config_repo = config_repo

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_full(self, symbol: str) -> pd.DataFrame:
        """
        Download the complete available history for a ticker.

        Fetches from yfinance using period="max" to get all available
        data going back as far as the exchange has records. Validates
        that the result meets the minimum row threshold, normalizes
        the column schema, and writes to MarketCache.

        Used during ticker onboarding. Overwrites any existing cache entry.

        Args:
            symbol: The ticker symbol to fetch, e.g. "AAPL".

        Returns:
            The normalized DataFrame that was written to cache.
            Has DatetimeIndex named "date" and columns: open, high,
            low, close, volume. Sorted ascending by date.

        Raises:
            DataFetchError: If yfinance returns an empty result or raises
                an exception. Empty results happen when the symbol is
                invalid or delisted.
            InsufficientDataError: If the fetch succeeds but the result
                has fewer rows than the configured minimum. The ticker
                exists but doesn't have enough history for training.
        """
        raw = self._fetch_from_yfinance(symbol, period="max")
        df = self._normalize(symbol, raw)
        self._validate_min_rows(symbol, df)
        self._cache.write(symbol, df)
        return df

    def fetch_incremental(self, symbol: str) -> pd.DataFrame:
        """
        Download only rows newer than the last cached date.

        Checks the cache for the most recent date, then fetches only
        the data after that date from yfinance. Merges the new rows
        with the existing cache and writes the combined result back.

        If no cache exists, falls back to fetch_full().

        Used by the daily scheduled job to keep cached data current
        without re-fetching years of historical data every day.

        Args:
            symbol: The ticker symbol to update.

        Returns:
            The full updated DataFrame (existing + new rows) written
            to cache.

        Raises:
            DataFetchError: If the incremental fetch fails.
            InsufficientDataError: If the updated cache still has too
                few rows (shouldn't happen after a successful full fetch,
                but checked as a safety net).
        """
        last_date = self._cache.get_last_date(symbol)

        if last_date is None:
            # No cache exists yet — do a full fetch instead
            return self.fetch_full(symbol)

        # Fetch data starting the day after the last cached date.
        # yfinance start= is inclusive, so we add one day to avoid
        # re-fetching the row we already have.
        start_date = last_date + timedelta(days=1)
        today = date.today()

        if start_date > today:
            # Cache is already up to date — no fetch needed.
            # Return the existing cached data.
            return self._cache.read(symbol)

        # Fetch the new rows
        raw = self._fetch_from_yfinance(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
        )

        if raw.empty:
            # No new rows available — market may have been closed since
            # last fetch. Return the existing cache unchanged.
            return self._cache.read(symbol)

        new_rows = self._normalize(symbol, raw)

        # Merge: load existing cache, append new rows, deduplicate,
        # re-sort chronologically, write back.
        existing = self._cache.read(symbol)
        combined = pd.concat([existing, new_rows])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index(ascending=True)

        self._validate_min_rows(symbol, combined)
        self._cache.write(symbol, combined)
        return combined

    def fetch_macro(self, macro_symbol: str) -> pd.DataFrame:
        """
        Fetch data for a macro indicator using the same logic as fetch_full.

        Macro indicators (VIX, DXY, GLD, SLV, sector ETFs, etc.) are
        fetched the same way as equity tickers — the only difference is
        that the minimum row validation is skipped, because macro data
        availability varies by source and the correlation analyzer handles
        sparse data gracefully.

        The result is written to MarketCache under the macro_symbol key.

        Args:
            macro_symbol: The yfinance symbol for the macro indicator,
                          e.g. "^VIX", "DX-Y.NYB", "GLD", "XLK".

        Returns:
            Normalized DataFrame written to cache. May be empty if the
            macro source has no data — callers handle that case.

        Raises:
            DataFetchError: If yfinance raises an exception fetching this
                            macro symbol.
        """
        raw = self._fetch_from_yfinance(macro_symbol, period="max")

        if raw.empty:
            # Macro data unavailable — return empty DataFrame.
            # MacroCorrelationAnalyzer handles missing macro data by
            # excluding that indicator from correlation scoring.
            return pd.DataFrame()

        df = self._normalize(macro_symbol, raw)
        self._cache.write(macro_symbol, df)

        # Small delay between macro fetches to avoid rate limiting
        time.sleep(_INTER_FETCH_SLEEP_SECONDS)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_from_yfinance(
        self,
        symbol: str,
        period: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Call yfinance and return the raw DataFrame.

        Wraps the yfinance call in a try/except so all network errors,
        invalid symbol errors, and timeout errors surface as DataFetchError
        rather than raw yfinance or requests exceptions.

        Args:
            symbol: The yfinance ticker symbol.
            period: yfinance period string, e.g. "max", "2y".
                    Used for full fetches. Mutually exclusive with start/end.
            start: Start date string "YYYY-MM-DD". Used for incremental fetches.
            end: End date string "YYYY-MM-DD". Used for incremental fetches.

        Returns:
            Raw DataFrame from yfinance. May be empty — callers check.

        Raises:
            DataFetchError: If yfinance raises any exception.
        """
        try:
            ticker = yf.Ticker(symbol)

            if period is not None:
                df = ticker.history(period=period, auto_adjust=True)
            else:
                df = ticker.history(
                    start=start,
                    end=end,
                    auto_adjust=True,
                )

            return df

        except Exception as exc:
            raise DataFetchError(
                f"yfinance fetch failed for {symbol}: {exc}",
                cause=exc,
            )

    def _normalize(self, symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the raw yfinance DataFrame into our internal schema.

        Steps:
            1. Verify the expected columns are present.
            2. Keep only the columns we need (drop Dividends, Stock Splits).
            3. Rename to lowercase column names.
            4. Ensure the index is a DatetimeIndex named "date".
            5. Remove timezone info from the index (yfinance returns
               timezone-aware timestamps, we store timezone-naive).
            6. Sort ascending by date.
            7. Drop any rows with NaN in critical columns.

        Args:
            symbol: Ticker symbol (for error messages).
            raw: Raw DataFrame from yfinance.

        Returns:
            Normalized DataFrame matching our internal schema.

        Raises:
            DataFetchError: If the raw DataFrame is empty after normalization.
        """
        if raw.empty:
            raise DataFetchError(
                f"yfinance returned empty DataFrame for {symbol}. "
                f"The symbol may be invalid or delisted."
            )

        # Step 1 — check expected columns exist in the raw result
        missing = [col for col in _KEEP_COLUMNS if col not in raw.columns]
        if missing:
            raise DataFetchError(
                f"yfinance response for {symbol} is missing expected columns: "
                f"{missing}. Got: {list(raw.columns)}."
            )

        # Step 2 — keep only our columns
        df = raw[_KEEP_COLUMNS].copy()

        # Step 3 — rename to lowercase
        df = df.rename(columns=_RENAME_MAP)

        # Step 4 — ensure the index is named "date"
        df.index.name = "date"

        # Step 5 — remove timezone info
        # yfinance returns timestamps like "2024-01-02 00:00:00-05:00"
        # We store them as naive datetimes: "2024-01-02 00:00:00"
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Step 6 — sort ascending
        df = df.sort_index(ascending=True)

        # Step 7 — drop rows with NaN in any required column
        # This handles split-adjustment artifacts and market holiday edge cases
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        if df.empty:
            raise DataFetchError(
                f"DataFrame for {symbol} was empty after dropping NaN rows. "
                f"All rows may have had missing price data."
            )

        return df

    def _validate_min_rows(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Check that the DataFrame has enough rows for training.

        Reads the minimum row threshold from SYSTEM_CONFIG. If the
        DataFrame has fewer rows than the threshold, raises
        InsufficientDataError.

        One trading year is approximately 252 rows. The default minimum
        is 252 rows — roughly one calendar year of daily data — which
        ensures the model has enough history for meaningful patterns.

        Args:
            symbol: Ticker symbol (for error messages).
            df: The normalized DataFrame to validate.

        Raises:
            InsufficientDataError: If len(df) < min_rows threshold.
        """
        min_rows = self._config_repo.get(
            ConfigCategory.TRAINING,
            "min_training_rows",
        )

        if len(df) < min_rows:
            raise InsufficientDataError(
                f"{symbol} has {len(df)} rows of historical data, "
                f"but the minimum required for training is {min_rows} rows "
                f"(approximately one calendar year of trading days). "
                f"This ticker does not have enough history to train a model."
            )