"""
MarketCache — Parquet-based on-disk cache for OHLCV market data.

Every ticker's historical price data is stored as a Parquet file on disk.
DataLoader writes to this cache after every successful yfinance fetch.
FeatureEngineer reads from it rather than hitting the API on every
training or inference run.

Why Parquet?
    Parquet is a columnar binary file format designed for analytical
    workloads. Compared to CSV:
        - Much faster to read (columnar layout, compressed)
        - Preserves data types — dates stay dates, floats stay floats
        - Handles large datasets (years of OHLCV data) efficiently
        - Supported natively by pandas via pyarrow

File layout on disk:
    {MARKET_CACHE_DIR}/
        AAPL_ohlcv.parquet
        MSFT_ohlcv.parquet
        TSLA_ohlcv.parquet
        ...

DataFrame schema:
    Every cached DataFrame must have these columns exactly:
        date        — datetime64[ns], the trading day (index after read)
        open        — float64, opening price
        high        — float64, daily high
        low         — float64, daily low
        close       — float64, closing price (adjusted)
        volume      — float64, trading volume

    The date column is stored as a regular column in Parquet (Parquet
    does not preserve the index). read() restores it as the DataFrame
    index after loading.

Depends on:
    pandas      — DataFrame read/write
    pyarrow     — Parquet engine used by pandas

Exposes:
    write(symbol, df)       — serialize and cache a DataFrame
    read(symbol)            — load and return a cached DataFrame
    exists(symbol)          — check if a cache file exists
    delete(symbol)          — remove a cache file
    get_last_date(symbol)   — most recent trading date in the cache
    list_cached()           — all ticker symbols with a cache file
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


# Expected columns in every cached DataFrame.
# FeatureEngineer relies on these names exactly — do not rename them.
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class MarketCache:
    """
    Reads and writes per-ticker OHLCV DataFrames as Parquet files.

    Only DataLoader writes to this cache. Only FeatureEngineer reads
    from it. No other component interacts with MarketCache directly.

    One instance is created at startup and injected wherever needed.

    Usage:
        cache = MarketCache(cache_dir="data/market_cache")

        # After a successful yfinance fetch:
        cache.write("AAPL", df)

        # Before feature engineering:
        df = cache.read("AAPL")
    """

    def __init__(self, cache_dir: str) -> None:
        """
        Set up the MarketCache pointed at a directory on disk.

        Creates the directory if it does not already exist.

        Args:
            cache_dir: Path to the directory where .parquet files are stored.
                       Typically "data/market_cache" from the .env file.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Serialize a DataFrame to disk as {symbol}_ohlcv.parquet.

        Validates that the DataFrame has all required columns before
        writing. The date index is reset to a column before serialization
        because Parquet does not preserve the DataFrame index.

        Args:
            df: A DataFrame with columns: open, high, low, close, volume,
                and either a DatetimeIndex named "date" or a "date" column.
                Must not be empty.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        self._validate_dataframe(symbol, df)

        df_to_write = df.copy()

        # If date is the index, reset it to a regular column for Parquet.
        # Parquet does not preserve the DataFrame index, so we always
        # store date as a column and restore it on read.
        if df_to_write.index.name == "date":
            df_to_write = df_to_write.reset_index()
        elif "date" not in df_to_write.columns:
            raise ValueError(
                f"DataFrame for {symbol} has no 'date' column or DatetimeIndex "
                f"named 'date'. Cannot write to cache."
            )

        path = self._path_for(symbol)
        df_to_write.to_parquet(path, engine="pyarrow", index=False)

    def read(self, symbol: str) -> pd.DataFrame:
        """
        Load a cached DataFrame from disk and restore the date index.

        Returns the DataFrame with date set as the index, sorted
        chronologically ascending (oldest row first). FeatureEngineer
        and DataLoader both expect this ordering.

        Args:
            symbol: The ticker symbol to read, e.g. "AAPL".

        Returns:
            A DataFrame with DatetimeIndex named "date" and columns:
            open, high, low, close, volume. Sorted ascending by date.

        Raises:
            FileNotFoundError: If no cache file exists for this ticker.
                Callers should check exists() before calling read() if
                they need to handle the missing-file case gracefully.
                DataLoader always writes before FeatureEngineer reads,
                so in normal operation this should never be raised.
        """
        path = self._path_for(symbol)

        if not path.exists():
            raise FileNotFoundError(
                f"No market cache file found for {symbol} at {path}. "
                f"Has DataLoader run a successful fetch for this ticker?"
            )

        df = pd.read_parquet(path, engine="pyarrow")

        # Restore date as the DataFrame index.
        # parse_dates ensures the column is read as datetime64, not string.
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # Ensure chronological order — Parquet preserves write order,
        # but we sort explicitly to guarantee it regardless of how the
        # file was written.
        df = df.sort_index(ascending=True)

        return df

    def exists(self, symbol: str) -> bool:
        """
        Check whether a cache file exists for a ticker.

        Does not validate the file contents — only checks presence.

        Args:
            symbol: The ticker symbol to check.

        Returns:
            True if {symbol}_ohlcv.parquet exists in the cache directory.
        """
        return self._path_for(symbol).exists()

    def delete(self, symbol: str) -> None:
        """
        Delete the cache file for a ticker.

        Used when removing a ticker or forcing a full re-fetch.
        Does nothing if the file does not exist.

        Args:
            symbol: The ticker symbol whose cache to delete.
        """
        path = self._path_for(symbol)
        if path.exists():
            path.unlink()

    def get_last_date(self, symbol: str) -> date | None:
        """
        Return the most recent trading date in the cache for a ticker.

        Used by DataLoader to implement incremental fetching — instead of
        re-fetching the full history on every run, it fetches only the
        rows newer than the last cached date and appends them.

        This reads only the index metadata from the Parquet file rather
        than loading the entire DataFrame, which is much faster for large
        cache files.

        Args:
            symbol: The ticker symbol to check.

        Returns:
            The most recent date as a Python date object, or None if
            no cache file exists for this ticker.
        """
        if not self.exists(symbol):
            return None

        # Read only the date column — much faster than loading all columns
        # when the file has years of daily data.
        df = pd.read_parquet(
            self._path_for(symbol),
            engine="pyarrow",
            columns=["date"],
        )

        if df.empty:
            return None

        last_timestamp = pd.to_datetime(df["date"]).max()
        return last_timestamp.date()

    def list_cached(self) -> list[str]:
        """
        Return all ticker symbols that have a cache file.

        Scans the cache directory for files matching *_ohlcv.parquet.

        Returns:
            Sorted list of ticker symbols with cached data.
            Empty list if no cache files exist yet.
        """
        symbols = []
        for path in self._cache_dir.glob("*_ohlcv.parquet"):
            # Filename pattern: {SYMBOL}_ohlcv.parquet
            stem = path.stem  # e.g. "AAPL_ohlcv"
            if stem.endswith("_ohlcv"):
                symbol = stem[: -len("_ohlcv")]
                symbols.append(symbol)

        return sorted(symbols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path_for(self, symbol: str) -> Path:
        """
        Return the Path for the cache file for a ticker.

        Args:
            symbol: The ticker symbol.

        Returns:
            Path object for {symbol}_ohlcv.parquet.
        """
        return self._cache_dir / f"{symbol}_ohlcv.parquet"

    def _validate_dataframe(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Validate a DataFrame before writing it to the cache.

        Checks:
            1. DataFrame is not empty.
            2. All required columns are present.

        Args:
            symbol: Ticker symbol (for error messages).
            df: The DataFrame to validate.

        Raises:
            ValueError: If the DataFrame is empty or missing columns.
        """
        if df.empty:
            raise ValueError(
                f"Cannot cache empty DataFrame for {symbol}. "
                f"DataLoader should not be writing empty data."
            )

        # Check using the column names, accounting for the case where
        # date is the index rather than a column.
        available = set(df.columns)
        if df.index.name == "date":
            available.add("date")

        missing = REQUIRED_COLUMNS - available
        if missing:
            raise ValueError(
                f"DataFrame for {symbol} is missing required columns: "
                f"{sorted(missing)}. "
                f"Expected columns: {sorted(REQUIRED_COLUMNS)}. "
                f"Got: {sorted(available)}."
            )