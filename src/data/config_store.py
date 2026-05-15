"""
ConfigStore — single source of truth for per-ticker model configuration.

Every trained model produces a TickerConfig object that is serialized to
a .pkl file on disk. ConfigStore is the only component that reads or writes
these files. Everything else that needs model configuration goes through
ConfigStore.

Why this exists:
    The v1 system had a critical bug where inference scripts called
    get_feature_columns() to rebuild feat_cols at runtime. This caused
    silent feature count mismatches — TSLA had 57 features, other tickers
    had 50 — producing incorrect predictions with no error or warning.

    ConfigStore fixes this architecturally. feat_cols is written to
    config.pkl at train time and NEVER recomputed. Inference loads it
    from the file. If the file is missing or invalid, a clear exception
    is raised. There is no fallback, no guess, no silent failure.

File layout on disk:
    {MODEL_ARTIFACT_DIR}/
        AAPL_config.pkl
        AAPL_config_rejected_20260514_090000.pkl   ← rejected retrains
        MSFT_config.pkl
        TSLA_config.pkl
        ...

Schema versioning:
    TickerConfig has a schema_version field. CURRENT_SCHEMA_VERSION is
    defined here. On every load, ConfigStore checks that the file's
    schema_version matches CURRENT_SCHEMA_VERSION. If it doesn't, a
    ConfigSchemaError is raised — the config is too old to be trusted.

    Current schema version: 2

Depends on:
    TickerConfig        — the dataclass being serialized (from utils/types.py)
    ConfigNotFoundError — raised when the .pkl file does not exist
    ConfigSchemaError   — raised when the file fails validation

Exposes:
    save(config)            — serialize and write a TickerConfig to disk
    save_rejected(config)   — write a rejected config with timestamp suffix
    load(symbol)            — deserialize, validate, and return a TickerConfig
    exists(symbol)          — check if a config file exists for a ticker
    delete(symbol)          — remove a config file from disk
    list_trained()          — return all ticker symbols that have a config file
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path

from src.utils.exceptions import ConfigNotFoundError, ConfigSchemaError
from src.utils.types import TickerConfig


class ConfigStore:
    """
    Reads and writes per-ticker TickerConfig objects to disk as .pkl files.

    Only one instance of ConfigStore should exist at runtime. It is
    constructed once in run.py and injected into components that need it.

    Attributes:
        CURRENT_SCHEMA_VERSION: The schema version this code understands.
            Increment this when TickerConfig fields change in a breaking way.
            All existing config files with an older version become invalid
            and must be retrained.

    Usage:
        store = ConfigStore(artifact_dir="data/model_artifacts")

        # After training:
        store.save(ticker_config)

        # Before inference:
        config = store.load("AAPL")
        # config.feat_cols is now available — pass it to FeatureEngineer
    """

    CURRENT_SCHEMA_VERSION: int = 2

    def __init__(self, artifact_dir: str) -> None:
        """
        Set up the ConfigStore pointed at a directory on disk.

        Creates the directory if it does not already exist. Safe to call
        on every startup.

        Args:
            artifact_dir: Path to the directory where .pkl files are stored.
                          Typically "data/model_artifacts" from the .env file.
        """
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self, config: TickerConfig) -> str:
        """
        Serialize a TickerConfig to disk as {symbol}_config.pkl.

        Overwrites any existing config for this ticker. This is the
        accepted config path — the one inference will read.

        Args:
            config: The TickerConfig to persist. Must have a valid symbol
                    and schema_version == CURRENT_SCHEMA_VERSION.

        Returns:
            The full path to the file that was written, as a string.
            Trainer stores this path in ModelAudit.config_path.

        Raises:
            ConfigSchemaError: If config.schema_version does not match
                CURRENT_SCHEMA_VERSION. This prevents writing a config
                that load() would immediately reject.
        """
        self._validate(config)

        path = self._active_path(config.symbol)
        self._write(config, path)
        return str(path)

    def save_rejected(self, config: TickerConfig) -> str:
        """
        Write a rejected config with a timestamp suffix for audit/rollback.

        Rejected configs are written to:
            {symbol}_config_rejected_{YYYYMMDD_HHMMSS}.pkl

        They are never read by inference — they exist purely as an audit
        trail and rollback option. The active config (without timestamp)
        is left untouched when a retrain is rejected.

        Args:
            config: The rejected TickerConfig to persist.

        Returns:
            The full path to the rejected config file that was written.
        """
        self._validate(config)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.symbol}_config_rejected_{timestamp}.pkl"
        path = self._artifact_dir / filename
        self._write(config, path)
        return str(path)

    def load(self, symbol: str) -> TickerConfig:
        """
        Deserialize and validate the TickerConfig for a ticker.

        This method is called before EVERY inference run. Validation
        is never skipped. If the file is missing or invalid, a clear
        exception is raised immediately — there is no fallback.

        Validation checks:
            1. The file exists on disk.
            2. The file can be deserialized without errors.
            3. The deserialized object is a TickerConfig instance.
            4. schema_version matches CURRENT_SCHEMA_VERSION.
            5. feat_cols is a non-empty list of strings.
            6. fwd_days is a positive integer.
            7. symbol in the file matches the symbol requested.

        Args:
            symbol: The ticker symbol to load config for, e.g. "AAPL".

        Returns:
            A validated TickerConfig object ready to use for inference.

        Raises:
            ConfigNotFoundError: If no config file exists for this ticker.
                Inference cannot proceed — the ticker needs to be onboarded
                or retrained.
            ConfigSchemaError: If the file exists but fails any validation
                check. This indicates a corrupted file or schema version
                mismatch from an old training run.
        """
        path = self._active_path(symbol)

        if not path.exists():
            raise ConfigNotFoundError(
                f"No config file found for ticker: {symbol}. "
                f"Expected path: {path}. "
                f"Has this ticker been successfully onboarded?"
            )

        config = self._read(path, symbol)
        self._validate_loaded(config, symbol)
        return config

    def exists(self, symbol: str) -> bool:
        """
        Check whether an active config file exists for a ticker.

        Does not validate the file — only checks presence on disk.
        Use load() if you need a validated config.

        Args:
            symbol: The ticker symbol to check.

        Returns:
            True if {symbol}_config.pkl exists in the artifact directory.
        """
        return self._active_path(symbol).exists()

    def delete(self, symbol: str) -> None:
        """
        Delete the active config file for a ticker.

        Used when removing a ticker from the system entirely. Does nothing
        if the file does not exist — safe to call unconditionally.

        Note: This only deletes the active config. Rejected config files
        with timestamp suffixes are not affected.

        Args:
            symbol: The ticker symbol whose config to delete.
        """
        path = self._active_path(symbol)
        if path.exists():
            path.unlink()

    def list_trained(self) -> list[str]:
        """
        Return the ticker symbols that have an active config file.

        Scans the artifact directory for files matching the pattern
        {symbol}_config.pkl (without any timestamp suffix, so rejected
        configs are excluded).

        Used by the scheduler and dashboard to know which tickers have
        trained models ready for inference.

        Returns:
            Sorted list of ticker symbols with active config files.
            Empty list if no config files exist yet.

        Example:
            ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
        """
        symbols = []
        for path in self._artifact_dir.glob("*_config.pkl"):
            # Filename pattern: {SYMBOL}_config.pkl
            # We want to exclude: {SYMBOL}_config_rejected_{timestamp}.pkl
            # Check that the stem ends with exactly "_config"
            stem = path.stem  # e.g. "AAPL_config" or "AAPL_config_rejected_20260514"
            if stem.endswith("_config"):
                symbol = stem[: -len("_config")]
                symbols.append(symbol)

        return sorted(symbols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _active_path(self, symbol: str) -> Path:
        """
        Return the Path for the active config file for a ticker.

        Active config files use the pattern: {symbol}_config.pkl
        No timestamp suffix — this is the file inference reads.

        Args:
            symbol: The ticker symbol.

        Returns:
            Path object for the config file.
        """
        return self._artifact_dir / f"{symbol}_config.pkl"

    def _write(self, config: TickerConfig, path: Path) -> None:
        """
        Serialize a TickerConfig to a .pkl file using pickle.

        Uses protocol=4 which is supported by Python 3.8+ and produces
        smaller files than the default protocol for dataclass objects.

        Args:
            config: The TickerConfig to serialize.
            path: The full path to write to.
        """
        with open(path, "wb") as f:
            pickle.dump(config, f, protocol=4)

    def _read(self, path: Path, symbol: str) -> TickerConfig:
        """
        Deserialize a .pkl file and return its contents.

        Wraps pickle.load() in error handling so deserialization failures
        produce a clean ConfigSchemaError rather than a raw pickle exception.

        Args:
            path: The full path to the .pkl file.
            symbol: The ticker symbol (for error messages only).

        Returns:
            The deserialized object — validated by the caller.

        Raises:
            ConfigSchemaError: If pickle cannot deserialize the file.
                This means the file is corrupted or was written by an
                incompatible version of Python or pickle.
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            raise ConfigSchemaError(
                f"Failed to deserialize config file for {symbol} at {path}. "
                f"The file may be corrupted. Delete it and retrain.",
                cause=exc,
            )

    def _validate(self, config: TickerConfig) -> None:
        """
        Validate a TickerConfig before writing it to disk.

        Called by save() and save_rejected() to prevent writing a config
        that load() would immediately reject.

        Args:
            config: The TickerConfig to validate.

        Raises:
            ConfigSchemaError: If the config has an incorrect schema version.
        """
        if config.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ConfigSchemaError(
                f"Cannot save config for {config.symbol}: schema_version "
                f"{config.schema_version} does not match current version "
                f"{self.CURRENT_SCHEMA_VERSION}. "
                f"Rebuild the TickerConfig with the current schema."
            )

    def _validate_loaded(self, config: object, symbol: str) -> None:
        """
        Validate a deserialized object before returning it from load().

        Runs every check needed to ensure the config is safe to use
        for inference. Any failure raises ConfigSchemaError with a
        specific message explaining exactly what is wrong.

        Checks performed:
            1. Object is a TickerConfig instance.
            2. schema_version matches CURRENT_SCHEMA_VERSION.
            3. symbol in config matches the requested symbol.
            4. feat_cols is a non-empty list of strings.
            5. fwd_days is a positive integer.
            6. model_name is a non-empty string.
            7. auc is a float between 0 and 1.

        Args:
            config: The deserialized object (typed as object because we
                    haven't confirmed it's a TickerConfig yet).
            symbol: The ticker symbol that was requested.

        Raises:
            ConfigSchemaError: If any check fails.
        """
        # Check 1 — correct type
        if not isinstance(config, TickerConfig):
            raise ConfigSchemaError(
                f"Config file for {symbol} deserialized to "
                f"{type(config).__name__}, expected TickerConfig. "
                f"The file may have been written by an incompatible version."
            )

        # Check 2 — schema version
        if config.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ConfigSchemaError(
                f"Config for {symbol} has schema_version "
                f"{config.schema_version}, but current version is "
                f"{self.CURRENT_SCHEMA_VERSION}. "
                f"This ticker needs to be retrained to update its config."
            )

        # Check 3 — symbol matches
        if config.symbol != symbol:
            raise ConfigSchemaError(
                f"Config file for {symbol} contains symbol={config.symbol}. "
                f"Symbol mismatch — the wrong config file may have been loaded."
            )

        # Check 4 — feat_cols is a non-empty list of strings
        if not isinstance(config.feat_cols, list) or len(config.feat_cols) == 0:
            raise ConfigSchemaError(
                f"Config for {symbol} has invalid feat_cols: "
                f"{config.feat_cols!r}. "
                f"feat_cols must be a non-empty list of strings."
            )
        if not all(isinstance(col, str) for col in config.feat_cols):
            raise ConfigSchemaError(
                f"Config for {symbol} has feat_cols containing non-string "
                f"values. feat_cols must be a list of strings."
            )

        # Check 5 — fwd_days is a positive integer
        if not isinstance(config.fwd_days, int) or config.fwd_days <= 0:
            raise ConfigSchemaError(
                f"Config for {symbol} has invalid fwd_days: {config.fwd_days}. "
                f"fwd_days must be a positive integer."
            )

        # Check 6 — model_name is a non-empty string
        if not isinstance(config.model_name, str) or len(config.model_name) == 0:
            raise ConfigSchemaError(
                f"Config for {symbol} has invalid model_name: "
                f"{config.model_name!r}. model_name must be a non-empty string."
            )

        # Check 7 — auc is a float between 0 and 1
        if not isinstance(config.auc, float) or not (0.0 <= config.auc <= 1.0):
            raise ConfigSchemaError(
                f"Config for {symbol} has invalid auc: {config.auc}. "
                f"auc must be a float between 0.0 and 1.0."
            )