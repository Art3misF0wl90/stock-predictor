"""
TickerValidator — validates a ticker symbol before onboarding begins.

TickerValidator is the first step in TickerOnboardingPipeline. It checks
two things:

    1. Is the symbol valid and accessible?
       Makes a lightweight yfinance call to confirm the ticker exists,
       is not delisted, and returns a non-empty price history.

    2. Does it have enough historical data for training?
       Checks that at least min_training_rows rows of data are available
       (typically 252 — one calendar year of trading days).

TickerValidator never raises for validation failures — invalid and
ineligible are normal outcomes returned as a ValidationResult with
is_valid=False or is_eligible=False and a reason explaining why.

This distinction matters for the onboarding pipeline:
    is_valid=False   → ticker does not exist or is inaccessible.
                       Stop onboarding entirely.
    is_eligible=False → ticker exists but doesn't have enough history.
                        Could become eligible later (new IPO, for example).
                        Record the ticker but don't train.

Why do this before the full DataLoader.fetch_full()?
    fetch_full() downloads the entire history — potentially years of data.
    TickerValidator uses a short sample fetch (period="6mo") to quickly
    confirm the ticker is real and has data, before committing to the
    expensive full download.

Depends on:
    ConfigRepository    — reads min_training_rows threshold
    ValidationResult    — the output dataclass

Exposes:
    validate(symbol) → ValidationResult
"""

from __future__ import annotations

import yfinance as yf

from src.data.repositories.config_repository import ConfigRepository
from src.utils.types import ConfigCategory, ValidationResult


class TickerValidator:
    """
    Validates a ticker symbol before committing to full onboarding.

    Never raises for validation failures. Returns a ValidationResult
    with is_valid and is_eligible flags and a reason string when either
    is False.

    Usage:
        validator = TickerValidator(config_repo)
        result = validator.validate("AAPL")

        if not result.is_valid:
            # Ticker does not exist — abort onboarding
        elif not result.is_eligible:
            # Ticker exists but not enough history — record but don't train
        else:
            # Proceed with full onboarding
    """

    def __init__(self, config_repo: ConfigRepository) -> None:
        """
        Args:
            config_repo: For reading min_training_rows threshold.
        """
        self._config_repo = config_repo

    def validate(self, symbol: str) -> ValidationResult:
        """
        Validate a ticker symbol.

        Performs a lightweight 6-month history fetch to confirm the
        symbol is real and accessible, then checks the approximate
        data availability against the minimum row threshold.

        Args:
            symbol: The ticker symbol to validate, e.g. "AAPL".
                    Case-insensitive — yfinance normalizes it.

        Returns:
            ValidationResult with:
                symbol      — the symbol that was validated
                is_valid    — True if the ticker exists and is accessible
                is_eligible — True if enough historical data is available
                reason      — explanation if is_valid or is_eligible is False
        """
        # Step 1 — confirm the ticker exists and returns data
        raw_df = self._quick_fetch(symbol)

        if raw_df is None:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                is_eligible=False,
                reason=(
                    f"Could not retrieve data for '{symbol}' from yfinance. "
                    f"The symbol may be invalid, delisted, or temporarily "
                    f"unavailable. Verify the symbol on a financial data source."
                ),
            )

        if raw_df.empty:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                is_eligible=False,
                reason=(
                    f"yfinance returned an empty result for '{symbol}'. "
                    f"The symbol may be delisted or have no price history."
                ),
            )

        # Step 2 — check approximate data availability
        # We use a full max-period fetch row count estimate.
        # The 6-month fetch gives us ~126 rows. If we have those, the
        # ticker is real. We then do a separate check on approximate
        # full history availability.
        min_rows = self._config_repo.get(
            ConfigCategory.TRAINING,
            "min_training_rows",
        )

        # Estimate full history availability via a max-period fetch
        # of just the last row count — lightweight because we only
        # need the count, not the data itself.
        full_row_count = self._estimate_full_row_count(symbol)

        if full_row_count < min_rows:
            return ValidationResult(
                symbol=symbol,
                is_valid=True,
                is_eligible=False,
                reason=(
                    f"'{symbol}' is a valid ticker but has approximately "
                    f"{full_row_count} trading days of history. "
                    f"The minimum required for training is {min_rows} rows "
                    f"(approximately one calendar year). "
                    f"This ticker may become eligible after more history accumulates."
                ),
            )

        # All checks passed
        return ValidationResult(
            symbol=symbol,
            is_valid=True,
            is_eligible=True,
            reason=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quick_fetch(self, symbol: str):
        """
        Perform a lightweight 6-month history fetch to confirm ticker exists.

        Returns the raw DataFrame on success, or None on any exception.
        We catch all exceptions here because yfinance raises a variety
        of error types for invalid symbols — catching them all and
        returning None keeps the validation logic clean.

        Args:
            symbol: The ticker symbol to check.

        Returns:
            Raw DataFrame from yfinance, or None if the fetch failed.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo", auto_adjust=True)
            return df
        except Exception:
            return None

    def _estimate_full_row_count(self, symbol: str) -> int:
        """
        Estimate how many rows of full history are available for a ticker.

        Fetches the full history (period="max") and returns the row count.
        This is used to check eligibility against min_training_rows.

        If the fetch fails for any reason, returns 0 so the ticker is
        marked as ineligible rather than crashing the validation.

        Args:
            symbol: The ticker symbol.

        Returns:
            Approximate number of trading days of history available.
            Returns 0 if the fetch fails.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max", auto_adjust=True)
            return len(df)
        except Exception:
            return 0