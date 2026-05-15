"""
MacroCorrelationAnalyzer — selects relevant macro indicators for a ticker.

Not all macro indicators are useful for every ticker. A gold mining stock
correlates strongly with GLD and commodity prices. A tech stock may correlate
more with DXY and sector ETFs. Adding all macro indicators as features for
every ticker adds noise that hurts model performance.

This component solves that by computing the Pearson correlation between
each candidate macro indicator's returns and the ticker's returns, then
applying a tiered selection rule to pick the most relevant ones.

Tiered selection algorithm:
    Step 1 — Always include the 4 unconditional macros (VIX, DXY, GLD, SLV).
              These are included regardless of correlation score because they
              represent market-wide risk factors that affect every equity.

    Step 2 — For each remaining candidate macro, compute the absolute Pearson
              correlation with the ticker's 1-day returns over the overlapping
              date range.

    Step 3 — Find the maximum correlation score across all candidates.

    Step 4 — Apply tier rule:
              If max_correlation > CORRELATION_THRESHOLD (from SYSTEM_CONFIG):
                  Select the top 5 correlated macro candidates.
                  This ticker has strong macro relationships worth capturing.
              Else:
                  Select the top 3 correlated macro candidates.
                  Weaker macro relationships — fewer features is better.

    Step 5 — Final set = unconditionals + selected conditionals (deduplicated).

    Step 6 — Persist the full results to MACRO_RELEVANCE table via
              MacroRepository so every correlation score is auditable.

Depends on:
    MarketCache         — source of macro OHLCV data
    MacroRepository     — reads candidate universe, writes relevance results
    ConfigRepository    — reads correlation threshold from SYSTEM_CONFIG
    MacroRelevanceProfile — the result dataclass

Exposes:
    analyze(symbol) → MacroRelevanceProfile
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.data.database import MacroRelevanceModel
from src.data.market_cache import MarketCache
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.macro_repository import MacroRepository
from src.utils.exceptions import FeatureEngineeringError
from src.utils.types import ConfigCategory, MacroRelevanceProfile


class MacroCorrelationAnalyzer:
    """
    Computes per-ticker macro relevance and persists the results.

    Called during ticker onboarding and on every retrain. The results
    are stored in the MACRO_RELEVANCE table — the feature engineer reads
    from there to know which macro data to include.

    Usage:
        analyzer = MacroCorrelationAnalyzer(cache, macro_repo, config_repo)
        profile = analyzer.analyze("AAPL")
        # profile.selected_macros is the list to pass to FeatureEngineer
    """

    def __init__(
        self,
        cache: MarketCache,
        macro_repo: MacroRepository,
        config_repo: ConfigRepository,
    ) -> None:
        """
        Args:
            cache: MarketCache for reading ticker and macro OHLCV data.
            macro_repo: MacroRepository for reading candidates and writing results.
            config_repo: ConfigRepository for reading correlation threshold.
        """
        self._cache = cache
        self._macro_repo = macro_repo
        self._config_repo = config_repo

    def analyze(self, symbol: str) -> MacroRelevanceProfile:
        """
        Run the full macro correlation analysis for one ticker.

        Reads the candidate macro universe from the database, computes
        correlations, applies the tier rule, persists results, and returns
        a MacroRelevanceProfile.

        Called by TickerOnboardingPipeline during onboarding and by
        RetrainOrchestrator before every retrain. Old relevance records
        for this ticker are deleted and replaced on every call.

        Args:
            symbol: The ticker to analyze.

        Returns:
            MacroRelevanceProfile with selected_macros, correlation_scores,
            tier, and computed_at.

        Raises:
            FeatureEngineeringError: If the ticker's OHLCV data cannot be
                read or if correlation computation fails.
        """
        # Load the ticker's returns — this is what we correlate against
        ticker_returns = self._load_ticker_returns(symbol)

        # Read configuration
        threshold = self._config_repo.get(
            ConfigCategory.MACRO,
            "correlation_threshold",
        )

        # Get the full candidate universe from the database
        all_indicators = self._macro_repo.get_all_indicators()
        unconditionals = {
            ind.symbol for ind in all_indicators if ind.is_unconditional
        }
        conditionals = [
            ind for ind in all_indicators if not ind.is_unconditional
        ]

        # Compute correlation for every conditional macro
        correlation_scores: dict[str, float] = {}
        for indicator in conditionals:
            score = self._compute_correlation(ticker_returns, indicator.symbol)
            correlation_scores[indicator.symbol] = score

        # Apply the tier rule
        max_corr = max(correlation_scores.values()) if correlation_scores else 0.0
        tier = 5 if max_corr > threshold else 3

        # Select top N conditional macros by absolute correlation score
        sorted_conditionals = sorted(
            correlation_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        selected_conditionals = [sym for sym, _ in sorted_conditionals[:tier]]

        # Final selected set: unconditionals + selected conditionals
        selected_macros = sorted(unconditionals | set(selected_conditionals))

        # Persist results — delete old rows first, then insert new ones
        self._macro_repo.delete_relevance_for_ticker(symbol)

        relevance_records = []
        now = datetime.utcnow()

        for indicator in all_indicators:
            is_relevant = (
                indicator.symbol in unconditionals
                or indicator.symbol in selected_conditionals
            )
            score = correlation_scores.get(indicator.symbol, 1.0)
            # Unconditionals get score 1.0 — they are always included

            relevance_records.append(
                MacroRelevanceModel(
                    ticker=symbol,
                    macro_symbol=indicator.symbol,
                    correlation_score=score,
                    is_relevant=is_relevant,
                    tier=tier,
                    max_correlation_seen=max_corr,
                    computed_at=now,
                )
            )

        self._macro_repo.add_many_relevance(relevance_records)

        return MacroRelevanceProfile(
            symbol=symbol,
            selected_macros=selected_macros,
            correlation_scores=correlation_scores,
            tier=tier,
            computed_at=now,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_ticker_returns(self, symbol: str) -> pd.Series:
        """
        Load 1-day returns for the ticker from MarketCache.

        Args:
            symbol: The ticker symbol.

        Returns:
            Series of 1-day percent returns with DatetimeIndex.

        Raises:
            FeatureEngineeringError: If the cache read fails.
        """
        try:
            df = self._cache.read(symbol)
            returns = df["close"].pct_change(1).dropna()
            return returns
        except FileNotFoundError as exc:
            raise FeatureEngineeringError(
                f"Cannot run macro correlation for {symbol}: "
                f"no cached OHLCV data found. "
                f"DataLoader must run before MacroCorrelationAnalyzer.",
                cause=exc,
            )

    def _compute_correlation(
        self,
        ticker_returns: pd.Series,
        macro_symbol: str,
    ) -> float:
        """
        Compute the absolute Pearson correlation between ticker and macro returns.

        Aligns the two series on their overlapping date range before
        computing correlation. If the macro data is unavailable or the
        overlap is too short to be meaningful, returns 0.0 so the macro
        is ranked at the bottom of the selection list.

        Args:
            ticker_returns: 1-day returns for the ticker.
            macro_symbol: The macro indicator symbol to compare against.

        Returns:
            Absolute Pearson correlation coefficient in [0, 1].
            Returns 0.0 if correlation cannot be computed.
        """
        if not self._cache.exists(macro_symbol):
            return 0.0

        try:
            macro_df = self._cache.read(macro_symbol)
            macro_returns = macro_df["close"].pct_change(1).dropna()

            # Align on overlapping dates only
            aligned = pd.concat(
                [ticker_returns, macro_returns],
                axis=1,
                join="inner",
            ).dropna()

            # Need at least 60 overlapping data points for a
            # meaningful correlation estimate
            if len(aligned) < 60:
                return 0.0

            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

            # corr() can return NaN if one series has zero variance
            if np.isnan(corr):
                return 0.0

            return abs(float(corr))

        except Exception:
            return 0.0