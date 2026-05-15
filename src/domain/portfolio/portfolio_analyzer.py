"""
PortfolioAnalyzer — computes portfolio-level analytics from derived positions.

PortfolioAnalyzer takes a dict of current positions (from PositionDeriver)
and the current market prices (from MarketCache), then computes:

    1. Sector breakdown — what percentage of portfolio value is in each sector
    2. Concentration warnings — tickers or sectors exceeding the weight limit
    3. Correlation warnings — pairs of tickers with excessive return correlation

PortfolioAnalyzer is a pure computation component. It reads from MarketCache
and TickerRepository for metadata. It never writes to the database.
PortfolioSnapshotService (below) handles persistence.

Concentration analysis:
    For each open position, compute:
        position_value = shares * current_price
        weight = position_value / total_portfolio_value

    If any single ticker weight > max_ticker_weight (from SYSTEM_CONFIG),
    generate a ConcentrationWarning for that ticker.

    If any sector's aggregate weight > max_sector_weight (from SYSTEM_CONFIG),
    generate a ConcentrationWarning for that sector.

Correlation analysis:
    For each pair of open positions, compute the Pearson correlation of their
    daily returns over the last N days (from SYSTEM_CONFIG). If correlation
    > max_pair_correlation, generate a CorrelationWarning.

    High correlation between two positions means they are essentially the
    same bet — when one falls, the other falls too. This reduces the real
    diversification of the portfolio.

Depends on:
    PositionDeriver         — provides current positions
    MarketCache             — provides price data for valuation and correlation
    TickerRepository        — provides sector metadata per ticker
    ConfigRepository        — reads concentration/correlation thresholds
    PortfolioAnalysis       — the result dataclass
    ConcentrationWarning    — warning dataclass
    CorrelationWarning      — warning dataclass

Exposes:
    analyze(positions) → PortfolioAnalysis
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.data.market_cache import MarketCache
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.ticker_repository import TickerRepository
from src.utils.types import (
    ConfigCategory,
    ConcentrationWarning,
    CorrelationWarning,
    PortfolioAnalysis,
    Position,
    PositionDirection,
)


class PortfolioAnalyzer:
    """
    Computes concentration and correlation analytics for a portfolio.

    Usage:
        analyzer = PortfolioAnalyzer(cache, ticker_repo, config_repo)
        positions = position_deriver.derive_all()
        analysis = analyzer.analyze(positions)

        for warning in analysis.concentration_warnings:
            print(warning.message)
    """

    def __init__(
        self,
        cache: MarketCache,
        ticker_repo: TickerRepository,
        config_repo: ConfigRepository,
    ) -> None:
        """
        Args:
            cache: MarketCache for reading current prices and return history.
            ticker_repo: For reading sector metadata per ticker.
            config_repo: For reading concentration and correlation thresholds.
        """
        self._cache = cache
        self._ticker_repo = ticker_repo
        self._config_repo = config_repo

    def analyze(self, positions: dict[str, Position]) -> PortfolioAnalysis:
        """
        Compute portfolio analytics for the given positions.

        Args:
            positions: Dict mapping symbol → Position from PositionDeriver.
                       FLAT positions are included but contribute zero value.

        Returns:
            PortfolioAnalysis with sector_breakdown, total_invested,
            concentration_warnings, correlation_warnings, and analyzed_at.
        """
        # Read thresholds from config
        max_ticker_weight = self._config_repo.get(
            ConfigCategory.PORTFOLIO, "max_ticker_weight"
        )
        max_sector_weight = self._config_repo.get(
            ConfigCategory.PORTFOLIO, "max_sector_weight"
        )
        max_pair_correlation = self._config_repo.get(
            ConfigCategory.PORTFOLIO, "max_pair_correlation"
        )
        correlation_lookback_days = self._config_repo.get(
            ConfigCategory.PORTFOLIO, "correlation_lookback_days"
        )

        # Filter to open (non-FLAT) positions only
        open_positions = {
            sym: pos for sym, pos in positions.items()
            if pos.direction != PositionDirection.FLAT
        }

        # Get current prices and compute position values
        position_values = self._compute_position_values(open_positions)
        total_value = sum(position_values.values())
        total_invested = sum(
            pos.shares * pos.average_cost
            for pos in open_positions.values()
        )

        # Compute sector breakdown
        sector_breakdown = self._compute_sector_breakdown(
            open_positions, position_values, total_value
        )

        # Compute concentration warnings
        concentration_warnings = self._check_concentration(
            open_positions, position_values, total_value,
            sector_breakdown,
            max_ticker_weight, max_sector_weight,
        )

        # Compute correlation warnings
        correlation_warnings = self._check_correlations(
            open_positions,
            max_pair_correlation,
            correlation_lookback_days,
        )

        return PortfolioAnalysis(
            positions=positions,
            sector_breakdown=sector_breakdown,
            total_invested=total_invested,
            concentration_warnings=concentration_warnings,
            correlation_warnings=correlation_warnings,
            analyzed_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_position_values(
        self,
        open_positions: dict[str, Position],
    ) -> dict[str, float]:
        """
        Compute current market value for each open position.

        Reads the most recent close price from MarketCache for each ticker.
        If no cache data is available for a ticker, uses average_cost as
        a fallback (cost basis = market value, no gain/loss assumed).

        Args:
            open_positions: Dict of non-FLAT positions.

        Returns:
            Dict mapping symbol → current market value.
        """
        values = {}
        for symbol, position in open_positions.items():
            try:
                df = self._cache.read(symbol)
                current_price = float(df["close"].iloc[-1])
            except Exception:
                # Fallback to cost basis if price unavailable
                current_price = position.average_cost

            values[symbol] = position.shares * current_price

        return values

    def _compute_sector_breakdown(
        self,
        open_positions: dict[str, Position],
        position_values: dict[str, float],
        total_value: float,
    ) -> dict[str, float]:
        """
        Compute the fraction of portfolio value in each sector.

        Args:
            open_positions: Non-FLAT positions.
            position_values: Current market value per ticker.
            total_value: Total portfolio market value.

        Returns:
            Dict mapping sector name → weight (0.0 to 1.0).
            "Unknown" sector is used for tickers with no sector metadata.
        """
        if total_value == 0:
            return {}

        sector_values: dict[str, float] = {}

        for symbol, position in open_positions.items():
            ticker_row = self._ticker_repo.get(symbol)
            sector = (
                ticker_row.sector
                if ticker_row and ticker_row.sector
                else "Unknown"
            )
            value = position_values.get(symbol, 0.0)
            sector_values[sector] = sector_values.get(sector, 0.0) + value

        return {
            sector: round(value / total_value, 4)
            for sector, value in sector_values.items()
        }

    def _check_concentration(
        self,
        open_positions: dict[str, Position],
        position_values: dict[str, float],
        total_value: float,
        sector_breakdown: dict[str, float],
        max_ticker_weight: float,
        max_sector_weight: float,
    ) -> list[ConcentrationWarning]:
        """
        Generate ConcentrationWarnings for over-weighted tickers and sectors.

        Args:
            open_positions: Non-FLAT positions.
            position_values: Current market value per ticker.
            total_value: Total portfolio market value.
            sector_breakdown: Sector weights already computed.
            max_ticker_weight: Maximum allowed weight for any single ticker.
            max_sector_weight: Maximum allowed weight for any single sector.

        Returns:
            List of ConcentrationWarning objects. Empty if no violations.
        """
        warnings = []

        if total_value == 0:
            return warnings

        # Ticker-level concentration
        for symbol in open_positions:
            value = position_values.get(symbol, 0.0)
            weight = value / total_value

            if weight > max_ticker_weight:
                warnings.append(ConcentrationWarning(
                    symbol=symbol,
                    weight=round(weight, 4),
                    threshold=max_ticker_weight,
                    message=(
                        f"{symbol} is {weight:.1%} of portfolio value, "
                        f"exceeding the {max_ticker_weight:.1%} single-ticker limit. "
                        f"Consider trimming to reduce concentration risk."
                    ),
                ))

        # Sector-level concentration
        for sector, weight in sector_breakdown.items():
            if weight > max_sector_weight:
                warnings.append(ConcentrationWarning(
                    symbol=sector,
                    weight=round(weight, 4),
                    threshold=max_sector_weight,
                    message=(
                        f"{sector} sector is {weight:.1%} of portfolio value, "
                        f"exceeding the {max_sector_weight:.1%} sector limit. "
                        f"Consider diversifying across sectors."
                    ),
                ))

        return warnings

    def _check_correlations(
        self,
        open_positions: dict[str, Position],
        max_pair_correlation: float,
        lookback_days: int,
    ) -> list[CorrelationWarning]:
        """
        Generate CorrelationWarnings for pairs of highly correlated positions.

        Computes Pearson correlation of daily returns for every pair of
        open positions over the last lookback_days trading days.

        Args:
            open_positions: Non-FLAT positions.
            max_pair_correlation: Threshold above which a pair is warned.
            lookback_days: How many trading days of history to use.

        Returns:
            List of CorrelationWarning objects. Empty if no violations.
        """
        warnings = []
        symbols = list(open_positions.keys())

        if len(symbols) < 2:
            return warnings

        # Load return series for each ticker
        returns: dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                df = self._cache.read(symbol)
                ret = df["close"].pct_change(1).dropna().tail(lookback_days)
                returns[symbol] = ret
            except Exception:
                continue

        # Check every pair
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]

                if sym_a not in returns or sym_b not in returns:
                    continue

                # Align on overlapping dates
                aligned = pd.concat(
                    [returns[sym_a], returns[sym_b]],
                    axis=1,
                    join="inner",
                ).dropna()

                if len(aligned) < 20:
                    continue

                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

                if np.isnan(corr):
                    continue

                if corr > max_pair_correlation:
                    warnings.append(CorrelationWarning(
                        symbol_a=sym_a,
                        symbol_b=sym_b,
                        correlation=round(float(corr), 4),
                        threshold=max_pair_correlation,
                        message=(
                            f"{sym_a} and {sym_b} have a {corr:.2f} return "
                            f"correlation over the last {lookback_days} trading days, "
                            f"exceeding the {max_pair_correlation:.2f} threshold. "
                            f"These positions may not provide meaningful diversification."
                        ),
                    ))

        return warnings