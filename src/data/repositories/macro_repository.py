"""
MacroRepository — database operations for MACRO_INDICATOR and MACRO_RELEVANCE.

Covers two tables because they are tightly coupled — you never query
macro relevance without also needing the macro indicator metadata.

Depends on:
    BaseRepository          — session management and context manager
    MacroIndicatorModel     — the ORM class for macro_indicator
    MacroRelevanceModel     — the ORM class for macro_relevance

Exposes:
    add_indicator(indicator)            — insert a new macro indicator
    get_all_indicators()                — all candidate macro indicators
    get_unconditionals()                — VIX, DXY, GLD, SLV only
    add_relevance(relevance)            — insert a macro relevance row
    add_many_relevance(records)         — bulk insert relevance rows
    get_relevant_for_ticker(symbol)     — selected macros for one ticker
    get_all_relevance_for_ticker(symbol)— full correlation results
    delete_relevance_for_ticker(symbol) — clear before recomputing
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.data.database import MacroIndicatorModel, MacroRelevanceModel
from src.data.repositories.base import BaseRepository


class MacroRepository(BaseRepository):
    """
    Reads and writes the MACRO_INDICATOR and MACRO_RELEVANCE tables.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add_indicator(self, indicator: MacroIndicatorModel) -> None:
        """
        Insert a new macro indicator into the candidate universe.

        Args:
            indicator: A fully constructed MacroIndicatorModel instance.
        """
        self._session.add(indicator)

    def get_all_indicators(self) -> list[MacroIndicatorModel]:
        """
        Fetch all macro indicators in the candidate universe.

        Used by MacroCorrelationAnalyzer to know which symbols to fetch
        and correlate against a ticker's returns.

        Returns:
            All MacroIndicatorModel rows ordered by symbol ascending.
        """
        return (
            self._session.query(MacroIndicatorModel)
            .order_by(MacroIndicatorModel.symbol)
            .all()
        )

    def get_unconditionals(self) -> list[MacroIndicatorModel]:
        """
        Fetch only the unconditional macro indicators.

        Unconditionals (VIX, DXY, GLD, SLV) are always included as
        training features regardless of their correlation score with
        any particular ticker. They are fetched separately from
        conditional macros during feature engineering.

        Returns:
            MacroIndicatorModel rows where is_unconditional=True.
        """
        return (
            self._session.query(MacroIndicatorModel)
            .filter(MacroIndicatorModel.is_unconditional == True)
            .all()
        )

    def add_relevance(self, relevance: MacroRelevanceModel) -> None:
        """
        Insert a single macro relevance row.

        Args:
            relevance: A fully constructed MacroRelevanceModel instance.
        """
        self._session.add(relevance)

    def add_many_relevance(self, records: list[MacroRelevanceModel]) -> None:
        """
        Bulk insert multiple macro relevance rows.

        Used after MacroCorrelationAnalyzer computes the full correlation
        profile for a ticker — all rows are inserted in one operation.

        Args:
            records: List of MacroRelevanceModel instances to insert.
        """
        self._session.add_all(records)

    def get_relevant_for_ticker(self, symbol: str) -> list[MacroRelevanceModel]:
        """
        Fetch only the macros selected as relevant for one ticker.

        Returns rows where is_relevant=True — these are the macros that
        were included as features in the ticker's training run. Used by
        FeatureEngineer to know which macro data to fetch and include.

        Args:
            symbol: The ticker symbol.

        Returns:
            MacroRelevanceModel rows where is_relevant=True for this ticker.
        """
        return (
            self._session.query(MacroRelevanceModel)
            .filter(
                MacroRelevanceModel.ticker == symbol,
                MacroRelevanceModel.is_relevant == True,
            )
            .all()
        )

    def get_all_relevance_for_ticker(self, symbol: str) -> list[MacroRelevanceModel]:
        """
        Fetch the full correlation results for one ticker.

        Includes both relevant and non-relevant macros. Used for
        debugging and dashboard display — you can see exactly what
        correlation scores every macro got for a given ticker.

        Args:
            symbol: The ticker symbol.

        Returns:
            All MacroRelevanceModel rows for this ticker, ordered by
            correlation_score descending (most correlated first).
        """
        return (
            self._session.query(MacroRelevanceModel)
            .filter(MacroRelevanceModel.ticker == symbol)
            .order_by(MacroRelevanceModel.correlation_score.desc())
            .all()
        )

    def delete_relevance_for_ticker(self, symbol: str) -> None:
        """
        Delete all macro relevance rows for one ticker.

        Called before recomputing macro correlation on every retrain.
        MacroCorrelationAnalyzer always starts fresh — old relevance
        rows are deleted, new ones are inserted after correlation runs.

        This is the correct pattern rather than updating existing rows
        because the set of candidate macros may change between retrains
        (new indicators added to the universe, old ones removed).

        Args:
            symbol: The ticker symbol whose relevance rows to delete.
        """
        self._session.query(MacroRelevanceModel).filter(
            MacroRelevanceModel.ticker == symbol
        ).delete()
