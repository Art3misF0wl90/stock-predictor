"""
SuggestionRepository — all database operations for the SUGGESTION_LOG table.

Every suggestion the engine generates is logged here with full reasoning
inputs. The acted_on field is set by the user manually via the web interface.

Depends on:
    BaseRepository          — session management and context manager
    SuggestionLogModel      — the ORM class mapping to suggestion_log

Exposes:
    add(suggestion)                     — insert a new suggestion row
    get_latest_for_ticker(symbol)       — most recent suggestion for one ticker
    get_latest_timestamp(symbol)        — timestamp of most recent suggestion
    get_history(symbol, limit)          — recent suggestion history
    mark_acted_on(suggestion_id)        — set acted_on=True
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.data.database import SuggestionLogModel
from src.data.repositories.base import BaseRepository


class SuggestionRepository(BaseRepository):
    """
    Reads and writes the SUGGESTION_LOG table.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, suggestion: SuggestionLogModel) -> None:
        """
        Insert a new suggestion log row.

        Args:
            suggestion: A fully constructed SuggestionLogModel instance.
        """
        self._session.add(suggestion)

    def get_latest_for_ticker(self, symbol: str) -> SuggestionLogModel | None:
        """
        Fetch the most recent suggestion for one ticker.

        Used by SuggestionEngine's staleness check to determine whether
        any inputs have changed since the last suggestion was generated.

        Args:
            symbol: The ticker symbol.

        Returns:
            The most recent SuggestionLogModel, or None if no suggestions
            exist yet for this ticker.
        """
        return (
            self._session.query(SuggestionLogModel)
            .filter(SuggestionLogModel.ticker == symbol)
            .order_by(SuggestionLogModel.created_at.desc())
            .first()
        )

    def get_latest_timestamp(self, symbol: str) -> datetime | None:
        """
        Return the created_at timestamp of the most recent suggestion.

        Used by SuggestionEngine's staleness check alongside
        SignalRepository.get_latest_timestamp() and
        SentimentRepository.get_latest_timestamp(). If any input postdates
        this timestamp, a new suggestion is generated.

        Args:
            symbol: The ticker symbol.

        Returns:
            The datetime of the most recent suggestion, or None.
        """
        from sqlalchemy import func

        return (
            self._session.query(func.max(SuggestionLogModel.created_at))
            .filter(SuggestionLogModel.ticker == symbol)
            .scalar()
        )

    def get_history(self, symbol: str, limit: int = 10) -> list[SuggestionLogModel]:
        """
        Fetch recent suggestion history for one ticker, newest first.

        Used by ChatbotService to give the LLM context about what
        recommendations have been made and whether the user acted on them.

        Args:
            symbol: The ticker symbol.
            limit: Maximum rows to return. Defaults to 10.

        Returns:
            List of SuggestionLogModel rows, newest first.
        """
        return (
            self._session.query(SuggestionLogModel)
            .filter(SuggestionLogModel.ticker == symbol)
            .order_by(SuggestionLogModel.created_at.desc())
            .limit(limit)
            .all()
        )

    def mark_acted_on(self, suggestion_id: str) -> None:
        """
        Set acted_on=True for a suggestion the user has acted on.

        Called from the web interface when the user marks a suggestion
        as acted on. Used by ChatbotService and SuggestionEngine to
        understand whether past suggestions were followed.

        Args:
            suggestion_id: UUID of the suggestion to update.
        """
        suggestion = self._session.get(SuggestionLogModel, suggestion_id)
        if suggestion is None:
            return

        suggestion.acted_on = True
