"""
EarningsRepository — all database operations for the EARNINGS_EVENT table.

Tracks both upcoming and historical earnings reports. The is_upcoming flag
is flipped to False by a daily cron job after the report date passes.

Depends on:
    BaseRepository      — session management and context manager
    EarningsEventModel  — the ORM class mapping to the earnings_event table

Exposes:
    add(event)                  — insert a new earnings event
    get_upcoming()              — all events with is_upcoming=True
    get_upcoming_for_ticker(symbol) — upcoming events for one ticker
    get_by_ticker(symbol)       — all events for one ticker
    flip_past_events()          — mark past events as not upcoming
    get_by_id(event_id)         — fetch one event by UUID
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.data.database import EarningsEventModel
from src.data.repositories.base import BaseRepository


class EarningsRepository(BaseRepository):
    """
    Reads and writes the EARNINGS_EVENT table.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, event: EarningsEventModel) -> None:
        """
        Insert a new earnings event row.

        Args:
            event: A fully constructed EarningsEventModel instance.
        """
        self._session.add(event)

    def get_upcoming(self) -> list[EarningsEventModel]:
        """
        Fetch all earnings events marked as upcoming.

        Used by InferenceOrchestrator when assembling inputs for
        AlertEvaluator — any held ticker with an upcoming earnings event
        within the configured blackout window triggers an
        EARNINGS_APPROACHING alert.

        Returns:
            All EarningsEventModel rows where is_upcoming=True,
            ordered by report_date ascending (soonest first).
        """
        return (
            self._session.query(EarningsEventModel)
            .filter(EarningsEventModel.is_upcoming == True)
            .order_by(EarningsEventModel.report_date.asc())
            .all()
        )

    def get_upcoming_for_ticker(self, symbol: str) -> EarningsEventModel | None:
        """
        Fetch the next upcoming earnings event for one ticker.

        Returns the soonest upcoming event, or None if no upcoming
        events are scheduled for this ticker.

        Args:
            symbol: The ticker symbol.

        Returns:
            The next EarningsEventModel for this ticker, or None.
        """
        return (
            self._session.query(EarningsEventModel)
            .filter(
                EarningsEventModel.ticker == symbol,
                EarningsEventModel.is_upcoming == True,
            )
            .order_by(EarningsEventModel.report_date.asc())
            .first()
        )

    def get_by_ticker(self, symbol: str) -> list[EarningsEventModel]:
        """
        Fetch all earnings events for one ticker, newest first.

        Includes both upcoming and historical events. Used by the chatbot
        to give the LLM full earnings history context.

        Args:
            symbol: The ticker symbol.

        Returns:
            All EarningsEventModel rows for that ticker, newest first.
        """
        return (
            self._session.query(EarningsEventModel)
            .filter(EarningsEventModel.ticker == symbol)
            .order_by(EarningsEventModel.report_date.desc())
            .all()
        )

    def flip_past_events(self, before: datetime | None = None) -> int:
        """
        Mark all past earnings events as not upcoming.

        Called by the daily scheduler cron job. Sets is_upcoming=False
        for every event whose report_date is in the past.

        Args:
            before: Flip events with report_date before this datetime.
                    Defaults to datetime.utcnow() if not provided.

        Returns:
            Number of rows updated.
        """
        cutoff = before or datetime.utcnow()
        past_events = (
            self._session.query(EarningsEventModel)
            .filter(
                EarningsEventModel.is_upcoming == True,
                EarningsEventModel.report_date < cutoff,
            )
            .all()
        )

        for event in past_events:
            event.is_upcoming = False

        return len(past_events)

    def get_by_id(self, event_id: str) -> EarningsEventModel | None:
        """
        Fetch one earnings event by its UUID primary key.

        Args:
            event_id: The UUID string.

        Returns:
            The EarningsEventModel row, or None if not found.
        """
        return self._session.get(EarningsEventModel, event_id)
