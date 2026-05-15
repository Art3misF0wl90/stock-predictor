"""
SentimentRepository — all database operations for the SENTIMENT_RECORD table.

Raw sentiment scores from external sources (news, social media, analyst ratings)
are stored here. The SuggestionEngine computes a weighted aggregate at query
time — the aggregate is never stored.

Depends on:
    BaseRepository          — session management and context manager
    SentimentRecordModel    — the ORM class mapping to the sentiment_record table

Exposes:
    add(record)                     — insert a new sentiment record
    add_many(records)               — bulk insert multiple records efficiently
    get_by_ticker(symbol, days)     — recent sentiment records for one ticker
    get_latest_timestamp(symbol)    — timestamp of most recent record
    get_by_ticker_and_source(...)   — filter by source type
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from src.data.database import SentimentRecordModel
from src.data.repositories.base import BaseRepository


class SentimentRepository(BaseRepository):
    """
    Reads and writes the SENTIMENT_RECORD table.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, record: SentimentRecordModel) -> None:
        """
        Insert a single sentiment record.

        Args:
            record: A fully constructed SentimentRecordModel instance.
        """
        self._session.add(record)

    def add_many(self, records: list[SentimentRecordModel]) -> None:
        """
        Bulk insert multiple sentiment records in one operation.

        More efficient than calling add() in a loop because SQLAlchemy
        can batch the INSERT statements. Use this when ingesting a batch
        of sentiment scores from a data source.

        Args:
            records: List of SentimentRecordModel instances to insert.
        """
        self._session.add_all(records)

    def get_by_ticker(
        self,
        symbol: str,
        days: int = 30,
    ) -> list[SentimentRecordModel]:
        """
        Fetch recent sentiment records for one ticker.

        Returns records from the last `days` days, newest first. The
        SuggestionEngine uses this window to compute a weighted aggregate
        sentiment score that factors into its recommendations.

        Args:
            symbol: The ticker symbol.
            days: How many days back to look. Defaults to 30. The
                  appropriate window is configurable in SYSTEM_CONFIG
                  SIGNALS category — callers should read it from there.

        Returns:
            List of SentimentRecordModel rows within the window,
            newest first. Empty list if none exist.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        return (
            self._session.query(SentimentRecordModel)
            .filter(
                SentimentRecordModel.ticker == symbol,
                SentimentRecordModel.record_date >= cutoff,
            )
            .order_by(SentimentRecordModel.record_date.desc())
            .all()
        )

    def get_latest_timestamp(self, symbol: str) -> datetime | None:
        """
        Return the fetched_at timestamp of the most recent sentiment record.

        Used by SuggestionEngine's staleness check — if new sentiment
        data has arrived since the last suggestion, a fresh suggestion
        is needed.

        Args:
            symbol: The ticker symbol.

        Returns:
            The datetime of the most recent record, or None if none exist.
        """
        from sqlalchemy import func

        return (
            self._session.query(func.max(SentimentRecordModel.fetched_at))
            .filter(SentimentRecordModel.ticker == symbol)
            .scalar()
        )

    def get_by_ticker_and_source(
        self,
        symbol: str,
        source_type: str,
        days: int = 30,
    ) -> list[SentimentRecordModel]:
        """
        Fetch sentiment records filtered by source type.

        Useful when the chatbot or suggestion engine needs to reason about
        a specific source — e.g. "what does analyst sentiment look like
        for AAPL independent of social media noise?"

        Args:
            symbol: The ticker symbol.
            source_type: One of "NEWS", "SOCIAL_MEDIA", "ANALYST_RATING".
            days: Lookback window in days. Defaults to 30.

        Returns:
            Matching SentimentRecordModel rows, newest first.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        return (
            self._session.query(SentimentRecordModel)
            .filter(
                SentimentRecordModel.ticker == symbol,
                SentimentRecordModel.source_type == source_type,
                SentimentRecordModel.record_date >= cutoff,
            )
            .order_by(SentimentRecordModel.record_date.desc())
            .all()
        )
