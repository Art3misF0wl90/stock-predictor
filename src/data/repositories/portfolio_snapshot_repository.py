"""
PortfolioSnapshotRepository — all database operations for PORTFOLIO_SNAPSHOT.

Snapshots are point-in-time records of portfolio state. They are written
after every transaction and on a daily scheduled cron. The sector_breakdown
field is stored as JSON text and parsed on read.

Depends on:
    BaseRepository              — session management and context manager
    PortfolioSnapshotModel      — the ORM class mapping to portfolio_snapshot

Exposes:
    add(snapshot)           — insert a new snapshot row
    get_latest()            — most recent snapshot
    get_history(limit)      — recent snapshot history
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from src.data.database import PortfolioSnapshotModel
from src.data.repositories.base import BaseRepository


class PortfolioSnapshotRepository(BaseRepository):
    """
    Reads and writes the PORTFOLIO_SNAPSHOT table.

    sector_breakdown is stored as a JSON string in the database.
    The add() method serializes it automatically. get_latest() and
    get_history() return raw model objects — callers parse
    sector_breakdown with json.loads() if they need the dict.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, snapshot: PortfolioSnapshotModel) -> None:
        """
        Insert a new portfolio snapshot.

        sector_breakdown must already be serialized to a JSON string
        before being set on the model. PortfolioSnapshotService handles
        this serialization before calling add().

        Args:
            snapshot: A fully constructed PortfolioSnapshotModel instance.
        """
        self._session.add(snapshot)

    def get_latest(self) -> PortfolioSnapshotModel | None:
        """
        Fetch the most recent portfolio snapshot.

        Used by the dashboard to display current portfolio state and by
        ChatbotService to give the LLM portfolio context.

        Returns:
            The most recent PortfolioSnapshotModel, or None if no
            snapshots exist yet.
        """
        return (
            self._session.query(PortfolioSnapshotModel)
            .order_by(PortfolioSnapshotModel.snapshot_at.desc())
            .first()
        )

    def get_history(self, limit: int = 30) -> list[PortfolioSnapshotModel]:
        """
        Fetch recent snapshot history, newest first.

        Used for portfolio value trend visualization on the dashboard.

        Args:
            limit: Maximum rows to return. Defaults to 30.

        Returns:
            List of PortfolioSnapshotModel rows, newest first.
        """
        return (
            self._session.query(PortfolioSnapshotModel)
            .order_by(PortfolioSnapshotModel.snapshot_at.desc())
            .limit(limit)
            .all()
        )
