"""
PortfolioOrchestrator — handles transaction recording and portfolio snapshots.

Coordinates:
    record_transaction(txn_data) — persist a transaction and take a snapshot
    expire_options()             — mark past-expiry options as EXPIRED
    take_scheduled_snapshot()    — daily portfolio snapshot

Exposes:
    record_transaction(txn_data, session) → TransactionModel
    expire_options(session)               → int (count expired)
    take_scheduled_snapshot(session)      → PortfolioSnapshotModel
"""

from __future__ import annotations

from datetime import datetime

from src.data.database import TransactionModel
from src.data.repositories.option_repository import OptionRepository
from src.data.repositories.transaction_repository import TransactionRepository
from src.domain.portfolio.portfolio_snapshot_service import PortfolioSnapshotService
from src.utils.types import SnapshotTrigger


class PortfolioOrchestrator:
    """
    Handles transaction recording and portfolio state management.

    Usage:
        orchestrator = PortfolioOrchestrator(...)
        orchestrator.record_transaction(txn_data, session)
    """

    def __init__(
        self,
        transaction_repo: TransactionRepository,
        option_repo: OptionRepository,
        snapshot_service: PortfolioSnapshotService,
    ) -> None:
        self._transaction_repo = transaction_repo
        self._option_repo = option_repo
        self._snapshot_service = snapshot_service

    def record_transaction(
        self,
        txn_data: dict,
        session,
    ) -> TransactionModel:
        """
        Persist a transaction and immediately take a portfolio snapshot.

        Args:
            txn_data: Dict with transaction fields: ticker, type, shares,
                      price_per_share, direction, position_intent, executed_at.
            session: Active SQLAlchemy Session.

        Returns:
            The persisted TransactionModel row.
        """
        txn = TransactionModel(
            ticker=txn_data["ticker"],
            type=txn_data["type"],
            shares=float(txn_data["shares"]),
            price_per_share=float(txn_data["price_per_share"]),
            direction=txn_data["direction"],
            position_intent=txn_data["position_intent"],
            executed_at=txn_data.get("executed_at", datetime.utcnow()),
            notes=txn_data.get("notes"),
        )
        self._transaction_repo.add(txn)
        session.flush()

        # Take a snapshot after every transaction so portfolio state
        # is always current after any position change
        self._snapshot_service.take_snapshot(
            trigger=SnapshotTrigger.TRANSACTION,
            session=session,
        )
        session.commit()

        return txn

    def expire_options(self, session) -> int:
        """
        Mark all open options past their expiration date as EXPIRED.

        Called by the daily scheduler job.

        Args:
            session: Active SQLAlchemy Session.

        Returns:
            Number of options expired.
        """
        count = self._option_repo.expire_options(before=datetime.utcnow())
        session.commit()
        return count

    def take_scheduled_snapshot(self, session) -> object:
        """
        Take a daily scheduled portfolio snapshot.

        Args:
            session: Active SQLAlchemy Session.

        Returns:
            The persisted PortfolioSnapshotModel row.
        """
        snapshot = self._snapshot_service.take_snapshot(
            trigger=SnapshotTrigger.SCHEDULED,
            session=session,
        )
        session.commit()
        return snapshot