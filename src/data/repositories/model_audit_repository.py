"""
ModelAuditRepository — all database operations for the MODEL_AUDIT table.

Every training run is recorded here — accepted or rejected. This gives
you a full history of every model ever trained for every ticker, including
rejected ones and their rejection reasons.

Depends on:
    BaseRepository      — session management and context manager
    ModelAuditModel     — the ORM class mapping to the model_audit table

Exposes:
    add(audit)                  — insert a new model audit row
    get_latest(symbol)          — most recent audit for one ticker
    get_history(symbol, limit)  — recent training history for one ticker
    get_accepted(symbol)        — only accepted training runs
    get_by_id(audit_id)         — fetch one audit by UUID
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.data.database import ModelAuditModel
from src.data.repositories.base import BaseRepository


class ModelAuditRepository(BaseRepository):
    """
    Reads and writes the MODEL_AUDIT table.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, audit: ModelAuditModel) -> None:
        """
        Insert a new model audit row.

        Called by Trainer after every training run — accepted or rejected.
        Never skip this write. The audit trail must be complete.

        Args:
            audit: A fully constructed ModelAuditModel instance.
        """
        self._session.add(audit)

    def get_latest(self, symbol: str) -> ModelAuditModel | None:
        """
        Fetch the most recent training audit for one ticker.

        Includes rejected runs. Used by Trainer to retrieve auc_before
        when deciding whether a new model meets the improvement threshold.

        Args:
            symbol: The ticker symbol.

        Returns:
            The most recent ModelAuditModel row, or None if this ticker
            has never been trained.
        """
        return (
            self._session.query(ModelAuditModel)
            .filter(ModelAuditModel.ticker == symbol)
            .order_by(ModelAuditModel.trained_at.desc())
            .first()
        )

    def get_latest_accepted(self, symbol: str) -> ModelAuditModel | None:
        """
        Fetch the most recent accepted training run for one ticker.

        Used when you need the AUC of the currently active model,
        not just the most recent training attempt.

        Args:
            symbol: The ticker symbol.

        Returns:
            The most recent accepted ModelAuditModel, or None if no
            accepted run exists yet.
        """
        return (
            self._session.query(ModelAuditModel)
            .filter(
                ModelAuditModel.ticker == symbol,
                ModelAuditModel.accepted == True,
            )
            .order_by(ModelAuditModel.trained_at.desc())
            .first()
        )

    def get_history(self, symbol: str, limit: int = 20) -> list[ModelAuditModel]:
        """
        Fetch recent training history for one ticker, newest first.

        Args:
            symbol: The ticker symbol.
            limit: Maximum rows to return. Defaults to 20.

        Returns:
            List of ModelAuditModel rows, newest first.
        """
        return (
            self._session.query(ModelAuditModel)
            .filter(ModelAuditModel.ticker == symbol)
            .order_by(ModelAuditModel.trained_at.desc())
            .limit(limit)
            .all()
        )

    def get_by_id(self, audit_id: str) -> ModelAuditModel | None:
        """
        Fetch one audit row by its UUID primary key.

        Args:
            audit_id: The UUID string.

        Returns:
            The ModelAuditModel row, or None if not found.
        """
        return self._session.get(ModelAuditModel, audit_id)
