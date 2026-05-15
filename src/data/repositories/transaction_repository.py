"""
TransactionRepository — all database operations for the TRANSACTION table.

Transactions are the source of truth for portfolio positions. Current
position state is ALWAYS derived from transaction history by PositionDeriver
— it is never stored directly. This means every buy, sell, partial close,
and direction change is just another row appended here.

Depends on:
    BaseRepository      — session management and context manager
    TransactionModel    — the ORM class mapping to the transaction table

Exposes:
    add(transaction)            — insert a new transaction row
    get_by_ticker(symbol)       — all transactions for one ticker, chronological
    get_all()                   — all transactions across all tickers
    get_by_id(transaction_id)   — fetch one transaction by UUID

Key rules:
    - Transactions are never updated or deleted after being written.
      They are the permanent ledger. PositionDeriver replays them to
      derive current state.
    - get_by_ticker() always returns rows in chronological order
      (executed_at ascending) because PositionDeriver must replay them
      in the correct sequence.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.data.database import TransactionModel
from src.data.repositories.base import BaseRepository


class TransactionRepository(BaseRepository):
    """
    Reads and writes the TRANSACTION table.

    Like the signal table, transactions are append-only in practice —
    you never update or delete a transaction after it is written.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, transaction: TransactionModel) -> None:
        """
        Insert a new transaction row.

        Args:
            transaction: A fully constructed TransactionModel instance.
        """
        self._session.add(transaction)

    def get_by_ticker(self, symbol: str) -> list[TransactionModel]:
        """
        Fetch all transactions for one ticker in chronological order.

        Chronological order (executed_at ascending) is required because
        PositionDeriver replays these rows sequentially to derive the
        current position. Returning them out of order would produce
        incorrect position state.

        Args:
            symbol: The ticker symbol.

        Returns:
            List of TransactionModel rows ordered by executed_at ascending.
            Empty list if no transactions exist for this ticker.
        """
        return (
            self._session.query(TransactionModel)
            .filter(TransactionModel.ticker == symbol)
            .order_by(TransactionModel.executed_at.asc())
            .all()
        )

    def get_all(self) -> list[TransactionModel]:
        """
        Fetch all transactions across all tickers, chronological order.

        Used by PositionDeriver.derive_all() to compute positions for
        every ticker in one pass.

        Returns:
            All TransactionModel rows ordered by executed_at ascending.
        """
        return (
            self._session.query(TransactionModel)
            .order_by(TransactionModel.executed_at.asc())
            .all()
        )

    def get_latest_timestamp(self, symbol: str):
        """
        Return the executed_at timestamp of the most recent transaction.

        Used by SuggestionEngine staleness check — if a new transaction
        has occurred since the last suggestion, a fresh suggestion is needed.

        Args:
            symbol: The ticker symbol.

        Returns:
            The datetime of the most recent transaction, or None if none exist.
        """
        from sqlalchemy import func

        return (
            self._session.query(func.max(TransactionModel.executed_at))
            .filter(TransactionModel.ticker == symbol)
            .scalar()
        )

    def get_by_id(self, transaction_id: str) -> TransactionModel | None:
        """
        Fetch one transaction by its UUID primary key.

        Args:
            transaction_id: The UUID string of the transaction row.

        Returns:
            The TransactionModel row, or None if not found.
        """
        return self._session.get(TransactionModel, transaction_id)
