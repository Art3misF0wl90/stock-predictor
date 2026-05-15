"""
PositionDeriver — derives current equity positions from transaction history.

Position state is never stored directly in the database. It is always
derived by replaying the full transaction history for a ticker in
chronological order. This means partial closes, averaging down/up,
and direction changes are all handled naturally — the ledger is the
truth, not a cached snapshot.

Why derive instead of store?
    Storing a "current position" creates two sources of truth — the
    transaction log and the stored position. If they ever diverge
    (bug, failed write, manual correction), you have a consistency
    problem. By always deriving from the transaction log, there is
    only one source of truth. The position is always exactly what the
    transactions say it is.

Derivation algorithm for one ticker:
    Replay transactions in chronological order (executed_at ascending).
    Maintain a running state: shares, average_cost, direction, intent.

    BUY_LONG:
        If currently FLAT or LONG:
            weighted average the new cost into average_cost
            add shares to position
            set direction=LONG
        If currently SHORT:
            reduce short position by shares (partial cover)
            if shares >= current_short: position becomes FLAT

    SELL_LONG:
        Reduce long position by shares
        If shares >= current_long: position becomes FLAT

    SELL_SHORT:
        If currently FLAT or SHORT:
            add shares to short position
            set direction=SHORT
        If currently LONG:
            reduce long position (closing a long)

    BUY_SHORT (cover):
        Reduce short position by shares
        If shares >= current_short: position becomes FLAT

Depends on:
    TransactionRepository   — source of transaction history
    Position                — the result dataclass
    PositionDirection       — LONG / SHORT / FLAT
    TransactionType         — BUY_LONG / SELL_LONG / SELL_SHORT / BUY_SHORT

Exposes:
    derive(symbol) → Position
    derive_all()   → dict[str, Position]
"""

from __future__ import annotations

from datetime import datetime

from src.data.repositories.transaction_repository import TransactionRepository
from src.utils.types import (
    Position,
    PositionDirection,
    PositionIntent,
    TransactionType,
)


class PositionDeriver:
    """
    Derives current equity positions by replaying transaction history.

    Usage:
        deriver = PositionDeriver(transaction_repo)

        position = deriver.derive("AAPL")
        # position.direction is LONG, SHORT, or FLAT
        # position.shares is the current share count
        # position.average_cost is the weighted average cost basis
    """

    def __init__(self, transaction_repo: TransactionRepository) -> None:
        """
        Args:
            transaction_repo: Source of transaction history.
                              Returns rows in chronological order.
        """
        self._transaction_repo = transaction_repo

    def derive(self, symbol: str) -> Position:
        """
        Derive the current position for one ticker.

        Reads all transactions for the ticker in chronological order
        and replays them to compute the current state.

        Args:
            symbol: The ticker symbol.

        Returns:
            Position with direction, shares, average_cost, intent,
            and last_transaction_at. Returns a FLAT position with
            zero shares if no transactions exist for this ticker.
        """
        transactions = self._transaction_repo.get_by_ticker(symbol)

        if not transactions:
            return self._flat_position(symbol)

        return self._replay(symbol, transactions)

    def derive_all(self) -> dict[str, Position]:
        """
        Derive current positions for every ticker that has transactions.

        More efficient than calling derive() per ticker because it loads
        all transactions in one query and groups them in Python rather
        than making one DB call per ticker.

        Returns:
            Dict mapping symbol → Position for every ticker with at least
            one transaction. Tickers with no transactions are not included.
        """
        all_transactions = self._transaction_repo.get_all()

        if not all_transactions:
            return {}

        # Group transactions by ticker — preserve chronological order
        # because get_all() returns them sorted by executed_at ascending
        grouped: dict[str, list] = {}
        for txn in all_transactions:
            if txn.ticker not in grouped:
                grouped[txn.ticker] = []
            grouped[txn.ticker].append(txn)

        return {
            symbol: self._replay(symbol, txns)
            for symbol, txns in grouped.items()
        }

    # ------------------------------------------------------------------
    # Private — derivation logic
    # ------------------------------------------------------------------

    def _replay(self, symbol: str, transactions: list) -> Position:
        """
        Replay a chronologically ordered list of transactions.

        Maintains running position state across all transactions and
        returns the final state after all have been processed.

        Args:
            symbol: The ticker symbol (for the returned Position).
            transactions: Transactions in chronological order (asc).

        Returns:
            The derived Position after all transactions are applied.
        """
        # Running state
        shares: float = 0.0
        average_cost: float = 0.0
        direction: PositionDirection = PositionDirection.FLAT
        intent: PositionIntent = PositionIntent.NONE
        last_at: datetime | None = None

        for txn in transactions:
            txn_type = TransactionType(txn.type)
            txn_shares = float(txn.shares)
            txn_price = float(txn.price_per_share)
            last_at = txn.executed_at

            if txn_type == TransactionType.BUY_LONG:
                if direction == PositionDirection.SHORT:
                    # Covering a short position
                    shares -= txn_shares
                    if shares <= 0:
                        shares = 0.0
                        average_cost = 0.0
                        direction = PositionDirection.FLAT
                        intent = PositionIntent.NONE
                else:
                    # Opening or adding to a long position
                    # Weighted average cost basis
                    total_cost = (shares * average_cost) + (txn_shares * txn_price)
                    shares += txn_shares
                    average_cost = total_cost / shares if shares > 0 else 0.0
                    direction = PositionDirection.LONG
                    intent = PositionIntent(txn.position_intent)

            elif txn_type == TransactionType.SELL_LONG:
                # Reducing or closing a long position
                shares -= txn_shares
                if shares <= 0:
                    shares = 0.0
                    average_cost = 0.0
                    direction = PositionDirection.FLAT
                    intent = PositionIntent.NONE
                # average_cost does not change on a partial sell

            elif txn_type == TransactionType.SELL_SHORT:
                if direction == PositionDirection.LONG:
                    # Treating as closing a long (user sold long shares short)
                    shares -= txn_shares
                    if shares <= 0:
                        shares = 0.0
                        average_cost = 0.0
                        direction = PositionDirection.FLAT
                        intent = PositionIntent.NONE
                else:
                    # Opening or adding to a short position
                    total_cost = (shares * average_cost) + (txn_shares * txn_price)
                    shares += txn_shares
                    average_cost = total_cost / shares if shares > 0 else 0.0
                    direction = PositionDirection.SHORT
                    intent = PositionIntent(txn.position_intent)

            elif txn_type == TransactionType.BUY_SHORT:
                # Covering a short position
                shares -= txn_shares
                if shares <= 0:
                    shares = 0.0
                    average_cost = 0.0
                    direction = PositionDirection.FLAT
                    intent = PositionIntent.NONE

        return Position(
            symbol=symbol,
            direction=direction,
            shares=round(shares, 6),
            average_cost=round(average_cost, 6),
            intent=intent,
            last_transaction_at=last_at,
        )

    def _flat_position(self, symbol: str) -> Position:
        """
        Return a FLAT Position with zero shares.

        Used when no transactions exist for a ticker.

        Args:
            symbol: The ticker symbol.

        Returns:
            A FLAT Position with shares=0 and no transaction timestamp.
        """
        return Position(
            symbol=symbol,
            direction=PositionDirection.FLAT,
            shares=0.0,
            average_cost=0.0,
            intent=PositionIntent.NONE,
            last_transaction_at=None,
        )