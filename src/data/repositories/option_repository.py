"""
OptionRepository — all database operations for the OPTION_POSITION table.

Options are standalone contracts fully independent of equity transactions.
They have their own lifecycle: OPEN → CLOSED or OPEN → EXPIRED.

Depends on:
    BaseRepository          — session management and context manager
    OptionPositionModel     — the ORM class mapping to the option_position table
    OptionStatus            — enum for OPEN / CLOSED / EXPIRED

Exposes:
    add(option)                     — insert a new option position
    get_open()                      — all currently open options
    get_by_ticker(symbol)           — all options for one ticker
    get_expiring_before(date)       — options expiring before a given date
    close_option(id, premium, at)   — mark an option as closed
    expire_options(before)          — bulk-mark expired options
    get_by_id(option_id)            — fetch one option by UUID
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.data.database import OptionPositionModel
from src.data.repositories.base import BaseRepository
from src.utils.types import OptionStatus


class OptionRepository(BaseRepository):
    """
    Reads and writes the OPTION_POSITION table.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, option: OptionPositionModel) -> None:
        """
        Insert a new option position row.

        Args:
            option: A fully constructed OptionPositionModel instance.
        """
        self._session.add(option)

    def get_open(self) -> list[OptionPositionModel]:
        """
        Fetch all option positions with status=OPEN.

        Used by InferenceOrchestrator to assemble inputs for AlertEvaluator
        so it can check for contracts approaching expiration.

        Returns:
            List of open OptionPositionModel rows ordered by expiration_date
            ascending (soonest-expiring first).
        """
        return (
            self._session.query(OptionPositionModel)
            .filter(OptionPositionModel.status == OptionStatus.OPEN.value)
            .order_by(OptionPositionModel.expiration_date.asc())
            .all()
        )

    def get_by_ticker(self, symbol: str) -> list[OptionPositionModel]:
        """
        Fetch all option positions for one ticker regardless of status.

        Args:
            symbol: The ticker symbol.

        Returns:
            All OptionPositionModel rows for that ticker, newest first.
        """
        return (
            self._session.query(OptionPositionModel)
            .filter(OptionPositionModel.ticker == symbol)
            .order_by(OptionPositionModel.opened_at.desc())
            .all()
        )

    def get_expiring_before(self, before: datetime) -> list[OptionPositionModel]:
        """
        Fetch open options that expire before a given datetime.

        Used by AlertEvaluator to find contracts approaching expiration
        that should trigger an OPTION_EXPIRING alert.

        Args:
            before: Fetch options whose expiration_date is before this datetime.
                    Typically set to now + the alert window from SYSTEM_CONFIG.

        Returns:
            Open OptionPositionModel rows expiring before `before`,
            ordered by expiration_date ascending.
        """
        return (
            self._session.query(OptionPositionModel)
            .filter(
                OptionPositionModel.status == OptionStatus.OPEN.value,
                OptionPositionModel.expiration_date < before,
            )
            .order_by(OptionPositionModel.expiration_date.asc())
            .all()
        )

    def close_option(
        self,
        option_id: str,
        close_premium: float,
        closed_at: datetime,
    ) -> None:
        """
        Mark an option position as closed.

        Sets status to CLOSED and records the closing premium and timestamp.
        The P&L can be derived from premium_paid vs close_premium.

        Args:
            option_id: UUID of the option to close.
            close_premium: The premium received or paid to close the position.
            closed_at: When the position was closed.
        """
        option = self.get_by_id(option_id)
        if option is None:
            return

        option.status = OptionStatus.CLOSED.value
        option.close_premium = close_premium
        option.closed_at = closed_at

    def expire_options(self, before: datetime) -> int:
        """
        Bulk-mark all open options past their expiration date as EXPIRED.

        Called by the PortfolioOrchestrator.expire_options() job on a daily
        schedule. Returns the count of rows updated so the caller can log it.

        Args:
            before: Mark options with expiration_date before this as EXPIRED.
                    Typically datetime.utcnow().

        Returns:
            Number of option rows updated to EXPIRED status.
        """
        expired = (
            self._session.query(OptionPositionModel)
            .filter(
                OptionPositionModel.status == OptionStatus.OPEN.value,
                OptionPositionModel.expiration_date < before,
            )
            .all()
        )

        for option in expired:
            option.status = OptionStatus.EXPIRED.value

        return len(expired)

    def get_by_id(self, option_id: str) -> OptionPositionModel | None:
        """
        Fetch one option position by its UUID primary key.

        Args:
            option_id: The UUID string of the option row.

        Returns:
            The OptionPositionModel row, or None if not found.
        """
        return self._session.get(OptionPositionModel, option_id)
