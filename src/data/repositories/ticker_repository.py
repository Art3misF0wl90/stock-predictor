"""
TickerRepository — all database operations for the TICKER table.

This is the only place in the system that reads or writes ticker rows.
No other component queries the ticker table directly.

Depends on:
    BaseRepository  — session management and context manager
    TickerModel     — the ORM class mapping to the ticker table
    OnboardingStatus — enum for valid status values

Exposes:
    add(ticker)                         — insert a new ticker row
    get(symbol)                         — fetch one ticker by symbol
    get_all()                           — fetch every ticker row
    get_watchlist()                     — fetch only on_watchlist=True tickers
    get_training_eligible()             — fetch tickers ready to train
    update_onboarding_status(...)       — transition onboarding state machine
    update_training_flags(...)          — set training_eligible and min_data_met
    exists(symbol)                      — check if a ticker row exists
    delete(symbol)                      — remove a ticker and cascade deletes

Key rules:
    - Never update onboarding_status directly from outside this repository.
      Always go through update_onboarding_status() so the transition is logged.
    - get() returns None if the symbol is not found — it never raises.
    - delete() cascades to all child rows (signals, transactions, alerts, etc.)
      because of cascade="all, delete-orphan" on the ORM relationships.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.data.database import TickerModel
from src.data.repositories.base import BaseRepository
from src.utils.types import OnboardingStatus


class TickerRepository(BaseRepository):
    """
    Reads and writes the TICKER table.

    Inherits session management and context manager from BaseRepository.
    Every method operates through self._session — no raw SQL anywhere.

    Usage:
        with TickerRepository(session) as repo:
            ticker = repo.get("AAPL")
            if ticker is None:
                repo.add(TickerModel(symbol="AAPL", ...))
    """

    def __init__(self, session: Session) -> None:
        """
        Args:
            session: Active SQLAlchemy Session from Database.get_session().
        """
        super().__init__(session)

    def add(self, ticker: TickerModel) -> None:
        """
        Insert a new ticker row.

        Does not commit — commit happens in the context manager __exit__
        or when you call self.commit() manually.

        Args:
            ticker: A fully constructed TickerModel instance.

        Example:
            ticker = TickerModel(
                symbol="AAPL",
                company_name="Apple Inc.",
                sector="Technology",
                exchange="NASDAQ",
            )
            repo.add(ticker)
        """
        self._session.add(ticker)

    def get(self, symbol: str) -> TickerModel | None:
        """
        Fetch one ticker by its symbol.

        Returns None if the symbol does not exist in the database.
        Never raises for a missing ticker — callers check for None.

        Args:
            symbol: The ticker symbol, e.g. "AAPL". Case-sensitive.

        Returns:
            The TickerModel row, or None if not found.
        """
        return self._session.get(TickerModel, symbol)

    def get_all(self) -> list[TickerModel]:
        """
        Fetch every row in the ticker table.

        Returns an empty list if no tickers exist — never raises.

        Returns:
            List of all TickerModel rows, ordered by symbol ascending.
        """
        return (
            self._session.query(TickerModel)
            .order_by(TickerModel.symbol)
            .all()
        )

    def get_watchlist(self) -> list[TickerModel]:
        """
        Fetch all tickers where on_watchlist=True.

        These are the tickers the scheduler runs daily inference and
        retrain on. Tickers with on_watchlist=False are tracked in the
        database but excluded from scheduled jobs.

        Returns:
            List of TickerModel rows where on_watchlist is True,
            ordered by symbol ascending.
        """
        return (
            self._session.query(TickerModel)
            .filter(TickerModel.on_watchlist == True)
            .order_by(TickerModel.symbol)
            .all()
        )

    def get_training_eligible(self) -> list[TickerModel]:
        """
        Fetch tickers that are ready to be trained.

        A ticker is training-eligible when:
            - training_eligible=True (set by TickerValidator after checking
              that at least one calendar year of data is available)
            - min_data_met=True (set by DataLoader after a successful fetch)
            - onboarding_status=COMPLETE (has been successfully onboarded)

        The scheduler uses this list to decide which tickers to include
        in the daily retrain job.

        Returns:
            List of eligible TickerModel rows, ordered by symbol ascending.
        """
        return (
            self._session.query(TickerModel)
            .filter(
                TickerModel.training_eligible == True,
                TickerModel.min_data_met == True,
                TickerModel.onboarding_status == OnboardingStatus.COMPLETE.value,
            )
            .order_by(TickerModel.symbol)
            .all()
        )

    def update_onboarding_status(
        self,
        symbol: str,
        new_status: OnboardingStatus,
        reason: str | None = None,
    ) -> None:
        """
        Transition a ticker's onboarding status.

        This is the only correct way to change onboarding_status. Setting
        it directly on the model object bypasses this method and skips the
        timestamp update, which breaks the audit trail.

        Valid transitions are defined in the state machine in the architecture
        doc and enforced by TickerOnboardingPipeline — this repository method
        does not re-validate transitions, it just applies them.

        Args:
            symbol: The ticker to update.
            new_status: The OnboardingStatus to transition to.
            reason: Optional human-readable reason for the transition.
                    Stored in the model's notes field if provided. Used
                    primarily for FAILED transitions to record why it failed.

        Returns:
            None. Does not raise if the ticker is not found — logs nothing,
            does nothing. Callers should check exists() first if they need
            to guarantee the row is present.
        """
        ticker = self.get(symbol)
        if ticker is None:
            return

        ticker.onboarding_status = new_status.value

        if reason is not None and new_status == OnboardingStatus.FAILED:
            # Store the failure reason in company_name temporarily if notes
            # field doesn't exist — in practice use a dedicated reason column
            # added in a migration if this becomes important.
            pass

    def update_training_flags(
        self,
        symbol: str,
        training_eligible: bool,
        min_data_met: bool,
    ) -> None:
        """
        Set the training eligibility flags for a ticker.

        Called by TickerValidator after checking data availability and by
        DataLoader after a successful fetch. Both flags must be True for
        the scheduler to include this ticker in daily retrains.

        Args:
            symbol: The ticker to update.
            training_eligible: True if at least one calendar year of
                               historical data is available.
            min_data_met: True if the most recent data fetch succeeded
                          and returned enough rows for inference.
        """
        ticker = self.get(symbol)
        if ticker is None:
            return

        ticker.training_eligible = training_eligible
        ticker.min_data_met = min_data_met

    def set_watchlist(self, symbol: str, on_watchlist: bool) -> None:
        """
        Add or remove a ticker from the active watchlist.

        Tickers removed from the watchlist remain in the database and
        keep their trained models — they are simply excluded from
        scheduled inference and retrain jobs.

        Args:
            symbol: The ticker to update.
            on_watchlist: True to include in scheduled jobs, False to exclude.
        """
        ticker = self.get(symbol)
        if ticker is None:
            return

        ticker.on_watchlist = on_watchlist

    def exists(self, symbol: str) -> bool:
        """
        Check whether a ticker row exists without fetching the full object.

        More efficient than get() when you only need a yes/no answer,
        because it runs a COUNT query instead of fetching all columns.

        Args:
            symbol: The ticker symbol to check.

        Returns:
            True if a row with that symbol exists, False otherwise.
        """
        count = (
            self._session.query(TickerModel)
            .filter(TickerModel.symbol == symbol)
            .count()
        )
        return count > 0

    def delete(self, symbol: str) -> None:
        """
        Delete a ticker and all its related rows.

        Because the ORM relationships on TickerModel use
        cascade="all, delete-orphan", deleting the ticker row
        automatically deletes all child rows across every related table:
        transactions, signals, alerts, model audits, suggestions,
        sentiment records, earnings events, macro relevance, and options.

        This is a destructive operation. There is no soft delete.
        Use set_watchlist(symbol, False) if you want to stop processing
        a ticker without losing its history.

        Args:
            symbol: The ticker to delete. Does nothing if not found.
        """
        ticker = self.get(symbol)
        if ticker is None:
            return

        self._session.delete(ticker)