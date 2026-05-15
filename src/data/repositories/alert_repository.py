"""
AlertRepository — all database operations for the ALERT table.

Alerts are never silently dropped. Permanently failed alerts remain in
the database as audit records. The full lifecycle is tracked via the
AlertStatus state machine.

Depends on:
    BaseRepository  — session management and context manager
    AlertModel      — the ORM class mapping to the alert table
    AlertStatus     — enum for PENDING / DELIVERED / FAILED / ACKNOWLEDGED

Exposes:
    add(alert)                          — insert a new alert
    get_pending()                       — all PENDING alerts
    get_failed_retryable(max_attempts)  — FAILED alerts below retry limit
    get_unacknowledged()                — PENDING and DELIVERED alerts
    mark_delivered(alert_id, at)        — transition to DELIVERED
    mark_failed(alert_id)               — transition to FAILED, increment retry
    acknowledge(alert_id, at)           — transition to ACKNOWLEDGED
    exists_recent(symbol, type, window) — deduplication check
    get_by_id(alert_id)                 — fetch one alert by UUID
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from src.data.database import AlertModel
from src.data.repositories.base import BaseRepository
from src.utils.types import AlertStatus, AlertType


class AlertRepository(BaseRepository):
    """
    Reads and writes the ALERT table.

    The most important rule: alerts are never silently dropped.
    Every alert that fails to deliver stays in the database with
    status=FAILED and an incremented retry_count. The AlertOrchestrator
    retries them on a schedule until max_attempts is reached.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def add(self, alert: AlertModel) -> None:
        """
        Insert a new alert row.

        Args:
            alert: A fully constructed AlertModel instance.
        """
        self._session.add(alert)

    def get_pending(self) -> list[AlertModel]:
        """
        Fetch all alerts with status=PENDING.

        Called by AlertOrchestrator.process_pending() to find alerts
        that need to be delivered for the first time.

        Returns:
            All PENDING AlertModel rows, ordered by created_at ascending
            (oldest first — deliver in the order they were created).
        """
        return (
            self._session.query(AlertModel)
            .filter(AlertModel.status == AlertStatus.PENDING.value)
            .order_by(AlertModel.created_at.asc())
            .all()
        )

    def get_failed_retryable(self, max_attempts: int) -> list[AlertModel]:
        """
        Fetch FAILED alerts that are still eligible for retry.

        An alert is retryable if its retry_count is strictly less than
        max_attempts. Once retry_count reaches max_attempts, the alert
        is permanently failed and stays in the DB as an audit record
        with no further retry attempts.

        Args:
            max_attempts: The configured maximum retry count from
                          SYSTEM_CONFIG ALERTS category.

        Returns:
            FAILED AlertModel rows below the retry limit, ordered by
            created_at ascending.
        """
        return (
            self._session.query(AlertModel)
            .filter(
                AlertModel.status == AlertStatus.FAILED.value,
                AlertModel.retry_count < max_attempts,
            )
            .order_by(AlertModel.created_at.asc())
            .all()
        )

    def get_unacknowledged(self) -> list[AlertModel]:
        """
        Fetch all alerts the user has not yet acknowledged.

        Returns PENDING and DELIVERED alerts — these are the ones
        visible to the user in the dashboard alert panel.

        Returns:
            AlertModel rows with status PENDING or DELIVERED,
            ordered by created_at descending (newest first).
        """
        return (
            self._session.query(AlertModel)
            .filter(
                AlertModel.status.in_([
                    AlertStatus.PENDING.value,
                    AlertStatus.DELIVERED.value,
                ])
            )
            .order_by(AlertModel.created_at.desc())
            .all()
        )

    def mark_delivered(self, alert_id: str, delivered_at: datetime) -> None:
        """
        Transition an alert from PENDING or FAILED to DELIVERED.

        Sets delivered_at to record when delivery succeeded.

        Args:
            alert_id: UUID of the alert to update.
            delivered_at: When delivery completed successfully.
        """
        alert = self.get_by_id(alert_id)
        if alert is None:
            return

        alert.status = AlertStatus.DELIVERED.value
        alert.delivered_at = delivered_at

    def mark_failed(self, alert_id: str) -> None:
        """
        Transition an alert to FAILED and increment its retry count.

        Called when a delivery attempt fails. The AlertOrchestrator checks
        retry_count against max_attempts before attempting delivery —
        permanently failed alerts are never retried again but remain in
        the database as audit records.

        Args:
            alert_id: UUID of the alert to update.
        """
        alert = self.get_by_id(alert_id)
        if alert is None:
            return

        alert.status = AlertStatus.FAILED.value
        alert.retry_count += 1

    def acknowledge(self, alert_id: str, acknowledged_at: datetime) -> None:
        """
        Transition an alert to ACKNOWLEDGED — terminal state.

        Once acknowledged, no further transitions occur. The alert stays
        in the database permanently as part of the audit trail.

        Args:
            alert_id: UUID of the alert to acknowledge.
            acknowledged_at: When the user dismissed the alert.
        """
        alert = self.get_by_id(alert_id)
        if alert is None:
            return

        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_at = acknowledged_at

    def exists_recent(
        self,
        symbol: str,
        alert_type: AlertType,
        window_minutes: int,
    ) -> bool:
        """
        Check whether a matching alert was created within the dedup window.

        Used by AlertDeduplicator to prevent the same alert condition from
        firing multiple times in a short period. An alert is considered a
        duplicate if there is already an alert with the same ticker and
        alert_type created within the last `window_minutes` minutes.

        Args:
            symbol: The ticker symbol.
            alert_type: The type of alert to check for.
            window_minutes: The deduplication window from SYSTEM_CONFIG
                            ALERTS category.

        Returns:
            True if a matching alert exists within the window.
            False if no duplicate exists and this alert should be created.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        count = (
            self._session.query(AlertModel)
            .filter(
                AlertModel.ticker == symbol,
                AlertModel.alert_type == alert_type.value,
                AlertModel.created_at >= cutoff,
            )
            .count()
        )
        return count > 0

    def get_by_id(self, alert_id: str) -> AlertModel | None:
        """
        Fetch one alert by its UUID primary key.

        Args:
            alert_id: The UUID string.

        Returns:
            The AlertModel row, or None if not found.
        """
        return self._session.get(AlertModel, alert_id)