"""
AlertOrchestrator — delivers pending alerts and retries failed ones.

The delivery mechanism in v2 is WebSocket push via SocketDispatcher.
AlertOrchestrator marks alerts as delivered when the push succeeds,
or increments retry_count when it fails.

Exposes:
    process_pending(session) → dict with delivered and failed counts
    retry_failed(session)    → dict with delivered and failed counts
"""

from __future__ import annotations

from datetime import datetime

from src.data.repositories.alert_repository import AlertRepository
from src.data.repositories.config_repository import ConfigRepository
from src.utils.types import ConfigCategory


class AlertOrchestrator:
    """
    Delivers alerts via WebSocket and manages retry logic.

    Usage:
        orchestrator = AlertOrchestrator(alert_repo, config_repo, socket_dispatcher)
        orchestrator.process_pending(session)
    """

    def __init__(
        self,
        alert_repo: AlertRepository,
        config_repo: ConfigRepository,
        socket_dispatcher,
    ) -> None:
        self._alert_repo = alert_repo
        self._config_repo = config_repo
        self._socket_dispatcher = socket_dispatcher

    def process_pending(self, session) -> dict:
        """
        Attempt delivery for all PENDING alerts.

        Args:
            session: Active SQLAlchemy Session.

        Returns:
            Dict with "delivered" and "failed" counts.
        """
        pending = self._alert_repo.get_pending()
        delivered = 0
        failed = 0

        for alert in pending:
            success = self._attempt_delivery(alert)
            if success:
                self._alert_repo.mark_delivered(
                    alert.id, delivered_at=datetime.utcnow()
                )
                delivered += 1
            else:
                self._alert_repo.mark_failed(alert.id)
                failed += 1

        session.commit()
        return {"delivered": delivered, "failed": failed}

    def retry_failed(self, session) -> dict:
        """
        Retry delivery for FAILED alerts below the max retry limit.

        Args:
            session: Active SQLAlchemy Session.

        Returns:
            Dict with "delivered" and "failed" counts.
        """
        max_attempts = self._config_repo.get(
            ConfigCategory.ALERTS, "max_retry_attempts"
        )
        retryable = self._alert_repo.get_failed_retryable(max_attempts)
        delivered = 0
        failed = 0

        for alert in retryable:
            success = self._attempt_delivery(alert)
            if success:
                self._alert_repo.mark_delivered(
                    alert.id, delivered_at=datetime.utcnow()
                )
                delivered += 1
            else:
                self._alert_repo.mark_failed(alert.id)
                failed += 1

        session.commit()
        return {"delivered": delivered, "failed": failed}

    def _attempt_delivery(self, alert) -> bool:
        """
        Attempt to deliver one alert via WebSocket.

        Args:
            alert: An AlertModel row to deliver.

        Returns:
            True if delivery succeeded, False if it failed.
        """
        try:
            self._socket_dispatcher.emit_alert({
                "id": alert.id,
                "ticker": alert.ticker,
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "created_at": str(alert.created_at),
            })
            return True
        except Exception:
            return False