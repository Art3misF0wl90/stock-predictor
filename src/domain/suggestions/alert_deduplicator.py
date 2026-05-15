"""
AlertDeduplicator — filters alert candidates against existing alerts.

AlertDeduplicator prevents the same alert condition from firing
repeatedly in a short window. Without deduplication, every inference
run could generate the same stop-loss or signal-change alert over and
over until the condition resolves.

Deduplication rule:
    An alert candidate is suppressed if AlertRepository.exists_recent()
    returns True for the same (symbol, alert_type) pair within the
    configured deduplication window (from SYSTEM_CONFIG ALERTS category).

    The window is alert-type-specific:
        SIGNAL_CHANGE alerts deduplicate over a short window (e.g. 60 min)
            because a signal can genuinely change multiple times per day.
        STOP_LOSS and PROFIT_TARGET deduplicate over a longer window (e.g. 24h)
            to avoid repeated urgent alerts for the same breach.
        EARNINGS_APPROACHING and OPTION_EXPIRING deduplicate over a very
            long window (e.g. 48h) — you only need one reminder per event.

Depends on:
    AlertRepository     — checks for existing recent alerts
    AlertCandidate      — input from AlertEvaluator
    AlertModel          — output (candidates that passed deduplication)
    AlertType           — for window lookup

Exposes:
    filter(candidates, session) → list[AlertCandidate]
"""

from __future__ import annotations

from src.data.repositories.alert_repository import AlertRepository
from src.utils.types import AlertCandidate, AlertType


# Default deduplication windows per alert type, in minutes.
# Overridden by SYSTEM_CONFIG ALERTS category if keys are present.
_DEFAULT_WINDOWS: dict[AlertType, int] = {
    AlertType.SIGNAL_CHANGE: 60,
    AlertType.STOP_LOSS: 1440,        # 24 hours
    AlertType.PROFIT_TARGET: 1440,    # 24 hours
    AlertType.EARNINGS_APPROACHING: 2880,  # 48 hours
    AlertType.OPTION_EXPIRING: 2880,       # 48 hours
    AlertType.PORTFOLIO_COMPOSITION: 1440, # 24 hours
}

# SYSTEM_CONFIG key suffixes for per-type window overrides.
# e.g. "signal_change_dedup_minutes", "stop_loss_dedup_minutes"
_CONFIG_KEY_SUFFIXES: dict[AlertType, str] = {
    AlertType.SIGNAL_CHANGE: "signal_change_dedup_minutes",
    AlertType.STOP_LOSS: "stop_loss_dedup_minutes",
    AlertType.PROFIT_TARGET: "profit_target_dedup_minutes",
    AlertType.EARNINGS_APPROACHING: "earnings_dedup_minutes",
    AlertType.OPTION_EXPIRING: "option_expiry_dedup_minutes",
    AlertType.PORTFOLIO_COMPOSITION: "composition_dedup_minutes",
}


class AlertDeduplicator:
    """
    Filters alert candidates to remove duplicates of recently fired alerts.

    Usage:
        deduplicator = AlertDeduplicator(alert_repo)
        unique_candidates = deduplicator.filter(candidates, config)
        # unique_candidates contains only candidates that have no matching
        # alert in the database within their deduplication window
    """

    def __init__(self, alert_repo: AlertRepository) -> None:
        """
        Args:
            alert_repo: For checking existing recent alerts.
        """
        self._alert_repo = alert_repo

    def filter(
        self,
        candidates: list[AlertCandidate],
        config: dict,
    ) -> list[AlertCandidate]:
        """
        Return only candidates that have no recent duplicate in the database.

        For each candidate, checks whether an alert with the same symbol
        and alert_type was created within the deduplication window. If one
        exists, the candidate is suppressed.

        Args:
            candidates: List of AlertCandidates from AlertEvaluator.
            config: Dict from SYSTEM_CONFIG ALERTS category. Used to
                    override default deduplication windows per alert type.

        Returns:
            List of AlertCandidates that passed deduplication.
            May be empty if all candidates are duplicates.
        """
        unique = []

        for candidate in candidates:
            window_minutes = self._get_window(candidate.alert_type, config)

            is_duplicate = self._alert_repo.exists_recent(
                symbol=candidate.symbol,
                alert_type=candidate.alert_type,
                window_minutes=window_minutes,
            )

            if not is_duplicate:
                unique.append(candidate)

        return unique

    def _get_window(self, alert_type: AlertType, config: dict) -> int:
        """
        Get the deduplication window for an alert type.

        Checks SYSTEM_CONFIG first, falls back to the hardcoded default.

        Args:
            alert_type: The alert type to look up.
            config: Dict from SYSTEM_CONFIG ALERTS category.

        Returns:
            Deduplication window in minutes.
        """
        config_key = _CONFIG_KEY_SUFFIXES.get(alert_type)
        if config_key and config_key in config:
            return int(config[config_key])

        return _DEFAULT_WINDOWS.get(alert_type, 60)