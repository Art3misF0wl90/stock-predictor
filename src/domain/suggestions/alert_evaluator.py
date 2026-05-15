"""
AlertEvaluator — evaluates current state and produces alert candidates.

AlertEvaluator is a pure function. It receives the current state of
the system — signals, positions, options, earnings events — and produces
a list of AlertCandidate objects describing conditions that warrant
an alert.

AlertEvaluator never reads from the database itself. InferenceOrchestrator
assembles all the inputs and passes them in. AlertEvaluator never writes
to the database. AlertDeduplicator filters the candidates before any are
persisted.

Alert conditions evaluated:
    SIGNAL_CHANGE:
        A ticker's signal has changed since the last recorded signal.
        Severity scales with the change:
            HOLD → BUY or SELL: WARNING
            BUY → SELL or SELL → BUY: URGENT (direction reversal)

    STOP_LOSS:
        An open long position's current price has fallen below the
        configured stop-loss percentage from average cost.
        e.g. if stop_loss_pct = 0.08, alert when price drops 8% below cost.

    PROFIT_TARGET:
        An open long position's current price has risen above the
        configured profit target percentage from average cost.

    EARNINGS_APPROACHING:
        An open position has an earnings event within the configured
        blackout window. Both long and short positions are alerted.

    OPTION_EXPIRING:
        An open option position is expiring within the configured
        alert window (days before expiration).

    PORTFOLIO_COMPOSITION:
        A concentration or correlation warning from PortfolioAnalyzer
        is present. Severity is INFO.

Depends on:
    AlertCandidate      — the output dataclass
    AlertType           — enum for alert categories
    AlertSeverity       — enum for INFO / WARNING / URGENT
    Position            — current equity position state
    SignalValue         — BUY / HOLD / SELL

Exposes:
    evaluate(inputs) → list[AlertCandidate]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.utils.types import (
    AlertCandidate,
    AlertSeverity,
    AlertType,
    ConcentrationWarning,
    CorrelationWarning,
    Position,
    PositionDirection,
    SignalValue,
)


@dataclass
class AlertEvaluatorInputs:
    """
    All inputs AlertEvaluator needs to evaluate alert conditions.

    Assembled by InferenceOrchestrator before calling AlertEvaluator.
    AlertEvaluator is stateless — it only reads from this object.

    Attributes:
        current_signals: Map of symbol → current SignalValue (after filtering).
        previous_signals: Map of symbol → previous SignalValue (from last DB record).
        signal_ids: Map of symbol → UUID of the current signal row (for FK reference).
        positions: Map of symbol → current derived Position.
        current_prices: Map of symbol → most recent close price.
        upcoming_earnings: Set of symbols with earnings within blackout window.
        expiring_option_ids: List of option UUIDs expiring within alert window.
        expiring_option_symbols: Map of option UUID → ticker symbol.
        concentration_warnings: From PortfolioAnalyzer.
        correlation_warnings: From PortfolioAnalyzer.
        config: Dict of alert thresholds from SYSTEM_CONFIG ALERTS category.
    """
    current_signals: dict[str, SignalValue]
    previous_signals: dict[str, SignalValue]
    signal_ids: dict[str, str]
    positions: dict[str, Position]
    current_prices: dict[str, float]
    upcoming_earnings: set[str]
    expiring_option_ids: list[str]
    expiring_option_symbols: dict[str, str]
    concentration_warnings: list[ConcentrationWarning]
    correlation_warnings: list[CorrelationWarning]
    config: dict


class AlertEvaluator:
    """
    Evaluates system state and produces AlertCandidate objects.

    Stateless — one instance can be reused across all inference runs.

    Usage:
        evaluator = AlertEvaluator()
        candidates = evaluator.evaluate(inputs)
        # Pass candidates to AlertDeduplicator before persisting
    """

    def evaluate(self, inputs: AlertEvaluatorInputs) -> list[AlertCandidate]:
        """
        Evaluate all alert conditions and return candidates.

        Args:
            inputs: AlertEvaluatorInputs assembled by InferenceOrchestrator.

        Returns:
            List of AlertCandidate objects. May be empty if no conditions
            are triggered. May contain multiple candidates for the same
            ticker if multiple conditions apply simultaneously.
        """
        candidates: list[AlertCandidate] = []

        candidates.extend(self._check_signal_changes(inputs))
        candidates.extend(self._check_stop_loss(inputs))
        candidates.extend(self._check_profit_target(inputs))
        candidates.extend(self._check_earnings_approaching(inputs))
        candidates.extend(self._check_options_expiring(inputs))
        candidates.extend(self._check_portfolio_composition(inputs))

        return candidates

    # ------------------------------------------------------------------
    # Individual condition checks
    # ------------------------------------------------------------------

    def _check_signal_changes(
        self, inputs: AlertEvaluatorInputs
    ) -> list[AlertCandidate]:
        """
        Check for signal changes since the last recorded signal.

        A signal change alert fires when the current signal differs from
        the previous signal for a ticker. The severity depends on the
        nature of the change:
            HOLD → BUY or SELL: WARNING (new directional signal)
            BUY → SELL or SELL → BUY: URGENT (direction reversal)
            BUY → HOLD or SELL → HOLD: INFO (signal neutralized)
        """
        candidates = []

        for symbol, current in inputs.current_signals.items():
            previous = inputs.previous_signals.get(symbol)

            if previous is None or current == previous:
                continue

            # Determine severity
            if (
                (previous == SignalValue.BUY and current == SignalValue.SELL)
                or (previous == SignalValue.SELL and current == SignalValue.BUY)
            ):
                severity = AlertSeverity.URGENT
                change_desc = f"direction reversal: {previous.value} → {current.value}"
            elif previous == SignalValue.HOLD:
                severity = AlertSeverity.WARNING
                change_desc = f"new signal: {previous.value} → {current.value}"
            else:
                severity = AlertSeverity.INFO
                change_desc = f"signal change: {previous.value} → {current.value}"

            candidates.append(AlertCandidate(
                symbol=symbol,
                alert_type=AlertType.SIGNAL_CHANGE,
                severity=severity,
                message=(
                    f"{symbol} signal {change_desc}. "
                    f"Check the dashboard for the full suggestion."
                ),
                triggering_signal_id=inputs.signal_ids.get(symbol),
            ))

        return candidates

    def _check_stop_loss(
        self, inputs: AlertEvaluatorInputs
    ) -> list[AlertCandidate]:
        """
        Check if any open long position has breached its stop-loss level.

        Stop-loss threshold is read from config as a decimal:
            stop_loss_pct = 0.08 means alert when price is 8% below cost.

        Only applies to LONG positions — short positions have different
        risk profiles and are not currently covered by this gate.
        """
        candidates = []
        stop_loss_pct = inputs.config.get("stop_loss_pct", 0.08)

        for symbol, position in inputs.positions.items():
            if position.direction != PositionDirection.LONG:
                continue
            if position.average_cost == 0:
                continue

            current_price = inputs.current_prices.get(symbol)
            if current_price is None:
                continue

            loss_pct = (position.average_cost - current_price) / position.average_cost

            if loss_pct >= stop_loss_pct:
                candidates.append(AlertCandidate(
                    symbol=symbol,
                    alert_type=AlertType.STOP_LOSS,
                    severity=AlertSeverity.URGENT,
                    message=(
                        f"{symbol} is down {loss_pct:.1%} from your average "
                        f"cost of ${position.average_cost:.2f}. "
                        f"Current price: ${current_price:.2f}. "
                        f"Stop-loss threshold: {stop_loss_pct:.1%}."
                    ),
                    triggering_signal_id=None,
                ))

        return candidates

    def _check_profit_target(
        self, inputs: AlertEvaluatorInputs
    ) -> list[AlertCandidate]:
        """
        Check if any open long position has reached its profit target.

        Profit target threshold is read from config as a decimal:
            profit_target_pct = 0.20 means alert when price is 20% above cost.
        """
        candidates = []
        profit_target_pct = inputs.config.get("profit_target_pct", 0.20)

        for symbol, position in inputs.positions.items():
            if position.direction != PositionDirection.LONG:
                continue
            if position.average_cost == 0:
                continue

            current_price = inputs.current_prices.get(symbol)
            if current_price is None:
                continue

            gain_pct = (current_price - position.average_cost) / position.average_cost

            if gain_pct >= profit_target_pct:
                candidates.append(AlertCandidate(
                    symbol=symbol,
                    alert_type=AlertType.PROFIT_TARGET,
                    severity=AlertSeverity.WARNING,
                    message=(
                        f"{symbol} is up {gain_pct:.1%} from your average "
                        f"cost of ${position.average_cost:.2f}. "
                        f"Current price: ${current_price:.2f}. "
                        f"Profit target: {profit_target_pct:.1%}."
                    ),
                    triggering_signal_id=None,
                ))

        return candidates

    def _check_earnings_approaching(
        self, inputs: AlertEvaluatorInputs
    ) -> list[AlertCandidate]:
        """
        Alert for open positions with earnings events approaching.

        Only alerts for tickers where there is an open position — no
        point alerting about earnings for tickers where you have no exposure.
        """
        candidates = []

        for symbol in inputs.upcoming_earnings:
            position = inputs.positions.get(symbol)
            if position is None or position.direction == PositionDirection.FLAT:
                continue

            candidates.append(AlertCandidate(
                symbol=symbol,
                alert_type=AlertType.EARNINGS_APPROACHING,
                severity=AlertSeverity.WARNING,
                message=(
                    f"{symbol} has an upcoming earnings report. "
                    f"You have an open {position.direction.value} position "
                    f"of {position.shares:.0f} shares. "
                    f"Signals are suppressed during earnings blackout."
                ),
                triggering_signal_id=None,
            ))

        return candidates

    def _check_options_expiring(
        self, inputs: AlertEvaluatorInputs
    ) -> list[AlertCandidate]:
        """
        Alert for open option positions approaching expiration.
        """
        candidates = []

        for option_id in inputs.expiring_option_ids:
            symbol = inputs.expiring_option_symbols.get(option_id, "UNKNOWN")

            candidates.append(AlertCandidate(
                symbol=symbol,
                alert_type=AlertType.OPTION_EXPIRING,
                severity=AlertSeverity.WARNING,
                message=(
                    f"An option position in {symbol} is approaching expiration. "
                    f"Review and decide whether to close, roll, or let expire."
                ),
                triggering_signal_id=None,
            ))

        return candidates

    def _check_portfolio_composition(
        self, inputs: AlertEvaluatorInputs
    ) -> list[AlertCandidate]:
        """
        Convert PortfolioAnalyzer warnings into AlertCandidates.
        """
        candidates = []

        for warning in inputs.concentration_warnings:
            candidates.append(AlertCandidate(
                symbol=warning.symbol,
                alert_type=AlertType.PORTFOLIO_COMPOSITION,
                severity=AlertSeverity.INFO,
                message=warning.message,
                triggering_signal_id=None,
            ))

        for warning in inputs.correlation_warnings:
            candidates.append(AlertCandidate(
                symbol=warning.symbol_a,
                alert_type=AlertType.PORTFOLIO_COMPOSITION,
                severity=AlertSeverity.INFO,
                message=warning.message,
                triggering_signal_id=None,
            ))

        return candidates