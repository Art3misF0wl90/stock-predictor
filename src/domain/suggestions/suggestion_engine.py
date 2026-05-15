"""
SuggestionEngine — generates natural language trading suggestions.

SuggestionEngine takes the current signal, position, sentiment, and
earnings context for a ticker and produces a Suggestion object containing
a short recommendation and a full explanation of the reasoning.

SuggestionEngine is a pure computation component. It reads inputs passed
in by InferenceOrchestrator. It never writes to the database — that is
handled by InferenceOrchestrator via SuggestionRepository.

Staleness check:
    Before generating a suggestion, SuggestionEngine checks whether any
    of the inputs have changed since the last suggestion was generated.
    If nothing has changed, the existing suggestion is still valid and
    no new suggestion is generated (returns None).

    Inputs that trigger a new suggestion when changed:
        - Signal value (BUY/HOLD/SELL changed)
        - Signal timestamp (new inference run produced a signal)
        - Sentiment data (new sentiment records arrived)
        - Transaction history (position changed)

Suggestion decision table:
    Signal  | Position  | Recommendation category
    --------|-----------|------------------------
    BUY     | FLAT      | Consider opening a long position
    BUY     | LONG      | Consider adding to existing long
    BUY     | SHORT     | Consider covering short, potentially going long
    HOLD    | FLAT      | No action — wait for a clearer signal
    HOLD    | LONG      | Hold existing position — signal is neutral
    HOLD    | SHORT     | Hold existing short — signal is neutral
    SELL    | FLAT      | No action (or consider short if intent allows)
    SELL    | LONG      | Consider trimming or closing long position
    SELL    | SHORT     | Consider adding to short position

The explanation is assembled from:
    - Current signal and model confidence
    - Current position direction, shares, and cost basis
    - Position intent (DIRECTIONAL / HEDGE / INCOME)
    - Weighted sentiment score and trend
    - Earnings context if applicable
    - Active filter gates that suppressed the signal (if HOLD via suppression)

Depends on:
    SignalRepository        — checks signal staleness
    SentimentRepository     — fetches sentiment for context
    TransactionRepository   — checks transaction staleness
    SuggestionRepository    — checks suggestion staleness
    Suggestion              — the output dataclass

Exposes:
    generate(inputs) → Suggestion | None
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.utils.types import (
    FilterResult,
    Position,
    PositionDirection,
    PositionIntent,
    Suggestion,
    SignalValue,
)


@dataclass
class SuggestionInputs:
    """
    All inputs SuggestionEngine needs to generate a suggestion.

    Assembled by InferenceOrchestrator before calling SuggestionEngine.

    Attributes:
        symbol: The ticker symbol.
        signal: Current signal value after filtering.
        probability: Model confidence for the signal.
        filter_result: Full FilterResult for gate audit trail.
        position: Current derived position.
        weighted_sentiment: Weighted aggregate sentiment score (-1 to 1).
                            0.0 if no sentiment data exists.
        sentiment_record_count: How many sentiment records contributed.
        has_upcoming_earnings: Whether earnings are in the blackout window.
        earnings_report_date: The upcoming earnings date if applicable.
        last_signal_at: Timestamp of the current signal row.
        last_suggestion_at: Timestamp of the most recent suggestion, or None.
        last_sentiment_at: Timestamp of the most recent sentiment record.
        last_transaction_at: Timestamp of the most recent transaction.
    """
    symbol: str
    signal: SignalValue
    probability: float
    filter_result: FilterResult
    position: Position
    weighted_sentiment: float
    sentiment_record_count: int
    has_upcoming_earnings: bool
    earnings_report_date: datetime | None
    last_signal_at: datetime
    last_suggestion_at: datetime | None
    last_sentiment_at: datetime | None
    last_transaction_at: datetime | None


class SuggestionEngine:
    """
    Generates Suggestion objects from signal and position context.

    Stateless — one instance reused across all inference runs.

    Usage:
        engine = SuggestionEngine()
        suggestion = engine.generate(inputs)
        if suggestion is not None:
            # Persist via SuggestionRepository
    """

    def generate(self, inputs: SuggestionInputs) -> Suggestion | None:
        """
        Generate a suggestion if inputs have changed since the last one.

        Returns None if the staleness check determines the existing
        suggestion is still valid (nothing has changed).

        Args:
            inputs: SuggestionInputs assembled by InferenceOrchestrator.

        Returns:
            A Suggestion object if a new suggestion should be generated.
            None if the existing suggestion is still current.
        """
        if not self._is_stale(inputs):
            return None

        recommendation = self._build_recommendation(inputs)
        explanation = self._build_explanation(inputs)
        sentiment_summary = self._build_sentiment_summary(inputs)
        earnings_context = self._build_earnings_context(inputs)

        return Suggestion(
            symbol=inputs.symbol,
            signal=inputs.signal,
            recommendation=recommendation,
            explanation=explanation,
            position_direction=inputs.position.direction,
            position_intent=inputs.position.intent,
            sentiment_summary=sentiment_summary,
            earnings_context=earnings_context,
            generated_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    def _is_stale(self, inputs: SuggestionInputs) -> bool:
        """
        Determine whether a new suggestion needs to be generated.

        A suggestion is stale (needs regeneration) if:
            - No suggestion has ever been generated (last_suggestion_at is None)
            - The signal timestamp is newer than the last suggestion
            - The most recent sentiment record is newer than the last suggestion
            - The most recent transaction is newer than the last suggestion

        Args:
            inputs: The SuggestionInputs to check.

        Returns:
            True if a new suggestion should be generated.
            False if the existing suggestion is still valid.
        """
        if inputs.last_suggestion_at is None:
            return True

        last = inputs.last_suggestion_at

        if inputs.last_signal_at > last:
            return True

        if inputs.last_sentiment_at and inputs.last_sentiment_at > last:
            return True

        if inputs.last_transaction_at and inputs.last_transaction_at > last:
            return True

        return False

    # ------------------------------------------------------------------
    # Content builders
    # ------------------------------------------------------------------

    def _build_recommendation(self, inputs: SuggestionInputs) -> str:
        """
        Build the short recommendation string from the decision table.

        Returns:
            A short action string like "Consider opening a long position."
        """
        signal = inputs.signal
        direction = inputs.position.direction
        intent = inputs.position.intent

        # HOLD via suppression — gates fired
        if signal == SignalValue.HOLD and inputs.filter_result.gates_triggered:
            gates = ", ".join(inputs.filter_result.gates_triggered)
            return (
                f"Hold — signal suppressed by market condition gates ({gates}). "
                f"No action recommended until conditions normalize."
            )

        if signal == SignalValue.BUY:
            if direction == PositionDirection.FLAT:
                return "Consider opening a long position."
            elif direction == PositionDirection.LONG:
                return "Consider adding to your existing long position."
            else:  # SHORT
                return (
                    "Consider covering your short position. "
                    "A BUY signal while short suggests adverse conditions for the trade."
                )

        elif signal == SignalValue.SELL:
            if direction == PositionDirection.FLAT:
                if intent == PositionIntent.DIRECTIONAL:
                    return "Consider a short position if your strategy permits."
                return "No position to act on — monitor for entry opportunity."
            elif direction == PositionDirection.LONG:
                return "Consider trimming or closing your long position."
            else:  # SHORT
                return "Consider adding to your short position."

        else:  # HOLD
            if direction == PositionDirection.FLAT:
                return "No action — wait for a clearer directional signal."
            elif direction == PositionDirection.LONG:
                return "Hold your existing long position — signal is neutral."
            else:
                return "Hold your existing short position — signal is neutral."

    def _build_explanation(self, inputs: SuggestionInputs) -> str:
        """
        Build the full reasoning explanation.

        Assembles multiple paragraphs covering:
            1. Model signal and confidence
            2. Gate suppression detail (if applicable)
            3. Current position context
            4. Sentiment influence (if meaningful)
            5. Earnings context (if applicable)

        Returns:
            Multi-sentence explanation string.
        """
        parts = []

        # 1 — Signal and confidence
        conf_pct = f"{inputs.probability:.0%}"
        parts.append(
            f"The model produced a {inputs.signal.value} signal with "
            f"{conf_pct} confidence."
        )

        # 2 — Gate suppression detail
        if inputs.filter_result.gates_triggered:
            gates = " and ".join(inputs.filter_result.gates_triggered)
            parts.append(
                f"This signal was suppressed to HOLD by the {gates}. "
                f"All gates evaluated: {', '.join(inputs.filter_result.gates_applied)}."
            )

        # 3 — Position context
        pos = inputs.position
        if pos.direction == PositionDirection.FLAT:
            parts.append("You currently have no open position in this ticker.")
        else:
            parts.append(
                f"You have an open {pos.direction.value} position of "
                f"{pos.shares:.0f} shares at an average cost of "
                f"${pos.average_cost:.2f} per share "
                f"(intent: {pos.intent.value})."
            )

        # 4 — Sentiment (only include if meaningful signal)
        if inputs.sentiment_record_count > 0:
            sentiment_dir = (
                "bullish" if inputs.weighted_sentiment > 0.1
                else "bearish" if inputs.weighted_sentiment < -0.1
                else "neutral"
            )
            parts.append(
                f"Sentiment across {inputs.sentiment_record_count} recent records "
                f"is {sentiment_dir} (weighted score: {inputs.weighted_sentiment:+.2f})."
            )

        # 5 — Earnings
        if inputs.has_upcoming_earnings and inputs.earnings_report_date:
            parts.append(
                f"An earnings report is scheduled for "
                f"{inputs.earnings_report_date.strftime('%b %d, %Y')}. "
                f"Signals are suppressed during the earnings blackout window."
            )

        return " ".join(parts)

    def _build_sentiment_summary(self, inputs: SuggestionInputs) -> str | None:
        """
        Build a short sentiment summary string, or None if no data.

        Returns:
            Short string like "Bullish sentiment (+0.34, 12 records)"
            or None if no sentiment records exist.
        """
        if inputs.sentiment_record_count == 0:
            return None

        direction = (
            "Bullish" if inputs.weighted_sentiment > 0.1
            else "Bearish" if inputs.weighted_sentiment < -0.1
            else "Neutral"
        )

        return (
            f"{direction} sentiment "
            f"({inputs.weighted_sentiment:+.2f}, "
            f"{inputs.sentiment_record_count} records)"
        )

    def _build_earnings_context(self, inputs: SuggestionInputs) -> str | None:
        """
        Build an earnings context string, or None if no upcoming earnings.

        Returns:
            String like "Earnings on May 28, 2026 — blackout active."
            or None if no upcoming earnings.
        """
        if not inputs.has_upcoming_earnings or inputs.earnings_report_date is None:
            return None

        return (
            f"Earnings on "
            f"{inputs.earnings_report_date.strftime('%b %d, %Y')} "
            f"— blackout active."
        )