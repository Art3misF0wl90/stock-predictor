"""
SignalFilter — applies market condition gates to raw model predictions.

SignalFilter is a pure computation component. It receives a raw signal
value, a probability, and a MarketContext, applies a set of configurable
gates, and returns a FilterResult describing the outcome.

SignalFilter never fetches data. It never writes to the database.
InferencePipeline assembles the MarketContext and calls filter() once
per inference run.

What "suppression" means:
    If any gate triggers, the raw signal is replaced with HOLD.
    The original probability is preserved in FilterResult — it reflects
    the model's confidence regardless of suppression. This matters for
    the chatbot, which can say "the model was 82% confident in a BUY,
    but it was suppressed due to high VIX."

Gates implemented:
    1. VIX gate    — suppress if VIX is above a configured threshold.
                     High VIX indicates elevated market fear/uncertainty.
                     Model predictions are less reliable in high-fear environments.

    2. RSI gate    — suppress BUY if RSI > overbought threshold.
                     Suppress SELL if RSI < oversold threshold.
                     Prevents signals that go against strong short-term momentum.

    3. Earnings blackout gate — suppress if an earnings report is scheduled
                                within the configured blackout window.
                                Earnings introduce binary risk that ML models
                                cannot reliably predict.

    4. Probability gate — suppress if the model's confidence is below the
                          configured minimum probability threshold.
                          Low-confidence predictions are not actionable.

All thresholds are read from SYSTEM_CONFIG SIGNALS category. Gates can be
individually enabled or disabled via SYSTEM_CONFIG.

Depends on:
    MarketContext       — input assembled by InferencePipeline
    FilterResult        — output dataclass
    SignalValue         — BUY / HOLD / SELL enum
    SignalFilterError   — raised if MarketContext is malformed

Exposes:
    filter(raw_signal, probability, context, config) → FilterResult
"""

from __future__ import annotations

from src.utils.exceptions import SignalFilterError
from src.utils.types import FilterResult, MarketContext, SignalValue


class SignalFilter:
    """
    Applies configurable market condition gates to raw model signals.

    SignalFilter is stateless — it holds no configuration itself.
    All thresholds and gate enable/disable flags are passed in via
    the config dict at call time. This makes it easy to test with
    different configurations without needing a database connection.

    Usage:
        signal_filter = SignalFilter()
        result = signal_filter.filter(
            raw_signal=SignalValue.BUY,
            probability=0.82,
            context=market_context,
            config=config_dict,
        )

        if result.gates_triggered:
            # Signal was suppressed — result.signal is HOLD
        else:
            # Signal passed all gates — result.signal == raw_signal
    """

    def filter(
        self,
        raw_signal: SignalValue,
        probability: float,
        context: MarketContext,
        config: dict,
    ) -> FilterResult:
        """
        Apply all enabled gates and return the final signal with audit info.

        Args:
            raw_signal: The signal value produced by the model (BUY/HOLD/SELL).
            probability: Model confidence for the predicted class (0.0 to 1.0).
            context: MarketContext assembled by InferencePipeline. Contains
                     VIX value, RSI value, price trend, and earnings flag.
            config: Dict of gate configuration values read from SYSTEM_CONFIG
                    SIGNALS category. Expected keys:
                        "vix_gate_enabled"          bool
                        "vix_suppression_threshold"  float
                        "rsi_gate_enabled"           bool
                        "rsi_overbought_threshold"   float (default 70.0)
                        "rsi_oversold_threshold"     float (default 30.0)
                        "earnings_gate_enabled"      bool
                        "prob_gate_enabled"          bool
                        "min_probability_threshold"  float (default 0.55)

        Returns:
            FilterResult with:
                signal          — final signal after gate evaluation
                probability     — original model probability (unchanged)
                gates_applied   — names of all gates that were checked
                gates_triggered — names of gates that caused suppression

        Raises:
            SignalFilterError: If a required context field is None when
                its gate is enabled. E.g. vix_value is None but VIX gate
                is enabled. This indicates InferencePipeline failed to
                assemble a complete MarketContext.
        """
        gates_applied: list[str] = []
        gates_triggered: list[str] = []

        # Run each gate — each appends to gates_applied and optionally
        # to gates_triggered if the condition is met.
        self._apply_vix_gate(context, config, gates_applied, gates_triggered)
        self._apply_rsi_gate(raw_signal, context, config, gates_applied, gates_triggered)
        self._apply_earnings_gate(context, config, gates_applied, gates_triggered)
        self._apply_probability_gate(probability, config, gates_applied, gates_triggered)

        # Determine final signal
        if gates_triggered:
            final_signal = SignalValue.HOLD
        else:
            final_signal = raw_signal

        return FilterResult(
            signal=final_signal,
            probability=probability,
            gates_applied=gates_applied,
            gates_triggered=gates_triggered,
        )

    # ------------------------------------------------------------------
    # Individual gate implementations
    # ------------------------------------------------------------------

    def _apply_vix_gate(
        self,
        context: MarketContext,
        config: dict,
        gates_applied: list[str],
        gates_triggered: list[str],
    ) -> None:
        """
        VIX gate — suppress all non-HOLD signals when VIX is elevated.

        High VIX means the options market is pricing in large near-term
        price swings. During these periods, directional signals from ML
        models are less reliable because the dominant driver of price
        movement is fear/uncertainty rather than the fundamental and
        technical patterns the model was trained on.

        Trigger condition:
            VIX gate is enabled AND context.vix_value > vix_threshold

        Both BUY and SELL are suppressed — during extreme fear, going
        short is also unreliable because volatility reversals are common.
        """
        gate_name = "vix_gate"

        if not config.get("vix_gate_enabled", True):
            return  # Gate disabled — skip entirely, don't add to gates_applied

        gates_applied.append(gate_name)

        if context.vix_value is None:
            raise SignalFilterError(
                f"MarketContext.vix_value is None for {context.symbol} "
                f"but the VIX gate is enabled. InferencePipeline must "
                f"populate vix_value before calling SignalFilter."
            )

        threshold = config.get("vix_suppression_threshold", 30.0)

        if context.vix_value > threshold:
            gates_triggered.append(gate_name)

    def _apply_rsi_gate(
        self,
        raw_signal: SignalValue,
        context: MarketContext,
        config: dict,
        gates_applied: list[str],
        gates_triggered: list[str],
    ) -> None:
        """
        RSI gate — suppress signals that go against extreme RSI readings.

        RSI measures short-term momentum. When RSI is extremely high
        (overbought), a BUY signal from the model is fighting against
        the reality that the stock has already run up significantly and
        a pullback is statistically more likely. Suppressing that BUY
        improves signal quality.

        Trigger conditions:
            BUY signal AND RSI > rsi_overbought_threshold (default 70)
            SELL signal AND RSI < rsi_oversold_threshold (default 30)

        HOLD signals are never suppressed by the RSI gate — they are
        already neutral.
        """
        gate_name = "rsi_gate"

        if not config.get("rsi_gate_enabled", True):
            return

        gates_applied.append(gate_name)

        if context.rsi_value is None:
            raise SignalFilterError(
                f"MarketContext.rsi_value is None for {context.symbol} "
                f"but the RSI gate is enabled."
            )

        overbought = config.get("rsi_overbought_threshold", 70.0)
        oversold = config.get("rsi_oversold_threshold", 30.0)

        if raw_signal == SignalValue.BUY and context.rsi_value > overbought:
            gates_triggered.append(gate_name)
        elif raw_signal == SignalValue.SELL and context.rsi_value < oversold:
            gates_triggered.append(gate_name)

    def _apply_earnings_gate(
        self,
        context: MarketContext,
        config: dict,
        gates_applied: list[str],
        gates_triggered: list[str],
    ) -> None:
        """
        Earnings blackout gate — suppress all signals near earnings reports.

        Earnings reports introduce binary, high-magnitude risk that ML
        models trained on historical price patterns cannot reliably predict.
        A stock can move 10-20% on an earnings surprise regardless of
        what the technical indicators say.

        The blackout window is configured in SYSTEM_CONFIG SIGNALS category
        as "earnings_blackout_days". If an upcoming earnings event is within
        that many trading days, all directional signals are suppressed.

        Trigger condition:
            Earnings gate enabled AND context.has_upcoming_earnings is True

        context.has_upcoming_earnings is set by InferencePipeline by checking
        EarningsRepository for upcoming events within the blackout window.
        """
        gate_name = "earnings_gate"

        if not config.get("earnings_gate_enabled", True):
            return

        gates_applied.append(gate_name)

        if context.has_upcoming_earnings:
            gates_triggered.append(gate_name)

    def _apply_probability_gate(
        self,
        probability: float,
        config: dict,
        gates_applied: list[str],
        gates_triggered: list[str],
    ) -> None:
        """
        Probability gate — suppress low-confidence predictions.

        When the model's predicted probability is only marginally above
        50%, the signal is not meaningfully different from random. Acting
        on a 52% confidence prediction is not sensible. The minimum
        probability threshold filters these out.

        The threshold is configurable in SYSTEM_CONFIG SIGNALS category
        as "min_probability_threshold". Default is 0.55 — at least 55%
        model confidence is required for a signal to pass.

        Trigger condition:
            Probability gate enabled AND probability < min_probability_threshold
        """
        gate_name = "probability_gate"

        if not config.get("prob_gate_enabled", True):
            return

        gates_applied.append(gate_name)

        min_prob = config.get("min_probability_threshold", 0.55)

        if probability < min_prob:
            gates_triggered.append(gate_name)