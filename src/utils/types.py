"""
Shared types for stock_predictor_v2.

All enums, dataclasses, and typed value objects used across more than one
layer live here. This module has no imports from anywhere else in the project —
it is the bottom of the dependency graph alongside exceptions.py.

Conventions:
    - Enums use string values so SQLAlchemy can store them directly without
      a separate mapping step. str(MyEnum.VALUE) == "VALUE".
    - Dataclasses use frozen=True where the object should never be mutated
      after creation (pure computation results, filter results, etc.).
    - Optional fields that are genuinely nullable use `| None` with a default
      of None. Fields that must always be provided have no default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums — stored as strings in SQLite via SQLAlchemy
# ---------------------------------------------------------------------------


class SignalValue(str, Enum):
    """Possible values for a generated trading signal."""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class OnboardingStatus(str, Enum):
    """
    Lifecycle states for ticker onboarding.

    Valid transitions are enforced by TickerOnboardingPipeline:
        PENDING   → FETCHING  (validation passed)
        PENDING   → FAILED    (validation failed)
        FETCHING  → ANALYZING (fetch succeeded)
        FETCHING  → FAILED    (DataFetchError)
        ANALYZING → TRAINING  (macro analysis + feature engineering succeeded)
        ANALYZING → FAILED    (analysis error)
        TRAINING  → COMPLETE  (training accepted, ConfigStore.load() succeeds)
        TRAINING  → FAILED    (first-train rejection or TrainingError)
        FAILED    → PENDING   (manual retry)
        COMPLETE  → PENDING   (retrain triggered — temporary)
    """
    PENDING = "PENDING"
    FETCHING = "FETCHING"
    ANALYZING = "ANALYZING"
    TRAINING = "TRAINING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class AlertStatus(str, Enum):
    """
    Lifecycle states for an alert.

    Permanently failed alerts (retry_count >= max_attempts) stay in the
    database as audit records. They are never silently dropped.

        PENDING      → DELIVERED    (delivery succeeds)
        PENDING      → FAILED       (delivery fails, retry_count=1)
        PENDING      → ACKNOWLEDGED (user dismisses before delivery)
        DELIVERED    → ACKNOWLEDGED (user dismisses)
        FAILED       → DELIVERED    (retry succeeds)
        FAILED       → FAILED       (retry fails, retry_count incremented)
        ACKNOWLEDGED → terminal (no further transitions)
    """
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class AlertType(str, Enum):
    """Categories of alert that the system can generate."""
    SIGNAL_CHANGE = "SIGNAL_CHANGE"
    STOP_LOSS = "STOP_LOSS"
    PROFIT_TARGET = "PROFIT_TARGET"
    PORTFOLIO_COMPOSITION = "PORTFOLIO_COMPOSITION"
    EARNINGS_APPROACHING = "EARNINGS_APPROACHING"
    OPTION_EXPIRING = "OPTION_EXPIRING"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    URGENT = "URGENT"


class TransactionType(str, Enum):
    """
    Direction-encoded transaction types for equity positions.

    BUY_LONG  — open or add to a long position
    SELL_LONG — close or reduce a long position
    SELL_SHORT — open or add to a short position
    BUY_SHORT  — close or reduce a short position (cover)
    """
    BUY_LONG = "BUY_LONG"
    SELL_LONG = "SELL_LONG"
    SELL_SHORT = "SELL_SHORT"
    BUY_SHORT = "BUY_SHORT"


class PositionDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class PositionIntent(str, Enum):
    DIRECTIONAL = "DIRECTIONAL"
    HEDGE = "HEDGE"
    INCOME = "INCOME"
    NONE = "NONE"


class ContractType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class OptionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class MacroCategory(str, Enum):
    VOLATILITY = "VOLATILITY"
    CURRENCY = "CURRENCY"
    COMMODITY = "COMMODITY"
    INDEX = "INDEX"
    SECTOR_ETF = "SECTOR_ETF"


class ConfigDataType(str, Enum):
    """How to cast a SYSTEM_CONFIG value string when reading it."""
    FLOAT = "FLOAT"
    INT = "INT"
    STRING = "STRING"
    BOOL = "BOOL"


class ConfigCategory(str, Enum):
    """
    SYSTEM_CONFIG categories. Each has its own version counter.
    Modules cache their config values and only re-read when the version
    for their category has changed.
    """
    MACRO = "MACRO"
    TRAINING = "TRAINING"
    SIGNALS = "SIGNALS"
    PORTFOLIO = "PORTFOLIO"
    ALERTS = "ALERTS"
    SCHEDULER = "SCHEDULER"


class SnapshotTrigger(str, Enum):
    TRANSACTION = "TRANSACTION"
    SCHEDULED = "SCHEDULED"


# ---------------------------------------------------------------------------
# Value objects — frozen dataclasses (pure data, never mutated after creation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    """
    Derived equity position for one ticker.

    Computed by PositionDeriver from transaction history. Never stored
    directly in the database — always derived on demand.

    Attributes:
        symbol: Ticker symbol.
        direction: LONG, SHORT, or FLAT.
        shares: Absolute number of shares held. Zero when FLAT.
        average_cost: Weighted average cost basis per share. Zero when FLAT.
        intent: How this position is being used. NONE when FLAT.
        last_transaction_at: Timestamp of the most recent transaction
            that affects this position. None when FLAT (no history).
    """
    symbol: str
    direction: PositionDirection
    shares: float
    average_cost: float
    intent: PositionIntent
    last_transaction_at: datetime | None


@dataclass(frozen=True)
class MarketContext:
    """
    Pre-assembled market context passed to SignalFilter.

    SignalFilter is a pure function — it receives this object and never
    fetches anything itself. InferencePipeline assembles this before calling
    SignalFilter.filter().

    Attributes:
        symbol: Ticker being evaluated.
        vix_value: Current VIX level. Required if VIX gate is enabled in config.
        rsi_value: Current RSI (14-period). Required if RSI gate is enabled.
        price_trend: Short-term price trend direction ('up', 'down', 'flat').
        has_upcoming_earnings: True if an earnings event is within the
            blackout window defined in SYSTEM_CONFIG SIGNALS category.
    """
    symbol: str
    vix_value: float | None
    rsi_value: float | None
    price_trend: str | None
    has_upcoming_earnings: bool


@dataclass(frozen=True)
class FilterResult:
    """
    Output of SignalFilter.filter().

    Suppression to HOLD is a valid result, not an error. Check
    gates_triggered to understand why a signal was suppressed.

    Attributes:
        signal: Final signal after gate evaluation (may differ from raw).
        probability: Model probability for the final signal class.
        gates_applied: Names of all gates that were evaluated.
        gates_triggered: Names of gates that caused suppression, if any.
            Empty list means no suppression occurred.
    """
    signal: SignalValue
    probability: float
    gates_applied: list[str]
    gates_triggered: list[str]


@dataclass(frozen=True)
class PredictionResult:
    """
    Output of InferencePipeline.predict().

    Returned to InferenceOrchestrator, which decides whether to persist
    the signal. InferencePipeline never writes to the database.

    Attributes:
        symbol: Ticker this prediction is for.
        signal: Final signal value after filtering.
        probability: Model probability for the signal class.
        fwd_days: Forward horizon this model was trained on.
        model_name: Identifier of the model that produced the raw prediction.
        filter_result: Full filter output for audit trail.
        predicted_at: When inference was run.
    """
    symbol: str
    signal: SignalValue
    probability: float
    fwd_days: int
    model_name: str
    filter_result: FilterResult
    predicted_at: datetime


@dataclass(frozen=True)
class TrainingResult:
    """
    Output of Trainer.train().

    A rejected model is a valid TrainingResult with accepted=False.
    Trainer never raises TrainingError for rejection — only for unexpected
    failures (OOM, corrupted data, etc.).

    Attributes:
        symbol: Ticker this training run was for.
        accepted: True if AUC improved by at least the configured minimum margin.
        auc_before: AUC of the previously accepted model. None on first train.
        auc_after: AUC of the newly trained model.
        n_train_rows: Number of rows the model was trained on.
        rejection_reason: Human-readable reason if accepted=False. None if accepted.
        config_path: Path where the new config was written (accepted or rejected).
            Rejected configs are written with a timestamp suffix for rollback.
        trained_at: When training completed.
    """
    symbol: str
    accepted: bool
    auc_before: float | None
    auc_after: float
    n_train_rows: int
    rejection_reason: str | None
    config_path: str
    trained_at: datetime


@dataclass(frozen=True)
class MacroRelevanceProfile:
    """
    Output of MacroCorrelationAnalyzer.analyze().

    Contains the selected macro symbols for a ticker after tiered selection:
        - If max correlation > threshold: top 5 correlated macros
        - If max correlation <= threshold: top 3 correlated macros
        - Plus 4 unconditionals always included: VIX, DXY, GLD, SLV

    Attributes:
        symbol: Ticker this profile applies to.
        selected_macros: Final list of macro symbols to use as features.
        correlation_scores: Raw correlation score for each candidate macro.
        tier: 3 or 5 — which tier threshold was applied for conditional macros.
        computed_at: When the analysis was run.
    """
    symbol: str
    selected_macros: list[str]
    correlation_scores: dict[str, float]
    tier: int
    computed_at: datetime


@dataclass(frozen=True)
class ValidationResult:
    """
    Output of TickerValidator.validate().

    TickerValidator never raises — invalid and insufficient are valid outcomes.

    Attributes:
        symbol: Symbol that was validated.
        is_valid: True if the symbol exists and data is accessible.
        is_eligible: True if enough historical data exists for training.
        reason: Human-readable explanation when is_valid or is_eligible is False.
    """
    symbol: str
    is_valid: bool
    is_eligible: bool
    reason: str | None


@dataclass(frozen=True)
class AlertCandidate:
    """
    A potential alert produced by AlertEvaluator.

    AlertEvaluator is a pure function — it produces candidates without
    knowing which alerts already exist. AlertDeduplicator filters these
    against existing alerts before any are written to the database.

    Attributes:
        symbol: Ticker this alert is about.
        alert_type: Category of the alert condition.
        severity: How urgent this alert is.
        message: Human-readable description of the condition.
        triggering_signal_id: UUID of the signal that caused this alert,
            if the alert_type is SIGNAL_CHANGE. None for other types.
    """
    symbol: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    triggering_signal_id: str | None


@dataclass(frozen=True)
class ConcentrationWarning:
    """
    A portfolio concentration concern from PortfolioAnalyzer.

    Attributes:
        symbol: Ticker or sector that is over-concentrated.
        weight: Current portfolio weight (0.0 to 1.0).
        threshold: Configured maximum weight that was exceeded.
        message: Human-readable description.
    """
    symbol: str
    weight: float
    threshold: float
    message: str


@dataclass(frozen=True)
class CorrelationWarning:
    """
    A portfolio correlation concern from PortfolioAnalyzer.

    Attributes:
        symbol_a: First ticker in the correlated pair.
        symbol_b: Second ticker in the correlated pair.
        correlation: Pearson correlation coefficient between the two.
        threshold: Configured maximum correlation that was exceeded.
        message: Human-readable description.
    """
    symbol_a: str
    symbol_b: str
    correlation: float
    threshold: float
    message: str


@dataclass
class PortfolioAnalysis:
    """
    Output of PortfolioAnalyzer.analyze().

    Mutable dataclass — PortfolioAnalyzer builds this incrementally.
    Not frozen because callers may annotate it before passing it to
    AlertEvaluator or SuggestionEngine.

    Attributes:
        positions: Current derived position for every tracked ticker.
        sector_breakdown: Sector name → portfolio weight (0.0 to 1.0).
        total_invested: Sum of cost basis across all open positions.
        concentration_warnings: Tickers or sectors over the weight threshold.
        correlation_warnings: Pairs of tickers with excessive correlation.
        analyzed_at: When this analysis was computed.
    """
    positions: dict[str, Position]
    sector_breakdown: dict[str, float]
    total_invested: float
    concentration_warnings: list[ConcentrationWarning]
    correlation_warnings: list[CorrelationWarning]
    analyzed_at: datetime


@dataclass(frozen=True)
class Suggestion:
    """
    A recommendation produced by SuggestionEngine.

    SuggestionEngine never writes — InferenceOrchestrator persists this
    via SuggestionRepository if the staleness check determines it is new.

    Attributes:
        symbol: Ticker this suggestion is for.
        signal: The signal value that drove this suggestion.
        recommendation: Short action string (e.g. "Consider trimming long position").
        explanation: Full reasoning, citing signal, position, intent, sentiment,
            and earnings context as applicable.
        position_direction: Direction of the current position. NONE if flat.
        position_intent: Intent of the current position. NONE if flat.
        sentiment_summary: Summary of sentiment influence, if applicable.
        earnings_context: Earnings context that influenced the suggestion, if applicable.
        generated_at: When this suggestion was produced.
    """
    symbol: str
    signal: SignalValue
    recommendation: str
    explanation: str
    position_direction: PositionDirection
    position_intent: PositionIntent
    sentiment_summary: str | None
    earnings_context: str | None
    generated_at: datetime


@dataclass(frozen=True)
class OnboardingResult:
    """
    Output of TickerOnboardingPipeline.run().

    run() never raises — failure is always a valid OnboardingResult with
    success=False and a reason explaining where the pipeline stopped.

    Attributes:
        symbol: Ticker that was onboarded.
        success: True if the ticker reached COMPLETE status.
        final_status: The OnboardingStatus the ticker ended in.
        reason: Explanation of failure if success=False. None if success.
    """
    symbol: str
    success: bool
    final_status: OnboardingStatus
    reason: str | None


@dataclass(frozen=True)
class RetrainResult:
    """
    Output of RetrainOrchestrator.retrain().

    retrain() never raises — failure is a valid result.

    Attributes:
        symbol: Ticker that was retrained.
        training_result: Full TrainingResult from Trainer. None if training
            could not be reached (e.g. data fetch failed).
        inference_run: True if a fresh inference run was triggered after
            an accepted retrain. False on rejection or failure.
        reason: Explanation if training could not be reached.
    """
    symbol: str
    training_result: TrainingResult | None
    inference_run: bool
    reason: str | None


@dataclass
class InferenceOrchestratorResult:
    """
    Output of InferenceOrchestrator.run_for_ticker().

    Mutable because run_for_all() builds a list of these incrementally.

    Attributes:
        symbol: Ticker this result is for.
        success: True if prediction and persistence completed without error.
        prediction: The PredictionResult, if inference succeeded.
        alerts_created: Number of net-new alerts written to the database.
        suggestion_created: True if a new suggestion was generated and persisted.
        error: The exception that caused failure, if success=False.
    """
    symbol: str
    success: bool
    prediction: PredictionResult | None
    alerts_created: int
    suggestion_created: bool
    error: Exception | None = None


@dataclass(frozen=True)
class TickerConfig:
    """
    Per-ticker model configuration written by Trainer and read by ConfigStore.

    This is the object that gets serialized to config.pkl. Schema version 2.

    The feat_cols field is the architectural invariant that fixes the v1 bug:
    feat_cols is locked at train time and inference NEVER recomputes it.
    ConfigStore validates this object on every load.

    Attributes:
        symbol: Ticker this config belongs to.
        schema_version: Must equal ConfigStore.CURRENT_SCHEMA_VERSION (2).
            If it doesn't, ConfigStore raises ConfigSchemaError.
        feat_cols: Ordered list of feature column names used during training.
            Inference passes this list to FeatureEngineer exactly as stored.
        fwd_days: Forward horizon (in trading days) this model predicts.
        model_name: Identifier for the model type (e.g. 'xgboost', 'logreg').
        auc: AUC score of this model on the holdout set at train time.
        trained_at: When this config was written.
        macro_symbols: Macro indicators included as features for this ticker.
            Stored so we can detect if the macro profile has changed.
        extra: Catch-all dict for model-specific hyperparameters or metadata
            that don't fit the fixed fields. Always serialized, may be empty.
    """
    symbol: str
    schema_version: int
    feat_cols: list[str]
    fwd_days: int
    model_name: str
    auc: float
    trained_at: datetime
    macro_symbols: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatbotResponse:
    """
    Output of ChatbotService.respond().

    Attributes:
        reply: The natural language response from the LLM.
        symbol: Symbol the response was scoped to, if provided.
        context_summary: Brief description of what context was assembled
            (for debugging and logging — not shown to the user).
    """
    reply: str
    symbol: str | None
    context_summary: str