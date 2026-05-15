"""
Database setup for stock_predictor_v2.

Responsibilities:
    1. Create and manage the SQLAlchemy engine (connection to SQLite file).
    2. Provide a session factory so repositories can get database sessions.
    3. Define all 15 table schemas as ORM models.
    4. Run migrations — create tables if they don't exist.

How SQLAlchemy works (brief):
    Engine      — knows HOW to talk to the database (driver, file path, pool).
    Session     — a unit of work. You read/write through a session, then
                  commit() to persist or rollback() to undo.
    Base        — the declarative base class all ORM models inherit from.
                  When you call Base.metadata.create_all(engine), SQLAlchemy
                  looks at every class that inherits from Base and creates
                  the corresponding table if it doesn't already exist.

Usage (startup only — repositories receive a session, not the Database object):
    db = Database.init("data/stock_predictor.db")
    db.migrate()
    session = db.get_session()
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from src.utils.types import (
    AlertSeverity,
    AlertStatus,
    AlertType,
    ConfigCategory,
    ConfigDataType,
    ContractType,
    MacroCategory,
    OnboardingStatus,
    OptionStatus,
    PositionDirection,
    PositionIntent,
    SignalValue,
    SnapshotTrigger,
    TransactionType,
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Parent class for all ORM models.

    Every table definition inherits from this. SQLAlchemy uses it to track
    all models and generate the correct CREATE TABLE statements.
    """
    pass


# ---------------------------------------------------------------------------
# Helper — generate UUID strings as primary keys
# ---------------------------------------------------------------------------


def _new_uuid() -> str:
    """Return a new UUID4 as a plain string. Used as default for PK columns."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# ORM models — one class per table, matching the 15 entities in the schema
# ---------------------------------------------------------------------------


class TickerModel(Base):
    """
    TICKER table — one row per tracked stock symbol.

    onboarding_status drives the TickerOnboardingPipeline state machine.
    training_eligible and min_data_met are set during validation and used
    by the scheduler to decide which tickers to retrain.
    """
    __tablename__ = "ticker"

    symbol: Mapped[str] = mapped_column(String(10), primary_key=True)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(50), nullable=True)
    has_earnings: Mapped[bool] = mapped_column(Boolean, default=False)
    on_watchlist: Mapped[bool] = mapped_column(Boolean, default=True)
    training_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    min_data_met: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarding_status: Mapped[str] = mapped_column(
        Enum(OnboardingStatus, values_callable=lambda x: [e.value for e in x]),
        default=OnboardingStatus.PENDING.value,
    )
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    transactions: Mapped[list[TransactionModel]] = relationship(
        "TransactionModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    signals: Mapped[list[SignalModel]] = relationship(
        "SignalModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    sentiment_records: Mapped[list[SentimentRecordModel]] = relationship(
        "SentimentRecordModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    earnings_events: Mapped[list[EarningsEventModel]] = relationship(
        "EarningsEventModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    alerts: Mapped[list[AlertModel]] = relationship(
        "AlertModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    model_audits: Mapped[list[ModelAuditModel]] = relationship(
        "ModelAuditModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    suggestions: Mapped[list[SuggestionLogModel]] = relationship(
        "SuggestionLogModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    macro_relevance: Mapped[list[MacroRelevanceModel]] = relationship(
        "MacroRelevanceModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )
    options: Mapped[list[OptionPositionModel]] = relationship(
        "OptionPositionModel", back_populates="ticker_ref", cascade="all, delete-orphan"
    )


class TransactionModel(Base):
    """
    TRANSACTION table — every equity buy/sell event.

    Current position state is ALWAYS derived from this history by
    PositionDeriver. It is never stored directly. This means partial
    closes, averaging down/up, and direction changes are all handled
    naturally — you just keep appending rows.
    """
    __tablename__ = "transaction"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    type: Mapped[str] = mapped_column(
        Enum(TransactionType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    price_per_share: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(
        Enum(PositionDirection, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    position_intent: Mapped[str] = mapped_column(
        Enum(PositionIntent, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    executed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    ticker_ref: Mapped[TickerModel] = relationship("TickerModel", back_populates="transactions")


class OptionPositionModel(Base):
    """
    OPTION_POSITION table — standalone options contracts.

    Options are fully independent of equity transactions in the same
    underlying. No FK to TransactionModel. Closed via close_premium
    and closed_at when the position is exited.
    """
    __tablename__ = "option_position"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    contract_type: Mapped[str] = mapped_column(
        Enum(ContractType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    position_direction: Mapped[str] = mapped_column(
        Enum(PositionDirection, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    strike_price: Mapped[float] = mapped_column(Float, nullable=False)
    expiration_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    contracts: Mapped[int] = mapped_column(Integer, nullable=False)
    premium_paid: Mapped[float] = mapped_column(Float, nullable=False)
    position_intent: Mapped[str] = mapped_column(
        Enum(PositionIntent, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Enum(OptionStatus, values_callable=lambda x: [e.value for e in x]),
        default=OptionStatus.OPEN.value,
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    close_premium: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    ticker_ref: Mapped[TickerModel] = relationship("TickerModel", back_populates="options")


class SignalModel(Base):
    """
    SIGNAL table — append-only log of every generated signal.

    NEVER update a row in this table. Current signal for a ticker is
    always the row with the latest created_at. This gives you a full
    audit trail of every signal ever generated.
    """
    __tablename__ = "signal"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    value: Mapped[str] = mapped_column(
        Enum(SignalValue, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    fwd_days: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ticker_ref: Mapped[TickerModel] = relationship("TickerModel", back_populates="signals")
    alerts: Mapped[list[AlertModel]] = relationship("AlertModel", back_populates="signal_ref")
    suggestions: Mapped[list[SuggestionLogModel]] = relationship(
        "SuggestionLogModel", back_populates="signal_ref"
    )


class SentimentRecordModel(Base):
    """
    SENTIMENT_RECORD table — raw sentiment scores from external sources.

    The suggestion engine computes a weighted aggregate of these at query
    time. The aggregate is never stored — it is always recomputed fresh
    from the raw rows.
    """
    __tablename__ = "sentiment_record"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    record_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    source_weight: Mapped[float] = mapped_column(Float, nullable=False)
    reference_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ticker_ref: Mapped[TickerModel] = relationship(
        "TickerModel", back_populates="sentiment_records"
    )


class EarningsEventModel(Base):
    """
    EARNINGS_EVENT table — scheduled and historical earnings reports.

    is_upcoming is flipped to False by a cron job after the report date
    passes. eps_actual and revenue_actual are null until the report is
    published.
    """
    __tablename__ = "earnings_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    report_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    eps_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps_expected: Mapped[float] = mapped_column(Float, nullable=False)
    eps_surprise: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_expected: Mapped[float] = mapped_column(Float, nullable=False)
    revenue_surprise: Mapped[float | None] = mapped_column(Float, nullable=True)
    guidance_sentiment: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_upcoming: Mapped[bool] = mapped_column(Boolean, default=True)

    ticker_ref: Mapped[TickerModel] = relationship(
        "TickerModel", back_populates="earnings_events"
    )


class AlertModel(Base):
    """
    ALERT table — every alert the system has ever generated.

    Permanently failed alerts (retry_count >= max_attempts from SYSTEM_CONFIG)
    stay in this table as audit records. They are never silently dropped.
    triggering_signal_id is nullable because not all alert types are
    triggered by a signal (e.g. EARNINGS_APPROACHING, OPTION_EXPIRING).
    """
    __tablename__ = "alert"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    triggering_signal_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("signal.id"), nullable=True
    )
    alert_type: Mapped[str] = mapped_column(
        Enum(AlertType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    severity: Mapped[str] = mapped_column(
        Enum(AlertSeverity, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(AlertStatus, values_callable=lambda x: [e.value for e in x]),
        default=AlertStatus.PENDING.value,
    )
    delivery_channel: Mapped[str | None] = mapped_column(String(100), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    ticker_ref: Mapped[TickerModel] = relationship("TickerModel", back_populates="alerts")
    signal_ref: Mapped[SignalModel | None] = relationship(
        "SignalModel", back_populates="alerts"
    )


class ModelAuditModel(Base):
    """
    MODEL_AUDIT table — one row per training run, accepted or rejected.

    auc_before is null only on the very first training run for a ticker
    (no previous model exists to compare against). Every subsequent run
    always has auc_before. This table is the full history of every model
    ever trained, including rejected ones.
    """
    __tablename__ = "model_audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    auc_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    auc_after: Mapped[float] = mapped_column(Float, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    config_path: Mapped[str] = mapped_column(String(500), nullable=False)
    n_train_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ticker_ref: Mapped[TickerModel] = relationship("TickerModel", back_populates="model_audits")


class PortfolioSnapshotModel(Base):
    """
    PORTFOLIO_SNAPSHOT table — point-in-time portfolio state.

    Recomputed after every transaction and on a daily scheduled cron.
    sector_breakdown is stored as JSON text — SQLAlchemy reads it back
    as a string; the repository layer parses it with json.loads().
    """
    __tablename__ = "portfolio_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    total_value: Mapped[float] = mapped_column(Float, nullable=False)
    total_invested: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    sector_breakdown: Mapped[str] = mapped_column(Text, nullable=False)
    trigger: Mapped[str] = mapped_column(
        Enum(SnapshotTrigger, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SuggestionLogModel(Base):
    """
    SUGGESTION_LOG table — every suggestion the engine has ever generated.

    acted_on is set manually by the user via the web interface. The full
    reasoning inputs (sentiment_summary, earnings_context) are stored so
    the chatbot can reference past suggestions with full context.
    """
    __tablename__ = "suggestion_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    signal_id: Mapped[str] = mapped_column(String(36), ForeignKey("signal.id"), nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    position_direction: Mapped[str] = mapped_column(
        Enum(PositionDirection, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    position_intent: Mapped[str] = mapped_column(
        Enum(PositionIntent, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    sentiment_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    earnings_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    acted_on: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ticker_ref: Mapped[TickerModel] = relationship("TickerModel", back_populates="suggestions")
    signal_ref: Mapped[SignalModel] = relationship("SignalModel", back_populates="suggestions")


class MacroIndicatorModel(Base):
    """
    MACRO_INDICATOR table — the universe of candidate macro indicators.

    is_unconditional=True for VIX, DXY, GLD, SLV — these are always
    included as training features regardless of correlation score.
    All other indicators are conditionally included by MacroCorrelationAnalyzer
    based on their correlation with the ticker's returns.
    """
    __tablename__ = "macro_indicator"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(
        Enum(MacroCategory, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    is_unconditional: Mapped[bool] = mapped_column(Boolean, default=False)
    fetch_source: Mapped[str] = mapped_column(String(100), nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    relevance_records: Mapped[list[MacroRelevanceModel]] = relationship(
        "MacroRelevanceModel", back_populates="macro_ref", cascade="all, delete-orphan"
    )


class MacroRelevanceModel(Base):
    """
    MACRO_RELEVANCE table — per-ticker macro correlation results.

    Recomputed on every onboarding and retrain by MacroCorrelationAnalyzer.
    tier is 3 or 5 — which conditional tier was applied.
    is_relevant=True means this macro was selected as a training feature.
    """
    __tablename__ = "macro_relevance"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    ticker: Mapped[str] = mapped_column(String(10), ForeignKey("ticker.symbol"), nullable=False)
    macro_symbol: Mapped[str] = mapped_column(
        String(20), ForeignKey("macro_indicator.symbol"), nullable=False
    )
    correlation_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_relevant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tier: Mapped[int] = mapped_column(Integer, nullable=False)
    max_correlation_seen: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ticker_ref: Mapped[TickerModel] = relationship(
        "TickerModel", back_populates="macro_relevance"
    )
    macro_ref: Mapped[MacroIndicatorModel] = relationship(
        "MacroIndicatorModel", back_populates="relevance_records"
    )


class SystemConfigCategoryModel(Base):
    """
    SYSTEM_CONFIG_CATEGORY table — one row per config category.

    version increments every time any key in that category changes.
    Modules compare their cached version against this before deciding
    whether to re-read their config values.
    """
    __tablename__ = "system_config_category"

    category: Mapped[str] = mapped_column(
        Enum(ConfigCategory, values_callable=lambda x: [e.value for e in x]),
        primary_key=True,
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    config_entries: Mapped[list[SystemConfigModel]] = relationship(
        "SystemConfigModel", back_populates="category_ref", cascade="all, delete-orphan"
    )
    history_entries: Mapped[list[SystemConfigHistoryModel]] = relationship(
        "SystemConfigHistoryModel", back_populates="category_ref"
    )


class SystemConfigModel(Base):
    """
    SYSTEM_CONFIG table — every tunable parameter in the system.

    Values are always stored as strings and cast to the correct Python
    type (float, int, bool, str) by ConfigRepository when reading.
    No threshold, count, or schedule is hardcoded anywhere in the system —
    everything operationally significant lives in this table.
    """
    __tablename__ = "system_config"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    category: Mapped[str] = mapped_column(
        String(50), ForeignKey("system_config_category.category"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    data_type: Mapped[str] = mapped_column(
        Enum(ConfigDataType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_value: Mapped[str] = mapped_column(String(500), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    category_ref: Mapped[SystemConfigCategoryModel] = relationship(
        "SystemConfigCategoryModel", back_populates="config_entries"
    )


class SystemConfigHistoryModel(Base):
    """
    SYSTEM_CONFIG_HISTORY table — append-only log of every config change.

    NEVER update a row in this table. reason is NOT nullable — every config
    change must have a written reason. category_version_at_change lets you
    reconstruct the exact config state at any point in time.
    """
    __tablename__ = "system_config_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    category: Mapped[str] = mapped_column(
        String(50), ForeignKey("system_config_category.category"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    previous_value: Mapped[str] = mapped_column(String(500), nullable=False)
    new_value: Mapped[str] = mapped_column(String(500), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    changed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    category_version_at_change: Mapped[int] = mapped_column(Integer, nullable=False)

    category_ref: Mapped[SystemConfigCategoryModel] = relationship(
        "SystemConfigCategoryModel", back_populates="history_entries"
    )


# ---------------------------------------------------------------------------
# Database class — the single object startup code interacts with
# ---------------------------------------------------------------------------


class Database:
    """
    Manages the SQLAlchemy engine and session factory.

    Only startup code (run.py) interacts with this class directly.
    Everything else receives a Session object — repositories never
    import Database or create their own connections.

    Usage:
        db = Database.init("data/stock_predictor.db")
        db.migrate()
        session = db.get_session()

    Attributes:
        _engine: The SQLAlchemy engine. Manages the connection pool.
        _Session: The session factory. Call it to get a new Session.
        _instance: Singleton — only one Database object exists at runtime.
    """

    _instance: Database | None = None

    def __init__(self, db_path: str) -> None:
        """
        Create the engine and session factory.

        Args:
            db_path: Path to the SQLite file, e.g. "data/stock_predictor.db".
                     SQLite creates the file if it doesn't exist.
        """
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            # echo=True prints every SQL statement to the console.
            # Set to True temporarily if you need to debug what queries
            # are being generated. Too noisy for normal use.
            echo=False,
            # check_same_thread=False allows connections from multiple threads.
            # Required because APScheduler runs background jobs on separate
            # threads that also need DB access.
            connect_args={"check_same_thread": False},
        )

        # Enable WAL mode for SQLite.
        # WAL (Write-Ahead Logging) allows concurrent reads while a write
        # is in progress. Without it, any write locks the entire DB file,
        # which causes Flask requests to block while APScheduler is writing.
        @event.listens_for(self._engine, "connect")
        def set_wal_mode(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        # Session factory — call _Session() to get a new session.
        # autocommit=False: you must call session.commit() explicitly.
        # autoflush=False: SQLAlchemy won't auto-write pending changes
        # before queries. Repositories control flushing explicitly.
        self._Session = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

    @classmethod
    def init(cls, db_path: str) -> "Database":
        """
        Create the singleton Database instance.

        Call this exactly once at startup in run.py before anything else.

        Args:
            db_path: Path to the SQLite file.

        Returns:
            The Database instance.
        """
        cls._instance = cls(db_path)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "Database":
        """
        Retrieve the singleton instance after init() has been called.

        Raises:
            RuntimeError: If init() has not been called yet.
        """
        if cls._instance is None:
            raise RuntimeError(
                "Database.init() has not been called. "
                "Call Database.init(db_path) in run.py before anything else."
            )
        return cls._instance

    def migrate(self) -> None:
        """
        Create all tables that don't already exist.

        Uses SQLAlchemy's create_all(), which is safe to call on every
        startup — it only creates missing tables and never modifies
        existing ones. For schema changes to existing tables, you would
        use Alembic. For this project, create_all() is sufficient.
        """
        Base.metadata.create_all(self._engine)

    def get_session(self):
        """
        Return a new SQLAlchemy Session.

        The caller is responsible for commit() and close(). In repositories
        this is handled via a context manager.

        Returns:
            A new SQLAlchemy Session bound to this engine.
        """
        return self._Session()