"""
Custom exception hierarchy for stock_predictor_v2

Every exception that crosses a component boundary must be one of these types.
Raw ValueError, KeyError, etc. must never propagate between components.

HTTP mapping (enforced by Flask error handler in presentation layer):
    DataFetchError              → 502 Bad Gateway
    InsufficientDataError       → 422 Unprocessable Entity
    ConfigNotFoundError         → 404 Not Found
    ConfigSchemaError           → 500 Internal Server Error
    FeatureEngineeringError     → 500 Internal Server Error
    FeatureCountMismatchError   → 500 Internal Server Error
    TrainingError               → 500 Internal Server Error
    InferenceError              → 500 Internal Server Error
    SignalFilterError           → 500 Internal Server Error
    ChatbotError                → 503 Service Unavailable
"""


class StockPredictorError(Exception):

    """
    Base exception for all stock_predictor_v2 errors. 

    Never raise this directly; always raise a more specific so the
    flask error handler can map it to the correct HTTP status code.

    Attributes:
        message: A human-readable error message describing what went wrong.
        cause: The original exception that triggered this one, if any.
                Always passe cause = when re-raising across layer boundaries
                so the root cause is never lost.
    """
    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause is not None:
            return (
                f"{self.message} (caused by: {type(self.cause).__name__}: {self.cause})"
            )
        return self.message


class DataFetchError(StockPredictorError):
    """
    Raised by DataLoader when the primary data source fails.
 
    This means the external API (yfinance or macro source) returned an error,
    timed out, or returned an empty response when data was expected.
 
    Maps to HTTP 502 — the app is acting as a gateway to an external source
    that failed.
 
    Example:
        raise DataFetchError("yfinance returned empty DataFrame for AAPL")
        raise DataFetchError("Macro fetch failed for VIX", cause=original_exc)
    """
    pass


class InsufficientDataError(StockPredictorError):
    """
    Raised by DataLoader or DataValidator when fetched data exists but does
    not meet the minimum row threshold required for training or inference.
 
    Maps to HTTP 422 — the request was understood but cannot be processed
    because the data precondition is not satisfied.
 
    Example:
        raise InsufficientDataError(
            "NVDA has 180 rows, minimum required is 252 (one calendar year)"
        )
    """
    pass


class ConfigNotFoundError(StockPredictorError):
    """
    Raised by ConfigStore when no config.pkl exists for the requested ticker.
 
    This typically means the ticker has not completed onboarding and has never
    been successfully trained. Inference cannot proceed without a config.
 
    Maps to HTTP 404 — the resource (trained model config) does not exist.
 
    Example:
        raise ConfigNotFoundError("No config found for ticker: TSLA")
    """
    pass

class ConfigSchemaError(StockPredictorError):
    """
    Raised by ConfigStore when a config.pkl is found but fails schema
    validation against the current TickerConfig model.
 
    This indicates either a corrupted file or a schema version mismatch
    from an incompatible older config format.
 
    Maps to HTTP 500 — the server has a config it cannot read.
 
    Example:
        raise ConfigSchemaError(
            "Config for AAPL failed schema validation: missing field 'fwd_days'",
            cause=validation_exc
        )
    """
    pass

class FeatureEngineeringError(StockPredictorError):
    """
    Raised by FeatureEngineer when feature matrix construction fails.
 
    This is the parent class for all feature-related errors. The Flask error
    handler catches this type and also catches FeatureCountMismatchError
    via inheritance — both map to HTTP 500.
 
    Example:
        raise FeatureEngineeringError(
            "Failed to compute RSI for MSFT: insufficient price history",
            cause=original_exc
        )
    """
    pass


class FeatureCountMismatchError(FeatureEngineeringError):

    """
    Raised by FeatureEngineer in inference mode when the feature matrix
    produced does not match the column list loaded from config.pkl.
 
    This is the architectural fix for the v1 bug where TSLA had 57 features
    and other tickers had 50, causing silent inference failures.
 
    In v2, feat_cols comes from config.pkl ONLY and is never recomputed.
    If a mismatch occurs anyway (e.g. a macro indicator was removed from
    the DB after training), this exception surfaces it explicitly.
 
    Maps to HTTP 500.
 
    Attributes:
        expected: Number of features expected from config.pkl.
        actual: Number of features produced by FeatureEngineer.
 
    Example:
        raise FeatureCountMismatchError(
            "Feature count mismatch for TSLA: expected 57, got 50",
            expected=57,
            actual=50
        )
    """

    def __init__(
        self,
        message: str,
        expected: int,
        actual: int,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.expected = expected
        self.actual = actual


class TrainingError(StockPredictorError):
    """
    Raised by Trainer for unexpected failures only.
 
    A model rejection (AUC did not improve enough) is NOT a TrainingError.
    Rejection is a valid TrainingResult with accepted=False. TrainingError
    is reserved for unrecoverable failures: out-of-memory, corrupted data
    that slipped past validation, or unexpected library exceptions.
 
    Maps to HTTP 500.
 
    Example:
        raise TrainingError("XGBoost OOM during fit for NVDA", cause=mem_exc)
    """
    pass


class InferenceError(StockPredictorError):
    """
    Raised by InferencePipeline when prediction fails for reasons other than
    a missing config or failed data fetch (those propagate as-is).
 
    InferencePipeline re-raises ConfigNotFoundError and DataFetchError
    unchanged. Everything else is wrapped in InferenceError so callers
    get a single catch point for unexpected inference failures.
 
    Maps to HTTP 500.
 
    Example:
        raise InferenceError(
            "Inference failed for GOOGL during model forward pass",
            cause=original_exc
        )
    """
    pass


class SignalFilterError(StockPredictorError):
    """
    Raised by SignalFilter when required context fields are missing or invalid.
 
    SignalFilter is a pure function — it never fetches data. If the
    MarketContext passed to it is missing required fields (e.g. VIX is None
    when the VIX gate is active), this exception is raised.
 
    Note: Suppression to HOLD (e.g. VIX above threshold) is NOT an error.
    It is a valid FilterResult. This exception is only for malformed input.
 
    Maps to HTTP 500.
 
    Example:
        raise SignalFilterError(
            "MarketContext missing vix_value but VIX gate is enabled"
        )
    """
    pass


class ChatbotError(StockPredictorError):
    """
    Raised by ChatbotService when the Groq API is unavailable after all
    retry attempts have been exhausted.
 
    Maps to HTTP 503 — the upstream LLM service is temporarily unavailable.
 
    Example:
        raise ChatbotError(
            "Groq API unreachable after 3 attempts",
            cause=connection_exc
        )
    """
    pass
