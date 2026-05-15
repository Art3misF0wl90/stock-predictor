"""
ChatbotService — assembles context and calls the Groq LLM for chat responses.

ChatbotService is the only component that calls the Groq API. It assembles
a system prompt from current portfolio state, signals, and suggestions,
then passes the user's message to llama-3.3-70b-versatile and returns
the response.

Context assembled per request:
    - Current signals for all watchlist tickers
    - Most recent suggestion per ticker (if any)
    - Current portfolio snapshot (sector breakdown, total value)
    - Recent signal history for the symbol in question (if specified)
    - Recent sentiment records (if any)

The system prompt instructs the LLM to:
    - Speak as a portfolio analysis assistant
    - Only discuss tickers in the watchlist
    - Reference the provided data, not general knowledge
    - Never give financial advice — present analysis only
    - Flag when it does not have data to answer a question

Retry logic:
    Up to max_retries attempts with exponential backoff.
    If all attempts fail, raises ChatbotError.

Depends on:
    groq SDK                — Groq API client
    SignalRepository        — current signals
    SuggestionRepository    — most recent suggestions
    PortfolioSnapshotRepository — current portfolio state
    SentimentRepository     — sentiment context
    ConfigRepository        — reads model name and retry settings
    ChatbotError            — raised when Groq is unreachable

Exposes:
    respond(message, symbol, conversation_history) → ChatbotResponse
"""

from __future__ import annotations

import json
import logging
import os
import time

from groq import Groq

from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.portfolio_snapshot_repository import (
    PortfolioSnapshotRepository,
)
from src.data.repositories.sentiment_repository import SentimentRepository
from src.data.repositories.signal_repository import SignalRepository
from src.data.repositories.suggestion_repository import SuggestionRepository
from src.utils.exceptions import ChatbotError
from src.utils.types import ChatbotResponse, ConfigCategory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You are a portfolio analysis assistant for a personal stock prediction system.

You have access to the following real-time data from the system:

CURRENT SIGNALS:
{signals_block}

RECENT SUGGESTIONS:
{suggestions_block}

PORTFOLIO SNAPSHOT:
{portfolio_block}

{sentiment_block}

Guidelines:
- Base your analysis on the data provided above, not general market knowledge.
- Never give direct financial advice. Present analysis and let the user decide.
- If you don't have data to answer a question, say so clearly.
- Be concise. The user can ask follow-up questions.
- Reference specific signals, probabilities, and suggestions when relevant.
- If asked about a ticker not in the data above, say it is not on the watchlist.
"""


class ChatbotService:
    """
    Assembles context and queries the Groq LLM for portfolio chat responses.

    Usage:
        service = ChatbotService(
            signal_repo, suggestion_repo, snapshot_repo,
            sentiment_repo, config_repo
        )
        response = service.respond(
            message="Should I add to my AAPL position?",
            symbol="AAPL",
            conversation_history=[],
        )
    """

    def __init__(
        self,
        signal_repo: SignalRepository,
        suggestion_repo: SuggestionRepository,
        snapshot_repo: PortfolioSnapshotRepository,
        sentiment_repo: SentimentRepository,
        config_repo: ConfigRepository,
    ) -> None:
        self._signal_repo = signal_repo
        self._suggestion_repo = suggestion_repo
        self._snapshot_repo = snapshot_repo
        self._sentiment_repo = sentiment_repo
        self._config_repo = config_repo
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def respond(
        self,
        message: str,
        symbol: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> ChatbotResponse:
        """
        Generate a response to a user message with full portfolio context.

        Args:
            message: The user's chat message.
            symbol: Optional ticker the message is specifically about.
                    If provided, additional signal history and sentiment
                    context for that ticker is included in the prompt.
            conversation_history: List of prior turns as dicts with
                    "role" and "content" keys. Pass [] for a new conversation.

        Returns:
            ChatbotResponse with reply, symbol, and context_summary.

        Raises:
            ChatbotError: If the Groq API is unreachable after all retries.
        """
        history = conversation_history or []

        system_prompt, context_summary = self._build_system_prompt(symbol)

        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": message}]
        )

        reply = self._call_groq(messages)

        return ChatbotResponse(
            reply=reply,
            symbol=symbol,
            context_summary=context_summary,
        )

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, symbol: str | None
    ) -> tuple[str, str]:
        """
        Assemble the system prompt with all available context.

        Args:
            symbol: Optional ticker to include extra context for.

        Returns:
            Tuple of (system_prompt_string, context_summary_string).
        """
        context_parts = []

        # Current signals for all tickers
        all_signals = self._signal_repo.get_latest_all_tickers()
        signals_lines = []
        for sym, sig in sorted(all_signals.items()):
            signals_lines.append(
                f"  {sym}: {sig.value} ({sig.probability:.0%} confidence, "
                f"fwd_days={sig.fwd_days})"
            )
        signals_block = "\n".join(signals_lines) if signals_lines else "  No signals yet."
        context_parts.append(f"{len(all_signals)} ticker signals")

        # Recent suggestions
        suggestions_lines = []
        for sym in sorted(all_signals.keys()):
            latest = self._suggestion_repo.get_latest_for_ticker(sym)
            if latest:
                suggestions_lines.append(
                    f"  {sym}: {latest.recommendation} "
                    f"(acted_on={latest.acted_on})"
                )
        suggestions_block = (
            "\n".join(suggestions_lines) if suggestions_lines
            else "  No suggestions yet."
        )
        context_parts.append(f"{len(suggestions_lines)} suggestions")

        # Portfolio snapshot
        snapshot = self._snapshot_repo.get_latest()
        if snapshot:
            sector_data = json.loads(snapshot.sector_breakdown)
            sector_str = ", ".join(
                f"{k}: {v:.1%}" for k, v in sorted(sector_data.items())
            )
            portfolio_block = (
                f"  Total value: ${snapshot.total_value:,.2f}\n"
                f"  Total invested: ${snapshot.total_invested:,.2f}\n"
                f"  Cash: ${snapshot.cash:,.2f}\n"
                f"  Sectors: {sector_str}"
            )
            context_parts.append("portfolio snapshot")
        else:
            portfolio_block = "  No portfolio snapshot available yet."

        # Sentiment block — only if a specific symbol is requested
        sentiment_block = ""
        if symbol:
            records = self._sentiment_repo.get_by_ticker(symbol, days=14)
            if records:
                total_weight = sum(r.source_weight for r in records)
                if total_weight > 0:
                    weighted = sum(
                        r.raw_score * r.source_weight for r in records
                    ) / total_weight
                    sentiment_block = (
                        f"SENTIMENT FOR {symbol} (last 14 days):\n"
                        f"  Weighted score: {weighted:+.2f} "
                        f"({len(records)} records)\n"
                        f"  Range: {min(r.raw_score for r in records):.2f} "
                        f"to {max(r.raw_score for r in records):.2f}"
                    )
                    context_parts.append(f"{symbol} sentiment")

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            signals_block=signals_block,
            suggestions_block=suggestions_block,
            portfolio_block=portfolio_block,
            sentiment_block=sentiment_block,
        )

        context_summary = ", ".join(context_parts)
        return system_prompt, context_summary

    # ------------------------------------------------------------------
    # Groq API call with retry
    # ------------------------------------------------------------------

    def _call_groq(self, messages: list[dict]) -> str:
        """
        Call the Groq API with exponential backoff retry.

        Args:
            messages: Full message list including system prompt and history.

        Returns:
            The LLM's reply text.

        Raises:
            ChatbotError: If all retry attempts fail.
        """
        max_retries = self._config_repo.get(
            ConfigCategory.ALERTS, "chatbot_max_retries"
        )
        model = self._config_repo.get(
            ConfigCategory.ALERTS, "chatbot_model"
        )

        last_exc = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                )
                return response.choices[0].message.content

            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Groq API attempt %d/%d failed: %s. Retrying in %ds.",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)

        raise ChatbotError(
            f"Groq API unreachable after {max_retries} attempts.",
            cause=last_exc,
        )