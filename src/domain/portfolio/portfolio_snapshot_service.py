"""
PortfolioSnapshotService — assembles and persists portfolio snapshots.

A portfolio snapshot is a point-in-time record of portfolio state:
total value, total invested, cash balance, and sector breakdown.
Snapshots are written after every transaction and on a daily schedule.

PortfolioSnapshotService is the only component that writes to the
PORTFOLIO_SNAPSHOT table. It coordinates PositionDeriver, PortfolioAnalyzer,
and PortfolioSnapshotRepository to produce and persist a complete snapshot.

Depends on:
    PositionDeriver             — derives current positions
    PortfolioAnalyzer           — computes sector breakdown and warnings
    PortfolioSnapshotRepository — persists the snapshot row
    ConfigRepository            — reads cash balance from SYSTEM_CONFIG
    SnapshotTrigger             — TRANSACTION or SCHEDULED

Exposes:
    take_snapshot(trigger) → PortfolioSnapshotModel
"""

from __future__ import annotations

import json
from datetime import datetime

from src.data.database import PortfolioSnapshotModel
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.portfolio_snapshot_repository import (
    PortfolioSnapshotRepository,
)
from src.domain.portfolio.portfolio_analyzer import PortfolioAnalyzer
from src.domain.portfolio.position_deriver import PositionDeriver
from src.utils.types import ConfigCategory, SnapshotTrigger


class PortfolioSnapshotService:
    """
    Coordinates position derivation, analysis, and snapshot persistence.

    Usage:
        service = PortfolioSnapshotService(
            position_deriver, portfolio_analyzer,
            snapshot_repo, config_repo
        )
        snapshot = service.take_snapshot(SnapshotTrigger.TRANSACTION)
    """

    def __init__(
        self,
        position_deriver: PositionDeriver,
        portfolio_analyzer: PortfolioAnalyzer,
        snapshot_repo: PortfolioSnapshotRepository,
        config_repo: ConfigRepository,
    ) -> None:
        self._position_deriver = position_deriver
        self._portfolio_analyzer = portfolio_analyzer
        self._snapshot_repo = snapshot_repo
        self._config_repo = config_repo

    def take_snapshot(
        self,
        trigger: SnapshotTrigger,
        session,
    ) -> PortfolioSnapshotModel:
        """
        Derive current positions, analyze the portfolio, and persist a snapshot.

        Args:
            trigger: Why this snapshot is being taken — TRANSACTION or SCHEDULED.
            session: Active SQLAlchemy Session. Passed in so the snapshot write
                     can be part of the same transaction as whatever triggered it.

        Returns:
            The persisted PortfolioSnapshotModel row.
        """
        # Derive all current positions
        positions = self._position_deriver.derive_all()

        # Run portfolio analysis
        analysis = self._portfolio_analyzer.analyze(positions)

        # Read cash balance from config
        cash = self._config_repo.get(ConfigCategory.PORTFOLIO, "cash_balance")

        # Compute total portfolio value = market value of all positions + cash
        total_value = analysis.total_invested + cash

        # Build and persist snapshot
        snapshot = PortfolioSnapshotModel(
            total_value=total_value,
            total_invested=analysis.total_invested,
            cash=cash,
            sector_breakdown=json.dumps(analysis.sector_breakdown),
            trigger=trigger.value,
            snapshot_at=datetime.utcnow(),
        )

        self._snapshot_repo.add(snapshot)

        return snapshot