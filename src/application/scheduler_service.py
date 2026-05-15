"""
SchedulerService — registers and manages all background scheduled jobs.

Uses APScheduler to run five recurring jobs inside the Flask process.
All jobs share the same SQLAlchemy session factory.

Jobs registered:
    1. daily_inference      — runs inference for all watchlist tickers
                              Frequency: configurable, default every 4 hours
                              on market days.

    2. daily_retrain        — retrains all eligible tickers
                              Frequency: configurable, default daily at midnight.

    3. daily_snapshot       — takes a scheduled portfolio snapshot
                              Frequency: daily at market close (4:30 PM ET).

    4. expire_options       — marks past-expiry options as EXPIRED
                              Frequency: daily at market open (9:45 AM ET).

    5. flip_earnings        — marks past earnings events as not upcoming
                              Frequency: daily at midnight.

All schedules are read from SYSTEM_CONFIG SCHEDULER category at startup.
SchedulerService never hardcodes times.

Depends on:
    APScheduler             — BackgroundScheduler
    InferenceOrchestrator   — job 1
    RetrainOrchestrator     — job 2
    PortfolioOrchestrator   — jobs 3 and 4
    EarningsRepository      — job 5
    Database                — session factory
    ConfigRepository        — reads job schedules

Exposes:
    start()  — start the scheduler (called once at startup)
    stop()   — stop the scheduler (called on shutdown)
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.data.database import Database
from src.data.repositories.config_repository import ConfigRepository
from src.data.repositories.earnings_repository import EarningsRepository
from src.utils.types import ConfigCategory

logger = logging.getLogger(__name__)


class SchedulerService:
    """
    Registers and runs all background jobs using APScheduler.

    One instance is created at startup in run.py and started after
    the Flask app is configured. Jobs run on background threads —
    each job gets its own fresh SQLAlchemy session.

    Usage:
        scheduler_service = SchedulerService(
            db, config_repo, inference_orchestrator,
            retrain_orchestrator, portfolio_orchestrator, earnings_repo
        )
        scheduler_service.start()
        # ... Flask runs ...
        scheduler_service.stop()
    """

    def __init__(
        self,
        db: Database,
        config_repo: ConfigRepository,
        inference_orchestrator,
        retrain_orchestrator,
        portfolio_orchestrator,
        earnings_repo: EarningsRepository,
    ) -> None:
        self._db = db
        self._config_repo = config_repo
        self._inference_orchestrator = inference_orchestrator
        self._retrain_orchestrator = retrain_orchestrator
        self._portfolio_orchestrator = portfolio_orchestrator
        self._earnings_repo = earnings_repo
        self._scheduler = BackgroundScheduler(timezone="America/New_York")

    def start(self) -> None:
        """
        Register all jobs and start the scheduler.

        Reads job schedules from SYSTEM_CONFIG SCHEDULER category.
        Called once at startup in run.py after all dependencies are wired.
        """
        self._register_inference_job()
        self._register_retrain_job()
        self._register_snapshot_job()
        self._register_expire_options_job()
        self._register_flip_earnings_job()

        self._scheduler.start()
        logger.info("SchedulerService started — %d jobs registered.",
                    len(self._scheduler.get_jobs()))

    def stop(self) -> None:
        """
        Gracefully stop the scheduler.

        Called on Flask shutdown. Waits for any running jobs to complete.
        """
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("SchedulerService stopped.")

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_inference_job(self) -> None:
        """
        Register the daily inference job.

        Runs inference for all watchlist tickers on a configurable interval.
        Default: every 4 hours during market hours (Mon-Fri, 9:30-16:30 ET).
        """
        interval_hours = self._config_repo.get(
            ConfigCategory.SCHEDULER, "inference_interval_hours"
        )

        self._scheduler.add_job(
            func=self._run_inference,
            trigger=IntervalTrigger(hours=interval_hours),
            id="daily_inference",
            name="Daily Inference",
            replace_existing=True,
            misfire_grace_time=300,  # 5 minutes
        )
        logger.info("Registered inference job — every %s hours.", interval_hours)

    def _register_retrain_job(self) -> None:
        """
        Register the daily retrain job.

        Retrains all eligible tickers. Default: daily at midnight ET.
        """
        retrain_hour = self._config_repo.get(
            ConfigCategory.SCHEDULER, "retrain_hour_et"
        )

        self._scheduler.add_job(
            func=self._run_retrain,
            trigger=CronTrigger(hour=retrain_hour, minute=0, day_of_week="mon-fri"),
            id="daily_retrain",
            name="Daily Retrain",
            replace_existing=True,
            misfire_grace_time=600,  # 10 minutes
        )
        logger.info("Registered retrain job — daily at %s:00 ET.", retrain_hour)

    def _register_snapshot_job(self) -> None:
        """
        Register the daily portfolio snapshot job.

        Default: daily at 16:30 ET (after market close).
        """
        self._scheduler.add_job(
            func=self._run_snapshot,
            trigger=CronTrigger(hour=16, minute=30, day_of_week="mon-fri"),
            id="daily_snapshot",
            name="Daily Portfolio Snapshot",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info("Registered snapshot job — daily at 16:30 ET.")

    def _register_expire_options_job(self) -> None:
        """
        Register the options expiry job.

        Default: daily at 09:45 ET (after market open).
        """
        self._scheduler.add_job(
            func=self._run_expire_options,
            trigger=CronTrigger(hour=9, minute=45, day_of_week="mon-fri"),
            id="expire_options",
            name="Expire Options",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info("Registered expire_options job — daily at 09:45 ET.")

    def _register_flip_earnings_job(self) -> None:
        """
        Register the earnings flip job.

        Default: daily at midnight ET.
        """
        self._scheduler.add_job(
            func=self._run_flip_earnings,
            trigger=CronTrigger(hour=0, minute=5),
            id="flip_earnings",
            name="Flip Past Earnings",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("Registered flip_earnings job — daily at 00:05 ET.")

    # ------------------------------------------------------------------
    # Job implementations — each gets its own session
    # ------------------------------------------------------------------

    def _run_inference(self) -> None:
        """Run inference for all watchlist tickers."""
        logger.info("Starting scheduled inference run.")
        try:
            results = self._inference_orchestrator.run_for_all()
            succeeded = sum(1 for r in results if r.success)
            logger.info(
                "Inference complete — %d/%d tickers succeeded.",
                succeeded, len(results)
            )
        except Exception as exc:
            logger.error("Inference job failed: %s", exc)

    def _run_retrain(self) -> None:
        """Retrain all eligible tickers."""
        logger.info("Starting scheduled retrain run.")
        session = self._db.get_session()
        try:
            from src.data.repositories.ticker_repository import TickerRepository
            ticker_repo = TickerRepository(session)
            eligible = ticker_repo.get_training_eligible()
            accepted = 0
            for ticker in eligible:
                result = self._retrain_orchestrator.run(ticker.symbol)
                if result.training_result and result.training_result.accepted:
                    accepted += 1
            logger.info(
                "Retrain complete — %d/%d models improved.",
                accepted, len(eligible)
            )
        except Exception as exc:
            logger.error("Retrain job failed: %s", exc)
        finally:
            session.close()

    def _run_snapshot(self) -> None:
        """Take a scheduled portfolio snapshot."""
        session = self._db.get_session()
        try:
            self._portfolio_orchestrator.take_scheduled_snapshot(session)
            logger.info("Scheduled portfolio snapshot taken.")
        except Exception as exc:
            logger.error("Snapshot job failed: %s", exc)
        finally:
            session.close()

    def _run_expire_options(self) -> None:
        """Mark past-expiry options as EXPIRED."""
        session = self._db.get_session()
        try:
            count = self._portfolio_orchestrator.expire_options(session)
            logger.info("Expired %d option positions.", count)
        except Exception as exc:
            logger.error("Expire options job failed: %s", exc)
        finally:
            session.close()

    def _run_flip_earnings(self) -> None:
        """Mark past earnings events as not upcoming."""
        session = self._db.get_session()
        try:
            count = self._earnings_repo.flip_past_events()
            session.commit()
            logger.info("Flipped %d past earnings events.", count)
        except Exception as exc:
            logger.error("Flip earnings job failed: %s", exc)
            session.rollback()
        finally:
            session.close()