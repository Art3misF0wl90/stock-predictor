"""
Base repository for stock_predictor_v2.

Every repository in data/repositories/ inherits from this class.
It provides session management, commit/rollback handling, and a
context manager so repositories can be used in a `with` block.

Why a base class?
    Every repository needs the same boilerplate: hold a session,
    commit on success, rollback on failure, close when done.
    Writing that 12 times across 12 repositories would mean 12 places
    to fix if anything changes. The base class writes it once.

How sessions work in this system:
    - Database.get_session() produces a new Session object.
    - A Session is a unit of work — a staging area for DB operations.
    - Nothing hits the database until session.commit() is called.
    - If an error occurs, session.rollback() throws away all pending changes.
    - session.close() returns the connection back to the pool.

Usage — two patterns:

    Pattern 1: context manager (preferred for single operations)
        with TickerRepository(session) as repo:
            repo.add(ticker)
        # commit and close happen automatically

    Pattern 2: manual (preferred when multiple repositories share a session
                        and you need one commit for all of them)
        repo = TickerRepository(session)
        repo.add(ticker)
        session.commit()
        session.close()
"""

from __future__ import annotations

from sqlalchemy.orm import Session


class BaseRepository:
    """
    Parent class for all repositories.

    Provides session management and a context manager interface.
    Subclasses add their own query methods — one method per operation
    on that table.

    Attributes:
        _session: The SQLAlchemy Session this repository operates through.
                  Provided at construction time by the caller. The repository
                  never creates its own session.
    """

    def __init__(self, session: Session) -> None:
        """
        Store the session. Do nothing else.

        The session is provided by whoever constructs the repository —
        either startup code, an orchestrator, or a test. The repository
        never calls Database.get_session() itself. This is called
        dependency injection: the dependency (session) is injected from
        outside rather than created internally.

        Why dependency injection?
            If repositories created their own sessions, you could never
            test them without a real database. By injecting the session,
            tests can pass in a mock or an in-memory SQLite session instead.

        Args:
            session: An active SQLAlchemy Session bound to the engine.
        """
        self._session = session

    def commit(self) -> None:
        """
        Persist all pending changes to the database.

        Wraps session.commit() with rollback on failure so the session
        is always left in a clean state. If commit fails, the rollback
        runs and the original exception is re-raised so the caller knows
        what went wrong.

        When to call this:
            Call commit() after one or more add/update/delete operations
            when you are ready to persist them. In the context manager
            pattern (__exit__), commit() is called automatically on
            clean exit.
        """
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

    def rollback(self) -> None:
        """
        Discard all pending changes since the last commit.

        Use this when you detect an error condition and want to throw
        away partially completed work before raising an exception.
        """
        self._session.rollback()

    def close(self) -> None:
        """
        Close the session and return the connection to the pool.

        Always call this when you are done with a repository. In the
        context manager pattern, close() is called automatically in
        __exit__. In manual usage, call it explicitly.

        After close(), the session is no longer usable.
        """
        self._session.close()

    def __enter__(self) -> "BaseRepository":
        """
        Enter the context manager — return self so the `with` block
        can use the repository.

        Called automatically when you write:
            with SomeRepository(session) as repo:
                ...

        Returns:
            self — the repository instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context manager.

        Called automatically at the end of a `with` block, whether it
        finished normally or raised an exception.

        Args:
            exc_type: The exception class, if an exception was raised.
                      None if the block finished without error.
            exc_val:  The exception instance, if raised. None otherwise.
            exc_tb:   The traceback, if an exception was raised. None otherwise.

        Behavior:
            - No exception (exc_type is None): commit, then close.
            - Exception raised: rollback, then close, then let the
              exception propagate normally.

        Returns:
            False — tells Python not to suppress the exception.
            Returning True would swallow it, which we never want.
        """
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
        return False