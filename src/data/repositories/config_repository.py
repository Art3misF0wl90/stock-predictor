"""
ConfigRepository — database operations for SYSTEM_CONFIG and related tables.

This is the most behaviorally complex repository because it handles:
    1. Type casting — values are stored as strings, read back as the
       correct Python type (float, int, bool, str).
    2. Version checking — modules cache config values and only re-read
       when the category version has changed.
    3. History logging — every change is written to SYSTEM_CONFIG_HISTORY
       with a required reason, previous value, and category version.

Depends on:
    BaseRepository              — session management and context manager
    SystemConfigModel           — the ORM class for system_config
    SystemConfigCategoryModel   — the ORM class for system_config_category
    SystemConfigHistoryModel    — the ORM class for system_config_history
    ConfigCategory              — enum for the six config categories
    ConfigDataType              — enum for how to cast stored strings

Exposes:
    get(category, key)              — read one config value, cast to correct type
    get_category_version(category)  — current version number for a category
    set(category, key, value, reason, changed_by) — update a value, log history
    get_all_for_category(category)  — all key/value pairs for one category
    seed(entries)                   — insert default values on first startup
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.data.database import (
    SystemConfigCategoryModel,
    SystemConfigHistoryModel,
    SystemConfigModel,
)
from src.data.repositories.base import BaseRepository
from src.utils.types import ConfigCategory, ConfigDataType


class ConfigRepository(BaseRepository):
    """
    Reads and writes the SYSTEM_CONFIG, SYSTEM_CONFIG_CATEGORY, and
    SYSTEM_CONFIG_HISTORY tables.

    Every module that needs a configurable threshold, count, or schedule
    reads it through this repository. Nothing operationally significant
    is hardcoded anywhere in the system.

    Type casting:
        All values are stored as VARCHAR strings in the database. When
        you call get(), the repository reads the data_type column and
        casts the string to the correct Python type before returning it.
        This means callers always receive a properly typed value — they
        never need to cast strings themselves.

    Version checking:
        Each config category has a version counter that increments every
        time any key in that category changes. Modules cache their config
        values alongside the version number they were read at. On each
        use, the module calls get_category_version() and compares it to
        their cached version — if it changed, they re-read. This avoids
        hitting the database on every inference run while still picking
        up config changes promptly.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def get(self, category: ConfigCategory, key: str) -> Any:
        """
        Read one config value and cast it to the correct Python type.

        The data_type column determines the cast:
            FLOAT  → float(value_str)
            INT    → int(value_str)
            BOOL   → value_str.lower() == "true"
            STRING → value_str (no cast needed)

        Args:
            category: The ConfigCategory enum value.
            key: The config key string, e.g. "auc_improvement_threshold".

        Returns:
            The config value cast to its correct Python type.

        Raises:
            KeyError: If the category/key combination does not exist.
                      This should never happen in production — all keys
                      are seeded at startup. If it does, it means a key
                      was referenced in code but never seeded in the DB.
        """
        row = (
            self._session.query(SystemConfigModel)
            .filter(
                SystemConfigModel.category == category.value,
                SystemConfigModel.key == key,
            )
            .first()
        )

        if row is None:
            raise KeyError(
                f"Config key not found: category={category.value}, key={key}. "
                f"Was this key seeded in config/default_config.json?"
            )

        return self._cast(row.value, row.data_type)

    def get_category_version(self, category: ConfigCategory) -> int:
        """
        Return the current version number for a config category.

        Modules cache this version alongside their cached config values.
        If the version has changed since last read, the module re-reads
        all its config values from the database.

        Args:
            category: The ConfigCategory to check.

        Returns:
            The current version integer. Starts at 1, increments on
            every set() call within this category.

        Raises:
            KeyError: If the category row does not exist. Should never
                      happen — all categories are seeded at startup.
        """
        row = self._session.get(SystemConfigCategoryModel, category.value)
        if row is None:
            raise KeyError(
                f"Config category not found: {category.value}. "
                f"Was the database seeded correctly?"
            )
        return row.version

    def set(
        self,
        category: ConfigCategory,
        key: str,
        new_value: Any,
        reason: str,
        changed_by: str | None = None,
    ) -> None:
        """
        Update a config value and write a history record.

        Every change to SYSTEM_CONFIG does three things atomically:
            1. Updates the value in SYSTEM_CONFIG.
            2. Increments the category version in SYSTEM_CONFIG_CATEGORY.
            3. Appends a row to SYSTEM_CONFIG_HISTORY with the previous
               value, new value, reason, and version at time of change.

        The reason parameter is NOT optional in spirit — callers must
        provide a meaningful explanation. The function signature accepts
        a str, not Optional[str], to make this explicit.

        Args:
            category: The ConfigCategory to update.
            key: The config key to update.
            new_value: The new value. Will be converted to string for storage.
            reason: REQUIRED explanation of why this change was made.
                    This is stored permanently in the history log.
            changed_by: Optional identifier of who made the change
                        (username, service name, etc.).

        Raises:
            KeyError: If the category/key combination does not exist.
        """
        config_row = (
            self._session.query(SystemConfigModel)
            .filter(
                SystemConfigModel.category == category.value,
                SystemConfigModel.key == key,
            )
            .first()
        )

        if config_row is None:
            raise KeyError(
                f"Config key not found: category={category.value}, key={key}."
            )

        previous_value = config_row.value
        new_value_str = str(new_value)

        category_row = self._session.get(SystemConfigCategoryModel, category.value)
        if category_row is None:
            raise KeyError(f"Config category not found: {category.value}.")

        new_version = category_row.version + 1

        history_row = SystemConfigHistoryModel(
            category=category.value,
            key=key,
            previous_value=previous_value,
            new_value=new_value_str,
            reason=reason,
            changed_by=changed_by,
            changed_at=datetime.utcnow(),
            category_version_at_change=new_version,
        )

        config_row.value = new_value_str
        config_row.updated_at = datetime.utcnow()
        config_row.updated_by = changed_by

        category_row.version = new_version
        category_row.updated_at = datetime.utcnow()

        self._session.add(history_row)

    def get_all_for_category(self, category: ConfigCategory) -> dict[str, Any]:
        """
        Fetch all key/value pairs for one category, cast to correct types.

        Used by modules that cache their entire config category at once
        rather than reading key by key. More efficient when a module
        needs several keys from the same category.

        Args:
            category: The ConfigCategory to read.

        Returns:
            Dict mapping key → cast Python value for all keys in the
            category. Empty dict if no keys exist (shouldn't happen
            after seeding).
        """
        rows = (
            self._session.query(SystemConfigModel)
            .filter(SystemConfigModel.category == category.value)
            .all()
        )
        return {row.key: self._cast(row.value, row.data_type) for row in rows}

    def seed(self, entries: list[dict]) -> None:
        """
        Insert default config values on first startup.

        Called by run.py after migrate() if the config tables are empty.
        Each entry in the list should have:
            category, key, value, data_type, description, default_value

        Does not overwrite existing rows — only inserts keys that don't
        already exist. Safe to call on every startup.

        Args:
            entries: List of dicts, one per config key to seed.
        """
        for entry in entries:
            exists = (
                self._session.query(SystemConfigModel)
                .filter(
                    SystemConfigModel.category == entry["category"],
                    SystemConfigModel.key == entry["key"],
                )
                .count()
            ) > 0

            if not exists:
                row = SystemConfigModel(
                    category=entry["category"],
                    key=entry["key"],
                    value=str(entry["value"]),
                    data_type=entry["data_type"],
                    description=entry.get("description"),
                    default_value=str(entry["value"]),
                )
                self._session.add(row)

        self._session.flush()

    @staticmethod
    def _cast(value_str: str, data_type: str) -> Any:
        """
        Cast a stored string value to the correct Python type.

        This is a private helper — only called internally by get()
        and get_all_for_category().

        Args:
            value_str: The raw string from the database.
            data_type: The ConfigDataType value string from the DB row.

        Returns:
            The value cast to float, int, bool, or str.

        Raises:
            ValueError: If the cast fails (e.g. "abc" stored as INT).
                        This indicates corrupted or incorrectly seeded data.
        """
        if data_type == ConfigDataType.FLOAT.value:
            return float(value_str)
        elif data_type == ConfigDataType.INT.value:
            return int(value_str)
        elif data_type == ConfigDataType.BOOL.value:
            return value_str.strip().lower() == "true"
        else:
            return value_str
