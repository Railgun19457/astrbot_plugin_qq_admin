"""Group-scoped blacklist with group-set sharing.

Scope model
-----------
Entries are stored per *scope*, not per group. A scope is normally the name of
the group set the group belongs to, which is what makes the blacklist shared
across every group in that set. A group that belongs to no configured set
falls back to a scope of its own id, so blacklisting still works on
unconfigured groups instead of erroring out.

The ``reason`` field is a first-class citizen: it is filled in by the command
and the LLM tool, shown in receipts, and included in the exported table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import JsonStore


@dataclass
class BlacklistEntry:
    """One blacklisted user.

    Attributes:
        user_id: QQ number.
        reason: Why the user was blacklisted.
        operator: QQ number of the person who added the entry.
        created_at: Unix timestamp of the addition.
        expires_at: Unix timestamp when the entry lapses, ``0`` for permanent.
        nickname: Best known display name at the time of addition.
    """

    user_id: str
    reason: str
    operator: str
    created_at: float
    expires_at: float = 0.0
    nickname: str = ""

    @property
    def is_expired(self) -> bool:
        """Whether the entry has passed its expiry time."""
        return bool(self.expires_at) and time.time() >= self.expires_at

    @property
    def is_temporary(self) -> bool:
        """Whether the entry is time limited rather than permanent."""
        return bool(self.expires_at)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry for storage.

        Returns:
            A JSON-compatible mapping.
        """
        return {
            "reason": self.reason,
            "operator": self.operator,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "nickname": self.nickname,
        }

    @classmethod
    def from_dict(cls, user_id: str, raw: Any) -> BlacklistEntry | None:
        """Rebuild an entry from stored data.

        Args:
            user_id: Key the entry was stored under.
            raw: Stored payload.

        Returns:
            The parsed entry, or ``None`` when the payload is unusable.
        """
        if not isinstance(raw, dict):
            return None
        return cls(
            user_id=str(user_id),
            reason=str(raw.get("reason") or ""),
            operator=str(raw.get("operator") or ""),
            created_at=float(raw.get("created_at") or 0.0),
            expires_at=float(raw.get("expires_at") or 0.0),
            nickname=str(raw.get("nickname") or ""),
        )


class BlacklistStore:
    """Persistent, group-set-scoped blacklist.

    Attributes:
        _store: Backing JSON store; shape is ``{scope: {user_id: entry}}``.
    """

    def __init__(self, data_dir: Path, *, enabled: bool = True) -> None:
        """Prepare the store.

        Args:
            data_dir: Plugin data directory.
            enabled: When ``False`` every lookup reports "not blacklisted" and
                no data is written.
        """
        self._enabled = bool(enabled)
        self._store = JsonStore(data_dir / "blacklist.json")

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def is_blacklisted(self, scope: str, user_id: str) -> bool:
        """Check whether a user is currently blacklisted.

        Expired entries are pruned lazily on lookup.

        Args:
            scope: Blacklist bucket, usually the group set name.
            user_id: QQ number to test.

        Returns:
            ``True`` when an active entry exists.
        """
        return self.get(scope, user_id) is not None

    def get(self, scope: str, user_id: str) -> BlacklistEntry | None:
        """Fetch a user's entry, dropping it when expired.

        Args:
            scope: Blacklist bucket.
            user_id: QQ number to look up.

        Returns:
            The active entry, or ``None``.
        """
        if not self._enabled:
            return None

        bucket = self._bucket(scope)
        raw = bucket.get(str(user_id))
        if raw is None:
            return None

        entry = BlacklistEntry.from_dict(str(user_id), raw)
        if entry is None:
            bucket.pop(str(user_id), None)
            self._store.save()
            return None

        if entry.is_expired:
            bucket.pop(str(user_id), None)
            self._store.save()
            return None
        return entry

    def list_entries(self, scope: str) -> list[BlacklistEntry]:
        """List every active entry in a scope.

        Expired entries are pruned as a side effect.

        Args:
            scope: Blacklist bucket.

        Returns:
            Entries sorted by creation time, oldest first.
        """
        if not self._enabled:
            return []

        bucket = self._bucket(scope, create=False)
        entries: list[BlacklistEntry] = []
        expired: list[str] = []
        for user_id, raw in bucket.items():
            entry = BlacklistEntry.from_dict(str(user_id), raw)
            if entry is None or entry.is_expired:
                expired.append(str(user_id))
                continue
            entries.append(entry)

        if expired:
            for user_id in expired:
                bucket.pop(user_id, None)
            self._store.save()

        entries.sort(key=lambda item: item.created_at)
        return entries

    def count(self, scope: str) -> int:
        """Count active entries in a scope.

        Args:
            scope: Blacklist bucket.

        Returns:
            Number of active entries.
        """
        return len(self.list_entries(scope))

    # ------------------------------------------------------------------ #
    #  Mutations
    # ------------------------------------------------------------------ #

    def add(
        self,
        scope: str,
        user_id: str,
        *,
        reason: str,
        operator: str,
        duration: int = 0,
        nickname: str = "",
    ) -> BlacklistEntry:
        """Add or replace an entry.

        Args:
            scope: Blacklist bucket.
            user_id: QQ number to blacklist.
            reason: Mandatory justification, already defaulted by the caller.
            operator: QQ number of the operator.
            duration: Lifetime in seconds; ``0`` means permanent.
            nickname: Display name to remember.

        Returns:
            The stored entry.
        """
        now = time.time()
        entry = BlacklistEntry(
            user_id=str(user_id),
            reason=str(reason or ""),
            operator=str(operator),
            created_at=now,
            expires_at=now + duration if duration > 0 else 0.0,
            nickname=str(nickname or ""),
        )
        if self._enabled:
            self._bucket(scope)[str(user_id)] = entry.to_dict()
            self._store.save()
        return entry

    def remove(self, scope: str, user_id: str) -> bool:
        """Remove an entry.

        Args:
            scope: Blacklist bucket.
            user_id: QQ number to remove.

        Returns:
            ``True`` when an entry was present and removed.
        """
        bucket = self._bucket(scope, create=False)
        if str(user_id) not in bucket:
            return False
        bucket.pop(str(user_id))
        self._store.save()
        return True

    def clear(self, scope: str) -> int:
        """Drop every entry in a scope.

        Args:
            scope: Blacklist bucket.

        Returns:
            Number of entries removed.
        """
        bucket = self._bucket(scope, create=False)
        count = len(bucket)
        if count:
            # Pop the same normalized key ``_bucket`` uses, not the raw scope.
            self._store.data.pop(str(scope or "unknown"), None)
            self._store.save()
        return count

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _bucket(self, scope: str, *, create: bool = True) -> dict[str, Any]:
        """Return the raw storage bucket for a scope.

        Args:
            scope: Blacklist bucket name.
            create: Whether to create the bucket when it is missing.

        Returns:
            The mutable bucket mapping. A throwaway dict is returned when
            ``create`` is ``False`` and the bucket does not exist.
        """
        key = str(scope or "unknown")
        bucket = self._store.data.get(key)
        if isinstance(bucket, dict):
            return bucket
        if not create:
            return {}
        new_bucket: dict[str, Any] = {}
        self._store.data[key] = new_bucket
        return new_bucket
