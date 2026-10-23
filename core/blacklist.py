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

Whether the blacklist is enabled, and whether blacklisted messages are
recalled, is decided per group by the caller (see ``core.config``); the store
itself always answers and persists.
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
        nickname: Best known display name at the time of addition.
        group_id: Group the entry was added from. The bucket is shared by a
            whole group set, so this is what tells an operator where the
            blacklisting happened; it is informational only.
    """

    user_id: str
    reason: str
    operator: str
    created_at: float
    nickname: str = ""
    group_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry for storage.

        Returns:
            A JSON-compatible mapping.
        """
        return {
            "reason": self.reason,
            "operator": self.operator,
            "created_at": self.created_at,
            "nickname": self.nickname,
            "group_id": self.group_id,
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
            nickname=str(raw.get("nickname") or ""),
            group_id=str(raw.get("group_id") or ""),
        )


class BlacklistStore:
    """Persistent, group-set-scoped blacklist.

    Attributes:
        _store: Backing JSON store; shape is ``{scope: {user_id: entry}}``.
    """

    def __init__(self, data_dir: Path) -> None:
        """Prepare the store.

        Args:
            data_dir: Plugin data directory.
        """
        self._store = JsonStore(data_dir / "blacklist.json")

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def is_blacklisted(self, scope: str, user_id: str) -> bool:
        """Check whether a user is blacklisted.

        Args:
            scope: Blacklist bucket, usually the group set name.
            user_id: QQ number to test.

        Returns:
            ``True`` when an entry exists.
        """
        return self.get(scope, user_id) is not None

    def get(self, scope: str, user_id: str) -> BlacklistEntry | None:
        """Fetch a user's entry.

        An unreadable entry is dropped so a damaged record does not linger.

        Args:
            scope: Blacklist bucket.
            user_id: QQ number to look up.

        Returns:
            The entry, or ``None``.
        """
        bucket = self._bucket(scope)
        raw = bucket.get(str(user_id))
        if raw is None:
            return None

        entry = BlacklistEntry.from_dict(str(user_id), raw)
        if entry is None:
            bucket.pop(str(user_id), None)
            self._store.save()
            return None
        return entry

    def list_entries(self, scope: str) -> list[BlacklistEntry]:
        """List every entry in a scope.

        Unreadable entries are pruned as a side effect.

        Args:
            scope: Blacklist bucket.

        Returns:
            Entries sorted by creation time, oldest first.
        """
        bucket = self._bucket(scope, create=False)
        entries: list[BlacklistEntry] = []
        broken: list[str] = []
        for user_id, raw in bucket.items():
            entry = BlacklistEntry.from_dict(str(user_id), raw)
            if entry is None:
                broken.append(str(user_id))
                continue
            entries.append(entry)

        if broken:
            for user_id in broken:
                bucket.pop(user_id, None)
            self._store.save()

        entries.sort(key=lambda item: item.created_at)
        return entries

    def scopes(self) -> list[str]:
        """List the buckets that hold entries.

        Returns:
            Bucket keys in the order they were first written.
        """
        return [str(scope) for scope in self._store.data]

    def count(self, scope: str) -> int:
        """Count entries in a scope.

        Args:
            scope: Blacklist bucket.

        Returns:
            Number of entries.
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
        nickname: str = "",
        group_id: str = "",
    ) -> BlacklistEntry:
        """Add or replace an entry.

        Args:
            scope: Blacklist bucket.
            user_id: QQ number to blacklist.
            reason: Mandatory justification, already defaulted by the caller.
            operator: QQ number of the operator.
            nickname: Display name to remember.
            group_id: Group the blacklisting happened in.

        Returns:
            The stored entry.
        """
        entry = BlacklistEntry(
            user_id=str(user_id),
            reason=str(reason or ""),
            operator=str(operator),
            created_at=time.time(),
            nickname=str(nickname or ""),
            group_id=str(group_id or ""),
        )
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
