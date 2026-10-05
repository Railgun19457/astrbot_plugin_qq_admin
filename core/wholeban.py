"""Persistent schedule for temporary whole-group mutes.

OneBot's ``set_group_whole_ban`` is a plain switch: the server keeps no timer, so
a timed mute has to be lifted by the plugin itself. Keeping that timer only in
memory would lose it across a restart, leaving a group muted forever.

The intended expiry time is therefore written to disk and the timer is
re-armed on load. Entries are also verified against the server on startup,
because the group may have been unmuted manually while the bot was down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import JsonStore

#: A timer that is already past its deadline when restored still gets this many
#: seconds of grace, so the unmute call is not skipped just because the bot
#: started a moment too late.
RESTORE_GRACE: int = 5


@dataclass
class WholeBanRecord:
    """A scheduled whole-group mute.

    Attributes:
        group_id: Group that is muted.
        expires_at: Unix timestamp when the mute should be lifted.
        duration: Originally requested duration in seconds.
        operator: QQ number of the person who muted the group.
    """

    group_id: str
    expires_at: float
    duration: int = 0
    operator: str = ""

    @property
    def remaining(self) -> int:
        """Seconds left before the mute expires (may be negative)."""
        return int(self.expires_at - time.time())

    @property
    def is_expired(self) -> bool:
        """Whether the deadline has already passed."""
        return time.time() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage.

        Returns:
            A JSON-compatible mapping.
        """
        return {
            "expires_at": self.expires_at,
            "duration": self.duration,
            "operator": self.operator,
        }

    @classmethod
    def from_dict(cls, group_id: str, raw: Any) -> WholeBanRecord | None:
        """Rebuild a record from stored data.

        Args:
            group_id: Key the record was stored under.
            raw: Stored payload.

        Returns:
            The parsed record, or ``None`` when the payload is unusable.
        """
        if not isinstance(raw, dict):
            return None
        try:
            expires_at = float(raw.get("expires_at") or 0)
        except (TypeError, ValueError):
            return None
        if expires_at <= 0:
            return None
        return cls(
            group_id=str(group_id),
            expires_at=expires_at,
            duration=int(raw.get("duration") or 0),
            operator=str(raw.get("operator") or ""),
        )


class WholeBanStore:
    """Persistent store of pending whole-group mute deadlines.

    Attributes:
        _store: Backing JSON store keyed by group id.
    """

    def __init__(self, data_dir: Path) -> None:
        """Load the schedule from disk.

        Loading happens here rather than in a separate ``load`` method so that
        recovery can stay synchronous and run from ``__init__``.

        Args:
            data_dir: Plugin data directory.
        """
        self._store = JsonStore(data_dir / "whole_ban.json")

    def record(
        self,
        group_id: str,
        *,
        expires_at: float,
        duration: int = 0,
        operator: str = "",
    ) -> WholeBanRecord:
        """Persist a scheduled mute.

        Args:
            group_id: Group that is muted.
            expires_at: Unix timestamp when the mute should be lifted.
            duration: Originally requested duration in seconds.
            operator: QQ number of the person who muted the group.

        Returns:
            The stored record.
        """
        entry = WholeBanRecord(
            group_id=str(group_id),
            expires_at=float(expires_at),
            duration=int(duration),
            operator=str(operator),
        )
        self._store.data[str(group_id)] = entry.to_dict()
        self._store.save()
        return entry

    def get(self, group_id: str) -> WholeBanRecord | None:
        """Look up a pending mute.

        Args:
            group_id: Group to look up.

        Returns:
            The record, or ``None`` when no mute is scheduled.
        """
        raw = self._store.data.get(str(group_id))
        if raw is None:
            return None
        record = WholeBanRecord.from_dict(str(group_id), raw)
        if record is None:
            # Drop the unusable entry so it does not linger forever.
            self.remove(group_id)
        return record

    def remove(self, group_id: str) -> bool:
        """Forget a group's scheduled mute.

        Called when the deadline passes, when an operator lifts the mute early,
        or when startup verification finds the group is no longer muted.

        Args:
            group_id: Group to forget.

        Returns:
            ``True`` when an entry was present and removed.
        """
        if str(group_id) not in self._store.data:
            return False
        self._store.data.pop(str(group_id))
        self._store.save()
        return True

    def load_all(self) -> list[WholeBanRecord]:
        """Load every pending record, dropping unusable ones.

        Returns:
            Records sorted by deadline, earliest first.
        """
        records: list[WholeBanRecord] = []
        broken: list[str] = []
        for group_id, raw in self._store.data.items():
            record = WholeBanRecord.from_dict(str(group_id), raw)
            if record is None:
                broken.append(str(group_id))
                continue
            records.append(record)

        if broken:
            for group_id in broken:
                self._store.data.pop(group_id, None)
            self._store.save()

        records.sort(key=lambda item: item.expires_at)
        return records

    def __len__(self) -> int:
        return len(self._store.data)
