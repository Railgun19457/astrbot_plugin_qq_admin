"""Append-only audit trail backed by a JSON Lines file.

Every management action goes through :meth:`AuditLog.record` so that operators
can be held accountable later. The file is trimmed once at plugin load rather
than on every write, keeping the hot path a plain append.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import logger


@dataclass
class AuditEntry:
    """A single recorded action.

    Attributes:
        ts: Unix timestamp of the action.
        action: Machine readable action name, e.g. ``ban`` or ``kick``.
        operator: QQ number of the person who triggered the action.
        target: QQ number the action applied to, when applicable.
        group_id: Group the action happened in.
        scope: Blacklist bucket key, when relevant to the action.
        detail: Free-form extra context such as the reason or duration.
        result: ``"ok"`` or ``"failed"``.
        error: Failure description, empty on success.
    """

    ts: float
    action: str
    operator: str
    target: str = ""
    group_id: str = ""
    scope: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    result: str = "ok"
    error: str = ""


class AuditLog:
    """JSON Lines audit log with a retention window.

    Attributes:
        _path: Location of the underlying ``.jsonl`` file.
        _retention_days: Entries older than this are dropped on load.
        _enabled: Whether writes are recorded at all.
    """

    def __init__(
        self, data_dir: Path, *, enabled: bool = True, retention_days: int = 30
    ) -> None:
        """Prepare the audit log.

        Args:
            data_dir: Plugin data directory.
            enabled: When ``False``, :meth:`record` becomes a no-op.
            retention_days: Entries older than this many days are discarded;
                values ``<= 0`` disable trimming entirely.
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "audit.jsonl"
        self._enabled = bool(enabled)
        self._retention_days = max(0, int(retention_days))

    @property
    def path(self) -> Path:
        """Absolute path of the audit file."""
        return self._path

    def trim(self) -> int:
        """Drop entries older than the retention window.

        Intended to be called once during plugin startup.

        Returns:
            Number of entries removed.
        """
        if self._retention_days <= 0 or not self._path.exists():
            return 0

        cutoff = time.time() - self._retention_days * 86400
        kept: list[str] = []
        removed = 0
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        removed += 1
                        continue
                    if float(payload.get("ts") or 0) < cutoff:
                        removed += 1
                        continue
                    kept.append(stripped)
        except OSError as exc:
            logger.warning(f"[QQAdmin] 读取审计日志失败：{exc}")
            return 0

        if removed == 0:
            return 0

        try:
            with self._path.open("w", encoding="utf-8") as handle:
                for line in kept:
                    handle.write(line + "\n")
        except OSError as exc:
            logger.warning(f"[QQAdmin] 重写审计日志失败：{exc}")
            return 0

        logger.info(f"[QQAdmin] 审计日志已清理 {removed} 条过期记录")
        return removed

    def record(self, entry: AuditEntry) -> None:
        """Append one entry to the log.

        Failures are logged and swallowed: audit problems must never break a
        group management action.

        Args:
            entry: The entry to persist.
        """
        if not self._enabled:
            return
        try:
            payload = asdict(entry)
            if not payload.get("ts"):
                payload["ts"] = time.time()
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"[QQAdmin] 写入审计日志失败：{exc}")

    def log(
        self,
        action: str,
        operator: str,
        *,
        target: str = "",
        group_id: str = "",
        scope: str = "",
        result: str = "ok",
        error: str = "",
        **detail: Any,
    ) -> None:
        """Convenience wrapper building an :class:`AuditEntry`.

        Args:
            action: Machine readable action name.
            operator: QQ number of the operator.
            target: QQ number the action applied to.
            group_id: Group the action happened in.
            scope: Blacklist bucket key.
            result: ``"ok"`` or ``"failed"``.
            error: Failure description.
            **detail: Extra context stored verbatim in the entry.
        """
        self.record(
            AuditEntry(
                ts=time.time(),
                action=action,
                operator=str(operator),
                target=str(target),
                group_id=str(group_id),
                scope=str(scope),
                detail=detail,
                result=result,
                error=error,
            )
        )
