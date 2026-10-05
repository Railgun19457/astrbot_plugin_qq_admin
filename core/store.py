"""Small JSON persistence helper.

The plugin keeps its mutable state in ``data/plugin_data/<plugin>/`` rather than
inside the plugin directory, so that reinstalling or updating the plugin never
clobbers user data. Writes are atomic (temp file plus :func:`os.replace`) to
avoid leaving a truncated file behind if the process dies mid-write.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from astrbot.api import logger


class JsonStore:
    """Load and save a single JSON document.

    Attributes:
        path: Location of the backing file.
        _data: In-memory document, always a dict.
    """

    def __init__(self, path: Path, *, default: dict[str, Any] | None = None) -> None:
        """Bind the store to a file and load its contents.

        Args:
            path: Absolute path of the JSON file.
            default: Document used when the file is missing or unreadable.
        """
        self.path = path
        self._data: dict[str, Any] = dict(default or {})
        self.load()

    @property
    def data(self) -> dict[str, Any]:
        """The in-memory document.

        Returns:
            The current document; mutating it directly is allowed but the
            change is not persisted until :meth:`save` runs.
        """
        return self._data

    def load(self) -> None:
        """Read the file into memory.

        A missing file is normal on first run. Corrupt JSON is reported and
        replaced by an empty document rather than crashing the plugin, so a
        damaged file degrades into "no data" instead of "plugin unusable".
        """
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"[QQAdmin] 读取 {self.path.name} 失败（{exc}），将从空数据开始",
            )
            self._data = {}
            return
        self._data = raw if isinstance(raw, dict) else {}

    def save(self) -> None:
        """Persist the document atomically.

        Failures are logged but never raised: losing a write must not break a
        group management action that has already been executed.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = handle.name
                json.dump(self._data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except OSError as exc:
            logger.warning(f"[QQAdmin] 写入 {self.path.name} 失败：{exc}")
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
