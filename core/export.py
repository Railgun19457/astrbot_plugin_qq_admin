"""Spreadsheet export for member lists and the blacklist.

The output is a real ``.xlsx`` written with openpyxl and uploaded to the group
file list through ``upload_group_file``. When the upload fails the local path is
returned instead, so the operator can still retrieve the data.

The blacklist sheet deliberately includes the ``reason`` column: it is a
first-class field collected from the command and the LLM tool.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from .blacklist import BlacklistEntry


@dataclass
class ExportResult:
    """Outcome of building an export file.

    Attributes:
        path: Absolute path of the generated file, empty on failure.
        filename: Suggested file name.
        row_count: Number of data rows written.
        truncated: Whether ``max_rows`` clipped the data set.
        error: Failure description, empty on success.
    """

    path: str = ""
    filename: str = ""
    row_count: int = 0
    truncated: bool = False
    error: str = ""

    def __bool__(self) -> bool:
        return bool(self.path) and not self.error


def _timestamp() -> str:
    """Render a compact timestamp suitable for file names.

    Returns:
        A string such as ``20261002-181500``.
    """
    return time.strftime("%Y%m%d-%H%M%S")


def _autosize(worksheet: Any) -> None:
    """Approximate column widths from cell contents.

    Args:
        worksheet: An openpyxl worksheet.
    """
    for column in worksheet.columns:
        longest = 0
        letter = ""
        for cell in column:
            letter = cell.column_letter
            text = str(cell.value) if cell.value is not None else ""
            # CJK characters occupy roughly two character widths.
            width = sum(2 if ord(char) > 0x2E80 else 1 for char in text)
            longest = max(longest, width)
        if letter:
            worksheet.column_dimensions[letter].width = min(max(longest + 2, 8), 60)


class Exporter:
    """Builds xlsx exports for a group.

    Attributes:
        _output_dir: Directory the generated files are written to.
        _max_members: Row cap for the member listing.
        _cleanup_hours: Age at which previously generated files are deleted.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        max_members: int = 3000,
        cleanup_hours: int = 24,
    ) -> None:
        """Prepare the exporter.

        Args:
            output_dir: Directory for generated files.
            max_members: Row cap for member listings, so a huge group cannot
                produce a multi-hundred-megabyte upload.
            cleanup_hours: Files older than this are deleted on startup.
        """
        self._output_dir = output_dir
        self._max_members = max(1, int(max_members))
        self._cleanup_hours = max(0, int(cleanup_hours))

    def cleanup(self) -> int:
        """Delete export files older than the retention window.

        Intended to run once during plugin startup.

        Returns:
            Number of files removed.
        """
        if self._cleanup_hours <= 0 or not self._output_dir.exists():
            return 0

        cutoff = time.time() - self._cleanup_hours * 3600
        removed = 0
        for path in self._output_dir.glob("*.xlsx"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning(f"[QQAdmin] 清理导出文件失败（{path.name}）：{exc}")
        if removed:
            logger.info(f"[QQAdmin] 已清理 {removed} 个过期导出文件")
        return removed

    def export_members(
        self,
        group_id: str,
        members: list[dict[str, Any]],
        *,
        group_name: str = "",
    ) -> ExportResult:
        """Export a group member roster.

        Args:
            group_id: Group the roster belongs to.
            members: Raw member payloads from ``get_group_member_list``.
            group_name: Group display name, used in the sheet title.

        Returns:
            The export outcome.
        """
        truncated = len(members) > self._max_members
        rows = members[: self._max_members]

        headers = ["QQ号", "群名片", "昵称", "群身份", "等级", "入群时间"]
        data: list[list[Any]] = []
        for member in rows:
            data.append(
                [
                    str(member.get("user_id") or ""),
                    str(member.get("card") or ""),
                    str(member.get("nickname") or ""),
                    _role_label(str(member.get("role") or "")),
                    str(member.get("level") or ""),
                    _format_timestamp(member.get("join_time")),
                ],
            )

        title = f"群成员名单 - {group_name or group_id}"
        return self._write(
            sheet_title="群成员",
            headers=headers,
            rows=data,
            filename=f"members-{group_id}-{_timestamp()}.xlsx",
            heading=title,
            truncated=truncated,
            note=(
                f"共 {len(members)} 人，仅显示前 {self._max_members} 条"
                if truncated
                else f"共 {len(members)} 人"
            ),
        )

    def export_blacklist(
        self,
        scope: str,
        entries: list[BlacklistEntry],
        *,
        group_ids: list[str] | None = None,
    ) -> ExportResult:
        """Export the blacklist of a group set.

        Args:
            scope: Blacklist bucket name (usually the group set name).
            entries: Entries to export.
            group_ids: Groups sharing this bucket, listed in the sheet note.

        Returns:
            The export outcome.
        """
        headers = ["QQ号", "昵称", "拉黑原因", "操作人", "拉黑时间"]
        data: list[list[Any]] = []
        for entry in entries:
            data.append(
                [
                    entry.user_id,
                    entry.nickname,
                    entry.reason,
                    entry.operator,
                    _format_timestamp(entry.created_at),
                ],
            )

        covered = f"，覆盖 {len(group_ids)} 个群" if group_ids else ""
        # The scope is a user chosen group set name, so it cannot be copied
        # into a file name verbatim: separators would escape the output
        # directory and characters such as ``:`` are invalid on Windows.
        safe_scope = re.sub(r"[^\w.-]", "_", str(scope)).strip("._")[:60] or "scope"
        return self._write(
            sheet_title="黑名单",
            headers=headers,
            rows=data,
            filename=f"blacklist-{safe_scope}-{_timestamp()}.xlsx",
            heading=f"黑名单 - {scope}",
            note=f"共 {len(entries)} 人{covered}",
        )

    def _write(
        self,
        *,
        sheet_title: str,
        headers: list[str],
        rows: list[list[Any]],
        filename: str,
        heading: str,
        note: str,
        truncated: bool = False,
    ) -> ExportResult:
        """Write a single-sheet workbook.

        Args:
            sheet_title: Worksheet name.
            headers: Column headers.
            rows: Data rows.
            filename: File name to use.
            heading: Merged title row content.
            note: Subtitle describing the data set.
            truncated: Whether ``max_members`` clipped the data set.

        Returns:
            The export outcome.
        """
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font
        except ImportError:
            return ExportResult(error="未安装 openpyxl，请先执行 pip install openpyxl")

        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = sheet_title

        # Title row, then a subtitle, then the header row.
        worksheet.append([heading])
        first_column = worksheet.cell(row=1, column=1)
        first_column.font = Font(bold=True, size=13)

        worksheet.append([note])
        worksheet.append(headers)
        header_row = worksheet[3]
        for cell in header_row:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

        for row in rows:
            worksheet.append(row)
        worksheet.freeze_panes = "A4"
        _autosize(worksheet)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        target = self._output_dir / filename
        try:
            workbook.save(target)
        except OSError as exc:
            logger.warning(f"[QQAdmin] 保存导出文件失败：{exc}")
            return ExportResult(error=str(exc))

        return ExportResult(
            path=str(target.resolve()),
            filename=filename,
            row_count=len(rows),
            truncated=truncated,
        )


def _role_label(role: str) -> str:
    """Translate an OneBot role into Chinese.

    Args:
        role: ``owner``, ``admin``, ``member`` or empty.

    Returns:
        A display label.
    """
    return {"owner": "群主", "admin": "管理员", "member": "成员"}.get(
        role, role or "未知"
    )


def _format_timestamp(value: Any) -> str:
    """Render a Unix timestamp as local time.

    Args:
        value: Seconds since the epoch; ``0`` or ``None`` yields an empty
            string.

    Returns:
        A ``YYYY-MM-DD HH:MM:SS`` string, or an empty string.
    """
    try:
        seconds = float(value or 0)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


def to_file_uri(path: str) -> str:
    """Convert a local path into a ``file://`` URI for OneBot uploads.

    Protocol implementations accept either an absolute path or a ``file://``
    URI; the URI form is used because it survives paths containing spaces.

    Args:
        path: Absolute local path.

    Returns:
        The ``file://`` URI string.
    """
    import pathlib

    return pathlib.Path(path).absolute().as_uri()
