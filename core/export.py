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
class MemberSheet:
    """One group's roster inside an export.

    Attributes:
        group_id: Group the roster belongs to.
        members: Raw member payloads from ``get_group_member_list``.
        group_name: Group display name, used in the sheet title.
    """

    group_id: str
    members: list[dict[str, Any]]
    group_name: str = ""


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


#: Columns of a group member roster.
_MEMBER_HEADERS = ["QQ号", "群名片", "昵称", "群身份", "等级", "入群时间"]

#: Columns of a blacklist sheet. ``拉黑所在群`` exists because the blacklist is
#: shared by a whole group set: it records where the entry came from.
_BLACKLIST_HEADERS = ["QQ号", "昵称", "拉黑原因", "拉黑所在群", "操作人", "拉黑时间"]


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
        sheets: list[MemberSheet],
        *,
        scope: str = "",
    ) -> ExportResult:
        """Export one or more group rosters into a single workbook.

        Every group gets its own worksheet, so a manage session can export all
        of its groups at once without merging unrelated rosters into one table.

        Args:
            sheets: Rosters to export, in display order.
            scope: Group set name the rosters belong to, used in the file name.

        Returns:
            The export outcome.
        """
        try:
            from openpyxl import Workbook
        except ImportError:
            return ExportResult(error="未安装 openpyxl，请先执行 pip install openpyxl")

        workbook = Workbook()
        workbook.remove(workbook.active)
        total = 0
        truncated = False
        for sheet in sheets:
            clipped = len(sheet.members) > self._max_members
            rows = sheet.members[: self._max_members]
            truncated = truncated or clipped
            data = [
                [
                    str(member.get("user_id") or ""),
                    str(member.get("card") or ""),
                    str(member.get("nickname") or ""),
                    _role_label(str(member.get("role") or "")),
                    str(member.get("level") or ""),
                    _format_timestamp(member.get("join_time")),
                ]
                for member in rows
            ]
            total += len(data)
            self._fill_sheet(
                workbook.create_sheet(_sheet_title(sheet)),
                headers=_MEMBER_HEADERS,
                rows=data,
                heading=f"群成员名单 - {sheet.group_name or sheet.group_id}",
                note=(
                    f"共 {len(sheet.members)} 人，仅显示前 {self._max_members} 条"
                    if clipped
                    else f"共 {len(sheet.members)} 人"
                ),
            )

        safe_scope = _safe_name(scope)
        return self._save(
            workbook,
            filename=f"members-{safe_scope}-{_timestamp()}.xlsx",
            row_count=total,
            truncated=truncated,
        )

    def export_blacklist(
        self,
        entries: list[BlacklistEntry],
        *,
        scope: str = "",
        groups: str = "",
    ) -> ExportResult:
        """Export every blacklist entry an operator can see.

        Args:
            entries: Entries to export.
            scope: Group set name, used in the file name.
            groups: Human readable group list for the subtitle.

        Returns:
            The export outcome.
        """
        try:
            from openpyxl import Workbook
        except ImportError:
            return ExportResult(error="未安装 openpyxl，请先执行 pip install openpyxl")

        data = [
            [
                entry.user_id,
                entry.nickname,
                entry.reason,
                entry.group_id,
                entry.operator,
                _format_timestamp(entry.created_at),
            ]
            for entry in entries
        ]
        workbook = Workbook()
        self._fill_sheet(
            workbook.active,
            headers=_BLACKLIST_HEADERS,
            rows=data,
            heading=f"黑名单 - {scope or '全部'}",
            note=f"共 {len(entries)} 人{groups}",
        )
        safe_scope = _safe_name(scope)
        return self._save(
            workbook,
            filename=f"blacklist-{safe_scope}-{_timestamp()}.xlsx",
            row_count=len(data),
        )

    @staticmethod
    def _fill_sheet(
        worksheet: Any,
        *,
        headers: list[str],
        rows: list[list[Any]],
        heading: str,
        note: str,
    ) -> None:
        """Write a title, subtitle, header row and data into a worksheet.

        Args:
            worksheet: An openpyxl worksheet.
            headers: Column headers.
            rows: Data rows.
            heading: Merged title row content.
            note: Subtitle describing the data set.
        """
        from openpyxl.styles import Alignment, Font

        worksheet.append([heading])
        worksheet.cell(row=1, column=1).font = Font(bold=True, size=13)
        worksheet.append([note])
        worksheet.append(headers)
        for cell in worksheet[3]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        for row in rows:
            worksheet.append(row)
        worksheet.freeze_panes = "A4"
        _autosize(worksheet)

    def _save(
        self,
        workbook: Any,
        *,
        filename: str,
        row_count: int,
        truncated: bool = False,
    ) -> ExportResult:
        """Write the workbook to disk.

        Args:
            workbook: An openpyxl workbook.
            filename: File name to use.
            row_count: Number of data rows written.
            truncated: Whether ``max_members`` clipped the data set.

        Returns:
            The export outcome.
        """
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
            row_count=row_count,
            truncated=truncated,
        )


def _safe_name(text: str) -> str:
    """Turn a user supplied name into a safe file name fragment.

    A scope is a user chosen group set name, so it cannot be copied into a file
    name verbatim: separators would escape the output directory and characters
    such as ``:`` are invalid on Windows.

    Args:
        text: Raw name.

    Returns:
        A sanitized fragment, or ``all`` when nothing usable is left.
    """
    return re.sub(r"[^\w.-]", "_", str(text)).strip("._")[:60] or "all"


def _sheet_title(sheet: MemberSheet) -> str:
    """Build a worksheet title for a group roster.

    Excel rejects sheet titles longer than 31 characters and the characters
    ``[]:*?/\\``, so the group id is kept and the name is trimmed around it.

    Args:
        sheet: Roster to title.

    Returns:
        A worksheet title.
    """
    raw = f"{sheet.group_id} {sheet.group_name}".strip() or "群成员"
    return re.sub(r"[\[\]:*?/\\]", "_", raw)[:31]


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
