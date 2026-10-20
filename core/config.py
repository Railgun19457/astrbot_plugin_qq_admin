"""Configuration parsing and normalization.

This module deliberately contains only pure functions (plus logging) so that
the parsing rules stay easy to reason about and reuse from both the command
handlers and the LLM tools.
"""

from __future__ import annotations

import re
from typing import Any

from astrbot.api import logger

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

#: ``set_group_ban`` refuses anything longer than 30 days.
MAX_BAN_DURATION: int = 30 * 24 * 3600

#: Fallback duration when the user omits it entirely (10 minutes).
DEFAULT_BAN_DURATION: int = 10 * 60

#: Multipliers for the ``10s`` / ``10m`` / ``2h`` / ``1d`` duration syntax.
_TIME_UNITS: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400}

#: Capability keys used by ``permissions.capability_level``.
CAPABILITY_KEYS: tuple[str, ...] = (
    "ban",
    "whole_ban",
    "kick",
    "recall",
    "notice",
    "blacklist",
    "query",
    "review",
    "export",
)

#: Permission levels, ordered from loosest to strictest.
PERMISSION_LEVELS: tuple[str, ...] = ("any", "group_admin", "astrbot_admin")

#: The only group role values OneBot may return for a privileged member.
GROUP_PRIVILEGED_ROLES: frozenset[str] = frozenset({"owner", "admin"})

#: Special argument meaning "everyone in this group".
ALL_TARGET_KEYWORD = "全体"

_TIME_RE = re.compile(r"^(\d+)\s*([smhd])?$", re.IGNORECASE)


# ------------------------------------------------------------------ #
#  Duration helpers
# ------------------------------------------------------------------ #


def parse_duration(text: str, default: int | None = DEFAULT_BAN_DURATION) -> int | None:
    """Parse a human duration such as ``10``, ``10m``, ``2h`` or ``1d``.

    A bare number is interpreted as **seconds**.

    Args:
        text: Raw text from the command argument.
        default: Value returned when ``text`` is empty. Pass ``None`` to make
            an empty input an error instead.

    Returns:
        Duration in seconds, clamped to ``MAX_BAN_DURATION``. Returns ``default``
        when ``text`` is blank, or ``None`` when parsing fails and no default
        applies.
    """
    raw = str(text or "").strip()
    if not raw:
        return default

    match = _TIME_RE.match(raw)
    if not match:
        return None

    amount = int(match.group(1))
    unit = (match.group(2) or "s").lower()
    seconds = amount * _TIME_UNITS[unit]
    if seconds <= 0:
        return None
    return min(seconds, MAX_BAN_DURATION)


def format_duration(seconds: int) -> str:
    """Render a duration in seconds as a compact human readable string.

    Args:
        seconds: Duration in seconds. ``0`` is rendered as ``0 秒``.

    Returns:
        A string such as ``10 分钟``, ``2 小时`` or ``1 天 2 小时``.
    """
    if seconds <= 0:
        return "0 秒"

    parts: list[str] = []
    remaining = int(seconds)
    for unit_seconds, label in ((86400, "天"), (3600, "小时"), (60, "分钟")):
        if remaining >= unit_seconds:
            count, remaining = divmod(remaining, unit_seconds)
            parts.append(f"{count} {label}")
    if remaining:
        parts.append(f"{remaining} 秒")
    return " ".join(parts)


# ------------------------------------------------------------------ #
#  Command argument helpers
# ------------------------------------------------------------------ #


def is_all_target(text: str) -> bool:
    """Return whether the argument means "the whole group".

    Args:
        text: Raw argument text.

    Returns:
        ``True`` for ``全体``, ``所有人`` or ``all`` (case insensitive).
    """
    return str(text or "").strip().lower() in {ALL_TARGET_KEYWORD, "所有人", "all"}


# ------------------------------------------------------------------ #
#  Group set normalization
# ------------------------------------------------------------------ #


def normalize_id_list(raw: Any) -> list[str]:
    """Normalize a config list of identifiers into a list of plain strings.

    Accepts ints, floats (e.g. ``123.0`` from JSON round-trips) and strings.
    Empty entries are dropped and duplicates removed while preserving order.

    Args:
        raw: Raw value from the plugin configuration.

    Returns:
        A de-duplicated list of non-empty strings.
    """
    if not isinstance(raw, (list, tuple, set)):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalize_group_sets(raw: Any) -> list[dict[str, Any]]:
    """Normalize the ``group_sets`` configuration into a predictable shape.

    Args:
        raw: Raw value from the plugin configuration.

    Returns:
        A list of dicts with the keys ``name``, ``group_ids``,
        ``manage_session`` and ``notice_enabled``. Entries without a name and
        without any group id are skipped.
    """
    if not isinstance(raw, (list, tuple)):
        return []

    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning(f"[QQAdmin] 群组配置第 {index + 1} 项不是对象，已忽略")
            continue

        group_ids = normalize_id_list(item.get("group_ids"))
        name = str(item.get("name") or "").strip()
        if not group_ids and not name:
            continue
        if not name:
            name = f"群组{index + 1}"

        result.append(
            {
                "name": name,
                "group_ids": group_ids,
                "manage_session": str(item.get("manage_session") or "").strip(),
                "notice_enabled": bool(item.get("notice_enabled", True)),
            }
        )
    return result


def build_group_index(group_sets: list[dict[str, Any]]) -> dict[str, str]:
    """Build a ``group_id -> group set name`` index.

    When the same group id appears in multiple sets the first occurrence wins
    and a warning is logged; conflicting entries are never silently dropped.
    A duplicate *set name* is warned about as well, because the name doubles as
    the blacklist bucket key and would silently merge two sets' entries.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.

    Returns:
        Mapping from group id to the owning set name.
    """
    names = [group_set["name"] for group_set in group_sets]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        logger.warning(
            f"[QQAdmin] 群组名称重复：{'、'.join(duplicated)}。"
            "同名群组会共享同一份黑名单，请改用不重名的群组名称",
        )

    index: dict[str, str] = {}
    for group_set in group_sets:
        name = group_set["name"]
        for group_id in group_set["group_ids"]:
            existing = index.get(group_id)
            if existing is not None:
                logger.warning(
                    f"[QQAdmin] 群 {group_id} 同时出现在群组「{existing}」和"
                    f"「{name}」中，以先出现的「{existing}」为准",
                )
                continue
            index[group_id] = name
    return index


def find_group_set(
    group_sets: list[dict[str, Any]],
    group_id: str,
) -> dict[str, Any] | None:
    """Find the group set owning ``group_id``.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.
        group_id: QQ group id.

    Returns:
        The owning group set, or ``None`` when the group belongs to no set.
    """
    if not group_id:
        return None
    for group_set in group_sets:
        if group_id in group_set["group_ids"]:
            return group_set
    return None


def blacklist_scope(group_id: str) -> str:
    """Return the fallback blacklist bucket key for a group.

    Only used when the group belongs to no configured set, so that
    blacklisting keeps working on unconfigured groups.

    Args:
        group_id: QQ group id.

    Returns:
        The bucket key used by the blacklist store.
    """
    return group_id or "unknown"


def resolve_blacklist_scope(
    group_sets: list[dict[str, Any]],
    group_id: str,
) -> str:
    """Resolve the blacklist bucket a group reads from and writes to.

    Groups belonging to a configured set share that set's bucket, which is what
    implements the "shared blacklist across a group set" behaviour. Groups
    outside every set get a private bucket instead of an error.

    Args:
        group_sets: Normalized group set definitions.
        group_id: QQ group id.

    Returns:
        The bucket key; the group id itself when the group is unconfigured.
    """
    group_set = find_group_set(group_sets, group_id)
    if group_set:
        return str(group_set["name"])
    return blacklist_scope(group_id)


# ------------------------------------------------------------------ #
#  Permission normalization
# ------------------------------------------------------------------ #


def normalize_permission_level(value: Any, *, fallback: str = "group_admin") -> str:
    """Normalize a configured permission level.

    Args:
        value: Raw value from configuration.
        fallback: Level used when ``value`` is missing or unknown.

    Returns:
        One of :data:`PERMISSION_LEVELS`.
    """
    text = str(value or "").strip().lower()
    return text if text in PERMISSION_LEVELS else fallback


def normalize_bool_list(raw: Any) -> list[str]:
    """Normalize a list of plain string flags (lowercased, de-duplicated).

    Args:
        raw: Raw value from configuration.

    Returns:
        Lowercased, de-duplicated list of non-empty strings.
    """
    return [item.lower() for item in normalize_id_list(raw) if item.strip()]


def normalize_words(raw: Any, fallback: list[str]) -> list[str]:
    """Normalize a list of confirmation keywords.

    Args:
        raw: Raw value from configuration.
        fallback: Words used when the configured list is empty.

    Returns:
        The normalized word list, falling back to ``fallback``.
    """
    words = normalize_id_list(raw)
    return words or list(fallback)
