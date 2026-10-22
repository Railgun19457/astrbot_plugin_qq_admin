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

#: Fallback blacklist reason when neither the command nor the config sets one.
DEFAULT_BLACKLIST_REASON: str = "未填写原因"

#: Fallback rejection reason shown to the applicant.
DEFAULT_REVIEW_REJECT_REASON: str = "未通过进群审核"

#: Request types a keyword answer may be limited to.
REVIEW_QUESTION_TYPES: tuple[str, ...] = ("any", "add", "invite")

#: Actions taken when no review rule matched the applicant.
REVIEW_DEFAULT_ACTIONS: tuple[str, ...] = ("hold", "approve", "reject")

#: Built-in join review policy of a group set.
DEFAULT_REVIEW_POLICY: dict[str, Any] = {
    "enabled": False,
    "blacklist_reject": True,
    "keyword_allow": True,
    "keywords": [],
    "question_type": "any",
    "min_account_level": 0,
    "default_action": "hold",
    "reject_reason": DEFAULT_REVIEW_REJECT_REASON,
}

#: Built-in behaviour settings used by groups that belong to no group set.
DEFAULT_GROUP_SETTINGS: dict[str, Any] = {
    "default_ban_duration": DEFAULT_BAN_DURATION,
    "blacklist": {
        "enabled": True,
        "default_reason": DEFAULT_BLACKLIST_REASON,
        "block_message": False,
        "kick_on_add": False,
    },
}

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

#: Message categories whose delivery route can be configured per session.
#: One category per management function, so an operator can keep, for example,
#: ban receipts in the group while sending join review notices to the manage
#: session only.
NOTIFY_CATEGORY_BAN = "禁言"
NOTIFY_CATEGORY_KICK = "踢人"
NOTIFY_CATEGORY_RECALL = "撤回"
NOTIFY_CATEGORY_BLACKLIST = "黑名单"
NOTIFY_CATEGORY_EXPORT = "群导出"
NOTIFY_CATEGORY_REVIEW_NOTICE = "进群审核通知"

#: Every routable category, in the order the WebUI offers them.
NOTIFY_CATEGORIES: tuple[str, ...] = (
    NOTIFY_CATEGORY_BAN,
    NOTIFY_CATEGORY_KICK,
    NOTIFY_CATEGORY_RECALL,
    NOTIFY_CATEGORY_BLACKLIST,
    NOTIFY_CATEGORY_EXPORT,
    NOTIFY_CATEGORY_REVIEW_NOTICE,
)

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


def normalize_review_policy(raw: Any) -> dict[str, Any]:
    """Normalize the conditions and actions of one join review policy.

    Args:
        raw: Raw ``join_review`` object of a group set, or a merged override.

    Returns:
        A policy dict with every key filled from the built-in defaults.
    """
    source = raw if isinstance(raw, dict) else {}

    try:
        min_level = max(0, int(source.get("min_account_level", 0) or 0))
    except (TypeError, ValueError):
        min_level = 0

    question_type = str(source.get("question_type") or "any").strip().lower()
    if question_type not in REVIEW_QUESTION_TYPES:
        question_type = "any"

    unattended = str(source.get("default_action") or "hold").strip().lower()
    if unattended not in REVIEW_DEFAULT_ACTIONS:
        unattended = "hold"

    keyword_allow = bool(source.get("keyword_allow", True))
    keywords, keywords_by_group = _split_scoped_entries(
        source.get("keywords"),
        split_values=False,
    )
    if not keyword_allow:
        keywords, keywords_by_group = [], {}

    return {
        "enabled": bool(source.get("enabled", False)),
        "blacklist_reject": bool(source.get("blacklist_reject", True)),
        "keyword_allow": keyword_allow,
        "keywords": keywords,
        "keywords_by_group": keywords_by_group,
        "question_type": question_type,
        "min_account_level": min_level,
        "default_action": unattended,
        "reject_reason": (
            str(source.get("reject_reason") or DEFAULT_REVIEW_REJECT_REASON).strip()
            or DEFAULT_REVIEW_REJECT_REASON
        ),
    }


def _split_scope(text: str) -> tuple[str, str]:
    """Split a ``group:value`` entry into its scope and payload.

    A leading number followed by a colon is treated as a QQ group id; anything
    else is an unscoped entry. Both ASCII and full-width colons are accepted.

    Args:
        text: Raw list item.

    Returns:
        ``(group_id, value)``; ``group_id`` is empty for unscoped entries.
    """
    raw = str(text or "").strip()
    for separator in (":", "："):
        if separator in raw:
            head, _, tail = raw.partition(separator)
            if head.strip().isdigit():
                return head.strip(), tail.strip()
    return "", raw


def _split_id_values(raw: Any) -> list[str]:
    """Normalize an identifier list that may be a string or a sequence.

    Args:
        raw: JSON value from the configuration; strings are split on commas,
            Chinese commas, enumeration commas and whitespace.

    Returns:
        A de-duplicated list of non-empty strings.
    """
    if isinstance(raw, str):
        return normalize_id_list(re.split(r"[,，、\s]+", raw))
    return normalize_id_list(raw)


def _split_scoped_entries(
    raw: Any,
    *,
    split_values: bool,
) -> tuple[list[str], dict[str, list[str]]]:
    """Split a list of plain and ``group:value`` entries.

    Args:
        raw: Raw list from the configuration.
        split_values: When ``True`` the payload is split on commas, Chinese
            commas and whitespace (used for group id lists). Otherwise the
            payload is kept verbatim (used for keywords, which may contain
            spaces).

    Returns:
        ``(general, per_group)``: unscoped values in order, and scoped values
        grouped by group id.
    """
    general: list[str] = []
    per_group: dict[str, list[str]] = {}
    if not isinstance(raw, (list, tuple)):
        return general, per_group

    for item in raw:
        group_id, value = _split_scope(str(item or ""))
        if not value:
            continue
        values = _split_id_values(value) if split_values else [value]
        if not group_id:
            general.extend(values)
            continue
        bucket = per_group.setdefault(group_id, [])
        for entry in values:
            if entry not in bucket:
                bucket.append(entry)
    return general, per_group


def normalize_review_settings(raw: Any) -> dict[str, Any]:
    """Normalize a group set's ``join_review`` block.

    Args:
        raw: Raw ``join_review`` object of a group set.

    Returns:
        A dict with ``policy`` (the general rule) and ``exemptions`` (normalized
        cross-group exemption entries).
    """
    source = raw if isinstance(raw, dict) else {}
    return {
        "policy": normalize_review_policy(source),
        "exemptions": normalize_review_exemptions(source.get("exemptions")),
    }


def normalize_review_exemptions(raw: Any) -> list[dict[str, Any]]:
    """Normalize the cross-group review exemptions.

    Entries are plain strings: ``"来源群1,来源群2"`` applies to every group of
    the set, while ``"目标群:来源群1,来源群2"`` applies only to that target. A
    template-style list of objects is accepted too.

    Args:
        raw: Raw ``join_review.exemptions`` value.

    Returns:
        Entries with ``target_group_ids`` (empty means every group) and
        ``source_group_ids``. Entries without sources are skipped.
    """
    if not isinstance(raw, (list, tuple)):
        return []

    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            sources = _split_id_values(item.get("source_group_ids"))
            targets = _split_id_values(item.get("target_group_ids"))
        else:
            target, value = _split_scope(str(item or ""))
            sources = _split_id_values(value)
            targets = [target] if target else []
        if not sources:
            logger.warning(f"[QQAdmin] 跨群免审核第 {index + 1} 项未填写来源群，已忽略")
            continue
        result.append(
            {
                "target_group_ids": targets,
                "source_group_ids": sources,
            }
        )
    return result


def review_keywords(policy: dict[str, Any], group_id: str) -> list[str]:
    """Return the keywords that apply to a group.

    Args:
        policy: Output of :func:`normalize_review_policy`.
        group_id: QQ group being joined.

    Returns:
        General keywords followed by the ones scoped to the group.
    """
    return list(policy["keywords"]) + list(
        policy["keywords_by_group"].get(group_id, [])
    )


def group_review_policy(
    group_sets: list[dict[str, Any]],
    target_group_id: str,
) -> dict[str, Any] | None:
    """Resolve the join review policy of a group.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.
        target_group_id: Group being joined; chooses the owning group set.

    Returns:
        The owning group set's policy, or ``None`` when the group belongs to no
        set (automatic review only runs for configured groups).
    """
    group_set = find_group_set(group_sets, target_group_id)
    return group_set["join_review"]["policy"] if group_set else None


def review_exempt_sources(
    group_sets: list[dict[str, Any]],
    target_group_id: str,
) -> list[str]:
    """Collect the source groups that exempt a target group from review.

    Exemptions are defined on the group set owning the target group; multiple
    matching entries are merged.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.
        target_group_id: QQ group being joined.

    Returns:
        De-duplicated source group ids; empty when no exemption applies.
    """
    group_set = find_group_set(group_sets, target_group_id)
    if group_set is None:
        return []
    sources: list[str] = []
    for entry in group_set["join_review"]["exemptions"]:
        if (
            entry["target_group_ids"]
            and target_group_id not in entry["target_group_ids"]
        ):
            continue
        for group_id in entry["source_group_ids"]:
            if group_id not in sources:
                sources.append(group_id)
    return sources


def normalize_group_settings(raw: Any) -> dict[str, Any]:
    """Normalize the per-group behaviour settings of a group set.

    Every group set carries its own ban duration, blacklist policy and general
    join review policy, so features can differ between group sets.

    Args:
        raw: Raw ``group_set`` entry from the plugin configuration.

    Returns:
        A dict with ``default_ban_duration`` (seconds), ``blacklist`` and
        ``join_review`` sub-dicts. Missing or malformed values fall back to the
        built-in defaults.
    """
    source = raw if isinstance(raw, dict) else {}
    raw_blacklist = source.get("blacklist")
    raw_blacklist = raw_blacklist if isinstance(raw_blacklist, dict) else {}

    duration = parse_duration(
        str(source.get("default_ban_duration") or ""),
        DEFAULT_BAN_DURATION,
    )

    return {
        "default_ban_duration": duration or DEFAULT_BAN_DURATION,
        "blacklist": {
            "enabled": bool(raw_blacklist.get("enabled", True)),
            "default_reason": (
                str(
                    raw_blacklist.get("default_reason") or DEFAULT_BLACKLIST_REASON,
                ).strip()
                or DEFAULT_BLACKLIST_REASON
            ),
            "block_message": bool(raw_blacklist.get("block_message", False)),
            "kick_on_add": bool(raw_blacklist.get("kick_on_add", False)),
        },
        "join_review": normalize_review_settings(source.get("join_review")),
    }


def normalize_group_sets(raw: Any) -> list[dict[str, Any]]:
    """Normalize the ``group_sets`` configuration into a predictable shape.

    Args:
        raw: Raw value from the plugin configuration.

    Returns:
        A list of dicts with the keys ``name``, ``group_ids``,
        ``manage_session`` and the per-group behaviour settings produced by
        :func:`normalize_group_settings`. Entries without a name and without any
        group id are skipped.
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
                **normalize_group_settings(item),
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


def find_group_set_by_name(
    group_sets: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    """Find a group set by its display name.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.
        name: Group set name as configured.

    Returns:
        The matching group set, or ``None`` when the name is unknown.
    """
    target = str(name or "").strip()
    if not target:
        return None
    for group_set in group_sets:
        if str(group_set["name"]) == target:
            return group_set
    return None


def manage_group_sets(
    group_sets: list[dict[str, Any]],
    umo: str,
) -> list[dict[str, Any]]:
    """List the group sets that use ``umo`` as their manage session.

    A manage session is the control room of the sets that declare it: it may
    run the review and blacklist commands for those sets even though it is not
    one of their managed groups.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.
        umo: ``unified_msg_origin`` of the session.

    Returns:
        The owning group sets, in configuration order. Empty when the session
        is not configured as a manage session.
    """
    target = str(umo or "").strip()
    if not target:
        return []
    return [
        group_set
        for group_set in group_sets
        if str(group_set.get("manage_session") or "") == target
    ]


def group_settings(
    group_sets: list[dict[str, Any]],
    group_id: str,
) -> dict[str, Any]:
    """Resolve the behaviour settings that apply to a group.

    Args:
        group_sets: Output of :func:`normalize_group_sets`.
        group_id: QQ group id.

    Returns:
        The owning group set (which carries ``default_ban_duration`` and
        ``blacklist``), or :data:`DEFAULT_GROUP_SETTINGS` for groups outside
        every set. The mapping is read-only for callers.
    """
    group_set = find_group_set(group_sets, group_id)
    return group_set if group_set else DEFAULT_GROUP_SETTINGS


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


def normalize_notify_routes(raw: Any) -> dict[str, list[str]]:
    """Normalize the ``notify`` configuration into per-session category lists.

    Every routable message category (see :data:`NOTIFY_CATEGORIES`) can be
    delivered to the session the action happened in, to the manage session, to
    both or to neither. Unknown categories are dropped so a stale config cannot
    resurrect a removed one.

    Args:
        raw: Raw ``notify`` mapping from the plugin configuration.

    Returns:
        A dict with the keys ``current`` and ``manage``, each holding the
        categories delivered to that session.
    """
    source = raw if isinstance(raw, dict) else {}
    return {
        "current": [
            str(item)
            for item in normalize_id_list(source.get("to_current_session"))
            if item in NOTIFY_CATEGORIES
        ],
        "manage": [
            str(item)
            for item in normalize_id_list(source.get("to_manage_session"))
            if item in NOTIFY_CATEGORIES
        ],
    }


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
