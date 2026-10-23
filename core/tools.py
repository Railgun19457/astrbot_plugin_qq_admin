"""LLM function tools for group management.

Security model
--------------
Three rules shape every tool here:

1. **The set of groups a tool acts on comes from the session, not from the
   model.** Inside a group that is always the current group. A manage session
   covers every group it manages, and the model may narrow that to one of them
   with the ``group_id`` parameter; any value outside the session's scope is
   rejected, so a model steered by a crafted message cannot reach into a
   different group.
2. **Tools that share their state per group set never take a group.**
   ``qq_blacklist_*`` act on every set the session can see, mirroring the
   ``/拉黑`` family, because the blacklist is shared by a whole set.
2. **Tool-level permission checks are mandatory.** AstrBot's
   ``FunctionToolManager._default_permission`` returns ``"member"`` for every
   non-builtin tool, i.e. the framework gate does not restrict these tools at
   all. Each implementation therefore re-checks the caller through
   ``core.permission`` and returns an error string instead of raising, so the
   model can explain the refusal in natural language.

High-risk tools take a ``confirm`` flag: the first call with ``confirm=False``
only reports what would happen and asks the model to confirm with the user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astrbot.api import FunctionTool, logger

from .config import MAX_BAN_DURATION, format_duration
from .messages import render_template

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from ..main import QQAdminPlugin

#: Tool names exposed to the model, matching the ``llm_tool_options`` schema.
TOOL_NAMES: tuple[str, ...] = (
    "qq_ban_member",
    "qq_whole_ban",
    "qq_kick_member",
    "qq_send_group_notice",
    "qq_group_notice_query",
    "qq_group_notice_delete",
    "qq_blacklist_add",
    "qq_blacklist_remove",
    "qq_blacklist_query",
    "qq_group_info",
    "qq_member_list",
)


def _denied(plugin: QQAdminPlugin, reason: str) -> str:
    """Render a permission refusal for the model.

    Args:
        plugin: Plugin instance supplying the templates.
        reason: Explanation produced by the permission layer.

    Returns:
        A message the model can relay to the user.
    """
    return "error: " + render_template(plugin.templates, "no_permission", reason=reason)


def _failed(plugin: QQAdminPlugin, error: str, *, unsupported: bool = False) -> str:
    """Render an action failure for the model.

    Args:
        plugin: Plugin instance supplying the templates.
        error: Failure description.
        unsupported: Whether the protocol端 does not implement the action.

    Returns:
        A message the model can relay to the user.
    """
    if unsupported:
        return "error: " + render_template(plugin.templates, "api_unsupported")
    return "error: " + render_template(plugin.templates, "api_failed", error=error)


def _no_group(
    plugin: QQAdminPlugin,
    event: AstrMessageEvent,
    selector: str = "",
) -> str:
    """Message returned when a tool cannot resolve a target group.

    Args:
        plugin: Plugin instance supplying the session scope.
        event: The tool call event.
        selector: The rejected selector, empty when nothing was selected.

    Returns:
        The refusal text.
    """
    text = str(selector or "").strip().lstrip("#")
    if text:
        scope = plugin._tool_group_scope(event)
        reachable = (
            f"可管理的群：{'、'.join(scope)}" if scope else "当前会话没有可管理的群"
        )
        return f"error: 群 {text} 不在当前会话的可管理范围内。{reachable}"
    return "error: 当前会话不在已配置的群组中"


def _grouped(group_ids: list[str], results: list[str]) -> str:
    """Render one result per group, tagged with the group number.

    A single group keeps the plain message so the common case stays readable;
    several groups are prefixed so the model can tell which one failed.

    Args:
        group_ids: Groups the results belong to, in order.
        results: One message per group.

    Returns:
        The joined text.
    """
    if len(group_ids) == 1:
        return results[0] if results else ""
    return "\n".join(f"【{group}】{text}" for group, text in zip(group_ids, results))


def _format_notice_time(value: Any) -> str:
    """Render an announcement timestamp.

    Args:
        value: Unix timestamp supplied by the protocol implementation.

    Returns:
        A ``YYYY-MM-DD HH:MM`` string, or ``时间未知`` when unusable.
    """
    try:
        stamp = float(value or 0)
    except (TypeError, ValueError):
        return "时间未知"
    if stamp <= 0:
        return "时间未知"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


def _umo(event: AstrMessageEvent) -> str:
    """Return the event's ``unified_msg_origin``.

    Group configuration is matched by UMO, so a group bound to one platform
    instance is not confused with the same group number on another account.

    Args:
        event: The tool call event.

    Returns:
        The session UMO, or an empty string when unavailable.
    """
    return str(getattr(event, "unified_msg_origin", "") or "")


def _safe_int(value: Any) -> int | None:
    """Coerce a model supplied value into an integer without raising.

    Models occasionally pass strings such as ``"10m"`` or ``null`` for a
    numeric parameter; the tool contract is to answer with text instead of
    letting an exception escape. Booleans are rejected because ``int(True)``
    would silently turn a wrong parameter into ``1``.

    Args:
        value: Raw value from the model.

    Returns:
        The converted integer, or ``None`` when ``value`` is not a number.
    """
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _PluginTool(FunctionTool):
    """Base class wiring a tool to the plugin instance.

    Attributes:
        plugin: The owning plugin, used for permission checks and services.
    """

    plugin: Any = None

    async def resolve_target(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
    ) -> tuple[str, str]:
        """Work out which member a call refers to.

        An explicit ``user_id`` wins; otherwise the ``@`` segment or the quoted
        message author is used. This lets the model rely on the conversation
        context instead of guessing QQ numbers.

        Args:
            event: The tool call event.
            user_id: QQ number supplied by the model, possibly empty.

        Returns:
            ``(user_id, error)``; ``error`` is empty when resolution succeeded.
        """
        if user_id and str(user_id).strip().isdigit():
            return str(user_id).strip(), ""

        targets = self.plugin._extract_targets(event) if event else []
        if targets:
            return targets[0], ""
        return "", "error: 无法确定目标成员，请 @ 对方或提供 QQ 号"


@dataclass
class QqBanMemberTool(_PluginTool):
    """Mute or unmute a group member."""

    name: str = "qq_ban_member"
    description: str = (
        "禁言或解禁群成员。duration_seconds 为 0 表示解禁。"
        "不指定 user_id 时使用消息中 @ 的人或回复消息的对象。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
                "user_id": {
                    "type": "string",
                    "description": "目标成员的 QQ 号，可省略（省略时使用 @ 或回复对象）",
                },
                "duration_seconds": {
                    "type": "integer",
                    "description": "禁言秒数，0 表示解禁。最长 2592000（30 天）",
                },
            },
            "required": ["duration_seconds"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        duration_seconds: int,
        user_id: str = "",
        group_id: str = "",
    ) -> str:
        """Execute the mute.

        Args:
            event: The tool call event.
            duration_seconds: Mute duration; ``0`` lifts an existing mute.
            user_id: Target QQ number, optional.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)

        capability = "ban"
        allowed, deny = await plugin._tool_permission(event, capability)
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        raw_duration = _safe_int(duration_seconds)
        if raw_duration is None:
            # Defaulting to 0 here would silently *unmute* the member, so an
            # unusable value is refused instead.
            return "error: duration_seconds 必须是整数秒数，0 表示解禁"
        duration = max(0, raw_duration)
        action = "unban" if duration == 0 else "ban"

        results: list[str] = []
        for current_group in group_ids:
            result = await plugin.client.set_group_ban(
                event,
                current_group,
                target,
                duration,
            )
            plugin._tool_audit(
                action,
                event,
                target=target,
                group_id=current_group,
                result="ok" if result.ok else "failed",
                error=result.error,
                duration=duration,
            )
            if not result.ok:
                results.append(
                    _failed(plugin, result.error, unsupported=result.unsupported),
                )
                continue

            name = await plugin._resolve_member_name(event, current_group, target)
            if duration == 0:
                results.append(
                    render_template(
                        plugin.templates,
                        "unban_success",
                        target_name=name,
                        target_id=target,
                    ),
                )
            else:
                results.append(
                    render_template(
                        plugin.templates,
                        "ban_success",
                        target_name=name,
                        target_id=target,
                        duration=format_duration(duration),
                    ),
                )
        return _grouped(group_ids, results)


@dataclass
class QqWholeBanTool(_PluginTool):
    """Toggle the whole-group mute."""

    name: str = "qq_whole_ban"
    description: str = (
        "开启或关闭全群禁言。OneBot 的全群禁言是纯开关，服务端不会自动解除；"
        "开启时可用 duration_seconds 指定时长，到期后由插件自动解除。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
                "enabled": {"type": "boolean", "description": "true 开启，false 解除"},
                "duration_seconds": {
                    "type": "integer",
                    "description": "开启后自动解除的秒数，省略或 0 表示不自动解除，"
                    "最长 2592000（30 天）",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "高危操作。首次调用请传 false 向用户确认，用户同意后再传 true",
                },
            },
            "required": ["enabled"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        enabled: bool,
        duration_seconds: int = 0,
        confirm: bool = False,
        group_id: str = "",
    ) -> str:
        """Execute the toggle.

        Args:
            event: The tool call event.
            enabled: ``True`` mutes everyone, ``False`` lifts it.
            duration_seconds: When enabling, seconds until the mute is lifted
                automatically; ``0`` keeps it until lifted manually.
            confirm: Must be ``True`` to actually act.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)

        allowed, deny = await plugin._tool_permission(event, "whole_ban")
        if not allowed:
            return _denied(plugin, deny)

        raw_duration = _safe_int(duration_seconds)
        if raw_duration is None:
            return "error: duration_seconds 必须是整数秒数，0 或省略表示不自动解除"
        duration = min(max(0, raw_duration), MAX_BAN_DURATION)
        action_label = "开启全群禁言" if enabled else "解除全群禁言"
        if enabled and duration:
            action_label += f"（{format_duration(duration)} 后自动解除）"
        if not confirm:
            return (
                f"待确认：将在此次涉及的 {len(group_ids)} 个群{action_label}。"
                "请先向用户确认，用户同意后再次调用本工具并把 confirm 设为 true。"
            )

        action = "whole_ban" if enabled else "whole_ban_off"
        results: list[str] = []
        for current_group in group_ids:
            result = await plugin.client.set_group_whole_ban(
                event,
                current_group,
                enabled=enabled,
            )
            plugin._tool_audit(
                action,
                event,
                group_id=current_group,
                result="ok" if result.ok else "failed",
                error=result.error,
                duration=duration,
            )
            if not result.ok:
                results.append(
                    _failed(plugin, result.error, unsupported=result.unsupported),
                )
                continue

            if not enabled:
                plugin._cancel_whole_ban_timer(current_group)
                results.append(
                    render_template(plugin.templates, "whole_ban_off"),
                )
            elif duration:
                plugin._schedule_whole_ban_off(
                    event,
                    current_group,
                    duration,
                    operator=str(event.get_sender_id() or ""),
                )
                results.append(
                    render_template(
                        plugin.templates,
                        "whole_ban_timed",
                        duration=format_duration(duration),
                    ),
                )
            else:
                results.append(render_template(plugin.templates, "whole_ban_on"))
        return _grouped(group_ids, results)


@dataclass
class QqKickMemberTool(_PluginTool):
    """Remove a member from the group."""

    name: str = "qq_kick_member"
    description: str = (
        "将群成员移出本群。不指定 user_id 时使用消息中 @ 的人或回复对象。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
                "reject_add": {
                    "type": "boolean",
                    "description": "是否同时拒绝其再次申请，默认 false",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "高危操作。首次调用请传 false 向用户确认，用户同意后再传 true",
                },
            },
            "required": [],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        reject_add: bool = False,
        confirm: bool = False,
        group_id: str = "",
    ) -> str:
        """Execute the kick.

        Args:
            event: The tool call event.
            user_id: Target QQ number, optional.
            reject_add: Also reject future join requests.
            confirm: Must be ``True`` to actually act.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)

        allowed, deny = await plugin._tool_permission(event, "kick")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        preview = await plugin._resolve_member_name(event, group_ids[0], target)
        if not confirm:
            where = "本群" if len(group_ids) == 1 else f"以下 {len(group_ids)} 个群"
            return (
                f"待确认：将 {preview}({target}) 移出{where}。"
                "请先向用户确认，用户同意后再次调用本工具并把 confirm 设为 true。"
            )

        results: list[str] = []
        for current_group in group_ids:
            result = await plugin.client.set_group_kick(
                event,
                current_group,
                target,
                reject_add_request=bool(reject_add),
            )
            plugin._tool_audit(
                "kick",
                event,
                target=target,
                group_id=current_group,
                result="ok" if result.ok else "failed",
                error=result.error,
                reject_add=bool(reject_add),
            )
            if not result.ok:
                results.append(
                    _failed(plugin, result.error, unsupported=result.unsupported),
                )
                continue

            name = await plugin._resolve_member_name(event, current_group, target)
            results.append(
                render_template(
                    plugin.templates,
                    "kick_success",
                    target_name=name,
                    target_id=target,
                ),
            )
        return _grouped(group_ids, results)


@dataclass
class QqSendGroupNoticeTool(_PluginTool):
    """Publish a group announcement."""

    name: str = "qq_send_group_notice"
    description: str = (
        "发送群公告。内容会直接展示给全群成员。"
        "公告发布后可以用 qq_group_notice_query 查看、qq_group_notice_delete 删除。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
                "content": {"type": "string", "description": "公告正文"},
                "image": {
                    "type": "string",
                    "description": "可选配图，本地绝对路径或图片 URL；协议端不支持时会忽略",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "公告会对全群可见，首次调用请传 false 向用户确认",
                },
            },
            "required": ["content"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        content: str,
        image: str = "",
        confirm: bool = False,
        group_id: str = "",
    ) -> str:
        """Execute the announcement.

        Args:
            event: The tool call event.
            content: Notice body.
            image: Optional image path or URL.
            confirm: Must be ``True`` to actually publish.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)
        if not str(content or "").strip():
            return "error: 公告内容不能为空"

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        if not confirm:
            extra = f"（含配图 {image}）" if str(image or "").strip() else ""
            where = "本群" if len(group_ids) == 1 else f"以下 {len(group_ids)} 个群"
            return (
                f"待确认：向{where}发布群公告「{content[:50]}」{extra}。"
                "公告全群可见，请先向用户确认，同意后再把 confirm 设为 true 调用。"
            )

        results: list[str] = []
        for current_group in group_ids:
            result = await plugin.client.send_group_notice(
                event,
                current_group,
                content,
                image,
            )
            plugin._tool_audit(
                "notice",
                event,
                group_id=current_group,
                result="ok" if result.ok else "failed",
                error=result.error,
                content=content[:50],
                with_image=bool(str(image or "").strip()),
            )
            if not result.ok:
                results.append(
                    _failed(plugin, result.error, unsupported=result.unsupported),
                )
                continue
            results.append(render_template(plugin.templates, "notice_success"))
        return _grouped(group_ids, results)


@dataclass
class QqGroupNoticeQueryTool(_PluginTool):
    """List the current group announcements."""

    name: str = "qq_group_notice_query"
    description: str = (
        "查看本群当前的群公告（含编号、发布者和时间）。删除公告需要先用本工具拿到编号。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
            },
            "required": [],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        group_id: str = "",
    ) -> str:
        """Execute the lookup.

        Args:
            event: The tool call event.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        results: list[str] = []
        for current_group in group_ids:
            notices = await plugin.client.get_group_notice(event, current_group)
            if not notices:
                results.append(
                    "当前没有群公告，或协议端不支持读取公告（需 NapCat / Lagrange）",
                )
                continue

            lines = [f"共 {len(notices)} 条群公告（编号用于删除）："]
            for index, notice in enumerate(notices, start=1):
                body = notice.get("message")
                text = ""
                if isinstance(body, dict):
                    text = str(body.get("text") or "")
                elif isinstance(body, str):
                    text = body
                text = text.replace("\n", " ").strip() or "（无正文）"
                lines.append(
                    f"{index}. #{notice.get('notice_id') or '未知'} "
                    f"{text[:80]} · 发布者 {notice.get('sender_id') or '未知'}"
                    f" · {_format_notice_time(notice.get('publish_time'))}",
                )
            results.append("\n".join(lines))
        return _grouped(group_ids, results)


@dataclass
class QqGroupNoticeDeleteTool(_PluginTool):
    """Delete a group announcement."""

    name: str = "qq_group_notice_delete"
    description: str = (
        "删除本群的一条群公告。notice_id 先用 qq_group_notice_query 获取。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
                "notice_id": {"type": "string", "description": "公告编号"},
                "confirm": {
                    "type": "boolean",
                    "description": "高危操作。首次调用请传 false 向用户确认",
                },
            },
            "required": ["notice_id"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        notice_id: str,
        confirm: bool = False,
        group_id: str = "",
    ) -> str:
        """Execute the deletion.

        Args:
            event: The tool call event.
            notice_id: Identifier from :class:`QqGroupNoticeQueryTool`.
            confirm: Must be ``True`` to actually delete.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)
        target = str(notice_id or "").strip()
        if not target:
            return "error: 缺少 notice_id"

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        if not confirm:
            where = "本群" if len(group_ids) == 1 else f"以下 {len(group_ids)} 个群"
            return (
                f"待确认：删除{where}的群公告 #{target}。"
                "请先向用户确认，同意后再把 confirm 设为 true 调用。"
            )

        results: list[str] = []
        for current_group in group_ids:
            result = await plugin.client.delete_group_notice(
                event, current_group, target
            )
            plugin._tool_audit(
                "notice_delete",
                event,
                group_id=current_group,
                result="ok" if result.ok else "failed",
                error=result.error,
                notice_id=target,
            )
            if not result.ok:
                results.append(
                    _failed(plugin, result.error, unsupported=result.unsupported),
                )
                continue
            results.append(render_template(plugin.templates, "notice_deleted"))
        return _grouped(group_ids, results)


@dataclass
class QqBlacklistAddTool(_PluginTool):
    """Add a member to the shared blacklist."""

    name: str = "qq_blacklist_add"
    description: str = (
        "将成员永久加入本群组（管理会话中为它管理的全部群组）的共享黑名单。"
        "reason 是必填的拉黑原因，会记录并出现在导出表格中。"
        "拉黑不会自动解除，也不会自动踢出成员；需要时用 qq_blacklist_remove 移出。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "拉黑原因，必填"},
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
            },
            "required": ["reason"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        reason: str,
        user_id: str = "",
    ) -> str:
        """Execute the blacklisting.

        The blacklist is shared by a whole group set, so the entry never depends
        on one group: it is written to every set this session can see, matching
        the ``/拉黑`` command.

        Args:
            event: The tool call event.
            reason: Why the user is blacklisted.
            user_id: Target QQ number, optional.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, "")
        if not group_ids:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "blacklist")
        if not allowed:
            return _denied(plugin, deny)

        group_sets = [
            item
            for item in plugin._visible_group_sets(event)
            if item["blacklist"]["enabled"]
        ]
        if not group_sets:
            return "error: " + render_template(plugin.templates, "blacklist_disabled")

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        origin = group_ids[0]
        final_reason = (
            str(reason or "").strip() or group_sets[0]["blacklist"]["default_reason"]
        )
        name = await plugin._resolve_member_name(event, origin, target)
        for group_set in group_sets:
            scope = str(group_set["name"])
            plugin.blacklist.add(
                scope,
                target,
                reason=final_reason,
                operator=str(event.get_sender_id() or ""),
                nickname=name,
                group_id=origin,
            )
            plugin._tool_audit(
                "blacklist_add",
                event,
                target=target,
                group_id=origin,
                scope=scope,
                reason=final_reason,
            )

        text = render_template(
            plugin.templates,
            "blacklist_added",
            target_name=name,
            target_id=target,
            reason=final_reason,
        )
        if len(group_sets) > 1:
            text += "\n" + render_template(
                plugin.templates,
                "blacklist_scopes",
                scopes="、".join(str(item["name"]) for item in group_sets),
            )
        return text


@dataclass
class QqBlacklistRemoveTool(_PluginTool):
    """Remove a member from the blacklist."""

    name: str = "qq_blacklist_remove"
    description: str = "将成员移出本群组（管理会话中为它管理的全部群组）的共享黑名单。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
            },
            "required": [],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
    ) -> str:
        """Execute the removal.

        The entry is deleted from every set this session can see, matching the
        ``/解除拉黑`` command, so no group has to be named.

        Args:
            event: The tool call event.
            user_id: Target QQ number, optional.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, "")
        if not group_ids:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "blacklist")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        group_sets = plugin._visible_group_sets(event)
        if not group_sets:
            return _no_group(plugin, event)

        origin = group_ids[0]
        name = await plugin._resolve_member_name(event, origin, target)
        hit = False
        for group_set in group_sets:
            scope = str(group_set["name"])
            if not plugin.blacklist.remove(scope, target):
                continue
            hit = True
            plugin._tool_audit(
                "blacklist_remove",
                event,
                target=target,
                group_id=origin,
                scope=scope,
            )
        if not hit:
            return render_template(
                plugin.templates,
                "blacklist_not_found",
                target_name=name,
                target_id=target,
            )
        return render_template(
            plugin.templates,
            "blacklist_removed",
            target_name=name,
            target_id=target,
        )


@dataclass
class QqBlacklistQueryTool(_PluginTool):
    """List the shared blacklist."""

    name: str = "qq_blacklist_query"
    description: str = (
        "查询本群组（管理会话中为它管理的全部群组）共享的黑名单，包含拉黑原因。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        },
    )

    async def run(self, event: AstrMessageEvent) -> str:
        """Return the blacklist as plain text.

        Args:
            event: The tool call event.

        Returns:
            The formatted list or an ``error:`` string.
        """
        plugin = self.plugin
        if not plugin._tool_group_ids(event, ""):
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        group_sets = plugin._visible_group_sets(event)
        if not group_sets:
            return _no_group(plugin, event)

        blocks: list[str] = []
        for group_set in group_sets:
            scope = str(group_set["name"])
            if not group_set["blacklist"]["enabled"]:
                blocks.append(
                    render_template(
                        plugin.templates,
                        "blacklist_disabled_full",
                        scope=scope,
                    ),
                )
                continue
            entries = plugin.blacklist.list_entries(scope)
            if not entries:
                blocks.append(
                    render_template(plugin.templates, "blacklist_empty", scope=scope),
                )
                continue
            lines = [
                f"{index}. {entry.nickname or entry.user_id}({entry.user_id})"
                f" · {entry.reason}"
                for index, entry in enumerate(entries, start=1)
            ]
            blocks.append(
                f"黑名单（{scope}）共 {len(entries)} 人：\n" + "\n".join(lines)
            )
        return _grouped([str(item["name"]) for item in group_sets], blocks)


@dataclass
class QqGroupInfoTool(_PluginTool):
    """Report basic group information."""

    name: str = "qq_group_info"
    description: str = "获取当前群的基本信息，例如群名、群号、成员数、群主。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
            },
            "required": [],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        group_id: str = "",
    ) -> str:
        """Return group information.

        Args:
            event: The tool call event.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            The formatted information or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        results: list[str] = []
        for current_group in group_ids:
            info = await plugin.client.get_group_info(event, current_group)
            members = await plugin.client.get_group_member_list(event, current_group)
            owner = next(
                (m for m in members if str(m.get("role")) == "owner"),
                None,
            )
            lines = [
                f"群号：{current_group}",
                f"群名：{(info or {}).get('group_name') or '未知'}",
                f"成员数：{len(members)}",
            ]
            if owner:
                lines.append(
                    f"群主：{owner.get('nickname') or owner.get('user_id')}"
                    f"({owner.get('user_id')})",
                )
            group_set = plugin._group_set_name(current_group, umo=_umo(event))
            lines.append(f"所属群组：{group_set or '未加入任何群组'}")
            results.append("\n".join(lines))
        return _grouped(group_ids, results)


@dataclass
class QqMemberListTool(_PluginTool):
    """List group members."""

    name: str = "qq_member_list"
    description: str = "列出群成员，可选限制条数。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。省略时作用于当前会话可管理的所有群；"
                    "只填一个群号时仅作用于该群",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少条，默认 50，上限 200",
                },
            },
            "required": [],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        limit: int = 50,
        group_id: str = "",
    ) -> str:
        """Return a member listing.

        Args:
            event: The tool call event.
            limit: Maximum number of members to include.
            group_id: Target group number; every group of the session is
                used when it is omitted.

        Returns:
            The formatted listing or an ``error:`` string.
        """
        plugin = self.plugin
        group_ids = plugin._tool_group_ids(event, group_id)
        if not group_ids:
            return _no_group(plugin, event, group_id)

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        # ``or 50`` also covers a non-numeric or zero limit from the model.
        capped = max(1, min(_safe_int(limit) or 50, 200))
        results: list[str] = []
        for current_group in group_ids:
            members = await plugin.client.get_group_member_list(event, current_group)
            if not members:
                results.append("error: 无法获取群成员列表")
                continue

            lines = [
                f"{index}. {m.get('card') or m.get('nickname') or m.get('user_id')}"
                f"({m.get('user_id')}) {_role_text(str(m.get('role') or ''))}"
                for index, m in enumerate(members[:capped], start=1)
            ]
            header = f"群 {current_group} 共 {len(members)} 人"
            if len(members) > capped:
                header += f"，以下仅显示前 {capped} 条"
            results.append(header + "：\n" + "\n".join(lines))
        return _grouped(group_ids, results)


def _role_text(role: str) -> str:
    """Translate an OneBot role into Chinese.

    Args:
        role: ``owner``, ``admin`` or ``member``.

    Returns:
        A short label, empty for plain members.
    """
    return {"owner": "[群主]", "admin": "[管理员]"}.get(role, "")


def build_tools(plugin: QQAdminPlugin) -> dict[str, FunctionTool]:
    """Instantiate every tool bound to a plugin instance.

    Args:
        plugin: The owning plugin.

    Returns:
        Mapping from tool name to the ready-to-register tool object.
    """
    classes: tuple[type[_PluginTool], ...] = (
        QqBanMemberTool,
        QqWholeBanTool,
        QqKickMemberTool,
        QqSendGroupNoticeTool,
        QqGroupNoticeQueryTool,
        QqGroupNoticeDeleteTool,
        QqBlacklistAddTool,
        QqBlacklistRemoveTool,
        QqBlacklistQueryTool,
        QqGroupInfoTool,
        QqMemberListTool,
    )
    tools: dict[str, FunctionTool] = {}
    for tool_cls in classes:
        tool = tool_cls()
        tool.plugin = plugin
        tools[tool.name] = tool
    return tools


def log_registered(names: list[str]) -> None:
    """Log which tools were exposed to the model.

    Args:
        names: Registered tool names.
    """
    if names:
        logger.info(f"[QQAdmin] 已向 LLM 暴露 {len(names)} 个工具：{', '.join(names)}")
    else:
        logger.info("[QQAdmin] 未向 LLM 暴露任何工具（llm_tool_options 为空）")
