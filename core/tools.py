"""LLM function tools for group management.

Security model
--------------
Two rules shape every tool here:

1. **The chat a tool acts on comes from the session, not from the model.**
   Inside a group that is always the current group. A manage session may name
   one of its own groups through the ``group_id`` parameter, and the value is
   rejected unless it is part of that session's scope, so a model steered by a
   crafted message cannot reach into a different group.
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


def _no_group(plugin: QQAdminPlugin, event: AstrMessageEvent) -> str:
    """Message returned when a tool cannot resolve a target group.

    The refusal names the groups the session may act on, so the model can retry
    with an explicit ``group_id`` instead of guessing.

    Args:
        plugin: Plugin instance supplying the session scope.
        event: The tool call event.

    Returns:
        The refusal text.
    """
    scope = plugin._tool_group_scope(event)
    if len(scope) > 1:
        return "error: 当前会话对应多个群，请在 group_id 中指定其中一个：" + "、".join(
            scope
        )
    return "error: 当前会话不在已配置的群组中"


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

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
        result = await plugin.client.set_group_ban(event, group_id, target, duration)
        name = await plugin._resolve_member_name(event, group_id, target)
        if not result.ok:
            plugin._tool_audit(
                "ban" if duration else "unban",
                event,
                target=target,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        action = "unban" if duration == 0 else "ban"
        plugin._tool_audit(
            action,
            event,
            target=target,
            group_id=group_id,
            duration=duration,
        )
        if duration == 0:
            return render_template(
                plugin.templates,
                "unban_success",
                target_name=name,
                target_id=target,
            )
        return render_template(
            plugin.templates,
            "ban_success",
            target_name=name,
            target_id=target,
            duration=format_duration(duration),
        )


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

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
                f"待确认：{action_label}。请先向用户确认，用户同意后"
                "再次调用本工具并把 confirm 设为 true。"
            )

        result = await plugin.client.set_group_whole_ban(
            event, group_id, enabled=enabled
        )
        if not result.ok:
            plugin._tool_audit(
                "whole_ban" if enabled else "whole_ban_off",
                event,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit(
            "whole_ban" if enabled else "whole_ban_off",
            event,
            group_id=group_id,
            duration=duration,
        )

        if not enabled:
            plugin._cancel_whole_ban_timer(group_id)
            return render_template(plugin.templates, "whole_ban_off")

        if duration:
            plugin._schedule_whole_ban_off(
                event,
                group_id,
                duration,
                operator=str(event.get_sender_id() or ""),
            )
            return render_template(
                plugin.templates,
                "whole_ban_timed",
                duration=format_duration(duration),
            )
        return render_template(plugin.templates, "whole_ban_on")


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "kick")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        name = await plugin._resolve_member_name(event, group_id, target)
        if not confirm:
            return (
                f"待确认：将 {name}({target}) 移出本群。"
                "请先向用户确认，用户同意后再次调用本工具并把 confirm 设为 true。"
            )

        result = await plugin.client.set_group_kick(
            event,
            group_id,
            target,
            reject_add_request=bool(reject_add),
        )
        if not result.ok:
            plugin._tool_audit(
                "kick",
                event,
                target=target,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit(
            "kick",
            event,
            target=target,
            group_id=group_id,
            reject_add=bool(reject_add),
        )
        return render_template(
            plugin.templates,
            "kick_success",
            target_name=name,
            target_id=target,
        )


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)
        if not str(content or "").strip():
            return "error: 公告内容不能为空"

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        if not confirm:
            extra = f"（含配图 {image}）" if str(image or "").strip() else ""
            return (
                f"待确认：发布群公告「{content[:50]}」{extra}。"
                "公告全群可见，请先向用户确认，同意后再把 confirm 设为 true 调用。"
            )

        result = await plugin.client.send_group_notice(
            event,
            group_id,
            content,
            image,
        )
        if not result.ok:
            plugin._tool_audit(
                "notice",
                event,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit(
            "notice",
            event,
            group_id=group_id,
            content=content[:50],
            with_image=bool(str(image or "").strip()),
        )
        return render_template(plugin.templates, "notice_success")


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        notices = await plugin.client.get_group_notice(event, group_id)
        if not notices:
            return "当前没有群公告，或协议端不支持读取公告（需 NapCat / Lagrange）"

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
        return "\n".join(lines)


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)
        target = str(notice_id or "").strip()
        if not target:
            return "error: 缺少 notice_id"

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        if not confirm:
            return (
                f"待确认：删除群公告 #{target}。"
                "请先向用户确认，同意后再把 confirm 设为 true 调用。"
            )

        result = await plugin.client.delete_group_notice(event, group_id, target)
        if not result.ok:
            plugin._tool_audit(
                "notice_delete",
                event,
                group_id=group_id,
                result="failed",
                error=result.error,
                notice_id=target,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit(
            "notice_delete",
            event,
            group_id=group_id,
            notice_id=target,
        )
        return render_template(plugin.templates, "notice_deleted")


@dataclass
class QqBlacklistAddTool(_PluginTool):
    """Add a member to the shared blacklist."""

    name: str = "qq_blacklist_add"
    description: str = (
        "将成员永久加入本群组的共享黑名单。reason 是必填的拉黑原因，会记录并出现在导出表格中。"
        "拉黑不会自动解除，也不会自动踢出成员；需要时用 qq_blacklist_remove 移出。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
                },
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
        group_id: str = "",
    ) -> str:
        """Execute the blacklisting.

        Args:
            event: The tool call event.
            reason: Why the user is blacklisted.
            user_id: Target QQ number, optional.
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "blacklist")
        if not allowed:
            return _denied(plugin, deny)

        blacklist_policy = plugin._policy(group_id, umo=_umo(event))["blacklist"]
        if not blacklist_policy["enabled"]:
            return "error: " + render_template(plugin.templates, "blacklist_disabled")

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        final_reason = str(reason or "").strip() or blacklist_policy["default_reason"]
        scope = plugin._scope(group_id, umo=_umo(event))
        name = await plugin._resolve_member_name(event, group_id, target)
        plugin.blacklist.add(
            scope,
            target,
            reason=final_reason,
            operator=str(event.get_sender_id() or ""),
            nickname=name,
            group_id=group_id,
        )

        plugin._tool_audit(
            "blacklist_add",
            event,
            target=target,
            group_id=group_id,
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
        return text


@dataclass
class QqBlacklistRemoveTool(_PluginTool):
    """Remove a member from the blacklist."""

    name: str = "qq_blacklist_remove"
    description: str = "将成员移出本群组的共享黑名单。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
                },
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
            },
            "required": [],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        group_id: str = "",
    ) -> str:
        """Execute the removal.

        Args:
            event: The tool call event.
            user_id: Target QQ number, optional.
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "blacklist")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        scope = plugin._scope(group_id, umo=_umo(event))
        name = await plugin._resolve_member_name(event, group_id, target)
        if not plugin.blacklist.remove(scope, target):
            return render_template(
                plugin.templates,
                "blacklist_not_found",
                target_name=name,
                target_id=target,
            )

        plugin._tool_audit(
            "blacklist_remove",
            event,
            target=target,
            group_id=group_id,
            scope=scope,
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
    description: str = "查询本群组共享的黑名单，包含拉黑原因。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "group_id": {
                    "type": "string",
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
        """Return the blacklist as plain text.

        Args:
            event: The tool call event.
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            The formatted list or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        if not plugin._policy(group_id, umo=_umo(event))["blacklist"]["enabled"]:
            return "error: " + render_template(plugin.templates, "blacklist_disabled")

        scope = plugin._scope(group_id, umo=_umo(event))
        entries = plugin.blacklist.list_entries(scope)
        if not entries:
            return render_template(plugin.templates, "blacklist_empty", scope=scope)

        lines = [
            f"{index}. {entry.nickname or entry.user_id}({entry.user_id})"
            f" · {entry.reason}"
            for index, entry in enumerate(entries, start=1)
        ]
        return f"黑名单（{scope}）共 {len(entries)} 人：\n" + "\n".join(lines)


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            The formatted information or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        info = await plugin.client.get_group_info(event, group_id)
        members = await plugin.client.get_group_member_list(event, group_id)
        owner = next(
            (m for m in members if str(m.get("role")) == "owner"),
            None,
        )
        lines = [
            f"群号：{group_id}",
            f"群名：{(info or {}).get('group_name') or '未知'}",
            f"成员数：{len(members)}",
        ]
        if owner:
            lines.append(
                f"群主：{owner.get('nickname') or owner.get('user_id')}"
                f"({owner.get('user_id')})",
            )
        group_set = plugin._group_set_name(group_id, umo=_umo(event))
        lines.append(f"所属群组：{group_set or '未加入任何群组'}")
        return "\n".join(lines)


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
                    "description": "目标群号。管理会话对应多个群时必填；群内触发或管理会话只对应一个群时可省略",
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
            group_id: Target group number, needed only in a manage session
                that corresponds to several groups.

        Returns:
            The formatted listing or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event, group_id)
        if not group_id:
            return _no_group(plugin, event)

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        members = await plugin.client.get_group_member_list(event, group_id)
        if not members:
            return "error: 无法获取群成员列表"

        # ``or 50`` also covers a non-numeric or zero limit from the model.
        capped = max(1, min(_safe_int(limit) or 50, 200))
        lines = [
            f"{index}. {m.get('card') or m.get('nickname') or m.get('user_id')}"
            f"({m.get('user_id')}) {_role_text(str(m.get('role') or ''))}"
            for index, m in enumerate(members[:capped], start=1)
        ]
        header = f"群 {group_id} 共 {len(members)} 人"
        if len(members) > capped:
            header += f"，以下仅显示前 {capped} 条"
        return header + "：\n" + "\n".join(lines)


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
