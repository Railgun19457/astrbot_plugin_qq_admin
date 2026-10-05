"""LLM function tools for group management.

Security model
--------------
Two rules shape every tool here:

1. **The group id is never a tool parameter.** It is always derived from
   ``event.get_group_id()``, so a model steered by a crafted message cannot
   reach into a different group.
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astrbot.api import FunctionTool, logger

from .config import format_duration
from .messages import render_template

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from ..main import QQAdminPlugin

#: Tool names exposed to the model, matching the ``llm_tool_options`` schema.
TOOL_NAMES: tuple[str, ...] = (
    "qq_ban_member",
    "qq_whole_ban",
    "qq_kick_member",
    "qq_recall_message",
    "qq_send_group_notice",
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


def _no_group() -> str:
    """Message returned when a tool is called outside a group chat.

    Returns:
        The refusal text.
    """
    return "error: 此工具只能在群聊中使用"


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
                "user_id": {
                    "type": "string",
                    "description": "目标成员的 QQ 号，可省略（省略时使用 @ 或回复对象）",
                },
                "duration_seconds": {
                    "type": "integer",
                    "description": "禁言秒数，0 表示解禁。最长 2592000（30 天）",
                },
                "reason": {"type": "string", "description": "禁言理由，可省略"},
            },
            "required": ["duration_seconds"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        duration_seconds: int,
        user_id: str = "",
        reason: str = "",
    ) -> str:
        """Execute the mute.

        Args:
            event: The tool call event.
            duration_seconds: Mute duration; ``0`` lifts an existing mute.
            user_id: Target QQ number, optional.
            reason: Justification recorded in the audit log.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        capability = "ban"
        allowed, deny = await plugin._tool_permission(event, capability)
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        duration = max(0, int(duration_seconds))
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
            reason=reason,
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
        "开启或关闭全群禁言。注意 OneBot 的全群禁言是纯开关，"
        "duration_seconds 只用于提示自动解除时间，服务端不会自动解除。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "description": "true 开启，false 解除"},
                "reason": {"type": "string", "description": "操作理由，可省略"},
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
        reason: str = "",
        confirm: bool = False,
    ) -> str:
        """Execute the toggle.

        Args:
            event: The tool call event.
            enabled: ``True`` mutes everyone, ``False`` lifts it.
            reason: Justification recorded in the audit log.
            confirm: Must be ``True`` to actually act.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        allowed, deny = await plugin._tool_permission(event, "whole_ban")
        if not allowed:
            return _denied(plugin, deny)

        action_label = "开启全群禁言" if enabled else "解除全群禁言"
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
            reason=reason,
        )
        return render_template(
            plugin.templates,
            "whole_ban_on" if enabled else "whole_ban_off",
        )


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
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
                "reason": {"type": "string", "description": "踢出理由，可省略"},
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
        reason: str = "",
        reject_add: bool = False,
        confirm: bool = False,
    ) -> str:
        """Execute the kick.

        Args:
            event: The tool call event.
            user_id: Target QQ number, optional.
            reason: Justification recorded in the audit log.
            reject_add: Also reject future join requests.
            confirm: Must be ``True`` to actually act.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        allowed, deny = await plugin._tool_permission(event, "kick")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        name = await plugin._resolve_member_name(event, group_id, target)
        if not confirm:
            return (
                f"待确认：将 {name}({target}) 移出本群"
                f"{f'，理由：{reason}' if reason else ''}。"
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
                reason=reason,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit(
            "kick",
            event,
            target=target,
            group_id=group_id,
            reason=reason,
            reject_add=bool(reject_add),
        )
        return render_template(
            plugin.templates,
            "kick_success",
            target_name=name,
            target_id=target,
        )


@dataclass
class QqRecallMessageTool(_PluginTool):
    """Recall a message."""

    name: str = "qq_recall_message"
    description: str = "撤回群内的一条消息。需要提供消息 ID。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "要撤回的消息 ID"},
            },
            "required": ["message_id"],
        },
    )

    async def run(self, event: AstrMessageEvent, message_id: str) -> str:
        """Execute the recall.

        Args:
            event: The tool call event.
            message_id: Id of the message to recall.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()
        if not str(message_id).strip():
            return "error: 缺少 message_id"

        allowed, deny = await plugin._tool_permission(event, "recall")
        if not allowed:
            return _denied(plugin, deny)

        result = await plugin.client.delete_msg(event, str(message_id).strip())
        if not result.ok:
            plugin._tool_audit(
                "recall",
                event,
                group_id=group_id,
                result="failed",
                error=result.error,
                message_id=str(message_id),
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit(
            "recall",
            event,
            group_id=group_id,
            message_id=str(message_id),
        )
        return render_template(plugin.templates, "recall_success")


@dataclass
class QqSendGroupNoticeTool(_PluginTool):
    """Publish a group announcement."""

    name: str = "qq_send_group_notice"
    description: str = "发送群公告。内容会直接展示给全群成员。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "公告正文"},
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
        confirm: bool = False,
    ) -> str:
        """Execute the announcement.

        Args:
            event: The tool call event.
            content: Notice body.
            confirm: Must be ``True`` to actually publish.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()
        if not str(content or "").strip():
            return "error: 公告内容不能为空"

        allowed, deny = await plugin._tool_permission(event, "notice")
        if not allowed:
            return _denied(plugin, deny)

        if not confirm:
            return (
                f"待确认：发布群公告「{content[:50]}」。"
                "公告全群可见，请先向用户确认，同意后再把 confirm 设为 true 调用。"
            )

        result = await plugin.client.send_group_notice(event, group_id, content)
        if not result.ok:
            plugin._tool_audit(
                "notice",
                event,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            return _failed(plugin, result.error, unsupported=result.unsupported)

        plugin._tool_audit("notice", event, group_id=group_id, content=content[:50])
        return render_template(plugin.templates, "notice_success")


@dataclass
class QqBlacklistAddTool(_PluginTool):
    """Add a member to the shared blacklist."""

    name: str = "qq_blacklist_add"
    description: str = "将成员加入本群组的共享黑名单。reason 是必填的拉黑原因，会记录并出现在导出表格中。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "拉黑原因，必填"},
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
                "duration_seconds": {
                    "type": "integer",
                    "description": "临时拉黑的秒数，0 或不填表示永久",
                },
            },
            "required": ["reason"],
        },
    )

    async def run(
        self,
        event: AstrMessageEvent,
        reason: str,
        user_id: str = "",
        duration_seconds: int = 0,
    ) -> str:
        """Execute the blacklisting.

        Args:
            event: The tool call event.
            reason: Why the user is blacklisted.
            user_id: Target QQ number, optional.
            duration_seconds: Lifetime; ``0`` means permanent.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        allowed, deny = await plugin._tool_permission(event, "blacklist")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        final_reason = str(reason or "").strip() or plugin.blacklist_default_reason
        duration = max(0, int(duration_seconds or 0))
        scope = plugin._scope(group_id)
        name = await plugin._resolve_member_name(event, group_id, target)
        entry = plugin.blacklist.add(
            scope,
            target,
            reason=final_reason,
            operator=str(event.get_sender_id() or ""),
            duration=duration,
            nickname=name,
        )

        plugin._tool_audit(
            "blacklist_add",
            event,
            target=target,
            group_id=group_id,
            scope=scope,
            reason=final_reason,
            duration=duration,
        )
        return render_template(
            plugin.templates,
            "blacklist_added",
            target_name=name,
            target_id=target,
            reason=final_reason,
            duration=format_duration(duration) if entry.is_temporary else "永久",
        )


@dataclass
class QqBlacklistRemoveTool(_PluginTool):
    """Remove a member from the blacklist."""

    name: str = "qq_blacklist_remove"
    description: str = "将成员移出本群组的共享黑名单。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "目标 QQ 号，可省略"},
            },
            "required": [],
        },
    )

    async def run(self, event: AstrMessageEvent, user_id: str = "") -> str:
        """Execute the removal.

        Args:
            event: The tool call event.
            user_id: Target QQ number, optional.

        Returns:
            A human readable result or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        allowed, deny = await plugin._tool_permission(event, "blacklist")
        if not allowed:
            return _denied(plugin, deny)

        target, error = await self.resolve_target(event, user_id)
        if error:
            return error

        scope = plugin._scope(group_id)
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
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        scope = plugin._scope(group_id)
        entries = plugin.blacklist.list_entries(scope)
        if not entries:
            return render_template(plugin.templates, "blacklist_empty", scope=scope)

        lines = [
            f"{index}. {entry.nickname or entry.user_id}({entry.user_id}) | "
            f"{entry.reason} | {'永久' if not entry.is_temporary else format_duration(int(entry.expires_at - entry.created_at))}"
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
            "properties": {},
            "required": [],
        },
    )

    async def run(self, event: AstrMessageEvent) -> str:
        """Return group information.

        Args:
            event: The tool call event.

        Returns:
            The formatted information or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

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
        group_set = plugin._group_set_name(group_id)
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
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少条，默认 50，上限 200",
                },
            },
            "required": [],
        },
    )

    async def run(self, event: AstrMessageEvent, limit: int = 50) -> str:
        """Return a member listing.

        Args:
            event: The tool call event.
            limit: Maximum number of members to include.

        Returns:
            The formatted listing or an ``error:`` string.
        """
        plugin = self.plugin
        group_id = plugin._tool_group_id(event)
        if not group_id:
            return _no_group()

        allowed, deny = await plugin._tool_permission(event, "query")
        if not allowed:
            return _denied(plugin, deny)

        members = await plugin.client.get_group_member_list(event, group_id)
        if not members:
            return "error: 无法获取群成员列表"

        capped = max(1, min(int(limit or 50), 200))
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
        QqRecallMessageTool,
        QqSendGroupNoticeTool,
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
