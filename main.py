"""QQ群管 — OneBot v11 group management for AstrBot.

Entry point registering the command handlers. Only the aiocqhttp platform is
supported: every handler is guarded by a platform filter so that the plugin
stays inert on adapters that cannot provide the required OneBot actions.

See ``plan.md`` for the full design; this file only wires things together, with
the reusable pieces living under ``core/``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import GreedyStr

from .core.args import parse_duration_token, parse_target_args
from .core.audit import AuditLog
from .core.blacklist import BlacklistStore
from .core.config import (
    CAPABILITY_KEYS,
    DEFAULT_BAN_DURATION,
    MAX_BAN_DURATION,
    build_group_index,
    find_group_set,
    format_duration,
    is_all_target,
    normalize_bool_list,
    normalize_group_sets,
    normalize_permission_level,
    normalize_words,
    parse_duration,
    resolve_blacklist_scope,
)
from .core.confirm import ConfirmManager, PendingAction
from .core.messages import render_template
from .core.notify import Notifier
from .core.onebot import ActionResult, OneBotClient
from .core.permission import (
    PermissionContext,
    PermissionSettings,
    build_permission_context,
    check_capability,
)
from .core.utils import (
    extract_at_targets,
    extract_reply_message_id,
    shorten,
)

#: Platform adapter this plugin is able to drive.
SUPPORTED_PLATFORM = "aiocqhttp"

#: Maximum number of members rendered by the ``/群成员`` listing.
MEMBER_PAGE_SIZE = 20


class QQAdminPlugin(Star):
    """Command handlers and shared services for the QQ群管 plugin.

    Attributes:
        config: Plugin configuration handed over by AstrBot.
        client: OneBot action facade.
        audit: Audit trail writer.
        confirm: High-risk action confirmation manager.
        group_sets: Normalized group set definitions.
        group_index: ``group_id -> set name`` lookup.
        permission_settings: Parsed permission policy.
        templates: Message template mapping.
    """

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """Initialize the plugin and its shared services.

        Args:
            context: AstrBot plugin context.
            config: Plugin configuration parsed from ``_conf_schema.json``.
        """
        super().__init__(context)
        self.config = config

        self.client = OneBotClient()
        self._whole_ban_tasks: dict[str, asyncio.Task[None]] = {}

        self.enabled: bool = bool(config.get("enabled", True))
        self._load_config(config)
        self._init_services()

        # Trim the audit log once at startup rather than on every write.
        self.audit.trim()

        logger.info(
            f"[QQAdmin] 已加载 | 群组 {len(self.group_sets)} 个 | "
            f"索引 {len(self.group_index)} 个群",
        )

    # ------------------------------------------------------------------ #
    #  Startup helpers
    # ------------------------------------------------------------------ #

    def _load_config(self, config: AstrBotConfig) -> None:
        """Parse the configuration into normalized structures.

        Args:
            config: Raw plugin configuration.
        """
        self.group_sets = normalize_group_sets(config.get("group_sets"))
        self.group_index = build_group_index(self.group_sets)

        permissions = config.get("permissions")
        permissions = permissions if isinstance(permissions, dict) else {}
        raw_levels = permissions.get("capability_level")
        raw_levels = raw_levels if isinstance(raw_levels, dict) else {}
        self.permission_settings = PermissionSettings(
            custom_allowlist=frozenset(
                normalize_bool_list(permissions.get("custom_whitelist")),
            ),
            allow_astrbot_admin=bool(permissions.get("allow_astrbot_admin", True)),
            allow_group_admin=bool(permissions.get("allow_group_admin", True)),
            allow_member_command=bool(permissions.get("allow_member_command", True)),
            capability_levels={
                key: normalize_permission_level(raw_levels.get(key))
                for key in CAPABILITY_KEYS
            },
        )

        notify = config.get("notify")
        notify = notify if isinstance(notify, dict) else {}
        audit_conf = config.get("audit")
        audit_conf = audit_conf if isinstance(audit_conf, dict) else {}
        confirm_conf = config.get("confirm")
        confirm_conf = confirm_conf if isinstance(confirm_conf, dict) else {}
        blacklist_conf = config.get("blacklist")
        blacklist_conf = blacklist_conf if isinstance(blacklist_conf, dict) else {}

        self.receipt_enabled = bool(notify.get("operation_receipt", True))
        self.receipt_to_manage_session = bool(
            notify.get("receipt_to_manage_session", False),
        )
        self._audit_enabled = bool(audit_conf.get("enabled", True))
        self._audit_retention_days = int(audit_conf.get("retention_days", 30))

        self._blacklist_enabled = bool(blacklist_conf.get("enabled", True))
        self.blacklist_default_reason = (
            str(blacklist_conf.get("default_reason") or "未填写原因").strip()
            or "未填写原因"
        )
        self._blacklist_block_message = bool(
            blacklist_conf.get("block_message", False),
        )

        self._confirm_enabled = bool(confirm_conf.get("enabled", True))
        self._confirm_timeout = max(5, int(confirm_conf.get("timeout", 60)))
        self._confirm_words = normalize_words(
            confirm_conf.get("confirm_words"),
            ["同意", "确认", "yes"],
        )
        self._confirm_cancel_words = normalize_words(
            confirm_conf.get("cancel_words"),
            ["取消", "否", "no"],
        )
        self._confirm_require_for = normalize_bool_list(
            confirm_conf.get("require_for"),
        ) or ["kick", "whole_ban"]

        self.default_ban_duration = parse_duration(
            str(config.get("default_ban_duration", DEFAULT_BAN_DURATION)),
            DEFAULT_BAN_DURATION,
        )
        if not self.default_ban_duration:
            self.default_ban_duration = DEFAULT_BAN_DURATION

        raw_templates = config.get("templates")
        self.templates: dict[str, Any] = (
            raw_templates if isinstance(raw_templates, dict) else {}
        )

    def _init_services(self) -> None:
        """Create services that depend on the parsed configuration."""
        data_dir = StarTools.get_data_dir("astrbot_plugin_qq_admin")
        self.audit = AuditLog(
            data_dir,
            enabled=self._audit_enabled,
            retention_days=self._audit_retention_days,
        )
        self.blacklist = BlacklistStore(data_dir, enabled=self._blacklist_enabled)
        self.notifier = Notifier(
            self.context,
            self.templates,
            receipt_enabled=self.receipt_enabled,
            receipt_to_manage_session=self.receipt_to_manage_session,
        )
        self._build_confirm_manager()

    def _build_confirm_manager(self) -> None:
        """Instantiate the confirmation manager from the parsed policy.

        The manager is the single source of truth for the confirmation policy;
        the raw values in ``_confirm_*`` are construction inputs only, so there
        is no second copy of the state to fall out of sync.
        """
        self.confirm = ConfirmManager(
            self.templates,
            enabled=self._confirm_enabled,
            timeout=self._confirm_timeout,
            confirm_words=self._confirm_words,
            cancel_words=self._confirm_cancel_words,
            require_for=self._confirm_require_for,
            permission_checker=self._check_capability_for_event,
        )

    # ------------------------------------------------------------------ #
    #  Shared helpers
    # ------------------------------------------------------------------ #

    async def _check_capability_for_event(
        self,
        event: AstrMessageEvent,
        capability: str,
    ) -> tuple[bool, str]:
        """Resolve permission for an arbitrary event.

        Used by the confirmation manager, which validates the responder rather
        than the original operator.

        Args:
            event: Event to evaluate.
            capability: Capability key to check.

        Returns:
            ``(allowed, reason)`` as returned by :func:`check_capability`.
        """
        if not self.enabled:
            return False, "插件当前已禁用"
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return False, "本插件仅支持 aiocqhttp(OneBot v11) 平台"
        context = await self._permission_context(event)
        return check_capability(context, self.permission_settings, capability)

    async def _permission_context(self, event: AstrMessageEvent) -> PermissionContext:
        """Build the operator identity for an event.

        Args:
            event: Event of the command.

        Returns:
            The resolved :class:`PermissionContext`.
        """
        return await build_permission_context(
            event,
            self.client,
            self.permission_settings,
        )

    def _should_skip(self, event: AstrMessageEvent) -> bool:
        """Check the cheap preconditions shared by every command handler.

        Commands are silently ignored rather than answered when the plugin is
        disabled, when the adapter is not aiocqhttp (the OneBot actions this
        plugin relies on do not exist elsewhere), or when the message is not
        from a group. Bailing out here also avoids the extra protocol calls
        that permission resolution would otherwise make.

        Args:
            event: Event of the command.

        Returns:
            ``True`` when the handler must return without doing anything.
        """
        if not self.enabled:
            return True
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return True
        return not event.get_group_id()

    def _text(self, key: str, **values: Any) -> str:
        """Render a template with this plugin's configuration.

        Args:
            key: Template key.
            **values: Placeholder values.

        Returns:
            The rendered message.
        """
        return render_template(self.templates, key, **values)

    async def _reply(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        stop: bool = True,
    ) -> AsyncGenerator[Any, None]:
        """Send a text reply and optionally stop the event.

        Args:
            event: Event to reply to.
            text: Message body. An empty string sends nothing.
            stop: Whether to stop event propagation afterwards.

        Yields:
            The message result for the pipeline to deliver.
        """
        if text:
            yield event.plain_result(text)
        if stop:
            event.stop_event()

    async def _resolve_member_name(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
    ) -> str:
        """Look up a member's display name.

        Args:
            event: Event providing the OneBot connection.
            group_id: Group the member belongs to.
            user_id: Member QQ number.

        Returns:
            The group card, nickname, or the raw QQ number as a last resort.
        """
        info = await self.client.get_group_member_info(event, group_id, user_id)
        if info:
            name = str(info.get("card") or info.get("nickname") or "").strip()
            if name:
                return name
        return user_id

    def _log_action(
        self,
        action: str,
        operator: PermissionContext,
        *,
        target: str = "",
        group_id: str = "",
        result: str = "ok",
        error: str = "",
        **detail: Any,
    ) -> None:
        """Write an entry to the audit trail.

        Args:
            action: Machine readable action name.
            operator: Identity of the operator.
            target: QQ number the action applied to.
            group_id: Group the action happened in.
            result: ``"ok"`` or ``"failed"``.
            error: Failure description.
            **detail: Extra context recorded verbatim.
        """
        self.audit.log(
            action,
            operator.user_id,
            target=target,
            group_id=group_id,
            result=result,
            error=error,
            **detail,
        )

    # ------------------------------------------------------------------ #
    #  Whole-group mute scheduling
    # ------------------------------------------------------------------ #

    def _schedule_whole_ban_off(
        self,
        event: AstrMessageEvent,
        group_id: str,
        duration: int,
    ) -> None:
        """Schedule automatic removal of a whole-group mute.

        OneBot offers no timed variant of ``set_group_whole_ban``, so the
        plugin has to lift it itself. The task is tracked so that
        :meth:`terminate` can cancel it.

        Args:
            event: Event providing the OneBot connection.
            group_id: Group that was muted.
            duration: Delay in seconds before unmuting.
        """

        async def _worker() -> None:
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                return
            result = await self.client.set_group_whole_ban(
                event,
                group_id,
                enabled=False,
            )
            if result.ok:
                logger.info(f"[QQAdmin] 群 {group_id} 全群禁言已到期解除")
            else:
                logger.warning(
                    f"[QQAdmin] 群 {group_id} 全群禁言自动解除失败：{result.error}",
                )

        existing = self._whole_ban_tasks.pop(group_id, None)
        if existing and not existing.done():
            existing.cancel()
        self._whole_ban_tasks[group_id] = asyncio.create_task(_worker())

    # ------------------------------------------------------------------ #
    #  Commands: mute
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("禁言", alias={"ban"})
    async def ban(
        self,
        event: AstrMessageEvent,
        args: GreedyStr = "",
    ) -> AsyncGenerator[Any, None]:
        """禁言群成员，或使用「禁言 全体」开启全群禁言。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人 10m 刷屏」或「全体 1h」。允许省略。
        """
        if self._should_skip(event):
            return
        tokens = str(args or "").split()
        group_id = str(event.get_group_id())

        operator = await self._permission_context(event)

        if tokens and is_all_target(tokens[0]):
            async for item in self._whole_ban(event, operator, group_id, tokens[1:]):
                yield item
            return

        parsed = parse_target_args(tokens, extract_at_targets(event))

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "ban",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        if parsed.unknown_duration:
            async for item in self._reply(event, self._text("bad_duration")):
                yield item
            return

        if not parsed.targets:
            async for item in self._reply(event, self._text("no_target")):
                yield item
            return

        duration = (
            parsed.duration if parsed.duration_specified else self.default_ban_duration
        )
        if duration is None:
            duration = self.default_ban_duration

        lines: list[str] = []
        for user_id in parsed.targets:
            result = await self.client.set_group_ban(
                event,
                group_id,
                user_id,
                duration,
            )
            name = await self._resolve_member_name(event, group_id, user_id)
            if result.ok:
                lines.append(
                    self._text(
                        "ban_success",
                        target_name=name,
                        target_id=user_id,
                        duration=format_duration(duration),
                    ),
                )
                self._log_action(
                    "ban",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    duration=duration,
                    reason=parsed.reason,
                )
            else:
                lines.append(await self._describe_failure(result, user_id))
                self._log_action(
                    "ban",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    result="failed",
                    error=result.error,
                    duration=duration,
                    reason=parsed.reason,
                )

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    async def _whole_ban(
        self,
        event: AstrMessageEvent,
        operator: PermissionContext,
        group_id: str,
        rest: list[str],
    ) -> AsyncGenerator[Any, None]:
        """Toggle the whole-group mute, honouring confirmation and timers.

        Args:
            event: 消息事件。
            operator: 操作者身份。
            group_id: 目标群号。
            rest: 「全体」之后的参数。

        Yields:
            回复内容。
        """
        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "whole_ban",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        duration: int | None = None
        if rest:
            duration = parse_duration_token(rest)
            if duration is None:
                async for item in self._reply(event, self._text("bad_duration")):
                    yield item
                return

        summary = (
            f"开启全群禁言 {format_duration(duration)}" if duration else "开启全群禁言"
        )
        if self.confirm.is_required("whole_ban"):
            action = PendingAction(
                summary=summary,
                payload={"group_id": group_id, "duration": duration},
                capability="whole_ban",
            )
            async for item in self._reply(event, self.confirm.build_prompt(summary)):
                yield item
            if not await self.confirm.request(event, action, operator):
                self._log_action(
                    "whole_ban",
                    operator,
                    group_id=group_id,
                    result="failed",
                    error="cancelled",
                )
                return

        result = await self.client.set_group_whole_ban(event, group_id, enabled=True)
        if not result.ok:
            self._log_action(
                "whole_ban",
                operator,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            async for item in self._reply(event, await self._describe_failure(result)):
                yield item
            return

        self._log_action(
            "whole_ban",
            operator,
            group_id=group_id,
            duration=duration or 0,
        )
        if duration:
            self._schedule_whole_ban_off(
                event, group_id, min(duration, MAX_BAN_DURATION)
            )
            text = self._text("whole_ban_timed", duration=format_duration(duration))
        else:
            text = self._text("whole_ban_on")
        async for item in self._reply(event, text):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("解禁", alias={"unban"})
    async def unban(
        self,
        event: AstrMessageEvent,
        args: GreedyStr = "",
    ) -> AsyncGenerator[Any, None]:
        """解除禁言，或使用「解禁 全体」关闭全群禁言。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人」或「全体」。允许省略。
        """
        if self._should_skip(event):
            return
        text = str(args or "").strip()
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        if is_all_target(text):
            allowed, deny_reason = check_capability(
                operator,
                self.permission_settings,
                "whole_ban",
            )
            if not allowed:
                async for item in self._reply(
                    event,
                    self._text("no_permission", reason=deny_reason),
                ):
                    yield item
                return

            task = self._whole_ban_tasks.pop(group_id, None)
            if task and not task.done():
                task.cancel()

            result = await self.client.set_group_whole_ban(
                event,
                group_id,
                enabled=False,
            )
            if result.ok:
                self._log_action("whole_ban_off", operator, group_id=group_id)
                async for item in self._reply(event, self._text("whole_ban_off")):
                    yield item
            else:
                self._log_action(
                    "whole_ban_off",
                    operator,
                    group_id=group_id,
                    result="failed",
                    error=result.error,
                )
                async for item in self._reply(
                    event,
                    await self._describe_failure(result),
                ):
                    yield item
            return

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "ban",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        # 解禁只需要目标，不需要时长与理由，因此只需挑出 QQ 号。
        parsed = parse_target_args(str(text or "").split(), extract_at_targets(event))
        if not parsed.targets:
            async for item in self._reply(event, self._text("no_target")):
                yield item
            return

        lines: list[str] = []
        for user_id in parsed.targets:
            result = await self.client.set_group_ban(event, group_id, user_id, 0)
            name = await self._resolve_member_name(event, group_id, user_id)
            if result.ok:
                lines.append(
                    self._text("unban_success", target_name=name, target_id=user_id),
                )
                self._log_action("unban", operator, target=user_id, group_id=group_id)
            else:
                lines.append(await self._describe_failure(result, user_id))
                self._log_action(
                    "unban",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    result="failed",
                    error=result.error,
                )

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    # ------------------------------------------------------------------ #
    #  Commands: kick / recall
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("踢", alias={"kick", "踢人"})
    async def kick(
        self,
        event: AstrMessageEvent,
        args: GreedyStr = "",
    ) -> AsyncGenerator[Any, None]:
        """将成员移出本群，支持可选理由。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人 广告」。允许省略。
        """
        if self._should_skip(event):
            return
        text = str(args or "").strip()
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "kick",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        parsed = parse_target_args(str(text or "").split(), extract_at_targets(event))
        if not parsed.targets:
            async for item in self._reply(event, self._text("no_target")):
                yield item
            return
        targets = parsed.targets
        reason = parsed.reason

        names = [
            await self._resolve_member_name(event, group_id, user_id)
            for user_id in targets
        ]
        summary = "移出本群：" + "、".join(
            f"{name}({uid})" for name, uid in zip(names, targets, strict=False)
        )
        if reason:
            summary += f"（理由：{reason}）"

        if self.confirm.is_required("kick"):
            action = PendingAction(
                summary=summary,
                payload={"group_id": group_id, "targets": list(targets)},
                capability="kick",
            )
            async for item in self._reply(event, self.confirm.build_prompt(summary)):
                yield item
            if not await self.confirm.request(event, action, operator):
                self._log_action(
                    "kick",
                    operator,
                    group_id=group_id,
                    result="failed",
                    error="cancelled",
                )
                return

        lines: list[str] = []
        for user_id, name in zip(targets, names, strict=False):
            result = await self.client.set_group_kick(event, group_id, user_id)
            if result.ok:
                lines.append(
                    self._text("kick_success", target_name=name, target_id=user_id),
                )
                self._log_action(
                    "kick",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    reason=reason,
                )
            else:
                lines.append(await self._describe_failure(result, user_id))
                self._log_action(
                    "kick",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    result="failed",
                    error=result.error,
                    reason=reason,
                )

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("撤回", alias={"recall"})
    async def recall(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """撤回消息。回复目标消息后使用本指令。

        Args:
            event: 消息事件。
        """
        if self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "recall",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        message_id = extract_reply_message_id(event)
        if not message_id:
            async for item in self._reply(
                event,
                self._text("no_target") + "（请回复要撤回的消息）",
            ):
                yield item
            return

        result = await self.client.delete_msg(event, message_id)
        if result.ok:
            self._log_action(
                "recall",
                operator,
                group_id=group_id,
                message_id=message_id,
            )
            async for item in self._reply(event, self._text("recall_success")):
                yield item
        else:
            self._log_action(
                "recall",
                operator,
                group_id=group_id,
                result="failed",
                error=result.error,
                message_id=message_id,
            )
            async for item in self._reply(event, await self._describe_failure(result)):
                yield item

    # ------------------------------------------------------------------ #
    #  Commands: group notice / help / status
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("群公告", alias={"notice"})
    async def notice(
        self,
        event: AstrMessageEvent,
        content: GreedyStr = "",
    ) -> AsyncGenerator[Any, None]:
        """发送群公告；不带参数时查看当前公告。

        Args:
            event: 消息事件。
            content: 公告正文。允许省略。
        """
        if self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)
        text = str(content or "").strip()

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "notice",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        if not text:
            notices = await self.client.get_group_notice(event, group_id)
            if not notices:
                async for item in self._reply(
                    event,
                    self._text("notice_unsupported"),
                ):
                    yield item
                return
            latest = str(notices[-1].get("notice") or "").strip()
            async for item in self._reply(event, latest or self._text("notice_empty")):
                yield item
            return

        result = await self.client.send_group_notice(event, group_id, text)
        if result.ok:
            self._log_action(
                "notice",
                operator,
                group_id=group_id,
                content=shorten(text, 50),
            )
            async for item in self._reply(event, self._text("notice_success")):
                yield item
        else:
            self._log_action(
                "notice",
                operator,
                group_id=group_id,
                result="failed",
                error=result.error,
            )
            hint = (
                self._text("api_unsupported")
                if result.unsupported
                else self._text("api_failed", error=result.error)
            )
            async for item in self._reply(event, hint):
                yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("群管", alias={"qqadmin"})
    async def help_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查看 QQ群管插件的指令用法。

        Args:
            event: 消息事件。
        """
        if self._should_skip(event):
            return
        lines = [
            "QQ群管 指令列表",
            "  禁言 @某人 [时长] [理由]   如：禁言 @小王 10m 刷屏",
            "  禁言 全体 [时长]           全群禁言，带时长会自动解除",
            "  解禁 @某人 / 解禁 全体",
            "  踢 @某人 [理由]            将成员移出本群",
            "  撤回                       回复目标消息后使用",
            "  群公告 [内容]              不带内容则查看当前公告",
            "  群管状态                   查看运行状态与群组归属",
            "",
            "提示：指令与 @ 之间需要空格；若与其他插件指令重名，",
            "可在 WebUI 的「插件 → 管理行为」中重命名。",
        ]
        async for item in self._reply(event, "\n".join(lines)):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("群管状态", alias={"qqadminstatus"})
    async def status(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查看插件运行状态与当前群的群组归属。

        Args:
            event: 消息事件。
        """
        if self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "query",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        group_set = find_group_set(self.group_sets, group_id)
        lines = [
            "QQ群管 运行状态",
            f"  插件：{'已启用' if self.enabled else '已禁用'}",
            f"  当前群：{group_id}",
        ]
        if group_set:
            lines.append(f"  所属群组：{group_set['name']}")
            lines.append(f"  组内群数：{len(group_set['group_ids'])}")
            manage_session = group_set["manage_session"] or "未配置（回退到当前群）"
            lines.append(f"  管理会话：{manage_session}")
        else:
            lines.append(f"  所属群组：无（{self._text('group_unconfigured')}）")
        lines.append(f"  已配置群组总数：{len(self.group_sets)}")
        lines.append(f"  黑名单条目：{self.blacklist.count(self._scope(group_id))}")

        conflicted = self._conflicting_groups()
        if conflicted:
            lines.append(f"  配置冲突群号：{', '.join(conflicted)}")

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    # ------------------------------------------------------------------ #
    #  Commands: blacklist
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("拉黑", alias={"blacklist"})
    async def blacklist_add(
        self,
        event: AstrMessageEvent,
        args: GreedyStr = "",
    ) -> AsyncGenerator[Any, None]:
        """将成员加入本群组的黑名单，原因会记入导出表格。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人 发广告 7d」。允许省略。
        """
        if self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "blacklist",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        parsed = parse_target_args(
            str(args or "").split(),
            extract_at_targets(event),
        )
        if not parsed.targets:
            async for item in self._reply(event, self._text("no_target")):
                yield item
            return
        if parsed.unknown_duration:
            async for item in self._reply(event, self._text("bad_duration")):
                yield item
            return

        duration = parsed.duration if parsed.duration_specified else 0
        reason = parsed.reason or self.blacklist_default_reason
        scope = self._scope(group_id)

        lines: list[str] = []
        for user_id in parsed.targets:
            nickname = await self._resolve_member_name(event, group_id, user_id)
            entry = self.blacklist.add(
                scope,
                user_id,
                reason=reason,
                operator=operator.user_id,
                duration=duration or 0,
                nickname=nickname,
            )
            lines.append(
                self._text(
                    "blacklist_added",
                    target_name=nickname,
                    target_id=user_id,
                    reason=reason,
                    duration=(
                        format_duration(duration) if entry.is_temporary else "永久"
                    ),
                ),
            )
            self._log_action(
                "blacklist_add",
                operator,
                target=user_id,
                group_id=group_id,
                scope=scope,
                reason=reason,
                duration=duration or 0,
            )

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("解除拉黑", alias={"unblack"})
    async def blacklist_remove(
        self,
        event: AstrMessageEvent,
        args: GreedyStr = "",
    ) -> AsyncGenerator[Any, None]:
        """将成员移出本群组的黑名单。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人」。允许省略。
        """
        if self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "blacklist",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        parsed = parse_target_args(
            str(args or "").split(),
            extract_at_targets(event),
        )
        if not parsed.targets:
            async for item in self._reply(event, self._text("no_target")):
                yield item
            return

        scope = self._scope(group_id)
        lines: list[str] = []
        for user_id in parsed.targets:
            name = await self._resolve_member_name(event, group_id, user_id)
            if self.blacklist.remove(scope, user_id):
                lines.append(
                    self._text(
                        "blacklist_removed",
                        target_name=name,
                        target_id=user_id,
                    ),
                )
                self._log_action(
                    "blacklist_remove",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    scope=scope,
                )
            else:
                lines.append(
                    self._text(
                        "blacklist_not_found",
                        target_name=name,
                        target_id=user_id,
                    ),
                )

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("黑名单", alias={"blist"})
    async def blacklist_list(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[Any, None]:
        """查看本群组共享的黑名单。

        Args:
            event: 消息事件。
        """
        if self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "query",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        scope = self._scope(group_id)
        entries = self.blacklist.list_entries(scope)
        if not entries:
            async for item in self._reply(
                event,
                self._text("blacklist_empty", scope=scope),
            ):
                yield item
            return

        group_set = find_group_set(self.group_sets, group_id)
        header = self._text(
            "blacklist_header",
            scope=scope,
            count=len(entries),
            groups=len(group_set["group_ids"]) if group_set else 1,
        )
        lines = [header]
        for index, entry in enumerate(entries, start=1):
            name = entry.nickname or entry.user_id
            expiry = (
                format_duration(int(entry.expires_at - entry.created_at))
                if entry.is_temporary
                else "永久"
            )
            lines.append(
                f"{index}. {name}({entry.user_id}) | {entry.reason} | {expiry}",
            )
        async for item in self._reply(event, "\n".join(lines)):
            yield item

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _scope(self, group_id: str) -> str:
        """Resolve the blacklist bucket for a group.

        Args:
            group_id: QQ group id.

        Returns:
            The shared bucket key when the group belongs to a set, otherwise a
            bucket private to the group.
        """
        return resolve_blacklist_scope(self.group_sets, group_id)

    def _conflicting_groups(self) -> list[str]:
        """List group ids appearing in more than one group set.

        Returns:
            Sorted group ids that are configured ambiguously.
        """
        seen: dict[str, int] = {}
        for group_set in self.group_sets:
            for group_id in group_set["group_ids"]:
                seen[group_id] = seen.get(group_id, 0) + 1
        return sorted(gid for gid, count in seen.items() if count > 1)

    # ------------------------------------------------------------------ #
    #  Failure rendering
    # ------------------------------------------------------------------ #

    async def _describe_failure(
        self,
        result: ActionResult,
        target: str = "",
    ) -> str:
        """Turn a failed action result into a user-facing message.

        Args:
            result: The failed action result.
            target: Target QQ number, included when available.

        Returns:
            A localized description of the failure.
        """
        if result.unsupported:
            text = self._text("api_unsupported")
        else:
            text = self._text("api_failed", error=result.error)
        return f"{target}{'：' if target else ''}{text}"

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def terminate(self) -> None:
        """Cancel pending tasks when the plugin is unloaded or disabled.

        Whole-group mute timers cannot survive a restart, because OneBot has no
        server-side timer to restore. Any mute still running therefore stays on
        until an operator lifts it manually; a warning is logged to make that
        visible.
        """
        for group_id, task in list(self._whole_ban_tasks.items()):
            if not task.done():
                task.cancel()
                logger.warning(
                    f"[QQAdmin] 群 {group_id} 的全群禁言定时任务因插件卸载被取消，"
                    "该群的禁言状态需要手动解除",
                )
        self._whole_ban_tasks.clear()
        self.client.invalidate_member_cache()
