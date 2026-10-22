"""QQ群管 — OneBot v11 group management for AstrBot.

Entry point registering the command handlers. Only the aiocqhttp platform is
supported: every handler is guarded by a platform filter so that the plugin
stays inert on adapters that cannot provide the required OneBot actions.

See ``plan.md`` for the full design; this file only wires things together, with
the reusable pieces living under ``core/``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import AtAll
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import GreedyStr

from .core.args import parse_duration_token, parse_target_args
from .core.audit import AuditLog
from .core.blacklist import BlacklistStore
from .core.config import (
    CAPABILITY_KEYS,
    DEFAULT_REVIEW_REJECT_REASON,
    MAX_BAN_DURATION,
    NOTIFY_CATEGORIES,
    NOTIFY_CATEGORY_BAN,
    NOTIFY_CATEGORY_BLACKLIST,
    NOTIFY_CATEGORY_EXPORT,
    NOTIFY_CATEGORY_KICK,
    NOTIFY_CATEGORY_RECALL,
    NOTIFY_CATEGORY_REVIEW_ACTION,
    NOTIFY_CATEGORY_REVIEW_NOTICE,
    build_group_index,
    find_group_set,
    find_group_set_by_name,
    format_duration,
    group_review_policy,
    group_settings,
    is_all_target,
    manage_group_sets,
    normalize_bool_list,
    normalize_group_sets,
    normalize_notify_routes,
    normalize_permission_level,
    normalize_words,
    resolve_blacklist_scope,
    review_exempt_sources,
    review_keywords,
)
from .core.confirm import ConfirmManager, PendingAction
from .core.export import Exporter, to_file_uri
from .core.filters import mentioned_command
from .core.messages import render_template
from .core.notify import Notifier, parse_manage_session
from .core.onebot import ActionResult, OneBotClient
from .core.permission import (
    PermissionContext,
    PermissionSettings,
    build_permission_context,
    check_capability,
)
from .core.review import (
    PendingReviewStore,
    evaluate,
    parse_request_event,
)
from .core.tools import TOOL_NAMES, build_tools, log_registered
from .core.utils import (
    extract_at_targets,
    extract_reply_message_id,
)
from .core.wholeban import RESTORE_GRACE, WholeBanRecord, WholeBanStore

#: Platform adapter this plugin is able to drive.
SUPPORTED_PLATFORM = "aiocqhttp"


@dataclass
class _SimpleDecision:
    """Minimal decision holder used when an operator resolves a request.

    Attributes:
        action: ``"approve"`` or ``"reject"``.
        reason: Machine readable cause.
        detail: Human readable explanation passed to the notification.
    """

    action: str
    reason: str
    detail: str = ""


class QQAdminPlugin(Star):
    """Command handlers and shared services for the QQ群管 plugin.

    Attributes:
        config: Plugin configuration handed over by AstrBot.
        client: OneBot action facade.
        audit: Audit trail writer.
        confirm: High-risk action confirmation manager.
        group_sets: Normalized group set definitions.
        group_index: ``group_id -> set name`` lookup.
        group_sets: Normalized group set definitions, including each set's
            join review policy and cross-group exemptions.
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
        self.exporter.cleanup()

        logger.info(
            f"[QQAdmin] 已加载 | 群组 {len(self.group_sets)} 个 | "
            f"索引 {len(self.group_index)} 个群",
        )

    @filter.on_platform_loaded()
    async def restore_whole_ban_timers(self) -> None:
        """Re-arm whole-group mute timers that were pending before the reload.

        The deadlines are persisted, so a restart does not turn a timed mute
        into a permanent one. Each restored group is verified against the
        server first: the mute may have been lifted manually while the bot was
        down, in which case only the stale record is dropped.
        """
        records = self.whole_ban.load_all()
        if not records:
            return

        platform = self._find_onebot_platform()
        if platform is None:
            logger.warning(
                f"[QQAdmin] 有 {len(records)} 个全群禁言待恢复，但未找到可用的 "
                "aiocqhttp 平台实例，已保留记录待下次启动重试",
            )
            return

        for record in records:
            status = await self._probe_whole_ban(platform, record.group_id)
            if status is False:
                logger.info(
                    f"[QQAdmin] 群 {record.group_id} 的全群禁言已不在生效，"
                    "清理过期记录",
                )
                self.whole_ban.remove(record.group_id)
                continue

            if status is None:
                # The protocol implementation does not report the mute state.
                # Acting on a guess could lift a mute an operator set manually
                # while the bot was down, so the record is kept for inspection
                # instead of being trusted or deleted.
                logger.warning(
                    f"[QQAdmin] 无法确认群 {record.group_id} 的禁言状态"
                    f"（原定 {format_duration(max(0, record.remaining))} 后解除），"
                    "已保留 whole_ban.json 中的记录，请手动确认后处理",
                )
                continue

            delay = max(RESTORE_GRACE, record.remaining)
            self._arm_whole_ban_timer(platform, record.group_id, delay)
            logger.info(
                f"[QQAdmin] 已恢复群 {record.group_id} 的全群禁言定时，"
                f"将在 {format_duration(delay)} 后解除",
            )

    def _find_onebot_platform(self) -> Any | None:
        """Locate a live aiocqhttp platform instance.

        Timers that survive a reload must be rescheduled without an event, so
        the platform is resolved from the context instead.

        Returns:
            The platform instance, or ``None`` when unavailable.
        """
        manager = getattr(self.context, "platform_manager", None)
        insts = getattr(manager, "platform_insts", None)
        if insts is None:
            getter = getattr(manager, "get_insts", None)
            insts = getter() if callable(getter) else []
        for inst in insts or []:
            try:
                if inst.meta().name == SUPPORTED_PLATFORM:
                    return inst
            except Exception:  # noqa: BLE001 - a foreign instance must not break us
                continue
        return None

    async def _probe_whole_ban(
        self,
        platform: Any,
        group_id: str,
    ) -> bool | None:
        """Ask the server whether a group is currently muted.

        Args:
            platform: Live aiocqhttp platform instance.
            group_id: Group to check.

        Returns:
            ``True``/``False`` when the state could be read, ``None`` when the
            protocol implementation does not expose the information.
        """
        # Routed through the OneBot facade so error handling and the
        # multi-connection ``self_id`` parameter stay consistent.
        info = await self.client.get_group_info(platform, group_id)
        if not isinstance(info, dict) or "group_all_shut" not in info:
            logger.debug(f"[QQAdmin] 无法读取群 {group_id} 的全群禁言状态")
            return None
        return bool(info.get("group_all_shut"))

    # ------------------------------------------------------------------ #
    #  Startup helpers
    # ------------------------------------------------------------------ #

    def _load_config(self, config: AstrBotConfig) -> None:
        """Parse the configuration into normalized structures.

        Settings that only make sense per group live inside ``group_sets``:
        ban duration, blacklist policy and join review (general policy plus
        cross-group exemptions). They are resolved per group through
        :meth:`_policy` / :meth:`_review_policy`.

        Args:
            config: Raw plugin configuration.
        """
        self.group_sets = normalize_group_sets(config.get("group_sets"))
        self.group_index = build_group_index(self.group_sets)
        for group_set in self.group_sets:
            for exemption in group_set["join_review"]["exemptions"]:
                for group_id in exemption["target_group_ids"]:
                    if group_id not in self.group_index:
                        logger.warning(
                            f"[QQAdmin] 群组「{group_set['name']}」的跨群免审核中，"
                            f"目标群 {group_id} 未加入任何群组，该规则对它不会生效",
                        )

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
        self.notify_routes = normalize_notify_routes(notify)
        audit_conf = config.get("audit")
        audit_conf = audit_conf if isinstance(audit_conf, dict) else {}
        confirm_conf = config.get("confirm")
        confirm_conf = confirm_conf if isinstance(confirm_conf, dict) else {}

        export_conf = config.get("export")
        export_conf = export_conf if isinstance(export_conf, dict) else {}
        self._audit_enabled = bool(audit_conf.get("enabled", True))
        self._audit_retention_days = int(audit_conf.get("retention_days", 30))

        self._level_unsupported_warned = False

        self._export_upload = bool(export_conf.get("upload_to_group", True))
        self._export_max_members = max(1, int(export_conf.get("max_members", 3000)))
        self._export_cleanup_hours = max(0, int(export_conf.get("cleanup_hours", 24)))

        raw_tools = config.get("llm_tool_options")
        requested = (
            {str(item).strip() for item in raw_tools if str(item).strip()}
            if isinstance(raw_tools, (list, tuple, set))
            else set()
        )
        # Only names this build actually implements are accepted.
        self.enabled_tools: list[str] = [
            name for name in TOOL_NAMES if name in requested
        ]

        self._confirm_enabled = bool(confirm_conf.get("enabled", True))
        self._confirm_timeout = max(5, int(confirm_conf.get("timeout", 60)))
        self._confirm_words = normalize_words(
            confirm_conf.get("confirm_words"),
            ["同意", "确认", "yes"],
        )
        self._confirm_cancel_words = normalize_words(
            confirm_conf.get("cancel_words"),
            ["取消", "拒绝", "否", "no"],
        )
        self._confirm_require_for = normalize_bool_list(
            confirm_conf.get("require_for"),
        ) or ["kick", "whole_ban"]

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
        self.blacklist = BlacklistStore(data_dir)
        self.pending_reviews = PendingReviewStore(data_dir)
        self.whole_ban = WholeBanStore(data_dir)
        self.exporter = Exporter(
            data_dir / "exports",
            max_members=self._export_max_members,
            cleanup_hours=self._export_cleanup_hours,
        )
        self.notifier = Notifier(
            self.context,
            to_current=self.notify_routes["current"],
            to_manage=self.notify_routes["manage"],
        )
        self._register_tools()
        self._build_confirm_manager()

    def _register_tools(self) -> None:
        """Register the LLM tools the user opted into.

        Nothing is exposed by default: the framework's tool permission gate
        treats every non-builtin tool as available to ordinary members, so
        exposing these without an explicit opt-in would let anyone who can talk
        to the model mute or kick people.
        """
        self.tools = build_tools(self)
        selected = [
            self.tools[name] for name in self.enabled_tools if name in self.tools
        ]
        if selected:
            self.context.add_llm_tools(*selected)
        log_registered([tool.name for tool in selected])

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

    async def _should_skip(
        self,
        event: AstrMessageEvent,
        *,
        manage_ok: bool = False,
    ) -> bool:
        """Check the cheap preconditions shared by every command handler.

        Commands are silently ignored rather than answered when the plugin is
        disabled, when the adapter is not aiocqhttp (the OneBot actions this
        plugin relies on do not exist elsewhere), when the message does not come
        from a group, or when the group is not managed. Bailing out here also
        avoids the extra protocol calls that permission resolution would
        otherwise make.

        A manage session counts as a valid place to run the commands that make
        sense there (``manage_ok``). For the others the event is stopped with a
        short explanation, because leaving it alone would hand a group
        management command to the LLM.

        Args:
            event: Event of the command.
            manage_ok: Whether the command is available inside a manage session.

        Returns:
            ``True`` when the handler must return without doing anything.
        """
        if not self.enabled:
            return True
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return True
        group_id = str(event.get_group_id() or "")
        if not group_id:
            return True
        if group_id in self.group_index:
            return False

        if not self._manage_sets(event):
            logger.info(
                f"[QQAdmin] 群 {group_id} 未加入任何群组，已忽略该指令",
            )
            return True

        if manage_ok:
            return False

        # A manage session is ours, but this command needs a concrete group.
        await event.send(event.plain_result(self._text("manage_unsupported")))
        event.stop_event()
        event.should_call_llm(False)
        return True

    def _manage_sets(self, event: AstrMessageEvent) -> list[dict[str, Any]]:
        """List the group sets that use this session as their manage session.

        Args:
            event: Event of the command.

        Returns:
            The owning group sets; empty when the session is an ordinary chat.
        """
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return []
        return manage_group_sets(
            self.group_sets,
            str(event.unified_msg_origin or ""),
        )

    def _select_group_set(
        self,
        event: AstrMessageEvent,
        selector: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve which group set a blacklist command applies to.

        Inside a managed group the owning set is implicit. In a manage session
        the set that declared it is used, and an explicit selector (group id or
        group set name) picks one when several sets share the session.

        Args:
            event: Event of the command.
            selector: Optional group id or group set name from the arguments.

        Returns:
            ``(group_set, error)``; ``error`` is a user-facing message when the
            selector is unknown or the manage session is ambiguous.
        """
        text = str(selector or "").strip()
        if text:
            target = find_group_set(self.group_sets, text) or find_group_set_by_name(
                self.group_sets,
                text,
            )
            if target is None:
                return None, self._text("group_not_found", selector=text)
            return target, ""

        group_set = find_group_set(
            self.group_sets,
            str(event.get_group_id() or ""),
        )
        if group_set is not None:
            return group_set, ""

        bound = self._manage_sets(event)
        if len(bound) == 1:
            return bound[0], ""
        if not bound:
            return None, ""
        return None, self._text(
            "manage_ambiguous",
            names="、".join(str(item["name"]) for item in bound),
        )

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
        receipt: bool = False,
        group_id: str = "",
        category: str = "",
    ) -> AsyncGenerator[Any, None]:
        """Send a text reply and optionally stop the event.

        Args:
            event: Event to reply to.
            text: Message body. An empty string sends nothing.
            stop: Whether to stop event propagation afterwards. A stopped event
                also keeps the message away from the default LLM request.
            receipt: Mark ``text`` as the outcome of an executed management
                action. Receipts are routed by the ``notify`` configuration, so
                they may be withheld from the current session, mirrored into the
                manage session, or both.
            group_id: Group the action happened in; selects the group set whose
                manage session receives the mirrored receipt. Defaults to the
                event's group.
            category: Routing category of the receipt. Defaults to ``禁言``,
                which is the category of an unnamed management receipt.

        Yields:
            The message result for the pipeline to deliver.
        """
        origin = str(event.unified_msg_origin or "")
        mirror: list[str] = []
        if text and receipt:
            group_set = find_group_set(
                self.group_sets,
                group_id or str(event.get_group_id() or ""),
            )
            targets = self.notifier.targets(
                category or NOTIFY_CATEGORY_BAN,
                origin=origin,
                manage_session=str((group_set or {}).get("manage_session", "")),
            )
            if origin in targets:
                yield event.plain_result(text)
            mirror = [target for target in targets if target != origin]
        elif text:
            yield event.plain_result(text)

        for target in mirror:
            await self.notifier.send_to_session(target, text)

        if stop:
            event.stop_event()
            event.should_call_llm(False)

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
        *,
        operator: str = "",
    ) -> None:
        """Schedule automatic removal of a whole-group mute.

        OneBot offers no timed variant of ``set_group_whole_ban``, so the
        plugin has to lift it itself. The deadline is persisted before the
        timer is armed, so a restart can re-arm it instead of leaving the group
        muted forever (see :meth:`restore_whole_ban_timers`).

        Args:
            event: Event providing the OneBot connection.
            group_id: Group that was muted.
            duration: Delay in seconds before unmuting.
            operator: QQ number of the person who muted the group, recorded for
                the restored audit trail.
        """
        capped = min(max(0, int(duration)), MAX_BAN_DURATION)
        self.whole_ban.record(
            group_id,
            expires_at=time.time() + capped,
            duration=capped,
            operator=operator,
        )
        self._arm_whole_ban_timer(event, group_id, capped)

    def _arm_whole_ban_timer(
        self,
        source: Any,
        group_id: str,
        delay: int,
    ) -> None:
        """Create (or replace) the unmute timer for a group.

        Kept separate from :meth:`_schedule_whole_ban_off` because restoring a
        timer after a restart has no event to work with -- only the platform
        instance -- and the body must run with whichever source it was given.

        Args:
            source: Either an :class:`AstrMessageEvent` or a platform instance,
                passed straight through to the OneBot client.
            group_id: Group to unmute.
            delay: Seconds to wait before lifting the mute.
        """

        async def _worker() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            result = await self.client.set_group_whole_ban(
                source,
                group_id,
                enabled=False,
            )
            # The deadline has passed either way, so the record is cleared even
            # when the call fails; a failure is reported for manual follow-up.
            self.whole_ban.remove(group_id)
            self._whole_ban_tasks.pop(group_id, None)
            if result.ok:
                logger.info(f"[QQAdmin] 群 {group_id} 全群禁言已到期解除")
                self.audit.log(
                    "whole_ban_auto_off",
                    "system",
                    group_id=group_id,
                )
            else:
                logger.warning(
                    f"[QQAdmin] 群 {group_id} 全群禁言自动解除失败：{result.error}，"
                    "请手动解除",
                )
                self.audit.log(
                    "whole_ban_auto_off",
                    "system",
                    group_id=group_id,
                    result="failed",
                    error=result.error,
                )

        existing = self._whole_ban_tasks.pop(group_id, None)
        if existing and not existing.done():
            existing.cancel()
        self._whole_ban_tasks[group_id] = asyncio.create_task(_worker())

    def _cancel_whole_ban_timer(self, group_id: str) -> WholeBanRecord | None:
        """Stop and forget a pending unmute timer.

        Args:
            group_id: Group whose timer should be cancelled.

        Returns:
            The record that was pending, or ``None`` when there was none.
        """
        task = self._whole_ban_tasks.pop(group_id, None)
        if task and not task.done():
            task.cancel()
        record = self.whole_ban.get(group_id)
        self.whole_ban.remove(group_id)
        return record

    # ------------------------------------------------------------------ #
    #  Commands: mute
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("禁言", alias={"ban"})
    @filter.custom_filter(mentioned_command("禁言", "ban"))
    async def ban(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """禁言群成员，或使用「禁言 全体」开启全群禁言。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人 10m 刷屏」或「全体 1h」，顺序任意。
                允许省略。
        """
        if await self._should_skip(event):
            return
        tokens = str(args or "").split()
        group_id = str(event.get_group_id())

        operator = await self._permission_context(event)

        # 「全体」可以出现在任意位置（禁言 全体 10m / 禁言 10m 全体），@全体成员
        # 在 message_str 中不会留下文本，所以单独检查消息链。
        whole_index = next(
            (index for index, token in enumerate(tokens) if is_all_target(token)),
            None,
        )
        if whole_index is not None or any(
            isinstance(component, AtAll) for component in event.get_messages()
        ):
            rest = [token for index, token in enumerate(tokens) if index != whole_index]
            async for item in self._whole_ban(event, operator, group_id, rest):
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

        default_duration = self._policy(group_id)["default_ban_duration"]
        duration = parsed.duration if parsed.duration_specified else default_duration
        if duration is None:
            duration = default_duration

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

        async for item in self._reply(
            event,
            "\n".join(lines),
            receipt=True,
            category=NOTIFY_CATEGORY_BAN,
            group_id=group_id,
        ):
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
            outcome = await self.confirm.request(event, action, operator)
            if outcome != "confirmed":
                self._log_action(
                    "whole_ban",
                    operator,
                    group_id=group_id,
                    result="failed",
                    error=outcome,
                )
                # 超时与异常提示由确认管理器直接发出，这里只补取消提示。
                if outcome == "cancelled":
                    async for item in self._reply(
                        event,
                        self._text("confirm_cancelled"),
                    ):
                        yield item
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
            async for item in self._reply(
                event,
                await self._describe_failure(result),
                receipt=True,
                category=NOTIFY_CATEGORY_BAN,
                group_id=group_id,
            ):
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
                event,
                group_id,
                duration,
                operator=operator.user_id,
            )
            text = self._text("whole_ban_timed", duration=format_duration(duration))
        else:
            text = self._text("whole_ban_on")
        async for item in self._reply(
            event,
            text,
            receipt=True,
            category=NOTIFY_CATEGORY_BAN,
            group_id=group_id,
        ):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("解禁", alias={"unban"})
    @filter.custom_filter(mentioned_command("解禁", "unban"))
    async def unban(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """解除禁言，或使用「解禁 全体」关闭全群禁言。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人」或「全体」。允许省略。
        """
        if await self._should_skip(event):
            return
        text = str(args or "").strip()
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        if any(is_all_target(token) for token in text.split()) or any(
            isinstance(component, AtAll) for component in event.get_messages()
        ):
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

            previous = self._cancel_whole_ban_timer(group_id)
            if previous:
                logger.info(
                    f"[QQAdmin] 群 {group_id} 的定时解除被手动解禁取消"
                    f"（原定剩余 {format_duration(max(0, previous.remaining))}）",
                )

            result = await self.client.set_group_whole_ban(
                event,
                group_id,
                enabled=False,
            )
            if result.ok:
                self._log_action("whole_ban_off", operator, group_id=group_id)
                async for item in self._reply(
                    event,
                    self._text("whole_ban_off"),
                    receipt=True,
                    category=NOTIFY_CATEGORY_BAN,
                    group_id=group_id,
                ):
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
                    receipt=True,
                    category=NOTIFY_CATEGORY_BAN,
                    group_id=group_id,
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

        async for item in self._reply(
            event,
            "\n".join(lines),
            receipt=True,
            category=NOTIFY_CATEGORY_BAN,
            group_id=group_id,
        ):
            yield item

    # ------------------------------------------------------------------ #
    #  Commands: kick / recall
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("踢", alias={"kick", "踢人"})
    @filter.custom_filter(mentioned_command("踢", "kick", "踢人"))
    async def kick(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """将成员移出本群，支持可选理由。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人 广告」。允许省略。
        """
        if await self._should_skip(event):
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
            outcome = await self.confirm.request(event, action, operator)
            if outcome != "confirmed":
                self._log_action(
                    "kick",
                    operator,
                    group_id=group_id,
                    result="failed",
                    error=outcome,
                )
                # 超时与异常提示由确认管理器直接发出，这里只补取消提示。
                if outcome == "cancelled":
                    async for item in self._reply(
                        event,
                        self._text("confirm_cancelled"),
                    ):
                        yield item
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

        async for item in self._reply(
            event,
            "\n".join(lines),
            receipt=True,
            category=NOTIFY_CATEGORY_KICK,
            group_id=group_id,
        ):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("撤回", alias={"recall"})
    @filter.custom_filter(mentioned_command("撤回", "recall"))
    async def recall(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """撤回消息。回复目标消息后使用本指令。

        Args:
            event: 消息事件。
        """
        if await self._should_skip(event):
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
            async for item in self._reply(
                event,
                self._text("recall_success"),
                receipt=True,
                category=NOTIFY_CATEGORY_RECALL,
                group_id=group_id,
            ):
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
            async for item in self._reply(
                event,
                await self._describe_failure(result),
                receipt=True,
                category=NOTIFY_CATEGORY_RECALL,
                group_id=group_id,
            ):
                yield item

    # ------------------------------------------------------------------ #
    #  Commands: help / status
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("群管", alias={"qqadmin"})
    @filter.custom_filter(mentioned_command("群管", "qqadmin"))
    async def help_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查看 QQ群管插件的指令用法。

        Args:
            event: 消息事件。
        """
        if await self._should_skip(event, manage_ok=True):
            return
        lines = [
            "QQ群管 指令列表",
            "  禁言 @某人 [时长] [理由]   如：禁言 @小王 10m 刷屏",
            "  禁言 全体 [时长]           全群禁言，带时长会自动解除",
            "  解禁 @某人 / 解禁 全体",
            "  踢 @某人 [理由]            将成员移出本群",
            "  撤回                       回复目标消息后使用",
            "  拉黑 @某人 [原因] [时长]   加入本群组共享黑名单",
            "  解除拉黑 @某人             移出黑名单",
            "  黑名单 [群号/群组名]       查看本群组黑名单",
            "  群导出 成员 / 黑名单       导出 xlsx 并上传群文件",
            "  群管状态                   查看运行状态与群组归属",
            "",
            "进群申请需要人工确认时，引用通知消息回复「同意」或「拒绝」即可处理。",
            "管理会话（群组配置里的「管理会话」）专供群管理员使用：进群审核通知、",
            "黑名单 / 拉黑 / 解除拉黑 / 群管状态 均可直接在那里执行。",
            "  黑名单 通知设置 [群号或群组名] 查看该群组的通知路由",
            "",
            "提示：指令与 @ 之间需要空格；时长与目标的位置可以互换；若与其他插件指令重名，",
            "可在 WebUI 的「插件 → 管理行为」中重命名。",
        ]
        async for item in self._reply(event, "\n".join(lines)):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("群管状态", alias={"qqadminstatus"})
    @filter.custom_filter(mentioned_command("群管状态", "qqadminstatus"))
    async def status(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查看插件运行状态与当前群的群组归属。

        Args:
            event: 消息事件。
        """
        if await self._should_skip(event, manage_ok=True):
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
        if group_set is None:
            async for item in self._reply(event, self._manage_status(event)):
                yield item
            return
        lines = [
            "QQ群管 运行状态",
            f"  插件：{'已启用' if self.enabled else '已禁用'}",
            f"  当前群：{group_id}",
            f"  所属群组：{group_set['name']}",
            f"  组内群数：{len(group_set['group_ids'])}",
            f"  管理会话：{group_set['manage_session'] or '未配置（通知回退到当前群）'}",
            f"  已配置群组总数：{len(self.group_sets)}",
        ]
        policy = self._policy(group_id)
        blacklist_policy = policy["blacklist"]
        review = self._review_policy(group_id)
        lines.append(
            f"  默认禁言时长：{format_duration(policy['default_ban_duration'])}",
        )
        lines.append(
            f"  黑名单：{'已启用' if blacklist_policy['enabled'] else '已关闭'}"
            f" | 发言拦截：{'已开启' if blacklist_policy['block_message'] else '已关闭'}"
            f" | 拉黑踢出：{'已开启' if blacklist_policy['kick_on_add'] else '已关闭'}",
        )
        if review is None:
            review_line = "  进群审核：未配置（本群未加入任何群组）"
        else:
            review_line = f"  进群审核：{'已开启' if review['enabled'] else '已关闭'}"
            sources = self._review_exempt_sources(group_id)
            if sources and review["enabled"]:
                review_line += f" | 跨群免审核：{len(sources)} 个来源群"
        lines.append(review_line)
        lines.append(f"  黑名单条目：{self.blacklist.count(self._scope(group_id))}")

        conflicted = self._conflicting_groups()
        if conflicted:
            lines.append(f"  配置冲突群号：{', '.join(conflicted)}")

        names = [group_set["name"] for group_set in self.group_sets]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            lines.append(f"  群组重名：{'、'.join(duplicated)}（同名群组共享黑名单）")

        lines.append(self._text("notification_settings"))

        async for item in self._reply(event, "\n".join(lines)):
            yield item

    def _manage_status(self, event: AstrMessageEvent) -> str:
        """Render the status report of a manage session.

        A manage session is not itself a managed group (or it would already be
        handled by the per-group branch), so the report describes every group
        set that declared this session as its control room.

        Args:
            event: Event of the command.

        Returns:
            The status text.
        """
        lines = [
            "QQ群管 运行状态（管理会话）",
            f"  插件：{'已启用' if self.enabled else '已禁用'}",
            f"  当前会话：{event.unified_msg_origin}",
            f"  已配置群组总数：{len(self.group_sets)}",
        ]
        for group_set in self._manage_sets(event):
            review = group_set["join_review"]["policy"]
            lines.append(
                f"  群组「{group_set['name']}」："
                f"群 {', '.join(group_set['group_ids']) or '（未填写群号）'}"
                f" | 黑名单 {self.blacklist.count(str(group_set['name']))} 条"
                f" | 进群审核 {'已开启' if review['enabled'] else '已关闭'}",
            )
            lines.extend(self._notify_route_lines(group_set["name"], "    "))
        lines.append("  可用指令：审核 / 黑名单 / 拉黑 / 解除拉黑 / 群管状态")
        return "\n".join(lines)

    def _notify_route_lines(self, set_name: str, indent: str = "  ") -> list[str]:
        """Describe where each notification category of a group set goes.

        Args:
            set_name: Name of the group set.
            indent: Leading whitespace of every line.

        Returns:
            One line per category, plus a line listing the disabled categories.
        """
        group_set = find_group_set_by_name(self.group_sets, set_name)
        manage = str((group_set or {}).get("manage_session") or "")
        manage_ready = bool(parse_manage_session(manage))

        lines = [f"{indent}通知路由："]
        silent: list[str] = []
        for category in NOTIFY_CATEGORIES:
            targets: list[str] = []
            if self.notifier.wants(category, "current"):
                targets.append("当前会话")
            if self.notifier.wants(category, "manage"):
                targets.append(
                    "管理会话" if manage_ready else "管理会话未配置（回退当前会话）"
                )
            if not targets:
                silent.append(category)
                continue
            lines.append(f"{indent}  {category} → {'、'.join(targets)}")
        if silent:
            lines.append(f"{indent}  不发送：{'、'.join(silent)}")
        return lines

    # ------------------------------------------------------------------ #
    #  Commands: blacklist
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("拉黑", alias={"blacklist"})
    @filter.custom_filter(mentioned_command("拉黑", "blacklist"))
    async def blacklist_add(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """将成员加入本群组的黑名单，原因会记入导出表格。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人 发广告 7d」。允许省略。
        """
        if await self._should_skip(event, manage_ok=True):
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

        # In a manage session the owning group set is selected for us; inside a
        # managed group it is the one the command was issued in.
        group_set, error = self._select_group_set(event)
        if error:
            async for item in self._reply(event, error):
                yield item
            return

        blacklist_policy = self._policy(group_id, group_set)["blacklist"]
        if not blacklist_policy["enabled"]:
            # ``add`` cannot persist anything while the feature is off, so
            # reporting success here would be a lie.
            async for item in self._reply(event, self._text("blacklist_disabled")):
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
        reason = parsed.reason or blacklist_policy["default_reason"]
        scope = self._scope(group_id, group_set)

        # Kicking only makes sense while acting inside the group itself; from a
        # manage session the target may not even be a member of this chat.
        auto_kick = blacklist_policy["kick_on_add"] and group_id in (
            group_set["group_ids"] if group_set else []
        )
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

            if not auto_kick:
                continue
            kick = await self.client.set_group_kick(event, group_id, user_id)
            if kick.ok:
                lines.append(
                    self._text(
                        "kick_success",
                        target_name=nickname,
                        target_id=user_id,
                    ),
                )
                self._log_action(
                    "kick",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    reason=reason,
                    trigger="blacklist_auto",
                )
            else:
                lines.append(await self._describe_failure(kick, user_id))
                self._log_action(
                    "kick",
                    operator,
                    target=user_id,
                    group_id=group_id,
                    result="failed",
                    error=kick.error,
                    reason=reason,
                    trigger="blacklist_auto",
                )

        async for item in self._reply(
            event,
            "\n".join(lines),
            receipt=True,
            category=NOTIFY_CATEGORY_BLACKLIST,
            group_id=group_id,
        ):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("解除拉黑", alias={"unblack"})
    @filter.custom_filter(mentioned_command("解除拉黑", "unblack"))
    async def blacklist_remove(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """将成员移出本群组的黑名单。

        Args:
            event: 消息事件。
            args: 后续参数，形如「@某人」。允许省略。
        """
        if await self._should_skip(event, manage_ok=True):
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

        group_set, error = self._select_group_set(event)
        if error:
            async for item in self._reply(event, error):
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

        scope = self._scope(group_id, group_set)
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

        async for item in self._reply(
            event,
            "\n".join(lines),
            receipt=True,
            category=NOTIFY_CATEGORY_BLACKLIST,
            group_id=group_id,
        ):
            yield item

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("黑名单", alias={"blist"})
    @filter.custom_filter(mentioned_command("黑名单", "blist"))
    async def blacklist_list(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """查看群组共享的黑名单。

        Args:
            event: 消息事件。
            args: 可选的群号或群组名；在管理会话里对应多个群组时必须填写。
        """
        if await self._should_skip(event, manage_ok=True):
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

        arg = str(args or "").strip()
        if arg in ("设置", "通知设置", "settings"):
            group_set, error = self._select_group_set(event)
            if error:
                async for item in self._reply(event, error):
                    yield item
                return
            if group_set is None:
                async for item in self._reply(
                    event,
                    self._text("group_not_found", selector=group_id),
                ):
                    yield item
                return
            async for item in self._reply(
                event,
                "\n".join(self._notify_route_lines(str(group_set["name"]))),
            ):
                yield item
            return

        group_set, error = self._select_group_set(event, arg)
        if error:
            async for item in self._reply(event, error):
                yield item
            return
        if group_set is None:
            async for item in self._reply(
                event,
                self._text("group_not_found", selector=group_id),
            ):
                yield item
            return

        if not self._policy(group_id, group_set)["blacklist"]["enabled"]:
            async for item in self._reply(event, self._text("blacklist_disabled")):
                yield item
            return

        scope = self._scope(group_id, group_set)
        entries = self.blacklist.list_entries(scope)
        if not entries:
            async for item in self._reply(
                event,
                self._text("blacklist_empty", scope=scope),
            ):
                yield item
            return

        header = self._text(
            "blacklist_header",
            scope=scope,
            count=len(entries),
            groups=len(group_set["group_ids"]) or 1,
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

    def _scope(self, group_id: str, group_set: dict[str, Any] | None = None) -> str:
        """Resolve the blacklist bucket for a group.

        Args:
            group_id: QQ group id.
            group_set: Owning group set when it is already known, which is the
                case for commands run from a manage session.

        Returns:
            The shared bucket key when the group belongs to a set, otherwise a
            bucket private to the group.
        """
        if group_set is not None:
            return str(group_set["name"])
        return resolve_blacklist_scope(self.group_sets, group_id)

    def _policy(
        self,
        group_id: str,
        group_set: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve the behaviour settings that apply to a group.

        Args:
            group_id: QQ group id.
            group_set: Owning group set when it is already known.

        Returns:
            A mapping with ``default_ban_duration`` and ``blacklist``. Built-in
            defaults are only returned as a safety net; handlers already skip
            groups that belong to no group set.
        """
        if group_set is not None:
            return group_set
        return group_settings(self.group_sets, group_id)

    def _review_policy(self, target_group_id: str) -> dict[str, Any] | None:
        """Resolve the join review policy of a group.

        Args:
            target_group_id: QQ group being joined; selects the owning set.

        Returns:
            The group set's policy, or ``None`` when the group belongs to no
            group set (automatic review never runs for unconfigured groups).
        """
        return group_review_policy(self.group_sets, target_group_id)

    def _review_exempt_sources(self, target_group_id: str) -> list[str]:
        """Collect the source groups that exempt a target group from review.

        Args:
            target_group_id: QQ group being joined.

        Returns:
            Source group ids whose membership skips the remaining checks.
        """
        return review_exempt_sources(self.group_sets, target_group_id)

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
    #  Blacklist message blocking
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def block_blacklisted_message(self, event: AstrMessageEvent) -> None:
        """Recall messages sent by blacklisted members.

        The platform has already accepted the message by the time AstrBot sees
        it, so the only achievable interception is a recall. Events are not
        stopped, keeping other plugins unaffected.

        Args:
            event: The incoming group message event.
        """
        if not self.enabled:
            return
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return

        group_id = str(event.get_group_id() or "")
        user_id = str(event.get_sender_id() or "")
        if not group_id or not user_id or user_id == str(event.get_self_id() or ""):
            return
        if group_id not in self.group_index:
            return

        blacklist_policy = self._policy(group_id)["blacklist"]
        if not blacklist_policy["enabled"] or not blacklist_policy["block_message"]:
            return

        scope = self._scope(group_id)
        if not self.blacklist.is_blacklisted(scope, user_id):
            return

        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        if not message_id:
            return

        result = await self.client.delete_msg(event, message_id)
        self.audit.log(
            "blacklist_block",
            "system",
            target=user_id,
            group_id=group_id,
            scope=scope,
            result="ok" if result.ok else "failed",
            error=result.error,
            message_id=message_id,
        )
        if not result.ok:
            logger.warning(
                f"[QQAdmin] 撤回黑名单成员 {user_id} 在群 {group_id} 的消息失败："
                f"{result.error}",
            )

    # ------------------------------------------------------------------ #
    #  Join request review
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_raw_event(self, event: AstrMessageEvent) -> None:
        """Handle OneBot request events (group join applications).

        Request events carry an empty message chain, so they cannot be matched
        by a command filter and are inspected through ``raw_message`` instead.

        This handler returns nothing on purpose: yielding would send a message
        into an empty session, and calling ``stop_event`` would block other
        plugins from seeing legitimate traffic. ``should_call_llm(False)`` is
        what keeps the empty request from triggering a pointless LLM call.

        Args:
            event: The incoming adapter event.
        """
        if not self.enabled:
            return
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return

        raw = getattr(event.message_obj, "raw_message", None)
        request = parse_request_event(raw)
        if request is None:
            return

        event.should_call_llm(False)
        try:
            await self._process_join_request(event, request)
        except Exception as exc:  # noqa: BLE001 - a bad request must not crash
            logger.exception(f"[QQAdmin] 处理进群申请失败：{exc}")

    async def _process_join_request(
        self,
        event: AstrMessageEvent,
        request: Any,
    ) -> None:
        """Evaluate a join request and act on the decision.

        Args:
            event: Event carrying the request.
            request: Parsed join request.
        """
        from .core.review import JoinRequest

        assert isinstance(request, JoinRequest)

        group_set = find_group_set(self.group_sets, request.group_id)
        if group_set is None:
            logger.info(
                f"[QQAdmin] 群 {request.group_id} 未加入任何群组，跳过自动审核",
            )
            return

        policy = self._policy(request.group_id)
        review = self._review_policy(request.group_id)
        if review is None or not review["enabled"]:
            return

        scope = self._scope(request.group_id)
        is_blacklisted = (
            policy["blacklist"]["enabled"]
            and review["blacklist_reject"]
            and self.blacklist.is_blacklisted(
                scope,
                request.user_id,
            )
        )
        exemption_sources = self._review_exempt_sources(request.group_id)
        is_exempt_member = (
            await self._member_of_any_group(event, request, exemption_sources)
            if exemption_sources
            else False
        )
        min_level = review["min_account_level"]
        # An exempt member is approved before the level gate, so the extra
        # protocol call would be wasted.
        level = (
            None
            if is_exempt_member
            else await self._account_level(event, request, min_level)
        )

        decision = evaluate(
            request,
            is_blacklisted=is_blacklisted,
            is_exempt_member=is_exempt_member,
            account_level=level,
            min_account_level=min_level,
            keywords=review_keywords(review, request.group_id),
            question_type=review["question_type"],
            manual_confirm=review["manual_confirm"],
            unattended_action=review["unattended_action"],
        )
        logger.info(
            f"[QQAdmin] 进群申请 群{request.group_id} 用户{request.user_id} "
            f"→ {decision.action}（{decision.reason}：{decision.detail}）",
        )

        if decision.action == "hold":
            self.pending_reviews.add(request)
        else:
            await self._apply_decision(event, request, decision.action, decision.reason)

        await self._notify_review(event, group_set, request, decision)

    async def _member_of_any_group(
        self,
        event: AstrMessageEvent,
        request: Any,
        group_ids: list[str],
    ) -> bool:
        """Check whether the applicant already belongs to one of the groups.

        Used by the cross-group exemption list: membership of any listed source
        group lets the applicant skip the remaining review conditions. The
        target group itself is ignored.

        Args:
            event: Event providing the OneBot connection.
            request: The join request under evaluation.
            group_ids: Source groups whose membership grants the exemption.

        Returns:
            ``True`` when the protocol confirms membership of one of them.
        """
        for group_id in group_ids:
            if group_id == request.group_id:
                continue
            result = await self.client.call_action(
                event,
                "_get_group_member_info",
                group_id=group_id,
                user_id=request.user_id,
                no_cache=True,
            )
            if not result.ok or not isinstance(result.data, dict):
                continue
            if str(result.data.get("user_id") or "") == request.user_id:
                logger.info(
                    f"[QQAdmin] 申请人 {request.user_id} 已是群 {group_id} 的成员，"
                    f"申请加入群 {request.group_id} 时按跨群免审核放行",
                )
                return True
        return False

    async def _account_level(
        self,
        event: AstrMessageEvent,
        request: Any,
        min_level: int,
    ) -> int | None:
        """Look up an applicant's QQ level.

        Args:
            event: Event providing the OneBot connection.
            request: The join request.
            min_level: Configured minimum for the group; ``0`` skips the lookup
                entirely.

        Returns:
            The level, or ``None`` when the protocol implementation does not
            expose it (the ``level`` field comes from the private
            ``_get_group_member_info`` action).
        """
        if min_level <= 0:
            return None

        result = await self.client.call_action(
            event,
            "_get_group_member_info",
            group_id=request.group_id,
            user_id=request.user_id,
            no_cache=True,
        )
        if not result.ok or not isinstance(result.data, dict):
            if not self._level_unsupported_warned:
                self._level_unsupported_warned = True
                logger.warning(
                    "[QQAdmin] 协议端未提供 QQ 等级信息，已跳过等级校验",
                )
            return None

        raw_level = result.data.get("level")
        try:
            return int(raw_level)
        except (TypeError, ValueError):
            return None

    async def _apply_decision(
        self,
        event: AstrMessageEvent,
        request: Any,
        action: str,
        reason: str,
    ) -> bool:
        """Approve or reject a join request through the protocol.

        Args:
            event: Event providing the OneBot connection.
            request: The join request.
            action: ``"approve"`` or ``"reject"``.
            reason: Machine readable cause, recorded in the audit log.

        Returns:
            ``True`` when the protocol accepted the decision.
        """
        if action not in ("approve", "reject"):
            return False

        approve = action == "approve"
        review = self._review_policy(request.group_id)
        reject_reason = (
            review["reject_reason"] if review else DEFAULT_REVIEW_REJECT_REASON
        )
        result = await self.client.set_group_add_request(
            event,
            request.flag,
            approve=approve,
            reason="" if approve else reject_reason,
        )
        self.audit.log(
            f"review_{action}",
            "system",
            target=request.user_id,
            group_id=request.group_id,
            scope=self._scope(request.group_id),
            result="ok" if result.ok else "failed",
            error=result.error,
            reason=reason,
            comment=request.comment[:50],
        )
        if result.ok:
            self.pending_reviews.remove(request.flag)
        else:
            logger.warning(
                f"[QQAdmin] 处理进群申请失败（{request.user_id}@{request.group_id}）："
                f"{result.error}",
            )
        return result.ok

    def _pending_for_session(self, event: AstrMessageEvent) -> list[Any]:
        """List the pending join requests a session is responsible for.

        A manage session owns the groups of the sets that declared it, and a
        managed group owns its own set, so a decision made in either place only
        ever touches requests of that scope.

        Args:
            event: Event of the reply.

        Returns:
            Pending requests in scope, oldest first.
        """
        bound = list(self._manage_sets(event))
        group_set = find_group_set(
            self.group_sets,
            str(event.get_group_id() or ""),
        )
        if group_set is not None:
            bound.append(group_set)
        if not bound:
            return []
        scope = {group_id for item in bound for group_id in item["group_ids"]}
        return [
            request
            for request in self.pending_reviews.load_all()
            if request.group_id in scope
        ]

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def handle_review_reply(self, event: AstrMessageEvent) -> None:
        """Resolve a join request from a reply to its notification.

        The notification for a request that needs a human decision is answered
        by quoting it with an agreement or a refusal, so no separate command is
        needed. The words are the confirmation words of the plugin
        configuration, and the responder has to pass the ``review`` capability
        check.

        Args:
            event: The incoming message event.
        """
        if not self.enabled:
            return
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return

        verdict = self.confirm.verdict(event.message_str)
        if verdict is None:
            return

        message_id = extract_reply_message_id(event) or ""
        candidates = self._pending_for_session(event)
        if message_id:
            # A quote selects exactly one notification; quoting anything else is
            # not a decision on one of our requests.
            request = next(
                (item for item in candidates if item.notice_id == message_id),
                None,
            )
        elif candidates:
            # Without a quote the reply must be unambiguous.
            request = candidates[0] if len(candidates) == 1 else None
            if request is None:
                event.stop_event()
                event.should_call_llm(False)
                await event.send(
                    event.plain_result(
                        self._text("review_reply_ambiguous", count=len(candidates)),
                    ),
                )
                return
        else:
            request = None

        if request is None:
            return

        allowed, deny_reason = await self._check_capability_for_event(
            event,
            "review",
        )
        if not allowed:
            event.stop_event()
            event.should_call_llm(False)
            await event.send(
                event.plain_result(
                    self._text("no_permission", reason=deny_reason),
                ),
            )
            return

        action = "approve" if verdict else "reject"
        ok = await self._apply_decision(event, request, action, "manual")
        event.stop_event()
        event.should_call_llm(False)
        if not ok:
            async for item in self._reply(
                event,
                self._text("api_failed", error="协议端拒绝了该操作"),
                receipt=True,
                category=NOTIFY_CATEGORY_REVIEW_ACTION,
                group_id=request.group_id,
            ):
                await event.send(item)
            return

        group_set = find_group_set(self.group_sets, request.group_id)
        async for item in self._reply(
            event,
            self._text(
                "review_decided",
                user_id=request.user_id,
                group_id=request.group_id,
                action="同意" if action == "approve" else "拒绝",
            ),
            receipt=True,
            category=NOTIFY_CATEGORY_REVIEW_ACTION,
            group_id=request.group_id,
        ):
            await event.send(item)
        if group_set is not None:
            await self._notify_review(
                event,
                group_set,
                request,
                _SimpleDecision(
                    action,
                    "manual",
                    f"管理员 {event.get_sender_id()} 处理",
                ),
            )

    async def _send_review_notice(
        self,
        event: AstrMessageEvent,
        target: str,
        text: str,
    ) -> str:
        """Send a review notification and report its platform message id.

        The id is what a later reply has to quote, so it has to come from the
        protocol response instead of the platform-agnostic
        ``context.send_message``. When the target cannot be addressed through
        OneBot the message is still delivered, just without an id.

        Args:
            event: Event providing the OneBot connection.
            target: ``unified_msg_origin`` of the destination.
            text: Message body.

        Returns:
            The message id, or an empty string when it could not be determined.
        """
        parts = str(target or "").split(":")
        if len(parts) == 3 and parts[0] == SUPPORTED_PLATFORM:
            action = {
                "GroupMessage": ("send_group_msg", "group_id"),
                "PrivateMessage": ("send_private_msg", "user_id"),
            }.get(parts[1])
            if action:
                result = await self.client.call_action(
                    event,
                    action[0],
                    **{action[1]: parts[2]},
                    message=[{"type": "text", "data": {"text": text}}],
                )
                if result.ok and isinstance(result.data, dict):
                    return str(result.data.get("message_id") or "")
                logger.warning(
                    f"[QQAdmin] 发送审核通知到 {target} 失败：{result.error}",
                )

        await self.notifier.send_to_session(target, text)
        return ""

    async def _notify_review(
        self,
        event: AstrMessageEvent,
        group_set: dict[str, Any],
        request: Any,
        decision: Any,
    ) -> None:
        """Send the review notification to the configured destinations.

        Args:
            event: Event providing the platform instance id.
            group_set: Owning group set.
            request: The join request.
            decision: The decision that was applied (or held).
        """
        # A manual decision reuses the same actions, so it needs its own label
        # instead of reporting the bot as having decided on its own.
        if decision.reason == "manual" and decision.action in ("approve", "reject"):
            action_label = (
                "已人工放行" if decision.action == "approve" else "已人工拒绝"
            )
        else:
            action_label = {
                "approve": "已自动放行",
                "reject": "已自动拒绝",
                "hold": "待人工确认",
            }.get(decision.action, decision.action)

        # The template is keyed on the request content; the decision is
        # appended so an operator sees what the bot already did.
        text = render_template(
            self.templates,
            "review_new_request",
            group_id=request.group_id,
            user_id=request.user_id,
            user_name=request.user_id,
            comment=request.comment or "（无）",
            sub_type=request.sub_type,
            decision=action_label,
            reason=decision.detail or decision.reason,
        )
        if decision.action == "hold":
            text = f"{text}\n{render_template(self.templates, 'review_pending_hint')}"

        targets = self.notifier.targets(
            NOTIFY_CATEGORY_REVIEW_NOTICE,
            origin=str(event.unified_msg_origin or ""),
            manage_session=str(group_set.get("manage_session") or ""),
        )
        for target in targets:
            notice_id = await self._send_review_notice(event, target, text)
            if notice_id:
                self.pending_reviews.set_notice_id(request.flag, notice_id)

    # ------------------------------------------------------------------ #
    #  Helpers shared with the LLM tools
    # ------------------------------------------------------------------ #

    def _tool_group_id(self, event: AstrMessageEvent) -> str:
        """Resolve the group a tool call operates on.

        The group id is never a tool parameter; it always comes from the event,
        so a model cannot be steered into touching another group.

        Args:
            event: The tool call event.

        Returns:
            The group id, or an empty string when unavailable or when the group
            is not part of any configured group set.
        """
        if not self.enabled:
            return ""
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return ""
        group_id = str(event.get_group_id() or "")
        return group_id if group_id in self.group_index else ""

    def _extract_targets(self, event: AstrMessageEvent) -> list[str]:
        """Extract ``@`` targets from an event.

        Exposed for the tools module so target resolution stays in one place.

        Args:
            event: Event to inspect.

        Returns:
            De-duplicated QQ numbers.
        """
        return extract_at_targets(event)

    async def _tool_permission(
        self,
        event: AstrMessageEvent,
        capability: str,
    ) -> tuple[bool, str]:
        """Check whether the tool caller may use a capability.

        AstrBot does not gate these tools for us, so every tool re-checks the
        caller. This is the load-bearing check that stops a user from talking
        the model into muting someone.

        Args:
            event: The tool call event.
            capability: Capability key to check.

        Returns:
            ``(allowed, reason)``.
        """
        if not self.enabled:
            return False, "插件当前已禁用"
        if event.get_platform_name() != SUPPORTED_PLATFORM:
            return False, "本插件仅支持 aiocqhttp(OneBot v11) 平台"
        context = await self._permission_context(event)
        return check_capability(context, self.permission_settings, capability)

    def _tool_audit(
        self,
        action: str,
        event: AstrMessageEvent,
        *,
        target: str = "",
        group_id: str = "",
        result: str = "ok",
        error: str = "",
        **detail: Any,
    ) -> None:
        """Record a tool action in the audit trail.

        Args:
            action: Machine readable action name.
            event: The tool call event, used for the operator id.
            target: QQ number the action applied to.
            group_id: Group the action happened in.
            result: ``"ok"`` or ``"failed"``.
            error: Failure description.
            **detail: Extra context recorded verbatim.
        """
        self.audit.log(
            action,
            str(event.get_sender_id() or ""),
            target=target,
            group_id=group_id,
            result=result,
            error=error,
            via="llm_tool",
            **detail,
        )

    def _group_set_name(self, group_id: str) -> str:
        """Return the name of the group set owning a group.

        Args:
            group_id: QQ group id.

        Returns:
            The group set name, or an empty string when the group is
            unconfigured.
        """
        group_set = find_group_set(self.group_sets, group_id)
        return str(group_set["name"]) if group_set else ""

    # ------------------------------------------------------------------ #
    #  Commands: export
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.command("群导出", alias={"gexport"})
    @filter.custom_filter(mentioned_command("群导出", "gexport"))
    async def export_command(
        self,
        event: AstrMessageEvent,
        args: GreedyStr,
    ) -> AsyncGenerator[Any, None]:
        """导出群成员名单或黑名单表格并上传到群文件。

        Args:
            event: 消息事件。
            args: 「成员」或「黑名单」。允许省略。
        """
        if await self._should_skip(event):
            return
        group_id = str(event.get_group_id())
        operator = await self._permission_context(event)

        allowed, deny_reason = check_capability(
            operator,
            self.permission_settings,
            "export",
        )
        if not allowed:
            async for item in self._reply(
                event,
                self._text("no_permission", reason=deny_reason),
            ):
                yield item
            return

        kind = str(args or "").strip() or "成员"
        if kind not in ("成员", "黑名单", "members", "blacklist"):
            async for item in self._reply(
                event,
                "用法：群导出 成员 / 群导出 黑名单",
            ):
                yield item
            return

        if kind in ("黑名单", "blacklist"):
            scope = self._scope(group_id)
            entries = self.blacklist.list_entries(scope)
            if not entries:
                async for item in self._reply(
                    event,
                    self._text("export_empty", kind="黑名单"),
                ):
                    yield item
                return
            group_set = find_group_set(self.group_sets, group_id)
            result = self.exporter.export_blacklist(
                scope,
                entries,
                group_ids=list(group_set["group_ids"]) if group_set else [group_id],
            )
        else:
            info = await self.client.get_group_info(event, group_id)
            members = await self.client.get_group_member_list(
                event,
                group_id,
                use_cache=False,
            )
            if not members:
                async for item in self._reply(
                    event,
                    self._text("member_list_failed"),
                ):
                    yield item
                return
            result = self.exporter.export_members(
                group_id,
                members,
                group_name=str((info or {}).get("group_name") or ""),
            )

        if not result:
            async for item in self._reply(
                event,
                self._text("export_failed", error=result.error),
                receipt=True,
                category=NOTIFY_CATEGORY_EXPORT,
                group_id=group_id,
            ):
                yield item
            return

        self._log_action(
            "export",
            operator,
            group_id=group_id,
            kind=kind,
            rows=result.row_count,
        )

        uploaded = False
        if self._export_upload:
            upload = await self.client.upload_group_file(
                event,
                group_id,
                to_file_uri(result.path),
                result.filename,
            )
            uploaded = upload.ok
            if not upload.ok:
                logger.warning(
                    f"[QQAdmin] 上传导出文件失败：{upload.error}，改为返回本地路径",
                )

        if uploaded:
            text = self._text(
                "export_success",
                kind=kind,
                count=result.row_count,
            )
        else:
            text = self._text(
                "export_local_only",
                kind=kind,
                count=result.row_count,
                path=result.path,
            )
        async for item in self._reply(
            event,
            text,
            receipt=True,
            category=NOTIFY_CATEGORY_EXPORT,
            group_id=group_id,
        ):
            yield item

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
        """Release resources when the plugin is unloaded or disabled.

        Whole-group mute deadlines are deliberately **kept on disk**: the timers
        cannot survive the process, but the schedule does, and
        :meth:`restore_whole_ban_timers` re-arms them on the next load. Only the
        in-memory tasks are dropped here.

        Pending join requests are left alone for the same reason -- the protocol
        token usually outlives a reload, so they can still be resolved by
        quoting their notification afterwards.
        """
        for task in list(self._whole_ban_tasks.values()):
            if not task.done():
                task.cancel()
        self._whole_ban_tasks.clear()

        pending = self.whole_ban.load_all()
        if pending:
            logger.info(
                f"[QQAdmin] 插件卸载时仍有 {len(pending)} 个全群禁言定时，"
                "已保留在 whole_ban.json，下次加载时会自动恢复",
            )

        self.client.invalidate_member_cache()
        if len(self.pending_reviews):
            logger.info(
                f"[QQAdmin] 插件卸载时仍有 {len(self.pending_reviews)} 条待审核申请，"
                "已保留在 pending_reviews.json",
            )
