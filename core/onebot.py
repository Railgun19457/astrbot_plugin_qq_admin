"""Thin wrapper around the OneBot v11 actions used by this plugin.

Every call goes through :func:`call_action`, which normalizes the ``self_id``
routing parameter and converts protocol failures into a result object instead
of letting exceptions escape into the message pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from .config import MAX_BAN_DURATION

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


#: How long a cached group member list stays fresh.
MEMBER_CACHE_TTL: int = 60


@dataclass
class ActionResult:
    """Outcome of a OneBot action.

    Attributes:
        ok: Whether the action succeeded.
        data: Raw payload returned by the protocol implementation.
        error: Human readable failure description, empty on success.
        unsupported: Whether the failure looks like "this action does not
            exist on the connected protocol implementation".
    """

    ok: bool
    data: Any = None
    error: str = ""
    unsupported: bool = False

    def __bool__(self) -> bool:
        return self.ok


#: Substrings that indicate the connected OneBot implementation does not
#: implement the requested action at all.
_UNSUPPORTED_HINTS = (
    "not supported",
    "unsupported",
    "unknown action",
    "no such action",
    "未实现",
    "不支持",
    "未知的",
    "无效的 action",
    "invalid action",
)


class OneBotClient:
    """OneBot v11 action facade bound to a single plugin instance.

    The client keeps a short-lived cache of group member lists because
    ``get_group_member_list`` is comparatively expensive and is used by both
    the query commands and the table export.

    Attributes:
        _member_cache: ``group_id -> (timestamp, member list)``.
        _warned_actions: Actions already reported as unsupported, so the
            warning is emitted only once per action.
    """

    def __init__(self) -> None:
        self._member_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._warned_actions: set[str] = set()

    # ------------------------------------------------------------------ #
    #  Low level
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_bot(source: Any) -> Any | None:
        """Obtain the OneBot client from an event or a platform instance.

        Most call sites pass a message event. Timers restored after a restart
        have no event, so a platform instance is accepted as well.

        Args:
            source: An ``AstrMessageEvent`` or an aiocqhttp platform instance.

        Returns:
            The ``CQHttp`` client, or ``None`` when neither shape matched.
        """
        bot = getattr(source, "bot", None)
        if bot is not None:
            return bot
        getter = getattr(source, "get_client", None)
        if callable(getter):
            try:
                return getter()
            except Exception:  # noqa: BLE001 - treated as "no client available"
                return None
        return None

    @staticmethod
    def _routing(source: Any) -> dict[str, Any]:
        """Build the routing kwargs required by multi-connection setups.

        Args:
            source: An ``AstrMessageEvent`` or a platform instance.

        Returns:
            Either an empty dict or ``{"self_id": ...}``.
        """
        if source is None:
            return {}
        self_id = getattr(getattr(source, "message_obj", None), "self_id", None)
        return {"self_id": self_id} if self_id else {}

    async def call_action(
        self,
        event: Any,
        action: str,
        **params: Any,
    ) -> ActionResult:
        """Invoke a raw OneBot action.

        Args:
            event: Message event, or a platform instance for restored timers.
            action: OneBot action name.
            **params: Action parameters.

        Returns:
            An :class:`ActionResult`. Failures are reported through
            ``ok=False`` rather than raised.
        """
        bot = self._resolve_bot(event)
        if bot is None:
            return ActionResult(
                ok=False,
                error="当前平台适配器不支持 OneBot 接口（仅 aiocqhttp 可用）",
                unsupported=True,
            )

        payload = dict(params)
        payload.update(self._routing(event))
        try:
            ret = await bot.call_action(action, **payload)
        except Exception as exc:  # noqa: BLE001 - normalized into ActionResult
            return self._failure(action, exc)
        return ActionResult(ok=True, data=ret)

    def _failure(self, action: str, exc: Exception) -> ActionResult:
        """Convert an exception thrown by ``call_action`` into a result.

        Args:
            action: OneBot action name, used for logging.
            exc: The exception raised by the protocol layer.

        Returns:
            An :class:`ActionResult` describing the failure.
        """
        message = str(exc).strip() or exc.__class__.__name__
        unsupported = any(hint in message.lower() for hint in _UNSUPPORTED_HINTS)

        if unsupported and action not in self._warned_actions:
            self._warned_actions.add(action)
            logger.warning(
                f"[QQAdmin] 协议端不支持 API `{action}`：{message}。"
                "该功能将不可用，建议改用 NapCat 或 Lagrange。",
            )
        elif not unsupported:
            logger.debug(f"[QQAdmin] API `{action}` 调用失败: {message}")

        return ActionResult(ok=False, error=message, unsupported=unsupported)

    # ------------------------------------------------------------------ #
    #  Group management
    # ------------------------------------------------------------------ #

    async def set_group_ban(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
        duration: int,
    ) -> ActionResult:
        """Mute (or unmute) a member.

        Args:
            event: Message event.
            group_id: Target group.
            user_id: Target member.
            duration: Mute duration in seconds. ``0`` lifts the mute.

        Returns:
            The action result. ``duration`` is clamped to 30 days because the
            protocol rejects longer values.
        """
        return await self.call_action(
            event,
            "set_group_ban",
            group_id=self._as_int(group_id),
            user_id=self._as_int(user_id),
            duration=max(0, min(int(duration), MAX_BAN_DURATION)),
        )

    async def set_group_whole_ban(
        self,
        event: Any,
        group_id: str,
        enabled: bool = True,
    ) -> ActionResult:
        """Toggle the "mute everyone" switch for a group.

        Args:
            event: Message event, or a platform instance for restored timers.
            group_id: Target group.
            enabled: ``True`` mutes everyone, ``False`` lifts it.

        Returns:
            The action result.

        Note:
            OneBot exposes this as a plain on/off switch without a duration, so
            temporary whole-group mutes have to be scheduled by the caller.
        """
        return await self.call_action(
            event,
            "set_group_whole_ban",
            group_id=self._as_int(group_id),
            enable=bool(enabled),
        )

    async def set_group_kick(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
        *,
        reject_add_request: bool = False,
    ) -> ActionResult:
        """Kick a member out of a group.

        Args:
            event: Message event.
            group_id: Target group.
            user_id: Target member.
            reject_add_request: Also reject any pending join request from them.

        Returns:
            The action result.
        """
        return await self.call_action(
            event,
            "set_group_kick",
            group_id=self._as_int(group_id),
            user_id=self._as_int(user_id),
            reject_add_request=bool(reject_add_request),
        )

    async def delete_msg(
        self,
        event: AstrMessageEvent,
        message_id: str,
    ) -> ActionResult:
        """Recall a previously sent message.

        Args:
            event: Message event.
            message_id: Id of the message to recall.

        Returns:
            The action result.
        """
        return await self.call_action(
            event,
            "delete_msg",
            message_id=self._as_int(message_id),
        )

    async def get_group_member_info(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
        *,
        no_cache: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a single member's details.

        Args:
            event: Message event.
            group_id: Target group.
            user_id: Target member.
            no_cache: Forwarded to the protocol layer; when ``True`` the
                response reflects the latest server state.

        Returns:
            The member payload, or ``None`` when the lookup failed.
        """
        result = await self.call_action(
            event,
            "get_group_member_info",
            group_id=self._as_int(group_id),
            user_id=self._as_int(user_id),
            no_cache=bool(no_cache),
        )
        return result.data if result.ok and isinstance(result.data, dict) else None

    async def get_member_role(
        self,
        event: AstrMessageEvent,
        group_id: str,
        user_id: str,
    ) -> str | None:
        """Return a member's group role.

        Args:
            event: Message event.
            group_id: Target group.
            user_id: Target member.

        Returns:
            ``"owner"``, ``"admin"`` or ``"member"``; ``None`` when the role
            could not be determined.
        """
        info = await self.get_group_member_info(event, group_id, user_id)
        if not info:
            return None
        role = str(info.get("role") or "").strip()
        return role or None

    async def get_group_member_list(
        self,
        event: AstrMessageEvent,
        group_id: str,
        *,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch the full member list of a group.

        Args:
            event: Message event.
            group_id: Target group.
            use_cache: Reuse a result cached within :data:`MEMBER_CACHE_TTL`.

        Returns:
            The member list, or an empty list on failure.
        """
        now = time.time()
        if use_cache:
            cached = self._member_cache.get(group_id)
            if cached and now - cached[0] < MEMBER_CACHE_TTL:
                return cached[1]

        result = await self.call_action(
            event,
            "get_group_member_list",
            group_id=self._as_int(group_id),
        )
        if not result.ok or not isinstance(result.data, list):
            return []

        members = [m for m in result.data if isinstance(m, dict)]
        self._member_cache[group_id] = (now, members)
        return members

    async def get_group_info(
        self,
        event: AstrMessageEvent,
        group_id: str,
    ) -> dict[str, Any] | None:
        """Fetch basic information about a group.

        Args:
            event: Message event.
            group_id: Target group.

        Returns:
            The group payload, or ``None`` when the lookup failed.
        """
        result = await self.call_action(
            event,
            "get_group_info",
            group_id=self._as_int(group_id),
        )
        return result.data if result.ok and isinstance(result.data, dict) else None

    async def send_group_notice(
        self,
        event: AstrMessageEvent,
        group_id: str,
        content: str,
    ) -> ActionResult:
        """Publish a group announcement.

        Args:
            event: Message event.
            group_id: Target group.
            content: Notice body.

        Returns:
            The action result.
        """
        return await self.call_action(
            event,
            "_send_group_notice",
            group_id=self._as_int(group_id),
            content=str(content or ""),
        )

    async def set_group_add_request(
        self,
        event: AstrMessageEvent,
        flag: str,
        *,
        approve: bool,
        reason: str = "",
    ) -> ActionResult:
        """Approve or reject a pending join request.

        Args:
            event: Message event.
            flag: Opaque token taken from the original request event.
            approve: ``True`` to accept, ``False`` to reject.
            reason: Rejection reason shown to the applicant.

        Returns:
            The action result.
        """
        params: dict[str, Any] = {
            "flag": str(flag),
            "approve": bool(approve),
        }
        if not approve and reason:
            params["reason"] = str(reason)
        return await self.call_action(event, "set_group_add_request", **params)

    # ------------------------------------------------------------------ #
    #  Files
    # ------------------------------------------------------------------ #

    async def upload_group_file(
        self,
        event: AstrMessageEvent,
        group_id: str,
        file_path: str,
        name: str,
    ) -> ActionResult:
        """Upload a local file to the group's file list.

        Args:
            event: Message event.
            group_id: Target group.
            file_path: Absolute path of the local file.
            name: File name shown in the group.

        Returns:
            The action result.

        Note:
            Protocol servers expect either an absolute path or a ``file://``
            URI; the caller is responsible for producing one.
        """
        return await self.call_action(
            event,
            "upload_group_file",
            group_id=self._as_int(group_id),
            file=file_path,
            name=name,
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def invalidate_member_cache(self, group_id: str | None = None) -> None:
        """Drop cached member lists.

        Args:
            group_id: Only clear this group when given; otherwise clear all.
        """
        if group_id is None:
            self._member_cache.clear()
        else:
            self._member_cache.pop(group_id, None)

    @staticmethod
    def _as_int(value: Any) -> int | Any:
        """Convert numeric strings to ints, which most OneBot servers expect.

        Args:
            value: Raw value, usually a string.

        Returns:
            An ``int`` when the value looks numeric, otherwise the original
            value unchanged.
        """
        text = str(value or "").strip()
        if text.isdigit():
            return int(text)
        return value
