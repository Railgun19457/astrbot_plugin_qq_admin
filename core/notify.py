"""Notification routing.

A group set may declare a *manage session* (a group or a private chat) that
receives review notifications and optionally the operation receipts. When it is
not configured, notifications fall back to the group the action happened in.

Configuration writes the session in a friendly ``group:123456`` form, but
AstrBot routes by ``unified_msg_origin``, which looks like
``aiocqhttp_1:GroupMessage:123456``. The platform *instance id* is part of that
string and cannot be hard-coded, so it is always taken from the live event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.star import Context


#: Keys accepted in the ``manage_session`` configuration field.
_SESSION_KINDS: dict[str, str] = {
    "group": "GroupMessage",
    "private": "FriendMessage",
}


@dataclass
class SessionTarget:
    """A resolved destination for a proactive message.

    Attributes:
        value: ``unified_msg_origin`` string, ready for ``context.send_message``.
        kind: ``"group"`` or ``"private"``.
        session_id: Raw group id or QQ number.
    """

    value: str
    kind: str
    session_id: str

    def __str__(self) -> str:
        return self.value


def parse_manage_session(raw: str) -> tuple[str, str] | None:
    """Parse the ``manage_session`` configuration value.

    Args:
        raw: Raw configuration text, e.g. ``group:123456``.

    Returns:
        ``(kind, session_id)`` with ``kind`` either ``"group"`` or
        ``"private"``, or ``None`` when the value is empty or malformed.
    """
    text = str(raw or "").strip()
    if not text:
        return None

    if ":" not in text:
        # Bare numbers are ambiguous; default to a group rather than guessing.
        if text.isdigit():
            return "group", text
        logger.warning(f"[QQAdmin] 无法解析管理会话配置：{text}")
        return None

    kind_raw, _, session_id = text.partition(":")
    kind = kind_raw.strip().lower()
    session_id = session_id.strip()
    if kind not in _SESSION_KINDS or not session_id:
        logger.warning(f"[QQAdmin] 无法解析管理会话配置：{text}")
        return None
    return kind, session_id


def build_umo(
    event: AstrMessageEvent,
    kind: str,
    session_id: str,
) -> str | None:
    """Build a ``unified_msg_origin`` for an event's platform.

    Args:
        event: Any event from the target platform, used to read the platform
            instance id.
        kind: ``"group"`` or ``"private"``.
        session_id: Group id or QQ number.

    Returns:
        The umo string, or ``None`` when the platform id is unavailable.
    """
    message_type = _SESSION_KINDS.get(kind)
    if not message_type:
        return None

    platform_id = ""
    getter = getattr(event, "get_platform_id", None)
    if callable(getter):
        platform_id = str(getter() or "")
    if not platform_id:
        platform_id = str(getattr(event, "get_platform_name", lambda: "")() or "")
    if not platform_id:
        logger.warning("[QQAdmin] 无法确定平台实例 ID，通知未发送")
        return None
    return f"{platform_id}:{message_type}:{session_id}"


class Notifier:
    """Sends proactive messages to the right session.

    Attributes:
        _context: AstrBot context used for outbound messages.
        _receipt_to_manage_session: Whether operation receipts are mirrored to
            the manage session in addition to the originating group.
        _receipt_enabled: Whether operation receipts are sent at all.
    """

    def __init__(
        self,
        context: Context,
        *,
        receipt_enabled: bool = True,
        receipt_to_manage_session: bool = False,
    ) -> None:
        """Store the routing policy.

        Args:
            context: AstrBot context.
            receipt_enabled: Whether command handlers send receipts at all.
            receipt_to_manage_session: Mirror receipts into the manage session.
        """
        self._context = context
        self._receipt_enabled = bool(receipt_enabled)
        self._receipt_to_manage_session = bool(receipt_to_manage_session)

    @property
    def receipt_enabled(self) -> bool:
        """Whether operation receipts should be produced."""
        return self._receipt_enabled

    @property
    def mirror_receipts(self) -> bool:
        """Whether receipts are also sent to the manage session."""
        return self._receipt_to_manage_session

    def resolve_target(
        self,
        event: AstrMessageEvent,
        group_set: dict[str, Any] | None,
    ) -> SessionTarget:
        """Resolve where a group-set notification should go.

        Args:
            event: Event providing the platform instance id.
            group_set: Owning group set, or ``None`` when the group is
                unconfigured.

        Returns:
            The manage session when configured and resolvable, otherwise the
            originating group as a fallback.
        """
        if group_set:
            parsed = parse_manage_session(group_set.get("manage_session", ""))
            if parsed:
                kind, session_id = parsed
                umo = build_umo(event, kind, session_id)
                if umo:
                    return SessionTarget(umo, kind, session_id)

        return SessionTarget(
            value=str(getattr(event, "unified_msg_origin", "") or ""),
            kind="group",
            session_id=str(event.get_group_id() or ""),
        )

    async def send_to_session(
        self,
        target: SessionTarget,
        text: str,
    ) -> bool:
        """Send a message to a resolved target.

        Args:
            target: Destination resolved by :meth:`resolve_target` or built by
                the caller.
            text: Message body.

        Returns:
            ``True`` when the message was handed to a platform adapter.
        """
        if not text or not target.value:
            return False

        from astrbot.core.message.message_event_result import MessageChain

        try:
            sent = await self._context.send_message(
                target.value,
                MessageChain().message(text),
            )
        except Exception as exc:  # noqa: BLE001 - notification must not crash
            logger.warning(f"[QQAdmin] 发送通知到 {target.value} 失败：{exc}")
            return False
        if not sent:
            logger.warning(f"[QQAdmin] 未找到会话 {target.value}，通知未发送")
        return bool(sent)
