"""Notification routing.

A group set may declare a *manage session* that receives review notifications
and optionally the operation receipts. The value is a ``unified_msg_origin``
(UMO) such as ``aiocqhttp:GroupMessage:123456``, which can be copied from the
``/sid`` command in the target session. The UMO already embeds the message
type, so group chats and private chats are configured the same way and the
platform instance id does not have to be hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.star import Context


@dataclass
class SessionTarget:
    """A resolved destination for a proactive message.

    Attributes:
        value: ``unified_msg_origin`` string, ready for ``context.send_message``.
    """

    value: str

    def __str__(self) -> str:
        return self.value


def parse_manage_session(raw: str) -> str | None:
    """Validate the ``manage_session`` configuration value.

    Args:
        raw: Raw configuration text, expected to be a UMO such as
            ``aiocqhttp:GroupMessage:123456``.

    Returns:
        The normalized UMO, or ``None`` when the value is empty or malformed.
    """
    text = str(raw or "").strip()
    if not text:
        return None

    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 3 or not all(parts):
        logger.warning(
            f"[QQAdmin] 管理会话配置不是合法的 UMO：{text}。"
            "请在目标会话发送 /sid 获取，例如 aiocqhttp:GroupMessage:123456",
        )
        return None
    return ":".join(parts)


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
            event: Event whose session is used as the fallback destination.
            group_set: Owning group set, or ``None`` when the group is
                unconfigured.

        Returns:
            The manage session when configured and resolvable, otherwise the
            originating group as a fallback.
        """
        if group_set:
            umo = parse_manage_session(group_set.get("manage_session", ""))
            if umo:
                return SessionTarget(umo)

        return SessionTarget(
            value=str(getattr(event, "unified_msg_origin", "") or ""),
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
