"""Notification routing and receipt delivery.

Each routable message category (operation receipts, join review notices) can be
delivered to the session the action happened in, to the *manage session* of the
owning group set, to both, or to neither.

A manage session is declared per group set and holds a ``unified_msg_origin``
(UMO) such as ``aiocqhttp:GroupMessage:123456``, which can be copied from the
``/sid`` command in the target session. The UMO already embeds the message
type, so group chats and private chats are configured the same way and the
platform instance id does not have to be hard-coded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.star import Context


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
    """Resolves where a notification category is delivered.

    Attributes:
        _context: AstrBot context used for outbound messages.
        _to_current: Categories delivered to the session of the event.
        _to_manage: Categories delivered to the manage session.
    """

    def __init__(
        self,
        context: Context,
        *,
        to_current: list[str],
        to_manage: list[str],
    ) -> None:
        """Store the routing policy.

        Args:
            context: AstrBot context.
            to_current: Categories routed to the current session.
            to_manage: Categories routed to the manage session.
        """
        self._context = context
        self._to_current = set(to_current)
        self._to_manage = set(to_manage)

    def targets(
        self,
        category: str,
        *,
        origin: str,
        manage_session: str = "",
    ) -> list[str]:
        """Resolve the destinations of one notification.

        A category routed to the manage session without a usable manage session
        configured falls back to the current session, so enabling the route can
        never make a notification disappear silently.

        Args:
            category: One of ``config.NOTIFY_CATEGORIES``.
            origin: ``unified_msg_origin`` of the session the action happened
                in.
            manage_session: Raw ``manage_session`` value of the owning group
                set, empty when the group belongs to no set.

        Returns:
            De-duplicated UMOs in delivery order. Empty when the category is
            enabled on neither target.
        """
        origin = str(origin or "")
        destinations: list[str] = []

        if category in self._to_current and origin:
            destinations.append(origin)

        if category in self._to_manage:
            fallback = parse_manage_session(manage_session) or origin
            if fallback and fallback not in destinations:
                destinations.append(fallback)

        return destinations

    async def send_to_session(self, target: str, text: str) -> bool:
        """Send a proactive message to a resolved destination.

        Args:
            target: UMO returned by :meth:`targets`.
            text: Message body.

        Returns:
            ``True`` when the message was handed to a platform adapter.
        """
        if not text or not target:
            return False

        from astrbot.core.message.message_event_result import MessageChain

        try:
            sent = await self._context.send_message(
                target,
                MessageChain().message(text),
            )
        except Exception as exc:  # noqa: BLE001 - notification must not crash
            logger.warning(f"[QQAdmin] 发送通知到 {target} 失败：{exc}")
            return False
        if not sent:
            logger.warning(f"[QQAdmin] 未找到会话 {target}，通知未发送")
        return bool(sent)
