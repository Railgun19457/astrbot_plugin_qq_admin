"""Notification routing and receipt delivery.

Two kinds of messages are routed differently, because they are triggered
differently:

* **Command receipts** answer an operator, so they always stay in the session
  the command was issued in. Each of them can additionally be mirrored into the
  manage session of the owning group set, which lets the manage session double
  as a control room that sees every action without the group being spammed.
* **The join review notice** is triggered by an applicant rather than by a
  command, so it has no session to answer: its destination is configured
  directly (application group, manage session, both, or nowhere).

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
    """Resolves where a notification is delivered.

    Attributes:
        _context: AstrBot context used for outbound messages.
        _mirror: Receipt categories also sent to the manage session.
        _review_target: Destination of the join review notice.
    """

    def __init__(
        self,
        context: Context,
        *,
        mirror: list[str],
        review_target: str,
    ) -> None:
        """Store the routing policy.

        Args:
            context: AstrBot context.
            mirror: Receipt categories mirrored into the manage session.
            review_target: Destination of the join review notice.
        """
        self._context = context
        self._mirror = set(mirror)
        self._review_target = review_target

    @property
    def review_target(self) -> str:
        """Destination of the join review notice."""
        return self._review_target

    def mirrors(self, category: str) -> bool:
        """Report whether a receipt category is mirrored.

        Args:
            category: One of ``config.NOTIFY_RECEIPT_CATEGORIES``.

        Returns:
            ``True`` when the receipt is also sent to the manage session.
        """
        return category in self._mirror

    def manage_target(self, manage_session: str) -> str:
        """Resolve the manage session destination.

        Args:
            manage_session: Raw ``manage_session`` value of the owning group
                set, empty when the group belongs to no set.

        Returns:
            The manage session UMO, or an empty string when unusable.
        """
        return parse_manage_session(manage_session) or ""

    def review_destinations(self, *, origin: str, manage_session: str) -> list[str]:
        """Resolve the destinations of the join review notice.

        Args:
            origin: ``unified_msg_origin`` of the application's group.
            manage_session: Raw ``manage_session`` value of the owning group
                set.

        Returns:
            De-duplicated UMOs in delivery order. Empty when the notice is
            disabled. A configured-but-missing manage session degrades to the
            application group so the notice is never lost silently.
        """
        origin = str(origin or "")
        manage = self.manage_target(manage_session)

        if self._review_target == "none":
            return []
        if self._review_target == "origin":
            targets = [origin]
        elif self._review_target == "both":
            targets = [origin, manage]
        else:
            targets = [manage or origin]

        destinations: list[str] = []
        for target in targets:
            if target and target not in destinations:
                destinations.append(target)
        return destinations

    @staticmethod
    def describe_review_target(target: str, *, manage_ready: bool) -> str:
        """Render a review destination for display.

        Args:
            target: One of ``config.NOTIFY_REVIEW_TARGETS``.
            manage_ready: Whether the group set has a usable manage session.

        Returns:
            A human readable label such as ``管理会话``.
        """
        if target == "none":
            return "不发送"
        if target == "origin":
            return "申请所在群"
        manage = "管理会话" if manage_ready else "管理会话未配置（等同申请所在群）"
        if target == "both":
            return f"申请所在群、{manage}"
        return "管理会话" if manage_ready else "管理会话未配置（回退申请所在群）"

    async def send_to_session(self, target: str, text: str) -> bool:
        """Send a proactive message to a resolved destination.

        Args:
            target: UMO returned by :meth:`review_destinations`.
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
