"""Target extraction helpers for command arguments.

QQ numbers must be read from the message chain rather than scraped with a
regex from ``message_str``: nicknames may contain spaces or parentheses, which
makes textual parsing unreliable (see the design notes in ``plan.md`` §7.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


def extract_at_targets(
    event: AstrMessageEvent, *, include_self: bool = False
) -> list[str]:
    """Collect QQ numbers mentioned with an ``@`` segment.

    Args:
        event: The current message event.
        include_self: When ``False`` (the default) the bot's own id is filtered
            out so that a stray "@bot" never becomes a ban target.

    Returns:
        De-duplicated QQ ids in the order they appear in the message.
    """
    from astrbot.api.message_components import At, AtAll, Reply

    self_id = str(event.get_self_id() or "")
    targets: list[str] = []
    for component in event.get_messages():
        if isinstance(component, AtAll):
            # "@all" is not a real user id; callers handle it separately.
            continue
        if isinstance(component, At):
            qq = str(component.qq or "").strip()
            if not qq or qq.lower() == "all":
                continue
            if not include_self and qq == self_id:
                continue
            if qq not in targets:
                targets.append(qq)
        elif isinstance(component, Reply):
            # A reply implicitly targets its author.
            sender_id = str(component.sender_id or "").strip()
            if not sender_id or sender_id == "0":
                continue
            if not include_self and sender_id == self_id:
                continue
            if sender_id not in targets:
                targets.append(sender_id)
    return targets


def extract_reply_message_id(event: AstrMessageEvent) -> str | None:
    """Return the message id quoted by the event, if any.

    Args:
        event: The current message event.

    Returns:
        The quoted message id as a string, or ``None`` when the message does
        not reply to anything.
    """
    from astrbot.api.message_components import Reply

    for component in event.get_messages():
        if isinstance(component, Reply):
            message_id = str(component.id or "").strip()
            if message_id:
                return message_id
    return None


def shorten(text: str, limit: int = 20) -> str:
    """Trim ``text`` for inclusion in log lines and receipts.

    Args:
        text: Arbitrary user supplied text.
        limit: Maximum number of characters to keep.

    Returns:
        The original text when short enough, otherwise a truncated version
        suffixed with an ellipsis.
    """
    clean = str(text or "").replace("\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "…"
