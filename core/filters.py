"""Command matching helpers.

AstrBot renders every ``@`` segment into ``message_str`` as ``@nickname(qq)``.
When a member replies to the bot and mentions another member in the same
message, that mention text ends up *before* the command. The framework's
command filter requires ``message_str`` to start with the command name, so the
command is silently dropped. The filter below removes such leading mentions
before the command filter runs.
"""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Any

from astrbot.api.event.filter import CustomFilter

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

#: ``@nickname(qq)`` as rendered by the aiocqhttp adapter. Nicknames may contain
#: spaces, so the QQ number in parentheses is the only reliable delimiter.
_MENTION_RE = re.compile(r"^\s*@[^\n]*?\(\d{5,12}\)")


class MentionTolerantCommandFilter(CustomFilter):
    """Strip leading ``@`` mentions so a command is still matched.

    The filter only rewrites ``message_str``. Whether the message really is a
    command is still decided by the framework's command filter, which is
    evaluated right after this filter.
    """

    def __init__(self, names: set[str], raise_error: bool = True) -> None:
        """Bind the filter to a single command.

        Args:
            names: Command name followed by its aliases.
            raise_error: Passed through to :class:`CustomFilter`.
        """
        super().__init__(raise_error)
        self._names = {name.strip() for name in names if name.strip()}

    def filter(self, event: AstrMessageEvent, cfg: Any) -> bool:
        """Normalize the message text before command matching.

        Args:
            event: Incoming message event.
            cfg: AstrBot configuration, used for the wake prefixes.

        Returns:
            ``True`` unconditionally: the filter must never decide whether the
            handler runs, only which text the command filter gets to see.
        """
        if not event.is_at_or_wake_command:
            return True
        if event.get_platform_name() != "aiocqhttp":
            return True

        original = str(event.message_str or "")
        text = original
        while True:
            stripped = _MENTION_RE.sub("", text, count=1).lstrip()
            if stripped == text:
                break
            text = stripped
        if text == original:
            return True

        # The wake prefix could not be stripped by the framework because the
        # mentions came first, so drop it here as well.
        for prefix in cfg.get("wake_prefix") or []:
            if prefix and text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break

        if any(text == name or text.startswith(f"{name} ") for name in self._names):
            event.message_str = text
        return True


def mentioned_command(*names: str) -> Any:
    """Build the decorator argument for :class:`MentionTolerantCommandFilter`.

    Args:
        *names: Command name followed by its aliases.

    Returns:
        A factory accepted by ``@filter.custom_filter(...)``.
    """
    return partial(MentionTolerantCommandFilter, set(names))
