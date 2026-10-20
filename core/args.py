"""Command argument parsing.

The target syntax is ``<target> [duration] [reason...]``; only the target is
required and the tokens may appear in any order (``禁言 10m @某人 刷屏`` and
``禁言 @某人 刷屏 10m`` are equivalent). This module therefore classifies the
tokens instead of binding them to fixed positions:

1. a token with a unit suffix (``10m``, ``2h``, ``1d``) is always a duration;
2. the first QQ-like token is the target when no ``@`` target exists;
3. a remaining bare number is a duration when a target is already known, and
   otherwise becomes the target itself.

Kept as pure functions so the rules can be verified without a live platform.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import parse_duration


@dataclass
class TargetArgs:
    """Result of parsing a target command's arguments.

    Attributes:
        targets: QQ numbers the command applies to.
        duration: Parsed duration in seconds, ``None`` when not specified or
            not applicable.
        duration_specified: Whether the user actually wrote a duration, so
            callers can distinguish "not given" from "given but zero".
        reason: Free-form reason assembled from the leftover tokens.
        unknown_duration: A token that looked like it was meant to be a
            duration but could not be parsed, e.g. ``10x``.
    """

    targets: list[str] = field(default_factory=list)
    duration: int | None = None
    duration_specified: bool = False
    reason: str = ""
    unknown_duration: str = ""


def parse_target_args(
    tokens: list[str],
    at_targets: list[str],
) -> TargetArgs:
    """Parse the argument tokens of a target command.

    Args:
        tokens: Whitespace-split arguments following the command name.
        at_targets: QQ numbers already extracted from ``@`` segments and the
            quoted message author.

    Returns:
        A :class:`TargetArgs` describing the parsed intent. ``targets`` may be
        empty when the user supplied neither an ``@`` nor a QQ number.
    """
    # ``message_str`` renders every non-bot @ segment as ``@name(qq)``, so the
    # mention text arrives as ordinary argument tokens. Strip those segments
    # before parsing so a mention never leaks into the reason recorded in the
    # audit log. Matching on the QQ id also removes the fragments of nicknames
    # that contain spaces.
    text = " ".join(token for token in tokens if token)
    if at_targets:
        ids = "|".join(re.escape(target) for target in at_targets)
        text = re.sub(rf"@.*?\((?:{ids})\)", " ", text)
    remaining = text.split()
    targets = list(at_targets)

    # Drop hand-typed QQ numbers that duplicate an already known target, so a
    # following number is not mistaken for the duration.
    for target in targets:
        if target in remaining:
            remaining.remove(target)

    # A token with a unit suffix is unambiguously a duration, wherever it sits.
    duration: int | None = None
    unknown_duration = ""
    for index, token in enumerate(remaining):
        if not _is_unit_duration(token):
            continue
        parsed = parse_duration(token, default=None)
        if parsed is None:
            unknown_duration = token
        else:
            duration = parsed
        remaining.pop(index)
        break

    # With no @ present, the first QQ-like token is the target.
    if not targets:
        for index, token in enumerate(remaining):
            if _is_qq(token):
                targets.append(token)
                remaining.pop(index)
                break

    # A bare number is only a duration once a target is known; otherwise it was
    # meant as the target itself (``禁言 123456`` mutes that member).
    if duration is None and not unknown_duration:
        for index, token in enumerate(remaining):
            if not token.isdigit():
                continue
            parsed = parse_duration(token, default=None)
            if parsed is None:
                unknown_duration = token
            else:
                duration = parsed
            remaining.pop(index)
            break

    return TargetArgs(
        targets=targets,
        duration=duration,
        duration_specified=duration is not None,
        reason=" ".join(remaining).strip(),
        unknown_duration=unknown_duration,
    )


def _is_qq(token: str) -> bool:
    """Return whether a token is a plausible QQ number.

    Args:
        token: Raw argument token.

    Returns:
        ``True`` for 5-12 digit tokens.
    """
    return token.isdigit() and 5 <= len(token) <= 12


def _is_duration_like(token: str) -> bool:
    """Return whether a token is *meant* to be a duration.

    A token counts as a duration attempt when it is a bare number or a number
    followed by a single letter, which catches typos such as ``10x`` while
    keeping plain words like ``刷屏`` out of the duration slot.

    Args:
        token: Raw argument token.

    Returns:
        ``True`` when the token looks like a duration attempt.
    """
    lowered = token.lower()
    if lowered.isdigit():
        return True
    # e.g. "10m", "10x", "2H" -- a number plus one trailing letter.
    return len(lowered) >= 2 and lowered[:-1].isdigit() and lowered[-1].isalpha()


def _is_unit_duration(token: str) -> bool:
    """Return whether a token carries an explicit duration unit.

    Args:
        token: Raw argument token.

    Returns:
        ``True`` for tokens such as ``10m``, ``2h`` or ``30s``.
    """
    return _is_duration_like(token) and not token.isdigit()


def parse_duration_token(tokens: list[str]) -> int | None:
    """Pick the first parseable duration from a token list.

    Used by commands that take a duration but no target or reason, such as
    ``禁言 全体 1h``.

    Args:
        tokens: Argument tokens.

    Returns:
        Duration in seconds, or ``None`` when no token parses as a duration.
    """
    for token in tokens:
        parsed = parse_duration(token, default=None)
        if parsed is not None:
            return parsed
    return None
