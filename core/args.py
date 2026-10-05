"""Command argument parsing.

The target syntax is ``<target> [duration] [reason...]`` where the duration and
reason are both optional. Because the duration is optional, it cannot be bound
to a fixed position: ``禁言 @某人 刷屏`` legitimately omits it. This module
therefore scans the remaining tokens, treats the first duration-like token as
the duration, and everything else as the reason.

Kept as pure functions so the rules can be verified without a live platform.
"""

from __future__ import annotations

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
    remaining = [token for token in tokens if token]
    targets = list(at_targets)

    # Drop hand-typed QQ numbers that duplicate an already known target, so a
    # following number is not mistaken for the duration.
    for target in targets:
        if target in remaining:
            remaining.remove(target)

    # With no @ present, the first QQ-like token is the target.
    if not targets:
        for index, token in enumerate(remaining):
            if _is_qq(token):
                targets.append(token)
                remaining.pop(index)
                break

    duration: int | None = None
    unknown_duration = ""
    for index, token in enumerate(remaining):
        if not _is_duration_like(token):
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
