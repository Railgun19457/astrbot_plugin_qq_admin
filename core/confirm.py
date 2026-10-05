"""High-risk operation confirmation built on the framework's session waiter.

The framework's ``session_waiter`` already isolates sessions by
``unified_msg_origin`` and its built-in handler stops the event, so this module
only has to: check the responder's permission, decide the outcome, and cancel
any scheduled follow-up (the whole-group mute timer) when the user declines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.util import SessionController, session_waiter

from .messages import render_template

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from .permission import PermissionContext


@dataclass
class PendingAction:
    """A side effect that should only happen after the user confirms.

    Attributes:
        summary: Short description shown in the confirmation prompt.
        payload: Arbitrary data the caller needs to execute the action.
        capability: Capability key used for the permission check, so the
            confirmer must be authorised too.
    """

    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    capability: str = ""


class ConfirmManager:
    """Requests and resolves confirmations for high-risk actions.

    Attributes:
        _templates: Template mapping from the plugin configuration.
        _timeout: Seconds to wait for an answer.
        _confirm_words: Accepted affirmative replies.
        _cancel_words: Accepted negative replies.
        _enabled: Whether confirmation is required for any action.
        _require_for: Action keys that need confirmation.
        _permission_checker: Callable returning ``(allowed, reason)`` for the
            responder, supplied by the plugin to avoid a circular import.
    """

    def __init__(
        self,
        templates: dict[str, Any],
        *,
        enabled: bool,
        timeout: int,
        confirm_words: list[str],
        cancel_words: list[str],
        require_for: list[str],
        permission_checker: Any,
    ) -> None:
        """Store the confirmation policy.

        Args:
            templates: Template mapping for prompts.
            enabled: Master switch for the confirmation feature.
            timeout: Seconds to wait for a reply.
            confirm_words: Replies treated as "yes".
            cancel_words: Replies treated as "no".
            require_for: Action keys requiring confirmation.
            permission_checker: Coroutine ``(event, capability) ->
                (allowed, reason)`` used to validate the responder.
        """
        self._templates = templates
        self._enabled = bool(enabled)
        self._timeout = max(5, int(timeout))
        self._confirm_words = {word.strip() for word in confirm_words if word.strip()}
        self._cancel_words = {word.strip() for word in cancel_words if word.strip()}
        self._require_for = {item.strip() for item in require_for if item.strip()}
        self._permission_checker = permission_checker

    def is_required(self, action_key: str) -> bool:
        """Return whether an action needs confirmation.

        Args:
            action_key: Action identifier such as ``kick``.

        Returns:
            ``True`` when confirmation is enabled and the action is listed.
        """
        return self._enabled and action_key in self._require_for

    def build_prompt(self, summary: str) -> str:
        """Render the confirmation prompt.

        Args:
            summary: Description of the pending action.

        Returns:
            The prompt text sent to the group.
        """
        return render_template(
            self._templates,
            "confirm_prompt",
            summary=summary,
            timeout=self._timeout,
        )

    async def request(
        self,
        event: AstrMessageEvent,
        action: PendingAction,
        operator: PermissionContext,
    ) -> str:
        """Ask the group to confirm an action and wait for the answer.

        Args:
            event: The command message that triggered the action.
            action: Description of what will happen on confirmation.
            operator: Identity of the person who issued the command. Only this
                user may answer, so a bystander cannot confirm on their behalf.

        Returns:
            ``"confirmed"`` when the operator agreed. Otherwise ``"cancelled"``
            for an explicit rejection, ``"timeout"`` when nobody answered in
            time, or ``"error"`` when the waiter itself failed, so the caller
            can tell the group why nothing happened. Only ``"confirmed"`` means
            the action may proceed.
        """
        operator_id = operator.user_id
        confirm_words = self._confirm_words
        cancel_words = self._cancel_words
        checker = self._permission_checker
        templates = self._templates

        # The reply arrives as a *different* event object, so the decision has
        # to travel through this holder instead of an event extra.
        decision: list[bool] = []

        @session_waiter(timeout=self._timeout)
        async def _waiter(
            controller: SessionController,
            reply_event: AstrMessageEvent,
        ) -> None:
            text = reply_event.message_str.strip()
            if text not in confirm_words and text not in cancel_words:
                return

            # Re-check the responder: the framework's session handler does not
            # verify who is answering.
            responder_id = str(reply_event.get_sender_id() or "")
            if responder_id != operator_id:
                await reply_event.send(
                    reply_event.plain_result(
                        render_template(templates, "confirm_not_operator"),
                    ),
                )
                return

            allowed, reason = await checker(reply_event, action.capability)
            if not allowed:
                await reply_event.send(
                    reply_event.plain_result(
                        render_template(templates, "no_permission", reason=reason),
                    ),
                )
                return

            decision.append(text in confirm_words)
            controller.stop()

        try:
            await _waiter(event)
        except TimeoutError:
            logger.info("[QQAdmin] 二次确认超时，操作已取消")
            return "timeout"
        except Exception as exc:  # noqa: BLE001 - confirmation must not crash
            logger.warning(f"[QQAdmin] 二次确认过程出错：{exc}")
            return "error"

        return "confirmed" if decision and decision[0] else "cancelled"
