"""Join request review.

The aiocqhttp adapter surfaces ``post_type == "request"`` events with an empty
message chain, so this logic cannot be attached to a command filter. Instead the
plugin listens for every message event and inspects
``event.message_obj.raw_message`` directly.

Decision order for a group join request:

1. blacklist hit -> reject
2. already a member of a sibling group -> approve
3. account level below the configured minimum -> reject
4. keyword match -> approve
5. ``manual_confirm`` on -> notify the manage session and wait for an operator
6. otherwise -> fall back to ``unattended_action``

The pending queue is persisted so that a restart does not silently drop
requests that a human still has to decide on, and so that the opaque ``flag``
token survives until the decision is made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import UNATTENDED_ACTIONS
from .store import JsonStore

#: Request sub types that mean "this user wants to join the group".
GROUP_REQUEST_SUBTYPES: frozenset[str] = frozenset({"add", "invite"})

#: Pending entries older than this are dropped, because the protocol token
#: expires and answering it later would fail anyway.
PENDING_TTL: int = 7 * 24 * 3600


@dataclass
class JoinRequest:
    """A normalized group join request.

    Attributes:
        flag: Opaque token needed to approve or reject.
        sub_type: ``add`` (user applied) or ``invite`` (someone invited them).
        user_id: Applicant's QQ number.
        group_id: Group being joined.
        comment: Answer to the group question, or the invitation message.
        created_at: Unix timestamp when the request was observed.
    """

    flag: str
    sub_type: str
    user_id: str
    group_id: str
    comment: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence.

        Returns:
            A JSON-compatible mapping.
        """
        return {
            "flag": self.flag,
            "sub_type": self.sub_type,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "comment": self.comment,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> JoinRequest | None:
        """Rebuild a request from stored data.

        Args:
            raw: Stored payload.

        Returns:
            The parsed request, or ``None`` when the payload is unusable.
        """
        if not isinstance(raw, dict):
            return None
        flag = str(raw.get("flag") or "").strip()
        if not flag:
            return None
        return cls(
            flag=flag,
            sub_type=str(raw.get("sub_type") or "add"),
            user_id=str(raw.get("user_id") or ""),
            group_id=str(raw.get("group_id") or ""),
            comment=str(raw.get("comment") or ""),
            created_at=float(raw.get("created_at") or 0.0),
        )


@dataclass
class ReviewDecision:
    """Outcome of evaluating a join request.

    Attributes:
        action: ``"approve"``, ``"reject"`` or ``"hold"``.
        reason: Machine readable cause, e.g. ``blacklist`` or ``keyword``.
        detail: Human readable explanation for logs and notifications.
    """

    action: str
    reason: str
    detail: str = ""


def parse_request_event(raw: Any) -> JoinRequest | None:
    """Extract a group join request from a raw adapter event.

    Args:
        raw: ``event.message_obj.raw_message``.

    Returns:
        The parsed request, or ``None`` when the event is not a group join
        request (friend requests and non-request events return ``None``).
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("post_type") != "request":
        return None
    sub_type = str(raw.get("sub_type") or "").strip().lower()
    if sub_type not in GROUP_REQUEST_SUBTYPES:
        return None

    group_id = str(raw.get("group_id") or "").strip()
    flag = str(raw.get("flag") or "").strip()
    user_id = str(raw.get("user_id") or "").strip()
    if not group_id or not flag or not user_id:
        return None

    return JoinRequest(
        flag=flag,
        sub_type=sub_type,
        user_id=user_id,
        group_id=group_id,
        comment=str(raw.get("comment") or "").strip(),
        created_at=time.time(),
    )


def evaluate(
    request: JoinRequest,
    *,
    is_blacklisted: bool,
    is_sibling_member: bool,
    account_level: int | None,
    min_account_level: int,
    keywords: list[str],
    question_type: str,
    manual_confirm: bool,
    unattended_action: str,
) -> ReviewDecision:
    """Decide what to do with a join request.

    Args:
        request: The request under evaluation.
        is_blacklisted: Whether the applicant is on the group's blacklist.
        is_sibling_member: Whether the applicant already belongs to another
            group of the same group set.
        account_level: Applicant's QQ level, or ``None`` when the protocol
            implementation does not expose it.
        min_account_level: Lowest acceptable level; ``0`` disables the check.
        keywords: Answers that approve the request when matched.
        question_type: ``any``, ``add`` or ``invite``; limits the keyword check
            to one kind of request.
        manual_confirm: Whether unmatched requests are escalated to a human.
        unattended_action: Fallback when ``manual_confirm`` is off.

    Returns:
        The decision to apply.
    """
    if is_blacklisted:
        return ReviewDecision("reject", "blacklist", "申请人在黑名单中")

    if is_sibling_member:
        return ReviewDecision("approve", "sibling", "申请人已是同组其他群的成员")

    if (
        min_account_level > 0
        and account_level is not None
        and account_level < min_account_level
    ):
        return ReviewDecision(
            "reject",
            "level",
            f"QQ 等级 {account_level} 低于要求的 {min_account_level}",
        )

    applies = question_type in ("any", request.sub_type)
    if applies and keywords:
        answer = request.comment.lower()
        if any(word.lower() in answer for word in keywords):
            return ReviewDecision(
                "approve", "keyword", f"命中关键词：{'、'.join(keywords)}"
            )

    if manual_confirm:
        return ReviewDecision("hold", "manual", "未命中规则，转人工确认")

    if unattended_action not in UNATTENDED_ACTIONS:
        unattended_action = "hold"
    return ReviewDecision(
        unattended_action,
        "unattended",
        f"无人值守策略：{unattended_action}",
    )


class PendingReviewStore:
    """Persisted queue of requests awaiting a human decision.

    Attributes:
        _store: Backing JSON store keyed by ``flag``.
    """

    def __init__(self, data_dir: Path) -> None:
        """Prepare the queue.

        Args:
            data_dir: Plugin data directory.
        """
        self._store = JsonStore(data_dir / "pending_reviews.json")

    def load_all(self) -> list[JoinRequest]:
        """Load every non-expired pending request.

        Stale entries are dropped, since the protocol token behind them has
        almost certainly expired too.

        Returns:
            Pending requests, oldest first.
        """
        cutoff = time.time() - PENDING_TTL
        requests: list[JoinRequest] = []
        stale: list[str] = []
        for flag, raw in self._store.data.items():
            request = JoinRequest.from_dict(raw)
            if request is None or request.created_at < cutoff:
                stale.append(str(flag))
                continue
            requests.append(request)

        if stale:
            for flag in stale:
                self._store.data.pop(flag, None)
            self._store.save()

        requests.sort(key=lambda item: item.created_at)
        return requests

    def add(self, request: JoinRequest) -> None:
        """Record a pending request.

        Args:
            request: The request to remember.
        """
        self._store.data[request.flag] = request.to_dict()
        self._store.save()

    def get(self, flag: str) -> JoinRequest | None:
        """Look up a pending request by its token.

        Args:
            flag: Protocol token of the request.

        Returns:
            The request, or ``None`` when it is unknown or expired.
        """
        raw = self._store.data.get(str(flag))
        if raw is None:
            return None
        request = JoinRequest.from_dict(raw)
        if request is None:
            return None
        if request.created_at < time.time() - PENDING_TTL:
            self.remove(request.flag)
            return None
        return request

    def remove(self, flag: str) -> bool:
        """Drop a request from the queue.

        Args:
            flag: Protocol token of the request.

        Returns:
            ``True`` when an entry was removed.
        """
        if str(flag) not in self._store.data:
            return False
        self._store.data.pop(str(flag))
        self._store.save()
        return True

    def __len__(self) -> int:
        return len(self._store.data)
