"""Three-tier permission model.

``event.is_admin()`` only reflects AstrBot's global administrator list, which
has nothing to do with QQ group roles. Group ownership has to be queried from
the protocol layer, so this module combines both sources plus an optional
plugin-level allow-list.

See ``plan.md`` §3 for the decision table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import (
    GROUP_PRIVILEGED_ROLES,
    PERMISSION_LEVELS,
)

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from .onebot import OneBotClient


@dataclass
class PermissionContext:
    """Resolved identity of the user issuing a command.

    Attributes:
        user_id: QQ number of the operator.
        group_id: Group the command was sent in, empty for private chats.
        is_astrbot_admin: Whether the user is in AstrBot's global admin list.
        is_allowlisted: Whether the user is in the plugin's custom allow-list.
        group_role: OneBot group role (``owner`` / ``admin`` / ``member``),
            ``None`` when it could not be determined.
        is_manage_session: Whether the event came from a configured management
            session.
    """

    user_id: str
    group_id: str
    is_astrbot_admin: bool
    is_allowlisted: bool
    group_role: str | None
    is_manage_session: bool = False

    @property
    def is_group_privileged(self) -> bool:
        """Whether the user is the group owner or a group administrator."""
        return self.group_role in GROUP_PRIVILEGED_ROLES

    @property
    def satisfied_levels(self) -> set[str]:
        """Permission levels this identity satisfies.

        Returns:
            A subset of ``{"any", "group_admin", "astrbot_admin"}``.
        """
        levels = {"any"}
        if self.is_astrbot_admin or self.is_allowlisted or self.is_manage_session:
            levels.add("astrbot_admin")
            levels.add("group_admin")
        if self.is_group_privileged:
            levels.add("group_admin")
        return levels

    def describe(self) -> str:
        """Render a short identity summary for error messages.

        Returns:
            A string such as ``小王(123456)，群管理员``.
        """
        if self.is_astrbot_admin:
            role = "AstrBot 管理员"
        elif self.is_allowlisted:
            role = "白名单用户"
        elif self.is_manage_session:
            role = "管理会话用户"
        elif self.group_role == "owner":
            role = "群主"
        elif self.group_role == "admin":
            role = "群管理员"
        elif self.group_role == "member":
            role = "群成员"
        else:
            role = "身份未知"
        return f"{self.user_id}（{role}）"


@dataclass
class PermissionSettings:
    """Plugin configuration relevant to permission decisions.

    Attributes:
        custom_allowlist: QQ numbers granted the ``astrbot_admin`` level.
        allow_astrbot_admin: Whether AstrBot admins may manage groups.
        allow_group_admin: Whether group owners/admins may manage groups.
        allow_member_command: Whether plain members may use ``any`` commands.
        capability_levels: ``capability -> required level``.
    """

    custom_allowlist: frozenset[str]
    allow_astrbot_admin: bool
    allow_group_admin: bool
    allow_member_command: bool
    capability_levels: dict[str, str]


def resolve_capability_level(
    settings: PermissionSettings,
    capability: str,
) -> str:
    """Look up the required level for a capability.

    Args:
        settings: Parsed permission settings.
        capability: One of the keys in ``config.CAPABILITY_KEYS``.

    Returns:
        The configured level, defaulting to ``"group_admin"``.
    """
    level = str(settings.capability_levels.get(capability) or "").strip().lower()
    return level if level in PERMISSION_LEVELS else "group_admin"


async def build_permission_context(
    event: AstrMessageEvent,
    client: OneBotClient,
    settings: PermissionSettings,
    *,
    group_id: str | None = None,
    is_manage_session: bool = False,
) -> PermissionContext:
    """Resolve the identity of the command issuer.

    The group role is only queried when it can actually matter: AstrBot admins
    and allow-listed users already satisfy every level, so no extra API call is
    made for them.

    Args:
        event: Message event of the command.
        client: OneBot facade used to look up the group role.
        settings: Parsed permission settings.
        group_id: Override for the group id; defaults to the event's group.
        is_manage_session: Whether the event came from a configured management
            session. Users in a management group receive full plugin access.

    Returns:
        A :class:`PermissionContext` describing the operator.
    """
    user_id = str(event.get_sender_id() or "")
    if not user_id:
        sender = getattr(getattr(event, "message_obj", None), "sender", None)
        user_id = str(getattr(sender, "user_id", "") or "").strip()
    target_group = group_id if group_id is not None else str(event.get_group_id() or "")

    is_astrbot_admin = bool(settings.allow_astrbot_admin) and event.is_admin()
    is_allowlisted = user_id in settings.custom_allowlist

    group_role: str | None = None
    needs_role = (
        settings.allow_group_admin
        and bool(target_group)
        and not is_astrbot_admin
        and not is_allowlisted
    )
    if needs_role:
        group_role = await client.get_member_role(event, target_group, user_id)

    return PermissionContext(
        user_id=user_id,
        group_id=target_group,
        is_astrbot_admin=is_astrbot_admin,
        is_allowlisted=is_allowlisted,
        is_manage_session=is_manage_session,
        group_role=group_role,
    )


def check_capability(
    context: PermissionContext,
    settings: PermissionSettings,
    capability: str,
) -> tuple[bool, str]:
    """Decide whether the operator may use a capability.

    Args:
        context: Resolved operator identity.
        settings: Parsed permission settings.
        capability: Capability key to check.

    Returns:
        ``(allowed, reason)``. ``reason`` is empty when allowed and otherwise
        contains a user-facing explanation.
    """
    level = resolve_capability_level(settings, capability)

    if level == "any":
        if settings.allow_member_command or context.satisfied_levels - {"any"}:
            return True, ""
        return False, "本群已关闭普通成员使用查询类指令"

    if level in context.satisfied_levels:
        return True, ""

    if level == "group_admin":
        if not settings.allow_group_admin:
            return False, "本插件已关闭「群主/群管理员」权限"
        return False, f"需要群主或群管理员权限，当前身份：{context.describe()}"

    if not settings.allow_astrbot_admin:
        return False, "本插件已关闭「AstrBot 管理员」权限"
    return False, f"需要 AstrBot 管理员权限，当前身份：{context.describe()}"
