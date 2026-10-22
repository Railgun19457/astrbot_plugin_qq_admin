"""User-facing message templates.

Templates come from the plugin configuration and may reference placeholders
such as ``{target_name}``. Rendering never raises: a malformed template falls
back to a safe default so that a typo in the config cannot break a command.
"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger

#: Defaults used both for the config schema and as a last-resort fallback.
DEFAULT_TEMPLATES: dict[str, str] = {
    "ban_success": "已禁言 {target_name}({target_id}) {duration}",
    "unban_success": "已解除 {target_name}({target_id}) 的禁言",
    "whole_ban_on": "已开启全群禁言",
    "whole_ban_off": "已解除全群禁言",
    "whole_ban_timed": "已开启全群禁言，将在 {duration} 后自动解除",
    "kick_success": "已将 {target_name}({target_id}) 移出本群",
    "recall_success": "已撤回指定消息",
    "notice_success": "群公告已更新",
    "no_permission": "你没有权限执行该操作：{reason}",
    "no_target": "请 @ 一个目标，或回复对方的消息后再执行本指令",
    "bad_duration": "时长格式不正确，示例：10m、2h、1d、600",
    "blacklist_added": "已将 {target_name}({target_id}) 加入黑名单\n原因：{reason}\n期限：{duration}",
    "blacklist_removed": "已将 {target_name}({target_id}) 移出黑名单",
    "blacklist_not_found": "{target_name}({target_id}) 不在黑名单中",
    "blacklist_disabled": "黑名单功能已在插件配置中关闭",
    "blacklist_empty": "黑名单（{scope}）当前为空",
    "blacklist_header": "黑名单（{scope}）共 {count} 人，覆盖 {groups} 个群",
    "api_failed": "操作失败：{error}",
    "api_unsupported": "当前协议端不支持该操作，建议改用 NapCat 或 Lagrange",
    "confirm_prompt": "即将执行：{summary}\n回复「同意」继续，回复「取消」放弃（{timeout} 秒内有效）",
    "confirm_timeout": "操作已取消（等待确认超时）",
    "confirm_cancelled": "操作已取消",
    "confirm_not_operator": "只有发起该操作的人可以确认",
    "review_new_request": (
        "【进群申请】群 {group_id}\n"
        "申请人：{user_id}\n"
        "答案：{comment}\n"
        "处理：{decision}（{reason}）"
    ),
    "review_decided": "已{action} {user_id} 加入群 {group_id} 的申请",
    "review_pending_hint": "引用本条消息回复「同意」或「拒绝」即可处理",
    "review_reply_ambiguous": (
        "当前有 {count} 条待处理申请，请引用对应的通知消息后再回复同意/拒绝"
    ),
    "manage_unsupported": (
        "该指令需要在目标群内执行。管理会话仅支持：黑名单、拉黑、解除拉黑、群管状态"
    ),
    "manage_ambiguous": (
        "该管理会话同时对应多个群组（{names}），请写明群号或群组名再试"
    ),
    "group_not_found": "未找到群组：{selector}",
    "notification_settings": (
        "在管理会话发送「黑名单 通知设置 [群号或群组名]」可查看该群组的通知路由"
    ),
    "member_list_failed": "无法获取群成员列表，请稍后重试",
    "export_success": "已导出{kind}表格（{count} 条）并上传到群文件",
    "export_local_only": "已生成{kind}表格（{count} 条），但上传失败。\n本地路径：{path}",
    "export_empty": "{kind}为空，没有可导出的数据",
    "export_failed": "导出失败：{error}",
}


def render_template(
    templates: dict[str, Any],
    key: str,
    **values: Any,
) -> str:
    """Render a configured template.

    A template referencing a placeholder that was not supplied would make
    ``str.format`` raise, so any formatting failure falls back to the built-in
    default for ``key`` and logs a warning. This keeps a typo in the config from
    breaking a command.

    Args:
        templates: Template mapping from the plugin configuration.
        key: Template key, e.g. ``ban_success``.
        **values: Placeholder values.

    Returns:
        The rendered message. Never empty when a default exists for ``key``.
    """
    fallback = DEFAULT_TEMPLATES.get(key, "")
    raw = templates.get(key)
    template = str(raw) if isinstance(raw, str) and raw.strip() else fallback
    if not template:
        return ""

    safe_values = {name: _stringify(value) for name, value in values.items()}
    try:
        return template.format(**safe_values)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(
            f"[QQAdmin] 消息模板 `{key}` 渲染失败（{exc}），已回退到默认文案",
        )
        try:
            return fallback.format(**safe_values)
        except (KeyError, IndexError, ValueError):
            return fallback


def _stringify(value: Any) -> str:
    """Convert a placeholder value into display text.

    Args:
        value: Arbitrary placeholder value.

    Returns:
        A plain string; ``None`` becomes an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)
