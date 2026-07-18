from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flet as ft

from local_llm_chat.domain.models import ComputerPlan, DesktopSecurityContext
from local_llm_chat.domain.policies.computer_use import ComputerUsePolicy
from local_llm_chat.domain.states import (
    ComputerActionType,
    ComputerPolicyDecision,
    DesktopIntegrityLevel,
)


class ComputerUseReviewDialog(ft.AlertDialog):
    """Focused plan review; it never starts computer use by itself."""

    def __init__(
        self,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
        *,
        isolation_ready: bool,
        fake_only: bool = False,
        isolation_label: str = "未選定",
        remaining_seconds: int = 30,
        on_approve: Callable[[Any], Any] | None = None,
        on_cancel: Callable[[Any], Any] | None = None,
    ) -> None:
        policy_result = ComputerUsePolicy().evaluate_plan(plan, context)
        plan_allowed = (
            policy_result.decision is ComputerPolicyDecision.REQUIRE_APPROVAL
        )
        displayed_seconds = max(
            0,
            min(remaining_seconds, int(plan.limits.approval_timeout_seconds)),
        )
        can_approve_when_fresh = (
            plan_allowed
            and (isolation_ready or fake_only)
        )
        can_approve = can_approve_when_fresh and displayed_seconds > 0
        type_action = next(
            (
                action
                for action in plan.actions
                if action.action_type is ComputerActionType.TYPE_PLAIN_TEXT
            ),
            None,
        )
        input_text = type_action.text if type_action is not None else ""
        standard_context = (
            context.controller_integrity is DesktopIntegrityLevel.MEDIUM
            and context.target_integrity is DesktopIntegrityLevel.MEDIUM
            and not context.controller_elevated
            and not context.target_elevated
            and not context.ui_access
            and not context.secure_desktop
        )
        same_user = (
            bool(context.controller_user_sid)
            and context.controller_user_sid == context.target_user_sid
        )
        same_session = (
            context.controller_session_id >= 0
            and context.controller_session_id == context.target_session_id
        )
        status_lines = [
            (
                "✓ 標準権限・通常デスクトップ"
                if standard_context
                else "✕ 標準権限・通常デスクトップではありません"
            ),
            "✓ 同一ユーザー" if same_user else "✕ ユーザーが一致しません",
            "✓ 同一対話セッション" if same_session else "✕ セッションが一致しません",
            (
                f"✓ 隔離環境: {isolation_label}"
                if isolation_ready
                else "✕ 隔離環境が未選定 — このPCへは入力しません"
            ),
            "✓ 保存・削除・設定変更なし",
        ]
        if fake_only:
            status_lines = [
                "✓ Fake Brokerのみ — OS入力は0件",
                "✓ 固定3操作の契約検査だけを実行",
                "✓ 保存・削除・設定変更なし",
            ]
        if not plan_allowed:
            status_lines.append(
                f"✕ 安全検査: {policy_result.reason or 'plan_denied'}"
            )
        self._status_lines = tuple(status_lines)
        self._can_approve_when_fresh = can_approve_when_fresh
        self._max_remaining_seconds = int(plan.limits.approval_timeout_seconds)
        self.deadline_text = ft.Text(
            f"承認期限まで {displayed_seconds}秒",
            size=11,
            color="#F2A65A",
        )
        self.status_text = ft.Text(
            "\n".join(status_lines),
            size=11,
            color="#55C2A3" if can_approve else "#D87866",
        )

        self.approve_button = ft.Button(
            "Fakeで承認・実行" if fake_only else "この計画を承認",
            bgcolor="#F2A65A" if can_approve else "#34332F",
            color="#17120D" if can_approve else "#77736B",
            disabled=not can_approve,
            on_click=on_approve,
        )
        self.cancel_button = ft.Button("やめる", on_click=on_cancel)
        super().__init__(
            modal=True,
            title=ft.Column(
                [
                    ft.Text(
                        "実行前の安全確認",
                        size=11,
                        weight=ft.FontWeight.W_600,
                        color="#F2A65A",
                    ),
                    ft.Text(
                        "この計画だけを実行します",
                        size=23,
                        weight=ft.FontWeight.W_700,
                        color="#F3F0E8",
                    ),
                    self.deadline_text,
                ],
                spacing=3,
            ),
            content=ft.Column(
                [
                    ft.Text(
                        (
                            "Fake安全シミュレーション / メモ帳"
                            if fake_only
                            else f"{isolation_label} / メモ帳"
                        ),
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color="#F3F0E8",
                    ),
                    ft.Text(
                        (
                            "1. 固定メモ帳の起動要求を検査\n"
                            "2. 固定UIA対象のクリック要求を検査\n"
                            "3. 文字入力要求を検査（実入力なし）"
                            if fake_only
                            else "1. 固定メモ帳を起動\n"
                            "2. UI Automationで編集領域を一意確認\n"
                            "3. 入力後、保存せず停止"
                        ),
                        size=11,
                        color="#AAA69D",
                    ),
                    ft.Container(
                        content=ft.Text(
                            input_text or "入力なし",
                            size=13,
                            color="#F3F0E8",
                            selectable=True,
                        ),
                        bgcolor="#272822",
                        border_radius=10,
                        padding=12,
                    ),
                    self.status_text,
                    ft.Container(
                        content=ft.Text(
                            "計画が1文字でも変わると承認は失効します。"
                            "失敗時は自動再開しません。\n"
                            "停止: Ctrl + Alt + F12",
                            size=11,
                            color="#FFB0B0",
                        ),
                        bgcolor="#3A1717",
                        border_radius=10,
                        padding=12,
                    ),
                ],
                spacing=12,
                tight=True,
                width=680,
            ),
            actions=[self.cancel_button, self.approve_button],
        )
        self.set_remaining_seconds(displayed_seconds)

    def set_remaining_seconds(self, remaining_seconds: int) -> None:
        displayed_seconds = max(
            0, min(remaining_seconds, self._max_remaining_seconds)
        )
        fresh = displayed_seconds > 0
        can_approve = self._can_approve_when_fresh and fresh
        self.deadline_text.value = f"承認期限まで {displayed_seconds}秒"
        self.deadline_text.color = "#F2A65A" if fresh else "#D87866"
        lines = list(self._status_lines)
        if not fresh:
            lines.append("✕ 承認期限切れ")
        self.status_text.value = "\n".join(lines)
        self.status_text.color = "#55C2A3" if can_approve else "#D87866"
        self.approve_button.disabled = not can_approve
        self.approve_button.bgcolor = "#F2A65A" if can_approve else "#34332F"
        self.approve_button.color = "#17120D" if can_approve else "#77736B"
