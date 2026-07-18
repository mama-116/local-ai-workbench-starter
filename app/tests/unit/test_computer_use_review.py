from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import flet as ft

from local_llm_chat.domain.models import (
    ComputerActionRequest,
    ComputerPlan,
    DesktopSecurityContext,
)
from local_llm_chat.domain.states import (
    ComputerActionType,
    DesktopIntegrityLevel,
)
from local_llm_chat.presentation.components.computer_use_review import (
    ComputerUseReviewDialog,
)


def safe_context() -> DesktopSecurityContext:
    return DesktopSecurityContext(
        "same-user",
        "same-user",
        1,
        1,
        DesktopIntegrityLevel.MEDIUM,
        DesktopIntegrityLevel.MEDIUM,
        False,
        False,
        False,
        False,
    )


def fixed_plan() -> ComputerPlan:
    return ComputerPlan(
        "conversation",
        "メモ帳へ入力して保存しない",
        "observation",
        (
            ComputerActionRequest(
                "launch",
                ComputerActionType.LAUNCH_ALLOWED_APP,
                "windows_notepad",
            ),
            ComputerActionRequest(
                "click",
                ComputerActionType.CLICK_UIA_ELEMENT,
                "notepad_edit",
            ),
            ComputerActionRequest(
                "type",
                ComputerActionType.TYPE_PLAIN_TEXT,
                "notepad_edit",
                "安全確認🙂",
            ),
        ),
    )


def _texts(control: object) -> Iterator[str]:
    if isinstance(control, str):
        yield control
        return
    if isinstance(control, ft.Text):
        yield control.value
    for name in ("title", "content"):
        child = getattr(control, name, None)
        if child is not None and not isinstance(child, str):
            yield from _texts(child)
    for name in ("controls", "actions"):
        children = getattr(control, name, None)
        if children is not None:
            for child in children:
                yield from _texts(child)


def test_focus_review_blocks_approval_until_isolation_is_selected() -> None:
    dialog = ComputerUseReviewDialog(
        fixed_plan(),
        safe_context(),
        isolation_ready=False,
    )

    copy = "\n".join(_texts(dialog))

    assert dialog.modal is True
    assert dialog.approve_button.disabled is True
    assert "この計画だけを実行します" in copy
    assert "安全確認🙂" in copy
    assert "隔離環境が未選定 — このPCへは入力しません" in copy
    assert "停止: Ctrl + Alt + F12" in copy


def test_focus_review_enables_exact_plan_only_in_selected_isolation() -> None:
    dialog = ComputerUseReviewDialog(
        fixed_plan(),
        safe_context(),
        isolation_ready=True,
        isolation_label="専用Windows VM",
        remaining_seconds=24,
    )

    copy = "\n".join(_texts(dialog))

    assert dialog.approve_button.disabled is False
    assert "専用Windows VM / メモ帳" in copy
    assert "承認期限まで 24秒" in copy
    assert "保存・削除・設定変更なし" in copy


def test_focus_review_never_enables_unsafe_context() -> None:
    unsafe = replace(safe_context(), controller_elevated=True)
    dialog = ComputerUseReviewDialog(
        fixed_plan(), unsafe, isolation_ready=True, isolation_label="VM"
    )

    copy = "\n".join(_texts(dialog))

    assert dialog.approve_button.disabled is True
    assert "安全検査: controller_elevated" in copy
    assert "✕ 標準権限・通常デスクトップではありません" in copy
    assert "✓ 標準権限・通常デスクトップ" not in copy


def test_focus_review_never_displays_more_than_contract_approval_timeout() -> None:
    dialog = ComputerUseReviewDialog(
        fixed_plan(),
        safe_context(),
        isolation_ready=True,
        isolation_label="VM",
        remaining_seconds=999,
    )

    copy = "\n".join(_texts(dialog))

    assert "承認期限まで 30秒" in copy
    assert "999秒" not in copy


def test_focus_review_allows_fake_execution_without_claiming_host_isolation() -> None:
    dialog = ComputerUseReviewDialog(
        fixed_plan(),
        safe_context(),
        isolation_ready=False,
        fake_only=True,
    )

    copy = "\n".join(_texts(dialog))

    assert dialog.approve_button.disabled is False
    assert dialog.approve_button.content == "Fakeで承認・実行"
    assert "OS入力は0件" in copy
    assert "隔離環境" not in copy
    assert "固定メモ帳を起動" not in copy
    assert "起動要求を検査" in copy


def test_focus_review_disables_approval_when_live_deadline_expires() -> None:
    dialog = ComputerUseReviewDialog(
        fixed_plan(), safe_context(), isolation_ready=False, fake_only=True
    )

    dialog.set_remaining_seconds(0)

    copy = "\n".join(_texts(dialog))
    assert dialog.approve_button.disabled is True
    assert "承認期限まで 0秒" in copy
    assert "承認期限切れ" in copy
