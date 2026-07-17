from __future__ import annotations

from local_llm_chat.domain.models import ToolCallAudit, utc_now
from local_llm_chat.domain.states import ToolCallState
from local_llm_chat.presentation.flet_app import _latest_tool_audit


def _audit(identifier: str) -> ToolCallAudit:
    return ToolCallAudit(
        id=identifier,
        conversation_id="conversation-1",
        run_id="run-1",
        provider="builtin",
        tool_name="search_allowed_folder",
        input_arguments={"query": identifier},
        state=ToolCallState.COMPLETED,
        result_content=None,
        result_item_count=1,
        result_size_bytes=10,
        result_sha256="0" * 64,
        failure_reason=None,
        created_at=utc_now(),
    )


def test_recent_tool_result_uses_first_audit_from_newest_first_repository() -> None:
    newest = _audit("newest")
    oldest = _audit("oldest")

    assert _latest_tool_audit([newest, oldest]) is newest
    assert _latest_tool_audit([]) is None
