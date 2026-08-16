from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryEvent,
    CanonicalMemoryFact,
    CanonicalMemoryReviewItem,
    CanonicalMemoryLedger,
    MemoryApprovalDecision,
    MemoryProjectionQuery,
    transitive_superseded_event_ids,
)
from local_llm_chat.domain.errors import (
    ConversationNotFound,
    PersistenceError,
    ValidationError,
)
from local_llm_chat.domain.explicit_memory import (
    MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS,
    ExplicitMemoryDecision,
    ExplicitMemoryEvent,
    ExplicitMemoryLedger,
    ExplicitMemoryProjectionQuery,
    ExplicitMemoryReviewItem,
)
from local_llm_chat.domain.group_turns import (
    MAX_FORMAL_CHARACTERS,
    ConversationCast,
    ConversationGroupConfiguration,
    ConversationGroupSettings,
    FormalCastMember,
    TurnBatch,
    TurnBatchDraft,
    TurnSegment,
    validate_conversation_group_settings,
)
from local_llm_chat.domain.relationship_profile import (
    Continuity,
    EvidenceContext,
    LedgerActor,
    ProfileApproval,
    ProfileEvent,
    ProfileItem,
    ProfileOrigin,
    ProfilePurgeReceipt,
    ProfileScope,
    ProfileUsageState,
    RelationshipApproval,
    RelationshipAssignmentState,
    RelationshipDefinition,
    RelationshipDirection,
    RelationshipEvent,
    RelationshipInterpretation,
    RelationshipInterpretationState,
    RelationshipMeaning,
    RelationshipMetrics,
    RelationshipSeverity,
    UserProfile,
    reduce_relationship_events,
)
from local_llm_chat.domain.models import (
    AgentExecutionLimits,
    AgentRun,
    AgentStep,
    ComputerActionAudit,
    ComputerActionRequest,
    ComputerPlanApproval,
    ComputerUseLimits,
    ComputerUseRun,
    BranchInfo,
    CharacterVersion,
    Conversation,
    ContextSummary,
    ContextSummaryPreparation,
    DocumentChunk,
    DocumentRecord,
    LatestTelemetry,
    JobRun,
    Message,
    MessageCitation,
    MessageRagUsage,
    ModelProfile,
    RunRecord,
    RunSession,
    ScheduledJob,
    RagCitation,
    RagSearchResult,
    TelemetryMetric,
    ToolCallAudit,
    ToolCallRequest,
    ToolFolderGrant,
    Translation,
    TranslationPreparation,
)
from local_llm_chat.domain.states import (
    AgentRunState,
    AgentStepState,
    ComputerActionState,
    ComputerActionType,
    ComputerUseRunState,
    AgentToolEffect,
    DataClassification,
    CostClass,
    Locality,
    MemoryApprovalState,
    MemoryCardinality,
    MemoryKind,
    MessageRole,
    MessageState,
    ContextSummaryState,
    JobRunState,
    RunState,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
    TranslationState,
    ToolCallState,
)

T = TypeVar("T")

_DEFAULT_SYSTEM_PROMPT = (
    "あなたは親切で正確なローカルAIアシスタントです。"
    "質問には直接答え、依頼文の言い換えだけで終わらせないでください。"
    "ユーザーが言語、形式、内容を明示した場合は、その指定を優先してください。"
    "言語指定がない場合は、自然な日本語で簡潔に回答してください。"
    "実在を確認できない固有名詞を作らず、確信がなければ一般的な種類を提案してください。"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("時刻にはtimezoneが必要です。")
    return value.astimezone(UTC).isoformat()


class SQLiteAppRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._write_lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    async def _read(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        def execute() -> T:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                result = operation(connection)
                connection.commit()
                return result
            except sqlite3.Error as error:
                raise PersistenceError(str(error)) from error
            finally:
                if connection is not None:
                    connection.close()

        return await asyncio.to_thread(execute)

    async def _write(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        async with self._write_lock:
            return await self._read(operation)

    async def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        migration_paths = sorted(Path(__file__).with_name("migrations").glob("*.sql"))

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {
                int(row["version"])
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            for migration_path in migration_paths:
                version = int(migration_path.stem.split("_", maxsplit=1)[0])
                if version in applied:
                    continue
                applied_at = _now().replace("'", "''")
                migration_sql = migration_path.read_text(encoding="utf-8")
                dependent_trigger_rows: list[sqlite3.Row] = []
                if version == 22:
                    dependent_trigger_rows = connection.execute(
                        """
                        SELECT name, sql
                        FROM sqlite_master
                        WHERE type = 'trigger'
                          AND sql IS NOT NULL
                          AND lower(sql) LIKE '%conversations%'
                        ORDER BY name
                        """
                    ).fetchall()
                drop_dependent_triggers = "\n".join(
                    f'DROP TRIGGER "{str(row["name"]).replace('"', '""')}";'
                    for row in dependent_trigger_rows
                )
                restore_dependent_triggers = "\n".join(
                    str(row["sql"]) + ";" for row in dependent_trigger_rows
                )
                connection.commit()
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    f"{drop_dependent_triggers}\n"
                    f"{migration_sql}\n"
                    f"{restore_dependent_triggers}\n"
                    "INSERT INTO schema_migrations(version, applied_at) "
                    f"VALUES({version}, '{applied_at}');\n"
                    "COMMIT;"
                )
                connection.execute("PRAGMA foreign_keys = ON")
                foreign_key_error = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchone()
                if foreign_key_error is not None:
                    raise sqlite3.IntegrityError(
                        f"migration {version} left an invalid foreign key"
                    )

        await self._write(operation)

    async def recover_interrupted_runs(self) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            now = _now()
            connection.execute(
                """
                UPDATE messages
                SET state = 'failed', completed_at = ?
                WHERE state IN ('pending', 'streaming')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE tool_calls
                SET state = 'failed', completed_at = ?,
                    failure_reason = '前回のアプリ終了時に中断されました。'
                WHERE state IN ('pending', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE runs
                SET state = 'failed', completed_at = ?, error_code = 'previous_session_interrupted'
                WHERE state IN ('pending', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE message_translations
                SET state = 'failed', completed_at = ?, error_code = 'previous_session_interrupted'
                WHERE state IN ('pending', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE context_summaries
                SET state = 'failed', completed_at = ?, error_code = 'previous_session_interrupted'
                WHERE state IN ('pending', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE job_runs
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('pending', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE agent_steps
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('proposed', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE agent_runs
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('pending', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE computer_actions
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('proposed', 'approved', 'running')
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE computer_use_runs
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('pending', 'awaiting_approval', 'running')
                """,
                (now,),
            )

        await self._write(operation)

    async def create_agent_run(self, run: AgentRun) -> AgentRun:
        if run.state is not AgentRunState.RUNNING or run.started_at is None:
            raise ValidationError("Agent runはrunning状態で開始してください。")
        started_at = run.started_at
        allowed_tools_json = json.dumps(
            run.allowed_tools, ensure_ascii=False, separators=(",", ":")
        )

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            try:
                connection.execute(
                    """
                    INSERT INTO agent_runs(
                        id, conversation_id, objective, allowed_tools_json,
                        max_cost_units, max_steps, max_duration_seconds,
                        state, failure_reason, created_at, started_at, completed_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.conversation_id,
                        run.objective,
                        allowed_tools_json,
                        run.limits.max_cost_units,
                        run.limits.max_steps,
                        run.limits.max_duration_seconds,
                        run.state.value,
                        run.failure_reason,
                        _utc_iso(run.created_at),
                        _utc_iso(started_at),
                        None,
                    ),
                )
            except sqlite3.IntegrityError as error:
                if (
                    "agent_runs.1" in str(error)
                    or "uq_single_running_agent_run" in str(error)
                ):
                    raise ValidationError("別のAgent runが実行中です。") from error
                raise
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Agent runを作成できませんでした。")
            return cast(sqlite3.Row, row)

        return self._agent_run_from_row(await self._write(operation))

    async def get_agent_run(self, run_id: str) -> AgentRun:
        row = await self._read(
            lambda connection: connection.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
        )
        if row is None:
            raise ValidationError("Agent runが見つかりません。")
        return self._agent_run_from_row(cast(sqlite3.Row, row))

    async def create_agent_step(self, step: AgentStep) -> AgentStep:
        try:
            arguments_json = json.dumps(
                step.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValidationError("Agent stepの引数はJSON形式にしてください。") from error

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            run = connection.execute(
                "SELECT state FROM agent_runs WHERE id = ?", (step.run_id,)
            ).fetchone()
            if run is None or str(run["state"]) != AgentRunState.RUNNING.value:
                raise ValidationError("実行中のAgent runにだけstepを追加できます。")
            connection.execute(
                """
                INSERT INTO agent_steps(
                    id, run_id, ordinal, tool_name, arguments_json, action_hash,
                    data_classification, effect, destination, cost_class,
                    cost_units, state,
                    result_size_bytes, result_sha256, restore_token,
                    failure_reason, created_at, started_at, completed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.id,
                    step.run_id,
                    step.ordinal,
                    step.tool_name,
                    arguments_json,
                    step.action_hash,
                    (
                        step.data_classification.value
                        if step.data_classification is not None
                        else None
                    ),
                    step.effect.value if step.effect is not None else None,
                    (
                        step.destination.value if step.destination is not None else None
                    ),
                    step.cost_class.value if step.cost_class is not None else None,
                    step.cost_units,
                    step.state.value,
                    step.result_size_bytes,
                    step.result_sha256,
                    step.restore_token,
                    step.failure_reason,
                    _utc_iso(step.created_at),
                    _utc_iso(step.started_at) if step.started_at is not None else None,
                    (
                        _utc_iso(step.completed_at)
                        if step.completed_at is not None
                        else None
                    ),
                ),
            )
            row = connection.execute(
                "SELECT * FROM agent_steps WHERE id = ?", (step.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Agent stepを作成できませんでした。")
            return cast(sqlite3.Row, row)

        return self._agent_step_from_row(await self._write(operation))

    async def finish_agent_step(self, step: AgentStep) -> AgentStep:
        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            cursor = connection.execute(
                """
                UPDATE agent_steps
                SET state = ?, result_size_bytes = ?, result_sha256 = ?,
                    restore_token = ?, failure_reason = ?, started_at = ?, completed_at = ?
                WHERE id = ? AND run_id = ?
                """,
                (
                    step.state.value,
                    step.result_size_bytes,
                    step.result_sha256,
                    step.restore_token,
                    step.failure_reason,
                    _utc_iso(step.started_at) if step.started_at is not None else None,
                    (
                        _utc_iso(step.completed_at)
                        if step.completed_at is not None
                        else None
                    ),
                    step.id,
                    step.run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValidationError("Agent stepが見つかりません。")
            row = connection.execute(
                "SELECT * FROM agent_steps WHERE id = ?", (step.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Agent stepを更新できませんでした。")
            return cast(sqlite3.Row, row)

        return self._agent_step_from_row(await self._write(operation))

    async def finish_agent_run(self, run: AgentRun) -> AgentRun:
        if run.state not in {
            AgentRunState.COMPLETED,
            AgentRunState.FAILED,
            AgentRunState.CANCELLED,
            AgentRunState.DENIED,
        }:
            raise ValidationError("Agent runの終了状態が不正です。")

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET state = ?, failure_reason = ?, completed_at = ?
                WHERE id = ? AND state = 'running'
                """,
                (
                    run.state.value,
                    run.failure_reason,
                    _utc_iso(run.completed_at) if run.completed_at is not None else _now(),
                    run.id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValidationError("実行中のAgent runが見つかりません。")
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Agent runを更新できませんでした。")
            return cast(sqlite3.Row, row)

        return self._agent_run_from_row(await self._write(operation))

    async def list_agent_steps(self, run_id: str) -> list[AgentStep]:
        rows = await self._read(
            lambda connection: connection.execute(
                """
                SELECT * FROM agent_steps WHERE run_id = ?
                ORDER BY ordinal
                """,
                (run_id,),
            ).fetchall()
        )
        return [self._agent_step_from_row(row) for row in rows]

    async def recover_interrupted_agent_runs(self, now: datetime) -> int:
        value = _utc_iso(now)

        def operation(connection: sqlite3.Connection) -> int:
            connection.execute(
                """
                UPDATE agent_steps
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('proposed', 'running')
                """,
                (value,),
            )
            cursor = connection.execute(
                """
                UPDATE agent_runs
                SET state = 'failed', completed_at = ?,
                    failure_reason = 'previous_session_interrupted'
                WHERE state IN ('pending', 'running')
                """,
                (value,),
            )
            return cursor.rowcount

        return await self._write(operation)

    async def create_computer_use_run(
        self, run: ComputerUseRun
    ) -> ComputerUseRun:
        if run.state is not ComputerUseRunState.AWAITING_APPROVAL:
            raise ValidationError(
                "Computer Use runはawaiting_approval状態で開始してください。"
            )

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            connection.execute(
                """
                INSERT INTO computer_use_runs(
                    id, conversation_id, objective, observation_id, plan_hash,
                    max_actions, max_duration_seconds, approval_timeout_seconds,
                    planned_action_count, state, failure_reason,
                    created_at, started_at, completed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.conversation_id,
                    run.objective,
                    run.observation_id,
                    run.plan_hash,
                    run.limits.max_actions,
                    run.limits.max_duration_seconds,
                    run.limits.approval_timeout_seconds,
                    run.planned_action_count,
                    run.state.value,
                    run.failure_reason,
                    _utc_iso(run.created_at),
                    None,
                    None,
                ),
            )
            row = connection.execute(
                "SELECT * FROM computer_use_runs WHERE id = ?", (run.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Computer Use runを作成できませんでした。")
            return cast(sqlite3.Row, row)

        return self._computer_use_run_from_row(await self._write(operation))

    async def get_computer_use_run(self, run_id: str) -> ComputerUseRun:
        row = await self._read(
            lambda connection: connection.execute(
                "SELECT * FROM computer_use_runs WHERE id = ?", (run_id,)
            ).fetchone()
        )
        if row is None:
            raise ValidationError("Computer Use runが見つかりません。")
        return self._computer_use_run_from_row(cast(sqlite3.Row, row))

    async def list_computer_use_runs(
        self, conversation_id: str, limit: int = 50
    ) -> list[ComputerUseRun]:
        if not 1 <= limit <= 100:
            raise ValidationError("Computer Use履歴の件数は1〜100で指定してください。")
        rows = await self._read(
            lambda connection: connection.execute(
                """
                SELECT * FROM computer_use_runs
                WHERE conversation_id = ?
                ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        )
        return [self._computer_use_run_from_row(row) for row in rows]

    async def create_computer_action(
        self, action: ComputerActionAudit
    ) -> ComputerActionAudit:
        if action.state is not ComputerActionState.PROPOSED:
            raise ValidationError("Computer actionはproposed状態で作成してください。")

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            run = connection.execute(
                "SELECT state FROM computer_use_runs WHERE id = ?",
                (action.run_id,),
            ).fetchone()
            if (
                run is None
                or str(run["state"])
                != ComputerUseRunState.AWAITING_APPROVAL.value
            ):
                raise ValidationError("承認前のComputer Use runへだけ追加できます。")
            connection.execute(
                """
                INSERT INTO computer_actions(
                    id, run_id, ordinal, request_id, action_type,
                    target_profile_id, input_text, state, failure_reason,
                    created_at, started_at, completed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.id,
                    action.run_id,
                    action.ordinal,
                    action.request.id,
                    action.request.action_type.value,
                    action.request.target_profile_id,
                    action.request.text,
                    action.state.value,
                    action.failure_reason,
                    _utc_iso(action.created_at),
                    None,
                    None,
                ),
            )
            row = connection.execute(
                "SELECT * FROM computer_actions WHERE id = ?", (action.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Computer actionを作成できませんでした。")
            return cast(sqlite3.Row, row)

        return self._computer_action_from_row(await self._write(operation))

    async def list_computer_actions(
        self, run_id: str
    ) -> list[ComputerActionAudit]:
        rows = await self._read(
            lambda connection: connection.execute(
                """
                SELECT * FROM computer_actions
                WHERE run_id = ? ORDER BY ordinal
                """,
                (run_id,),
            ).fetchall()
        )
        return [self._computer_action_from_row(row) for row in rows]

    async def update_computer_action(
        self, action: ComputerActionAudit
    ) -> ComputerActionAudit:
        transitions = {
            ComputerActionState.PROPOSED: {
                ComputerActionState.APPROVED,
                ComputerActionState.DENIED,
                ComputerActionState.FAILED,
                ComputerActionState.CANCELLED,
            },
            ComputerActionState.APPROVED: {
                ComputerActionState.RUNNING,
                ComputerActionState.FAILED,
                ComputerActionState.CANCELLED,
            },
            ComputerActionState.RUNNING: {
                ComputerActionState.COMPLETED,
                ComputerActionState.FAILED,
                ComputerActionState.CANCELLED,
            },
        }

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            current = connection.execute(
                "SELECT state FROM computer_actions WHERE id = ? AND run_id = ?",
                (action.id, action.run_id),
            ).fetchone()
            if current is None:
                raise ValidationError("Computer actionが見つかりません。")
            current_state = ComputerActionState(str(current["state"]))
            if action.state not in transitions.get(current_state, set()):
                raise ValidationError("Computer actionの状態遷移が不正です。")
            connection.execute(
                """
                UPDATE computer_actions
                SET state = ?, failure_reason = ?, started_at = ?, completed_at = ?
                WHERE id = ? AND run_id = ?
                """,
                (
                    action.state.value,
                    action.failure_reason,
                    _utc_iso(action.started_at) if action.started_at else None,
                    _utc_iso(action.completed_at) if action.completed_at else None,
                    action.id,
                    action.run_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM computer_actions WHERE id = ?", (action.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Computer actionを更新できませんでした。")
            return cast(sqlite3.Row, row)

        return self._computer_action_from_row(await self._write(operation))

    async def finish_computer_use_run(
        self, run: ComputerUseRun
    ) -> ComputerUseRun:
        if run.state not in {
            ComputerUseRunState.COMPLETED,
            ComputerUseRunState.FAILED,
            ComputerUseRunState.CANCELLED,
            ComputerUseRunState.DENIED,
        }:
            raise ValidationError("Computer Use runの終了状態が不正です。")

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            if run.state is ComputerUseRunState.COMPLETED:
                audit = connection.execute(
                    """
                    SELECT runs.planned_action_count,
                           COUNT(actions.id) AS action_count,
                           SUM(CASE WHEN actions.state = 'completed' THEN 1 ELSE 0 END)
                               AS completed_count
                    FROM computer_use_runs runs
                    LEFT JOIN computer_actions actions ON actions.run_id = runs.id
                    WHERE runs.id = ?
                    GROUP BY runs.id
                    """,
                    (run.id,),
                ).fetchone()
                if (
                    audit is None
                    or int(audit["action_count"])
                    != int(audit["planned_action_count"])
                    or int(audit["completed_count"] or 0)
                    != int(audit["planned_action_count"])
                ):
                    raise ValidationError(
                        "全Computer action完了後だけrunを完了できます。"
                    )
            cursor = connection.execute(
                """
                UPDATE computer_use_runs
                SET state = ?, failure_reason = ?, completed_at = ?
                WHERE id = ? AND state IN ('awaiting_approval', 'running')
                """,
                (
                    run.state.value,
                    run.failure_reason,
                    _utc_iso(run.completed_at) if run.completed_at else _now(),
                    run.id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValidationError("終了可能なComputer Use runがありません。")
            row = connection.execute(
                "SELECT * FROM computer_use_runs WHERE id = ?", (run.id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("Computer Use runを更新できませんでした。")
            return cast(sqlite3.Row, row)

        return self._computer_use_run_from_row(await self._write(operation))

    async def create_computer_plan_approval(
        self, run_id: str, approval: ComputerPlanApproval
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            run = connection.execute(
                "SELECT plan_hash, state FROM computer_use_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if (
                run is None
                or str(run["state"])
                != ComputerUseRunState.AWAITING_APPROVAL.value
                or str(run["plan_hash"]) != approval.plan_hash
            ):
                raise ValidationError("承認対象の計画が一致しません。")
            connection.execute(
                """
                INSERT INTO computer_plan_approvals(
                    id, run_id, plan_hash, approved_at, consumed_at
                ) VALUES(?, ?, ?, ?, NULL)
                """,
                (
                    approval.id,
                    run_id,
                    approval.plan_hash,
                    _utc_iso(approval.approved_at),
                ),
            )

        await self._write(operation)

    async def verify_and_consume(
        self,
        approval: ComputerPlanApproval,
        plan_hash: str,
        verified_at: datetime,
    ) -> bool:
        approved_at = _utc_iso(approval.approved_at)
        _utc_iso(verified_at)
        verified_time = verified_at.astimezone(UTC)

        def operation(connection: sqlite3.Connection) -> bool:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT approvals.run_id, runs.created_at,
                       runs.approval_timeout_seconds
                FROM computer_plan_approvals approvals
                JOIN computer_use_runs runs ON runs.id = approvals.run_id
                WHERE approvals.id = ? AND approvals.plan_hash = ?
                  AND approvals.approved_at = ?
                  AND approvals.consumed_at IS NULL
                """,
                (approval.id, plan_hash, approved_at),
            ).fetchone()
            if row is None:
                return False
            created_at = datetime.fromisoformat(str(row["created_at"]))
            deadline = created_at + timedelta(
                seconds=float(row["approval_timeout_seconds"])
            )
            approval_time = approval.approved_at.astimezone(UTC)
            if not created_at <= approval_time <= verified_time < deadline:
                return False
            run_id = str(row["run_id"])
            audit = connection.execute(
                """
                SELECT runs.planned_action_count,
                       COUNT(actions.id) AS action_count,
                       SUM(CASE WHEN actions.state = 'proposed' THEN 1 ELSE 0 END)
                           AS proposed_count
                FROM computer_use_runs runs
                LEFT JOIN computer_actions actions ON actions.run_id = runs.id
                WHERE runs.id = ?
                GROUP BY runs.id
                """,
                (run_id,),
            ).fetchone()
            if (
                audit is None
                or int(audit["action_count"])
                != int(audit["planned_action_count"])
                or int(audit["proposed_count"] or 0)
                != int(audit["planned_action_count"])
            ):
                return False
            try:
                run_cursor = connection.execute(
                    """
                    UPDATE computer_use_runs
                    SET state = 'running', started_at = ?
                    WHERE id = ? AND plan_hash = ? AND state = 'awaiting_approval'
                    """,
                    (_now(), run_id, plan_hash),
                )
            except sqlite3.IntegrityError:
                return False
            if run_cursor.rowcount != 1:
                return False
            approval_cursor = connection.execute(
                """
                UPDATE computer_plan_approvals SET consumed_at = ?
                WHERE id = ? AND consumed_at IS NULL
                """,
                (_now(), approval.id),
            )
            return approval_cursor.rowcount == 1

        return await self._write(operation)

    async def create_scheduled_job(
        self,
        job_id: str,
        handler_name: str,
        interval_seconds: int,
        first_due_at: datetime,
        payload: dict[str, object],
    ) -> ScheduledJob:
        if not job_id.strip() or not handler_name.strip():
            raise ValidationError("ジョブIDとハンドラー名が必要です。")
        if interval_seconds < 1:
            raise ValidationError("実行間隔は1秒以上にしてください。")
        due = _utc_iso(first_due_at)
        created = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO scheduled_jobs(
                    id, handler_name, interval_seconds, first_due_at,
                    payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id.strip(),
                    handler_name.strip(),
                    interval_seconds,
                    due,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    created,
                ),
            )

        await self._write(operation)
        return ScheduledJob(
            job_id.strip(),
            handler_name.strip(),
            interval_seconds,
            datetime.fromisoformat(due),
            dict(payload),
            datetime.fromisoformat(created),
        )

    async def claim_due_job(
        self, now: datetime
    ) -> tuple[ScheduledJob, JobRun] | None:
        now_value = _utc_iso(now)

        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[sqlite3.Row, sqlite3.Row] | None:
            connection.execute("BEGIN IMMEDIATE")
            candidates = connection.execute(
                """
                SELECT jobs.*,
                    (
                        SELECT scheduled_for FROM job_runs
                        WHERE job_id = jobs.id AND attempt = 1
                        ORDER BY scheduled_for DESC LIMIT 1
                    ) AS last_scheduled_for
                FROM scheduled_jobs jobs
                WHERE NOT EXISTS (
                    SELECT 1 FROM job_runs running
                    WHERE running.job_id = jobs.id AND running.state = 'running'
                )
                ORDER BY jobs.first_due_at, jobs.id
                """
            ).fetchall()
            selected: tuple[sqlite3.Row, datetime] | None = None
            now_time = datetime.fromisoformat(now_value)
            for row in candidates:
                last_value = row["last_scheduled_for"]
                due_time = (
                    datetime.fromisoformat(str(row["first_due_at"]))
                    if last_value is None
                    else datetime.fromisoformat(str(last_value))
                    + timedelta(seconds=int(row["interval_seconds"]))
                )
                if due_time <= now_time and (
                    selected is None or due_time < selected[1]
                ):
                    selected = (row, due_time)
            if selected is None:
                return None
            job_row, due_time = selected
            run_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO job_runs(
                    id, job_id, scheduled_for, attempt, state,
                    retry_of_run_id, created_at, started_at
                ) VALUES(?, ?, ?, 1, 'running', NULL, ?, ?)
                """,
                (run_id, str(job_row["id"]), due_time.isoformat(), now_value, now_value),
            )
            run_row = connection.execute(
                "SELECT * FROM job_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run_row is None:  # pragma: no cover
                raise PersistenceError("実行記録を作成できませんでした。")
            return job_row, run_row

        result = await self._write(operation)
        if result is None:
            return None
        return self._scheduled_job_from_row(result[0]), self._job_run_from_row(result[1])

    async def create_job_retry(
        self, run_id: str, now: datetime
    ) -> tuple[ScheduledJob, JobRun]:
        now_value = _utc_iso(now)

        def operation(connection: sqlite3.Connection) -> tuple[sqlite3.Row, sqlite3.Row]:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                "SELECT * FROM job_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if source is None or str(source["state"]) != JobRunState.FAILED.value:
                raise ValidationError("失敗した実行だけを再実行できます。")
            attempt_row = connection.execute(
                """
                SELECT MAX(attempt) AS value FROM job_runs
                WHERE job_id = ? AND scheduled_for = ?
                """,
                (source["job_id"], source["scheduled_for"]),
            ).fetchone()
            attempt = int(attempt_row["value"]) + 1
            retry_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO job_runs(
                    id, job_id, scheduled_for, attempt, state,
                    retry_of_run_id, created_at, started_at
                ) VALUES(?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    retry_id,
                    source["job_id"],
                    source["scheduled_for"],
                    attempt,
                    run_id,
                    now_value,
                    now_value,
                ),
            )
            job_row = connection.execute(
                "SELECT * FROM scheduled_jobs WHERE id = ?", (source["job_id"],)
            ).fetchone()
            run_row = connection.execute(
                "SELECT * FROM job_runs WHERE id = ?", (retry_id,)
            ).fetchone()
            if job_row is None or run_row is None:  # pragma: no cover
                raise PersistenceError("再実行記録を作成できませんでした。")
            return job_row, run_row

        job_row, run_row = await self._write(operation)
        return self._scheduled_job_from_row(job_row), self._job_run_from_row(run_row)

    async def finish_job_run(
        self,
        run_id: str,
        state: JobRunState,
        failure_reason: str | None = None,
    ) -> JobRun:
        if state not in {JobRunState.COMPLETED, JobRunState.FAILED}:
            raise ValidationError("終了状態が不正です。")

        def operation(connection: sqlite3.Connection) -> sqlite3.Row:
            cursor = connection.execute(
                """
                UPDATE job_runs
                SET state = ?, failure_reason = ?, completed_at = ?
                WHERE id = ? AND state = 'running'
                """,
                (state.value, failure_reason, _now(), run_id),
            )
            if cursor.rowcount != 1:
                raise ValidationError("実行中のジョブが見つかりません。")
            row = connection.execute(
                "SELECT * FROM job_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:  # pragma: no cover
                raise PersistenceError("実行記録が見つかりません。")
            return cast(sqlite3.Row, row)

        return self._job_run_from_row(await self._write(operation))

    async def list_job_runs(self, job_id: str) -> list[JobRun]:
        rows = await self._read(
            lambda connection: connection.execute(
                """
                SELECT * FROM job_runs WHERE job_id = ?
                ORDER BY scheduled_for, attempt
                """,
                (job_id,),
            ).fetchall()
        )
        return [self._job_run_from_row(row) for row in rows]

    async def set_tool_folder_grant(
        self, conversation_id: str, root_path: Path
    ) -> ToolFolderGrant:
        resolved = Path(root_path).resolve(strict=True)
        if not resolved.is_dir():
            raise ValidationError("許可するフォルダーが見つかりません。")
        granted_at = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO conversation_tool_folder_grants(
                    conversation_id, root_path, granted_at, revoked_at
                ) VALUES(?, ?, ?, NULL)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    root_path = excluded.root_path,
                    granted_at = excluded.granted_at,
                    revoked_at = NULL
                """,
                (conversation_id, str(resolved), granted_at),
            )

        await self._write(operation)
        return ToolFolderGrant(
            conversation_id, resolved, datetime.fromisoformat(granted_at)
        )

    async def get_tool_folder_grant(
        self, conversation_id: str
    ) -> ToolFolderGrant | None:
        def operation(connection: sqlite3.Connection) -> sqlite3.Row | None:
            row = connection.execute(
                """
                SELECT conversation_id, root_path, granted_at
                FROM conversation_tool_folder_grants
                WHERE conversation_id = ? AND revoked_at IS NULL
                """,
                (conversation_id,),
            ).fetchone()
            return cast(sqlite3.Row | None, row)

        row = await self._read(operation)
        if row is None:
            return None
        return ToolFolderGrant(
            str(row["conversation_id"]),
            Path(str(row["root_path"])),
            datetime.fromisoformat(str(row["granted_at"])),
        )

    async def revoke_tool_folder_grant(self, conversation_id: str) -> None:
        await self._write(
            lambda connection: connection.execute(
                """
                UPDATE conversation_tool_folder_grants
                SET revoked_at = ?
                WHERE conversation_id = ? AND revoked_at IS NULL
                """,
                (_now(), conversation_id),
            )
        )

    async def create_tool_call(
        self,
        conversation_id: str,
        run_id: str | None,
        provider: str,
        request: ToolCallRequest,
    ) -> ToolCallAudit:
        # Ollama call IDs are only correlation tokens and may repeat across turns.
        # The audit trail needs an application-owned UUID for every attempt.
        call_id = str(uuid4())
        created_at = _now()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO tool_calls(
                    id, conversation_id, run_id, provider, tool_name,
                    input_json, state, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    call_id,
                    conversation_id,
                    run_id,
                    provider,
                    request.name,
                    json.dumps(request.arguments, ensure_ascii=False, sort_keys=True),
                    created_at,
                ),
            )

        await self._write(operation)
        return ToolCallAudit(
            call_id,
            conversation_id,
            run_id,
            provider,
            request.name,
            request.arguments,
            ToolCallState.PENDING,
            None,
            None,
            None,
            None,
            None,
            datetime.fromisoformat(created_at),
        )

    async def mark_tool_call_running(self, call_id: str) -> None:
        await self._write(
            lambda connection: connection.execute(
                """
                UPDATE tool_calls SET state = 'running', started_at = ?
                WHERE id = ? AND state = 'pending'
                """,
                (_now(), call_id),
            )
        )

    async def finish_tool_call(
        self,
        call_id: str,
        state: ToolCallState,
        result_content: str | None = None,
        result_item_count: int | None = None,
        failure_reason: str | None = None,
    ) -> None:
        payload = result_content.encode("utf-8") if result_content is not None else None
        await self._write(
            lambda connection: connection.execute(
                """
                UPDATE tool_calls SET
                    state = ?, result_content = ?, result_item_count = ?,
                    result_size_bytes = ?, result_sha256 = ?, failure_reason = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (
                    state.value,
                    result_content,
                    result_item_count,
                    len(payload) if payload is not None else None,
                    hashlib.sha256(payload).hexdigest() if payload is not None else None,
                    failure_reason,
                    _now(),
                    call_id,
                ),
            )
        )

    async def list_tool_calls(
        self, conversation_id: str, limit: int = 100
    ) -> list[ToolCallAudit]:
        def operation(connection: sqlite3.Connection) -> list[sqlite3.Row]:
            rows = connection.execute(
                """
                SELECT id, conversation_id, run_id, provider, tool_name,
                       input_json, state, result_item_count, result_size_bytes,
                       result_sha256, failure_reason, created_at, started_at,
                       completed_at
                FROM tool_calls WHERE conversation_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 100))),
            ).fetchall()
            rows.reverse()
            return rows

        rows = await self._read(operation)
        return [
            ToolCallAudit(
                id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                run_id=str(row["run_id"]) if row["run_id"] is not None else None,
                provider=str(row["provider"]),
                tool_name=str(row["tool_name"]),
                input_arguments=json.loads(str(row["input_json"])),
                state=ToolCallState(str(row["state"])),
                result_content=None,
                result_item_count=row["result_item_count"],
                result_size_bytes=row["result_size_bytes"],
                result_sha256=row["result_sha256"],
                failure_reason=row["failure_reason"],
                created_at=datetime.fromisoformat(str(row["created_at"])),
                started_at=_parse_time(row["started_at"]),
                completed_at=_parse_time(row["completed_at"]),
            )
            for row in rows
        ]

    async def ensure_default_character(self) -> CharacterVersion:
        def operation(connection: sqlite3.Connection) -> CharacterVersion:
            row = connection.execute(
                """
                SELECT cv.*, c.display_name
                FROM character_versions cv
                JOIN characters c ON c.id = cv.character_id
                WHERE c.archived_at IS NULL
                ORDER BY c.created_at, cv.version DESC
                LIMIT 1
                """
            ).fetchone()
            if row is not None:
                return self._character_from_row(row)
            now = _now()
            character_id = str(uuid4())
            version_id = str(uuid4())
            connection.execute(
                "INSERT INTO characters(id, display_name, created_at) VALUES(?, ?, ?)",
                (character_id, "アシスタント", now),
            )
            connection.execute(
                """
                INSERT INTO character_versions(id, character_id, version, system_prompt, created_at)
                VALUES(?, ?, 1, ?, ?)
                """,
                (
                    version_id,
                    character_id,
                    _DEFAULT_SYSTEM_PROMPT,
                    now,
                ),
            )
            return CharacterVersion(
                id=version_id,
                character_id=character_id,
                display_name="アシスタント",
                version=1,
                system_prompt=_DEFAULT_SYSTEM_PROMPT,
                created_at=datetime.fromisoformat(now),
            )

        return await self._write(operation)

    async def list_character_versions(self) -> list[CharacterVersion]:
        def operation(connection: sqlite3.Connection) -> list[CharacterVersion]:
            rows = connection.execute(
                """
                SELECT cv.*, c.display_name
                FROM character_versions cv
                JOIN characters c ON c.id = cv.character_id
                JOIN (
                    SELECT character_id, MAX(version) AS max_version
                    FROM character_versions GROUP BY character_id
                ) latest ON latest.character_id = cv.character_id AND latest.max_version = cv.version
                WHERE c.archived_at IS NULL
                ORDER BY c.display_name
                """
            ).fetchall()
            return [self._character_from_row(row) for row in rows]

        return await self._read(operation)

    async def create_character_version(
        self,
        display_name: str,
        system_prompt: str,
        character_id: str | None = None,
    ) -> CharacterVersion:
        if not display_name.strip() or not system_prompt.strip():
            raise ValidationError("キャラクター名と指示文は必須です。")

        def operation(connection: sqlite3.Connection) -> CharacterVersion:
            now = _now()
            target_id = character_id or str(uuid4())
            if character_id is None:
                connection.execute(
                    "INSERT INTO characters(id, display_name, created_at) VALUES(?, ?, ?)",
                    (target_id, display_name.strip(), now),
                )
                version = 1
            else:
                found = connection.execute(
                    "SELECT id FROM characters WHERE id = ? AND archived_at IS NULL",
                    (target_id,),
                ).fetchone()
                if found is None:
                    raise ValidationError("更新対象のキャラクターが見つかりません。")
                connection.execute(
                    "UPDATE characters SET display_name = ? WHERE id = ?",
                    (display_name.strip(), target_id),
                )
                row = connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM character_versions WHERE character_id = ?",
                    (target_id,),
                ).fetchone()
                version = int(row["version"])
            version_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO character_versions(id, character_id, version, system_prompt, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (version_id, target_id, version, system_prompt.strip(), now),
            )
            return CharacterVersion(
                id=version_id,
                character_id=target_id,
                display_name=display_name.strip(),
                version=version,
                system_prompt=system_prompt.strip(),
                created_at=datetime.fromisoformat(now),
            )

        return await self._write(operation)

    async def get_character_version(self, version_id: str) -> CharacterVersion:
        def operation(connection: sqlite3.Connection) -> CharacterVersion:
            row = connection.execute(
                """
                SELECT cv.*, c.display_name
                FROM character_versions cv JOIN characters c ON c.id = cv.character_id
                WHERE cv.id = ?
                """,
                (version_id,),
            ).fetchone()
            if row is None:
                raise ValidationError("キャラクター設定が見つかりません。")
            return self._character_from_row(row)

        return await self._read(operation)

    async def ensure_model_profile(
        self,
        provider: str,
        model_name: str,
        parameters: dict[str, Any],
    ) -> ModelProfile:
        def operation(connection: sqlite3.Connection) -> ModelProfile:
            now = _now()
            parameters_json = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
            row = connection.execute(
                "SELECT * FROM model_profiles WHERE provider = ? AND model_name = ?",
                (provider, model_name),
            ).fetchone()
            if row is None:
                profile_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO model_profiles(
                        id, provider, model_name, parameters_json, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (profile_id, provider, model_name, parameters_json, now, now),
                )
            else:
                profile_id = str(row["id"])
                connection.execute(
                    "UPDATE model_profiles SET parameters_json = ?, updated_at = ? WHERE id = ?",
                    (parameters_json, now, profile_id),
                )
            return ModelProfile(profile_id, provider, model_name, dict(parameters))

        return await self._write(operation)

    async def get_model_profile(self, profile_id: str) -> ModelProfile:
        def operation(connection: sqlite3.Connection) -> ModelProfile:
            row = connection.execute(
                "SELECT * FROM model_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            if row is None:
                raise ValidationError("モデル設定が見つかりません。")
            return self._profile_from_row(row)

        return await self._read(operation)

    async def create_conversation(
        self,
        title: str,
        character_version_id: str,
        model_profile_id: str,
        continuity_id: str | None = None,
    ) -> Conversation:
        def operation(connection: sqlite3.Connection) -> Conversation:
            now = _now()
            conversation_id = str(uuid4())
            branch_id = str(uuid4())
            selected_continuity_id = continuity_id
            if selected_continuity_id is None:
                user_profile_id = str(uuid4())
                selected_continuity_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO user_profiles(id, display_name, created_at)
                    VALUES(?, '利用者', ?)
                    """,
                    (user_profile_id, now),
                )
                connection.execute(
                    """
                    INSERT INTO continuities(
                        id, display_name, user_profile_id, created_at
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        selected_continuity_id,
                        title.strip() or "新しい世界線",
                        user_profile_id,
                        now,
                    ),
                )
            else:
                continuity_row = connection.execute(
                    """
                    SELECT id FROM continuities
                    WHERE id = ? AND archived_at IS NULL
                    """,
                    (selected_continuity_id,),
                ).fetchone()
                if continuity_row is None:
                    raise ValidationError("世界線が見つかりません。")
            connection.execute(
                """
                INSERT INTO conversations(
                    id, title, character_version_id, model_profile_id,
                    created_at, updated_at, continuity_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title.strip() or "新しい会話",
                    character_version_id,
                    model_profile_id,
                    now,
                    now,
                    selected_continuity_id,
                ),
            )
            connection.execute(
                "INSERT INTO branches(id, conversation_id, created_at) VALUES(?, ?, ?)",
                (branch_id, conversation_id, now),
            )
            connection.execute(
                "UPDATE conversations SET active_branch_id = ? WHERE id = ?",
                (branch_id, conversation_id),
            )
            character_row = connection.execute(
                "SELECT character_id FROM character_versions WHERE id = ?",
                (character_version_id,),
            ).fetchone()
            if character_row is None:
                raise ValidationError("conversation character version was not found")
            connection.execute(
                """
                INSERT INTO conversation_cast_members(
                    conversation_id, character_id, character_version_id,
                    position, added_at
                ) VALUES(?, ?, ?, 0, ?)
                """,
                (
                    conversation_id,
                    str(character_row["character_id"]),
                    character_version_id,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO conversation_group_settings(
                    conversation_id, enabled, mode,
                    spotlight_character_id, updated_at
                ) VALUES(?, 0, 'story', NULL, ?)
                """,
                (conversation_id, now),
            )
            return Conversation(
                id=conversation_id,
                title=title.strip() or "新しい会話",
                active_branch_id=branch_id,
                character_version_id=character_version_id,
                model_profile_id=model_profile_id,
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
                continuity_id=selected_continuity_id,
            )

        return await self._write(operation)

    async def list_conversations(self) -> list[Conversation]:
        def operation(connection: sqlite3.Connection) -> list[Conversation]:
            rows = connection.execute(
                """
                SELECT * FROM conversations
                WHERE archived_at IS NULL
                ORDER BY updated_at DESC
                """
            ).fetchall()
            return [self._conversation_from_row(row) for row in rows]

        return await self._read(operation)

    async def list_archived_conversations(self) -> list[Conversation]:
        def operation(connection: sqlite3.Connection) -> list[Conversation]:
            rows = connection.execute(
                """
                SELECT * FROM conversations
                WHERE archived_at IS NOT NULL
                ORDER BY archived_at DESC
                """
            ).fetchall()
            return [self._conversation_from_row(row) for row in rows]

        return await self._read(operation)

    async def get_conversation(self, conversation_id: str) -> Conversation:
        def operation(connection: sqlite3.Connection) -> Conversation:
            return self._require_conversation(connection, conversation_id)

        return await self._read(operation)

    async def rename_conversation(
        self, conversation_id: str, title: str
    ) -> Conversation:
        def operation(connection: sqlite3.Connection) -> Conversation:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND archived_at IS NULL
                """,
                (title, _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise ConversationNotFound("会話が見つかりません。")
            return self._require_conversation(connection, conversation_id)

        return await self._write(operation)

    async def get_continuity_for_conversation(
        self, conversation_id: str
    ) -> Continuity:
        def operation(connection: sqlite3.Connection) -> Continuity:
            row = connection.execute(
                """
                SELECT continuity.*
                FROM continuities continuity
                JOIN conversations conversation
                  ON conversation.continuity_id = continuity.id
                WHERE conversation.id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise ConversationNotFound("会話または世界線が見つかりません。")
            return self._continuity_from_row(row)

        return await self._read(operation)

    async def list_continuities(self) -> tuple[Continuity, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[Continuity, ...]:
            rows = connection.execute(
                """
                SELECT * FROM continuities
                WHERE archived_at IS NULL
                ORDER BY created_at, id
                """
            ).fetchall()
            return tuple(self._continuity_from_row(row) for row in rows)

        return await self._read(operation)

    async def get_user_profile(self, user_profile_id: str) -> UserProfile:
        def operation(connection: sqlite3.Connection) -> UserProfile:
            row = connection.execute(
                "SELECT * FROM user_profiles WHERE id = ?",
                (user_profile_id,),
            ).fetchone()
            if row is None:
                raise ValidationError("Profileが見つかりません。")
            return self._user_profile_from_row(row)

        return await self._read(operation)

    async def update_conversation_selection(
        self,
        conversation_id: str,
        character_version_id: str,
        model_profile_id: str,
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            version_row = connection.execute(
                """
                SELECT cv.character_id
                FROM character_versions cv
                JOIN characters c ON c.id = cv.character_id
                WHERE cv.id = ? AND c.archived_at IS NULL
                """,
                (character_version_id,),
            ).fetchone()
            if version_row is None:
                raise ValidationError("conversation character version was not found")
            cast_rows = connection.execute(
                """
                SELECT character_version_id
                FROM conversation_cast_members
                WHERE conversation_id = ? ORDER BY position
                """,
                (conversation_id,),
            ).fetchall()
            if len(cast_rows) > 1 and character_version_id != str(
                cast_rows[0]["character_version_id"]
            ):
                raise ValidationError(
                    "multi-person character changes must use the formal cast service"
                )
            cursor = connection.execute(
                """
                UPDATE conversations
                SET character_version_id = ?, model_profile_id = ?, updated_at = ?
                WHERE id = ? AND archived_at IS NULL
                """,
                (character_version_id, model_profile_id, _now(), conversation_id),
            )
            cast_count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM conversation_cast_members WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            assert cast_count is not None
            if cursor.rowcount == 1 and int(cast_count["count"]) == 1:
                connection.execute(
                    "DELETE FROM conversation_cast_members WHERE conversation_id = ?",
                    (conversation_id,),
                )
                connection.execute(
                    """
                    INSERT INTO conversation_cast_members(
                        conversation_id, character_id, character_version_id,
                        position, added_at
                    ) VALUES(?, ?, ?, 0, ?)
                    """,
                    (
                        conversation_id,
                        str(version_row["character_id"]),
                        character_version_id,
                        _now(),
                    ),
                )
                connection.execute(
                    """
                    UPDATE conversation_group_settings
                    SET mode = 'story', spotlight_character_id = NULL,
                        updated_at = ?
                    WHERE conversation_id = ?
                      AND spotlight_character_id IS NOT NULL
                      AND spotlight_character_id <> ?
                    """,
                    (
                        _now(),
                        conversation_id,
                        str(version_row["character_id"]),
                    ),
                )
            if cursor.rowcount != 1:
                raise ConversationNotFound("会話が見つかりません。")

        await self._write(operation)

    async def archive_conversation(self, conversation_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                "UPDATE conversations SET archived_at = ?, updated_at = ? WHERE id = ?",
                (_now(), _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise ConversationNotFound("会話が見つかりません。")

        await self._write(operation)

    async def restore_conversation(self, conversation_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET archived_at = NULL, updated_at = ?
                WHERE id = ? AND archived_at IS NOT NULL
                """,
                (_now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise ConversationNotFound("保管済みの会話が見つかりません。")

        await self._write(operation)

    async def delete_archived_conversations(
        self, conversation_ids: tuple[str, ...]
    ) -> int:
        if not conversation_ids:
            return 0
        if len(conversation_ids) != len(set(conversation_ids)):
            raise ValidationError("ゴミ箱の削除対象に重複があります。")

        def operation(connection: sqlite3.Connection) -> int:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id, archived_at FROM conversations"
            ).fetchall()
            conversations = {str(row["id"]): row["archived_at"] for row in rows}
            requested = set(conversation_ids)
            archived = {
                conversation_id
                for conversation_id, archived_at in conversations.items()
                if archived_at is not None
            }
            existing_requested = requested.intersection(conversations)
            if not existing_requested and not archived:
                return 0
            if any(
                conversations.get(conversation_id) is None
                for conversation_id in existing_requested
            ):
                raise ValidationError(
                    "ゴミ箱にない会話は完全削除できません。"
                )
            if requested != archived:
                raise ValidationError(
                    "ゴミ箱の内容が変更されました。もう一度確認してください。"
                )

            authorized_at = _now()
            connection.executemany(
                """
                INSERT INTO conversation_deletion_guards(
                    conversation_id, authorized_at
                ) VALUES(?, ?)
                """,
                (
                    (conversation_id, authorized_at)
                    for conversation_id in conversation_ids
                ),
            )
            affected_profile_rows = connection.execute(
                """
                WITH RECURSIVE affected_profile_events(id, user_profile_id) AS (
                    SELECT id, user_profile_id
                    FROM profile_events
                    WHERE source_conversation_id IN (
                        SELECT conversation_id
                        FROM conversation_deletion_guards
                    )
                    UNION
                    SELECT child.id, child.user_profile_id
                    FROM profile_events child
                    JOIN affected_profile_events parent
                      ON child.supersedes_event_id = parent.id
                )
                SELECT DISTINCT user_profile_id
                FROM affected_profile_events
                UNION
                SELECT DISTINCT user_profile_id
                FROM relationship_events
                WHERE source_conversation_id IN (
                    SELECT conversation_id
                    FROM conversation_deletion_guards
                )
                """
            ).fetchall()
            profile_authorizations = tuple(
                (str(uuid4()), str(row["user_profile_id"]))
                for row in affected_profile_rows
            )
            connection.executemany(
                """
                INSERT INTO profile_purge_authorizations(
                    request_id, user_profile_id
                ) VALUES(?, ?)
                """,
                profile_authorizations,
            )
            statements = (
                """
                DELETE FROM profile_capture_suppressions
                WHERE source_message_id IN (
                    SELECT id FROM messages
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM relationship_interpretations
                WHERE user_profile_id IN (
                    SELECT user_profile_id
                    FROM profile_purge_authorizations
                )
                """,
                """
                DELETE FROM profile_derived_data
                WHERE user_profile_id IN (
                    SELECT user_profile_id
                    FROM profile_purge_authorizations
                )
                """,
                """
                DELETE FROM relationship_decisions
                WHERE target_event_id IN (
                    SELECT id FROM relationship_events
                    WHERE source_conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM relationship_event_knowledge
                WHERE event_id IN (
                    SELECT id FROM relationship_events
                    WHERE source_conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM relationship_events
                WHERE source_conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM profile_decisions
                WHERE target_event_id IN (
                    WITH RECURSIVE affected_profile_events(id) AS (
                        SELECT id FROM profile_events
                        WHERE source_conversation_id IN (
                            SELECT conversation_id
                            FROM conversation_deletion_guards
                        )
                        UNION
                        SELECT child.id
                        FROM profile_events child
                        JOIN affected_profile_events parent
                          ON child.supersedes_event_id = parent.id
                    )
                    SELECT id FROM affected_profile_events
                )
                """,
                """
                DELETE FROM profile_event_scopes
                WHERE event_id IN (
                    WITH RECURSIVE affected_profile_events(id) AS (
                        SELECT id FROM profile_events
                        WHERE source_conversation_id IN (
                            SELECT conversation_id
                            FROM conversation_deletion_guards
                        )
                        UNION
                        SELECT child.id
                        FROM profile_events child
                        JOIN affected_profile_events parent
                          ON child.supersedes_event_id = parent.id
                    )
                    SELECT id FROM affected_profile_events
                )
                """,
                """
                DELETE FROM profile_events
                WHERE id IN (
                    WITH RECURSIVE affected_profile_events(id) AS (
                        SELECT id FROM profile_events
                        WHERE source_conversation_id IN (
                            SELECT conversation_id
                            FROM conversation_deletion_guards
                        )
                        UNION
                        SELECT child.id
                        FROM profile_events child
                        JOIN affected_profile_events parent
                          ON child.supersedes_event_id = parent.id
                    )
                    SELECT id FROM affected_profile_events
                )
                """,
                """
                DELETE FROM explicit_memory_decisions
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM explicit_memory_events
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM computer_actions
                WHERE run_id IN (
                    SELECT id FROM computer_use_runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM computer_plan_approvals
                WHERE run_id IN (
                    SELECT id FROM computer_use_runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM computer_use_runs
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM agent_steps
                WHERE run_id IN (
                    SELECT id FROM agent_runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM agent_runs
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM turn_segments
                WHERE turn_batch_id IN (
                    SELECT id FROM turn_batches
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM turn_batches
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM canonical_memory_decisions
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM canonical_memory_event_knowledge
                WHERE event_id IN (
                    SELECT id FROM canonical_memory_events
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM canonical_memory_events
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM context_summaries
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM app_events
                WHERE run_id IN (
                    SELECT id FROM runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM telemetry_samples
                WHERE run_id IN (
                    SELECT id FROM runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM run_citations
                WHERE run_id IN (
                    SELECT id FROM runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM run_rag_usage
                WHERE run_id IN (
                    SELECT id FROM runs
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                UPDATE message_translations
                SET reused_from_id = NULL
                WHERE reused_from_id IN (
                    SELECT translation.id
                    FROM message_translations translation
                    JOIN messages message ON message.id = translation.message_id
                    WHERE message.conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM message_translations
                WHERE message_id IN (
                    SELECT id FROM messages
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversation_deletion_guards
                    )
                )
                """,
                """
                DELETE FROM tool_calls
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM conversation_tool_folder_grants
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM conversation_documents
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM conversation_group_settings
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM conversation_cast_members
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM runs
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM branches
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
                """
                DELETE FROM messages
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                """,
            )
            for statement in statements:
                connection.execute(statement)
            connection.executemany(
                """
                DELETE FROM profile_purge_authorizations
                WHERE request_id = ?
                """,
                ((request_id,) for request_id, _ in profile_authorizations),
            )
            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE id IN (
                    SELECT conversation_id FROM conversation_deletion_guards
                )
                  AND archived_at IS NOT NULL
                """
            )
            if cursor.rowcount != len(conversation_ids):
                raise PersistenceError(
                    "ゴミ箱の会話をすべて削除できませんでした。"
                )
            return cursor.rowcount

        return await self._write(operation)

    async def set_conversation_auto_translate(
        self, conversation_id: str, enabled: bool
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            self._require_conversation(connection, conversation_id)
            cursor = connection.execute(
                """
                UPDATE conversations
                SET auto_translate = ?, updated_at = ?
                WHERE id = ? AND archived_at IS NULL
                """,
                (int(enabled), _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise ValidationError("会話の翻訳設定を保存できません。")

        await self._write(operation)

    async def list_active_messages(self, conversation_id: str) -> list[Message]:
        def operation(connection: sqlite3.Connection) -> list[Message]:
            conversation = self._require_conversation(connection, conversation_id)
            return self._message_path(connection, conversation.active_branch_id)

        return await self._read(operation)

    async def get_message(self, message_id: str) -> Message:
        return await self._read(
            lambda connection: self._require_message(connection, message_id)
        )

    async def get_memory_source_message(
        self, conversation_id: str, branch_id: str, source_message_id: str
    ) -> Message:
        def operation(connection: sqlite3.Connection) -> Message:
            self._require_conversation(connection, conversation_id)
            branch = connection.execute(
                "SELECT 1 FROM branches WHERE id = ? AND conversation_id = ?",
                (branch_id, conversation_id),
            ).fetchone()
            if branch is None:
                raise ValidationError("memory branch is not in the conversation")
            source = self._require_message(connection, source_message_id)
            if (
                source.conversation_id != conversation_id
                or source.role is not MessageRole.USER
                or source.state is not MessageState.COMPLETED
            ):
                raise ValidationError(
                    "memory source must be a completed user message in the conversation"
                )
            branch_message_ids = {
                message.id for message in self._message_path(connection, branch_id)
            }
            if source_message_id not in branch_message_ids:
                raise ValidationError("memory source is not in the selected branch")
            return source

        return await self._read(operation)

    async def get_response_model(self, message_id: str) -> tuple[str, str]:
        def operation(connection: sqlite3.Connection) -> tuple[str, str]:
            row = connection.execute(
                "SELECT provider, model FROM runs WHERE response_message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                raise ValidationError("この発言の生成モデルが見つかりません。")
            return str(row["provider"]), str(row["model"])

        return await self._read(operation)

    async def prepare_translation(
        self,
        message_id: str,
        target_language: str,
        provider: str,
        model: str,
        force: bool,
    ) -> TranslationPreparation:
        def operation(connection: sqlite3.Connection) -> TranslationPreparation:
            message = self._require_message(connection, message_id)
            if (
                message.role is not MessageRole.ASSISTANT
                or message.state is not MessageState.COMPLETED
            ):
                raise ValidationError("完了したAI回答だけを翻訳できます。")
            source_hash = hashlib.sha256(message.content.encode("utf-8")).hexdigest()
            if not force:
                existing = connection.execute(
                    """
                    SELECT * FROM message_translations
                    WHERE message_id = ? AND source_hash = ? AND target_language = ?
                      AND provider = ? AND model = ?
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (message_id, source_hash, target_language, provider, model),
                ).fetchone()
                if existing is not None:
                    return TranslationPreparation(
                        self._translation_from_row(existing), False
                    )

                cached = connection.execute(
                    """
                    SELECT * FROM message_translations
                    WHERE source_hash = ? AND target_language = ?
                      AND provider = ? AND model = ? AND state = 'completed'
                    ORDER BY completed_at DESC, rowid DESC LIMIT 1
                    """,
                    (source_hash, target_language, provider, model),
                ).fetchone()
                if cached is not None:
                    translation_id = str(uuid4())
                    now = _now()
                    connection.execute(
                        """
                        INSERT INTO message_translations(
                            id, message_id, source_hash, target_language, provider,
                            model, content, state, reused_from_id, created_at, completed_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?)
                        """,
                        (
                            translation_id,
                            message_id,
                            source_hash,
                            target_language,
                            provider,
                            model,
                            str(cached["content"]),
                            str(cached["id"]),
                            now,
                            now,
                        ),
                    )
                    return TranslationPreparation(
                        self._require_translation(connection, translation_id), False
                    )

            translation_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO message_translations(
                    id, message_id, source_hash, target_language, provider,
                    model, content, state, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, '', 'pending', ?)
                """,
                (
                    translation_id,
                    message_id,
                    source_hash,
                    target_language,
                    provider,
                    model,
                    _now(),
                ),
            )
            return TranslationPreparation(
                self._require_translation(connection, translation_id), True
            )

        return await self._write(operation)

    async def get_translation(self, translation_id: str) -> Translation:
        return await self._read(
            lambda connection: self._require_translation(connection, translation_id)
        )

    async def get_current_translation(self, message_id: str) -> Translation | None:
        def operation(connection: sqlite3.Connection) -> Translation | None:
            row = connection.execute(
                """
                SELECT * FROM message_translations
                WHERE message_id = ? ORDER BY rowid DESC LIMIT 1
                """,
                (message_id,),
            ).fetchone()
            return self._translation_from_row(row) if row is not None else None

        return await self._read(operation)

    async def list_current_translations(
        self, message_ids: list[str]
    ) -> dict[str, Translation]:
        if not message_ids:
            return {}

        def operation(connection: sqlite3.Connection) -> dict[str, Translation]:
            placeholders = ",".join("?" for _ in message_ids)
            rows = connection.execute(
                f"""
                SELECT current.*
                FROM message_translations current
                JOIN (
                    SELECT message_id, MAX(rowid) AS latest_rowid
                    FROM message_translations
                    WHERE message_id IN ({placeholders})
                    GROUP BY message_id
                ) latest ON latest.latest_rowid = current.rowid
                """,
                tuple(message_ids),
            ).fetchall()
            return {
                str(row["message_id"]): self._translation_from_row(row)
                for row in rows
            }

        return await self._read(operation)

    async def mark_translation_running(self, translation_id: str) -> Translation:
        def operation(connection: sqlite3.Connection) -> Translation:
            cursor = connection.execute(
                """
                UPDATE message_translations SET state = 'running'
                WHERE id = ? AND state = 'pending'
                """,
                (translation_id,),
            )
            if cursor.rowcount != 1:
                raise ValidationError("翻訳処理を開始できません。")
            return self._require_translation(connection, translation_id)

        return await self._write(operation)

    async def finish_translation(
        self,
        translation_id: str,
        content: str,
        state: TranslationState,
        error_code: str | None = None,
    ) -> Translation:
        if state not in (TranslationState.COMPLETED, TranslationState.FAILED):
            raise ValidationError("翻訳の終了状態が不正です。")

        def operation(connection: sqlite3.Connection) -> Translation:
            cursor = connection.execute(
                """
                UPDATE message_translations
                SET content = ?, state = ?, error_code = ?, completed_at = ?
                WHERE id = ? AND state IN ('pending', 'running')
                """,
                (content, state.value, error_code, _now(), translation_id),
            )
            if cursor.rowcount != 1:
                raise ValidationError("翻訳結果を保存できません。")
            return self._require_translation(connection, translation_id)

        return await self._write(operation)

    async def list_branches(self, conversation_id: str) -> list[BranchInfo]:
        def operation(connection: sqlite3.Connection) -> list[BranchInfo]:
            self._require_conversation(connection, conversation_id)
            rows = connection.execute(
                """
                SELECT * FROM branches
                WHERE conversation_id = ? AND hidden_at IS NULL
                ORDER BY created_at
                """,
                (conversation_id,),
            ).fetchall()
            return [self._branch_from_row(row) for row in rows]

        return await self._read(operation)

    async def list_all_branches(self, conversation_id: str) -> list[BranchInfo]:
        def operation(connection: sqlite3.Connection) -> list[BranchInfo]:
            self._require_conversation(connection, conversation_id)
            rows = connection.execute(
                "SELECT * FROM branches WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
            return [self._branch_from_row(row) for row in rows]

        return await self._read(operation)

    async def activate_branch(self, conversation_id: str, branch_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            self._require_conversation(connection, conversation_id)
            branch = connection.execute(
                """
                SELECT id FROM branches
                WHERE id = ? AND conversation_id = ? AND hidden_at IS NULL
                """,
                (branch_id, conversation_id),
            ).fetchone()
            if branch is None:
                raise ValidationError("会話の分岐が見つからないか、非表示です。")
            connection.execute(
                "UPDATE conversations SET active_branch_id = ?, updated_at = ? WHERE id = ?",
                (branch_id, _now(), conversation_id),
            )

        await self._write(operation)

    async def hide_branch(self, conversation_id: str, branch_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            conversation = self._require_conversation(connection, conversation_id)
            branch = self._require_branch(connection, branch_id)
            if branch.conversation_id != conversation_id:
                raise ValidationError("別の会話の分岐は非表示にできません。")
            if branch.parent_branch_id is None:
                raise ValidationError("最初の分岐は非表示にできません。")
            if branch.id == conversation.active_branch_id:
                raise ValidationError("使用中の分岐は非表示にできません。")
            cursor = connection.execute(
                "UPDATE branches SET hidden_at = ? WHERE id = ? AND hidden_at IS NULL",
                (_now(), branch_id),
            )
            if cursor.rowcount != 1:
                raise ValidationError("この分岐はすでに非表示です。")

        await self._write(operation)

    async def restore_branch(self, conversation_id: str, branch_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            self._require_conversation(connection, conversation_id)
            branch = self._require_branch(connection, branch_id)
            if branch.conversation_id != conversation_id:
                raise ValidationError("別の会話の分岐は復元できません。")
            cursor = connection.execute(
                "UPDATE branches SET hidden_at = NULL WHERE id = ? AND hidden_at IS NOT NULL",
                (branch_id,),
            )
            if cursor.rowcount != 1:
                raise ValidationError("この分岐は非表示になっていません。")

        await self._write(operation)

    async def get_conversation_cast(
        self, conversation_id: str
    ) -> ConversationCast:
        def operation(connection: sqlite3.Connection) -> ConversationCast:
            self._require_conversation(connection, conversation_id)
            return self._load_conversation_cast(connection, conversation_id)

        return await self._read(operation)

    async def set_conversation_cast(
        self, conversation_id: str, character_version_ids: tuple[str, ...]
    ) -> ConversationCast:
        if not 1 <= len(character_version_ids) <= MAX_FORMAL_CHARACTERS:
            raise ValidationError("formal cast must contain 1 to 5 characters")

        def operation(connection: sqlite3.Connection) -> ConversationCast:
            connection.execute("BEGIN IMMEDIATE")
            self._require_conversation(connection, conversation_id)
            now = _now()
            cast = self._replace_conversation_cast(
                connection, conversation_id, character_version_ids, now
            )
            stable_ids = tuple(member.character_id for member in cast.members)
            placeholders_for_characters = ",".join("?" for _ in stable_ids)
            connection.execute(
                f"""
                UPDATE conversation_group_settings
                SET mode = 'story', spotlight_character_id = NULL, updated_at = ?
                WHERE conversation_id = ?
                  AND spotlight_character_id IS NOT NULL
                  AND spotlight_character_id NOT IN ({placeholders_for_characters})
                """,
                (now, conversation_id, *stable_ids),
            )
            return cast

        return await self._write(operation)

    async def get_conversation_group_configuration(
        self, conversation_id: str
    ) -> ConversationGroupConfiguration:
        def operation(
            connection: sqlite3.Connection,
        ) -> ConversationGroupConfiguration:
            self._require_conversation(connection, conversation_id)
            return ConversationGroupConfiguration(
                self._load_conversation_cast(connection, conversation_id),
                self._load_conversation_group_settings(
                    connection, conversation_id
                ),
            )

        return await self._read(operation)

    async def set_conversation_group_configuration(
        self,
        conversation_id: str,
        character_version_ids: tuple[str, ...],
        enabled: bool,
        mode: TurnMode,
        spotlight_character_id: str | None,
    ) -> ConversationGroupConfiguration:
        if not 1 <= len(character_version_ids) <= MAX_FORMAL_CHARACTERS:
            raise ValidationError("formal cast must contain 1 to 5 characters")
        validate_conversation_group_settings(
            enabled, mode, spotlight_character_id
        )

        def operation(
            connection: sqlite3.Connection,
        ) -> ConversationGroupConfiguration:
            connection.execute("BEGIN IMMEDIATE")
            self._require_conversation(connection, conversation_id)
            now = _now()
            cast = self._replace_conversation_cast(
                connection, conversation_id, character_version_ids, now
            )
            stable_ids = {member.character_id for member in cast.members}
            if (
                spotlight_character_id is not None
                and spotlight_character_id not in stable_ids
            ):
                raise ValidationError(
                    "spotlight character must belong to the registered cast"
                )
            cursor = connection.execute(
                """
                UPDATE conversation_group_settings
                SET enabled = ?, mode = ?, spotlight_character_id = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    int(enabled),
                    mode.value,
                    spotlight_character_id,
                    now,
                    conversation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("conversation group settings were not found")
            return ConversationGroupConfiguration(
                cast,
                self._load_conversation_group_settings(
                    connection, conversation_id
                ),
            )

        return await self._write(operation)

    async def get_conversation_group_settings(
        self, conversation_id: str
    ) -> ConversationGroupSettings:
        def operation(connection: sqlite3.Connection) -> ConversationGroupSettings:
            self._require_conversation(connection, conversation_id)
            return self._load_conversation_group_settings(
                connection, conversation_id
            )

        return await self._read(operation)

    async def set_conversation_group_settings(
        self,
        conversation_id: str,
        enabled: bool,
        mode: TurnMode,
        spotlight_character_id: str | None,
    ) -> ConversationGroupSettings:
        validate_conversation_group_settings(
            enabled, mode, spotlight_character_id
        )

        def operation(connection: sqlite3.Connection) -> ConversationGroupSettings:
            connection.execute("BEGIN IMMEDIATE")
            conversation = self._require_conversation(connection, conversation_id)
            if conversation.archived_at is not None:
                raise ValidationError("archived conversation settings cannot be changed")
            if spotlight_character_id is not None:
                found = connection.execute(
                    """
                    SELECT 1 FROM conversation_cast_members
                    WHERE conversation_id = ? AND character_id = ?
                    """,
                    (conversation_id, spotlight_character_id),
                ).fetchone()
                if found is None:
                    raise ValidationError(
                        "spotlight character must belong to the registered cast"
                    )
            now = _now()
            cursor = connection.execute(
                """
                UPDATE conversation_group_settings
                SET enabled = ?, mode = ?, spotlight_character_id = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    int(enabled),
                    mode.value,
                    spotlight_character_id,
                    now,
                    conversation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("conversation group settings were not found")
            return self._load_conversation_group_settings(
                connection, conversation_id
            )

        return await self._write(operation)

    async def append_profile_event(
        self, event: ProfileEvent, actor: LedgerActor
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_profile_event_actor(connection, event, actor)
            existing = connection.execute(
                "SELECT * FROM profile_events WHERE id = ?", (event.id,)
            ).fetchone()
            if existing is not None:
                existing_event = self._profile_event_from_row(connection, existing)
                if existing_event == event:
                    return
                raise ValidationError("ProfileイベントIDは既に使われています。")
            self._validate_profile_source(connection, event)
            connection.execute(
                """
                INSERT INTO profile_events(
                    id, user_profile_id, item_kind, item_name, value,
                    origin, approval, scope, scope_count,
                    source_conversation_id, source_branch_id, source_message_id,
                    manual_operation_id, supersedes_event_id,
                    effective_at, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.user_profile_id,
                    event.item_kind.strip(),
                    event.item_name.strip(),
                    event.value.strip(),
                    event.origin.value,
                    event.approval.value,
                    event.scope.value,
                    len(event.known_by_character_ids),
                    event.source_conversation_id,
                    event.source_branch_id,
                    event.source_message_id,
                    event.manual_operation_id,
                    event.supersedes_event_id,
                    _utc_iso(event.effective_at),
                    _utc_iso(event.recorded_at),
                ),
            )
            for character_id in event.known_by_character_ids:
                connection.execute(
                    """
                    INSERT INTO profile_event_scopes(event_id, character_id)
                    VALUES(?, ?)
                    """,
                    (event.id, character_id),
                )
            if event.approval in {
                ProfileApproval.AUTO_SAVED,
                ProfileApproval.CONFIRMED,
            }:
                connection.execute(
                    """
                    UPDATE relationship_interpretations
                    SET state = 'invalidated'
                    WHERE user_profile_id = ? AND state = 'current'
                    """,
                    (event.user_profile_id,),
                )

        await self._write(operation)

    async def decide_profile_event(
        self,
        *,
        decision_id: str,
        user_profile_id: str,
        target_event_id: str,
        state: str,
        actor: LedgerActor,
        recorded_at: datetime,
    ) -> None:
        allowed_states = {"confirmed", "rejected", "undone", "disabled", "active"}
        if state not in allowed_states:
            raise ValidationError("Profile判断状態が不正です。")
        if actor is not LedgerActor.USER and state in {
            "confirmed",
            "rejected",
            "disabled",
            "active",
        }:
            raise ValidationError("このProfile操作は利用者だけが実行できます。")

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM profile_decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["user_profile_id"]) == user_profile_id
                    and str(existing["target_event_id"]) == target_event_id
                    and str(existing["state"]) == state
                    and str(existing["actor"]) == actor.value
                ):
                    return
                raise ValidationError("Profile判断IDは既に使われています。")
            event = connection.execute(
                """
                SELECT * FROM profile_events
                WHERE id = ? AND user_profile_id = ?
                """,
                (target_event_id, user_profile_id),
            ).fetchone()
            if event is None:
                raise ValidationError("Profile項目が見つかりません。")
            current_approval, current_usage = self._profile_event_state(
                connection, target_event_id, ProfileApproval(str(event["approval"]))
            )
            if state in {"confirmed", "rejected"} and (
                current_approval is not ProfileApproval.PENDING_CONFIRMATION
            ):
                raise ValidationError("確認待ちではないProfile項目です。")
            if state == "undone" and current_approval not in {
                ProfileApproval.AUTO_SAVED,
                ProfileApproval.CONFIRMED,
            }:
                raise ValidationError("このProfile項目はUndoできません。")
            if state == "disabled" and current_usage is ProfileUsageState.DISABLED:
                raise ValidationError("Profile項目は既に利用停止中です。")
            if state == "active" and current_usage is ProfileUsageState.ACTIVE:
                raise ValidationError("Profile項目は既に利用中です。")
            connection.execute(
                """
                INSERT INTO profile_decisions(
                    id, user_profile_id, target_event_id, state, actor, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    user_profile_id,
                    target_event_id,
                    state,
                    actor.value,
                    _utc_iso(recorded_at),
                ),
            )
            if state in {"confirmed", "rejected", "undone", "disabled"}:
                connection.execute(
                    """
                    UPDATE relationship_interpretations
                    SET state = 'invalidated'
                    WHERE user_profile_id = ? AND state = 'current'
                    """,
                    (user_profile_id,),
                )

        await self._write(operation)

    async def project_profile(
        self,
        user_profile_id: str,
        *,
        character_ids: tuple[str, ...] = (),
        include_disabled: bool = False,
    ) -> tuple[ProfileItem, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[ProfileItem, ...]:
            rows = connection.execute(
                """
                SELECT * FROM profile_events
                WHERE user_profile_id = ?
                ORDER BY recorded_at, rowid
                """,
                (user_profile_id,),
            ).fetchall()
            projected: list[ProfileItem] = []
            active_events: list[ProfileEvent] = []
            states: dict[
                str, tuple[ProfileApproval, ProfileUsageState]
            ] = {}
            for row in rows:
                event = self._profile_event_from_row(connection, row)
                approval, usage = self._profile_event_state(
                    connection, event.id, event.approval
                )
                states[event.id] = (approval, usage)
                if approval in {
                    ProfileApproval.AUTO_SAVED,
                    ProfileApproval.CONFIRMED,
                }:
                    active_events.append(event)
            superseded = {
                event.supersedes_event_id
                for event in active_events
                if event.supersedes_event_id is not None
            }
            requested = set(character_ids)
            for event in active_events:
                approval, usage = states[event.id]
                if event.id in superseded:
                    continue
                if not include_disabled and usage is ProfileUsageState.DISABLED:
                    continue
                if not self._profile_scope_is_visible(event, requested):
                    continue
                projected.append(
                    ProfileItem(
                        event_id=event.id,
                        user_profile_id=event.user_profile_id,
                        item_kind=event.item_kind,
                        item_name=event.item_name,
                        value=event.value,
                        origin=event.origin,
                        approval=approval,
                        usage=usage,
                        scope=event.scope,
                        known_by_character_ids=event.known_by_character_ids,
                        source_conversation_id=event.source_conversation_id,
                        source_message_id=event.source_message_id,
                        recorded_at=event.recorded_at,
                    )
                )
            return tuple(projected)

        return await self._read(operation)

    async def list_profile_history(
        self, user_profile_id: str, item_kind: str, item_name: str
    ) -> tuple[ProfileItem, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[ProfileItem, ...]:
            rows = connection.execute(
                """
                SELECT * FROM profile_events
                WHERE user_profile_id = ? AND item_kind = ? AND item_name = ?
                ORDER BY recorded_at DESC, rowid DESC
                """,
                (user_profile_id, item_kind, item_name),
            ).fetchall()
            items: list[ProfileItem] = []
            for row in rows:
                event = self._profile_event_from_row(connection, row)
                approval, usage = self._profile_event_state(
                    connection, event.id, event.approval
                )
                items.append(
                    ProfileItem(
                        event_id=event.id,
                        user_profile_id=event.user_profile_id,
                        item_kind=event.item_kind,
                        item_name=event.item_name,
                        value=event.value,
                        origin=event.origin,
                        approval=approval,
                        usage=usage,
                        scope=event.scope,
                        known_by_character_ids=event.known_by_character_ids,
                        source_conversation_id=event.source_conversation_id,
                        source_message_id=event.source_message_id,
                        recorded_at=event.recorded_at,
                    )
                )
            return tuple(items)

        return await self._read(operation)

    async def list_profile_items_for_management(
        self, user_profile_id: str
    ) -> tuple[ProfileItem, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[ProfileItem, ...]:
            rows = connection.execute(
                """
                SELECT * FROM profile_events
                WHERE user_profile_id = ? ORDER BY recorded_at, rowid
                """,
                (user_profile_id,),
            ).fetchall()
            events = [
                self._profile_event_from_row(connection, row) for row in rows
            ]
            superseded = {
                event.supersedes_event_id
                for event in events
                if event.supersedes_event_id is not None
                and self._profile_event_state(
                    connection, event.id, event.approval
                )[0]
                in {ProfileApproval.AUTO_SAVED, ProfileApproval.CONFIRMED}
            }
            items: list[ProfileItem] = []
            for event in events:
                if event.id in superseded:
                    continue
                approval, usage = self._profile_event_state(
                    connection, event.id, event.approval
                )
                if approval in {ProfileApproval.REJECTED, ProfileApproval.UNDONE}:
                    continue
                items.append(
                    ProfileItem(
                        event_id=event.id,
                        user_profile_id=event.user_profile_id,
                        item_kind=event.item_kind,
                        item_name=event.item_name,
                        value=event.value,
                        origin=event.origin,
                        approval=approval,
                        usage=usage,
                        scope=event.scope,
                        known_by_character_ids=event.known_by_character_ids,
                        source_conversation_id=event.source_conversation_id,
                        source_message_id=event.source_message_id,
                        recorded_at=event.recorded_at,
                    )
                )
            return tuple(items)

        return await self._read(operation)

    async def purge_profile(
        self,
        *,
        request_id: str,
        user_profile_id: str,
        actor: LedgerActor,
    ) -> ProfilePurgeReceipt:
        if actor is not LedgerActor.USER:
            raise ValidationError("Profile完全削除は利用者だけが実行できます。")

        def operation(connection: sqlite3.Connection) -> ProfilePurgeReceipt:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM profile_purge_receipts WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["user_profile_id"]) != user_profile_id:
                    raise ValidationError("削除要求IDは既に使われています。")
                return self._profile_purge_receipt_from_row(existing)
            profile = connection.execute(
                "SELECT id FROM user_profiles WHERE id = ?", (user_profile_id,)
            ).fetchone()
            if profile is None:
                raise ValidationError("Profileが見つかりません。")
            event_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM profile_events
                    WHERE user_profile_id = ?
                    """,
                    (user_profile_id,),
                ).fetchone()["count"]
            )
            relationship_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM relationship_events
                    WHERE user_profile_id = ?
                    """,
                    (user_profile_id,),
                ).fetchone()["count"]
            )
            connection.execute(
                """
                INSERT INTO profile_purge_authorizations(request_id, user_profile_id)
                VALUES(?, ?)
                """,
                (request_id, user_profile_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO profile_capture_suppressions(
                    user_profile_id, source_message_id, item_kind, item_name
                )
                SELECT user_profile_id, source_message_id, item_kind, item_name
                FROM profile_events
                WHERE user_profile_id = ? AND source_message_id IS NOT NULL
                """,
                (user_profile_id,),
            )
            connection.execute(
                "DELETE FROM relationship_interpretations WHERE user_profile_id = ?",
                (user_profile_id,),
            )
            connection.execute(
                "DELETE FROM profile_derived_data WHERE user_profile_id = ?",
                (user_profile_id,),
            )
            connection.execute(
                "DELETE FROM relationship_decisions WHERE user_profile_id = ?",
                (user_profile_id,),
            )
            connection.execute(
                """
                DELETE FROM relationship_event_knowledge
                WHERE event_id IN (
                    SELECT id FROM relationship_events WHERE user_profile_id = ?
                )
                """,
                (user_profile_id,),
            )
            connection.execute(
                "DELETE FROM relationship_events WHERE user_profile_id = ?",
                (user_profile_id,),
            )
            connection.execute(
                "DELETE FROM profile_decisions WHERE user_profile_id = ?",
                (user_profile_id,),
            )
            connection.execute(
                """
                DELETE FROM profile_event_scopes
                WHERE event_id IN (
                    SELECT id FROM profile_events WHERE user_profile_id = ?
                )
                """,
                (user_profile_id,),
            )
            connection.execute(
                "DELETE FROM profile_events WHERE user_profile_id = ?",
                (user_profile_id,),
            )
            completed_at = _now()
            connection.execute(
                """
                INSERT INTO profile_purge_receipts(
                    request_id, user_profile_id, deleted_event_count,
                    deleted_relationship_event_count, completed_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    user_profile_id,
                    event_count,
                    relationship_count,
                    completed_at,
                ),
            )
            connection.execute(
                "DELETE FROM profile_purge_authorizations WHERE request_id = ?",
                (request_id,),
            )
            return ProfilePurgeReceipt(
                request_id=request_id,
                user_profile_id=user_profile_id,
                deleted_event_count=event_count,
                deleted_relationship_event_count=relationship_count,
                completed_at=datetime.fromisoformat(completed_at),
            )

        return await self._write(operation)

    async def list_relationship_definitions(
        self,
    ) -> tuple[RelationshipDefinition, ...]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[RelationshipDefinition, ...]:
            rows = connection.execute(
                """
                SELECT * FROM relationship_definitions
                WHERE archived_at IS NULL
                ORDER BY category, group_name, display_name, id
                """
            ).fetchall()
            return tuple(self._relationship_definition_from_row(row) for row in rows)

        return await self._read(operation)

    async def append_relationship_event(
        self, event: RelationshipEvent, actor: LedgerActor
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            if actor is LedgerActor.AI and event.approval is not (
                RelationshipApproval.PENDING_CONFIRMATION
            ):
                raise ValidationError(
                    "初期版ではAIの関係イベントを自動反映できません。"
                )
            existing = connection.execute(
                "SELECT * FROM relationship_events WHERE id = ?", (event.id,)
            ).fetchone()
            if existing is not None:
                if self._relationship_event_from_row(connection, existing) == event:
                    return
                raise ValidationError("関係イベントIDは既に使われています。")
            self._validate_relationship_source(connection, event)
            if (
                actor is LedgerActor.AI
                and event.relationship_definition_id is not None
            ):
                definition = connection.execute(
                    """
                    SELECT caution_tags_json FROM relationship_definitions
                    WHERE id = ?
                    """,
                    (event.relationship_definition_id,),
                ).fetchone()
                if definition is None:
                    raise ValidationError("関係定義が見つかりません。")
                caution_tags = set(
                    self._string_tuple_from_json(
                        str(definition["caution_tags_json"]),
                        "関係定義の注意タグ",
                    )
                )
                if caution_tags.intersection(
                    {
                        "sexual_or_romantic",
                        "power_imbalance",
                        "coercion_or_confinement",
                        "harm_history",
                    }
                ):
                    raise ValidationError("この関係はAIから提案できません。")
            connection.execute(
                """
                INSERT INTO relationship_events(
                    id, continuity_id, user_profile_id, character_id,
                    source_conversation_id, source_branch_id, source_message_id,
                    meaning, severity, evidence_context,
                    evidence_start, evidence_end, reason, approval,
                    policy_version, knowledge_count,
                    relationship_definition_id, assignment_state, role,
                    recorded_at
                ) VALUES(
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    event.id,
                    event.continuity_id,
                    event.user_profile_id,
                    event.character_id,
                    event.source_conversation_id,
                    event.source_branch_id,
                    event.source_message_id,
                    event.meaning.value,
                    event.severity.value,
                    event.evidence_context.value,
                    event.evidence_start,
                    event.evidence_end,
                    event.reason.strip(),
                    event.approval.value,
                    event.policy_version,
                    len(event.known_by_character_ids),
                    event.relationship_definition_id,
                    (
                        event.assignment_state.value
                        if event.assignment_state is not None
                        else None
                    ),
                    event.role,
                    _utc_iso(event.recorded_at),
                ),
            )
            for character_id in event.known_by_character_ids:
                connection.execute(
                    """
                    INSERT INTO relationship_event_knowledge(event_id, character_id)
                    VALUES(?, ?)
                    """,
                    (event.id, character_id),
                )
            if event.approval in {
                RelationshipApproval.AUTO_APPLIED,
                RelationshipApproval.CONFIRMED,
            }:
                connection.execute(
                    """
                    UPDATE relationship_interpretations
                    SET state = 'invalidated'
                    WHERE continuity_id = ? AND user_profile_id = ?
                      AND character_id = ? AND state = 'current'
                    """,
                    (
                        event.continuity_id,
                        event.user_profile_id,
                        event.character_id,
                    ),
                )

        await self._write(operation)

    async def decide_relationship_event(
        self,
        *,
        decision_id: str,
        user_profile_id: str,
        target_event_id: str,
        state: str,
        actor: LedgerActor,
        recorded_at: datetime,
    ) -> None:
        if state not in {"confirmed", "rejected", "undone"}:
            raise ValidationError("関係判断状態が不正です。")
        if actor is not LedgerActor.USER:
            raise ValidationError("関係判断は利用者だけが実行できます。")

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM relationship_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["user_profile_id"]) == user_profile_id
                    and str(existing["target_event_id"]) == target_event_id
                    and str(existing["state"]) == state
                ):
                    return
                raise ValidationError("関係判断IDは既に使われています。")
            event = connection.execute(
                """
                SELECT approval FROM relationship_events
                WHERE id = ? AND user_profile_id = ?
                """,
                (target_event_id, user_profile_id),
            ).fetchone()
            if event is None:
                raise ValidationError("関係イベントが見つかりません。")
            current = self._relationship_event_approval(
                connection,
                target_event_id,
                RelationshipApproval(str(event["approval"])),
            )
            if state in {"confirmed", "rejected"} and current is not (
                RelationshipApproval.PENDING_CONFIRMATION
            ):
                raise ValidationError("確認待ちではない関係イベントです。")
            if state == "undone" and current not in {
                RelationshipApproval.AUTO_APPLIED,
                RelationshipApproval.CONFIRMED,
            }:
                raise ValidationError("この関係イベントはUndoできません。")
            connection.execute(
                """
                INSERT INTO relationship_decisions(
                    id, user_profile_id, target_event_id, state, actor, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    user_profile_id,
                    target_event_id,
                    state,
                    actor.value,
                    _utc_iso(recorded_at),
                ),
            )
            connection.execute(
                """
                UPDATE relationship_interpretations
                SET state = 'invalidated'
                WHERE user_profile_id = ? AND state = 'current'
                  AND evidence_event_ids_json LIKE ?
                """,
                (user_profile_id, f'%"{target_event_id}"%'),
            )

        await self._write(operation)

    async def list_relationship_events(
        self,
        continuity_id: str,
        user_profile_id: str,
        character_id: str,
        *,
        visible_to_character_ids: tuple[str, ...] = (),
        include_unapplied: bool = False,
    ) -> tuple[RelationshipEvent, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[RelationshipEvent, ...]:
            rows = connection.execute(
                """
                SELECT * FROM relationship_events
                WHERE continuity_id = ? AND user_profile_id = ? AND character_id = ?
                ORDER BY recorded_at, rowid
                """,
                (continuity_id, user_profile_id, character_id),
            ).fetchall()
            requested = set(visible_to_character_ids)
            selected: list[RelationshipEvent] = []
            for row in rows:
                item = self._relationship_event_from_row(connection, row)
                approval = self._relationship_event_approval(
                    connection, item.id, item.approval
                )
                if not include_unapplied and approval not in {
                    RelationshipApproval.AUTO_APPLIED,
                    RelationshipApproval.CONFIRMED,
                }:
                    continue
                if (
                    requested
                    and item.known_by_character_ids
                    and not requested.issubset(item.known_by_character_ids)
                ):
                    continue
                selected.append(replace(item, approval=approval))
            return tuple(selected)

        return await self._read(operation)

    async def project_relationship_metrics(
        self, continuity_id: str, user_profile_id: str, character_id: str
    ) -> RelationshipMetrics:
        events = await self.list_relationship_events(
            continuity_id, user_profile_id, character_id
        )
        return reduce_relationship_events(events)

    async def save_relationship_interpretation(
        self, interpretation: RelationshipInterpretation, actor: LedgerActor
    ) -> None:
        if actor not in {LedgerActor.AI, LedgerActor.SYSTEM}:
            raise ValidationError("関係解釈は再評価処理だけが保存できます。")
        if interpretation.state not in {
            RelationshipInterpretationState.CURRENT,
            RelationshipInterpretationState.RECOMPUTING,
        }:
            raise ValidationError("新しい関係解釈の状態が不正です。")

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            evidence_ids = interpretation.evidence_event_ids
            if evidence_ids:
                placeholders = ",".join("?" for _ in evidence_ids)
                rows = connection.execute(
                    f"""
                    SELECT id FROM relationship_events
                    WHERE continuity_id = ? AND user_profile_id = ?
                      AND character_id = ? AND id IN ({placeholders})
                    """,
                    (
                        interpretation.continuity_id,
                        interpretation.user_profile_id,
                        interpretation.character_id,
                        *evidence_ids,
                    ),
                ).fetchall()
                if {str(row["id"]) for row in rows} != set(evidence_ids):
                    raise ValidationError("関係解釈の根拠が台帳にありません。")
            existing = connection.execute(
                "SELECT * FROM relationship_interpretations WHERE id = ?",
                (interpretation.id,),
            ).fetchone()
            if existing is not None:
                if self._relationship_interpretation_from_row(existing) == interpretation:
                    return
                raise ValidationError("関係解釈IDは既に使われています。")
            connection.execute(
                """
                UPDATE relationship_interpretations
                SET state = 'superseded'
                WHERE continuity_id = ? AND user_profile_id = ?
                  AND character_id = ? AND state = 'current'
                """,
                (
                    interpretation.continuity_id,
                    interpretation.user_profile_id,
                    interpretation.character_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO relationship_interpretations(
                    id, continuity_id, user_profile_id, character_id,
                    character_version_id, relationship_definition_ids_json,
                    summary, evidence_event_ids_json, state, generated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interpretation.id,
                    interpretation.continuity_id,
                    interpretation.user_profile_id,
                    interpretation.character_id,
                    interpretation.character_version_id,
                    json.dumps(
                        interpretation.relationship_definition_ids,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    interpretation.summary.strip(),
                    json.dumps(
                        interpretation.evidence_event_ids,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    interpretation.state.value,
                    _utc_iso(interpretation.generated_at),
                ),
            )

        await self._write(operation)

    async def get_current_relationship_interpretation(
        self, continuity_id: str, user_profile_id: str, character_id: str
    ) -> RelationshipInterpretation | None:
        def operation(
            connection: sqlite3.Connection,
        ) -> RelationshipInterpretation | None:
            row = connection.execute(
                """
                SELECT * FROM relationship_interpretations
                WHERE continuity_id = ? AND user_profile_id = ?
                  AND character_id = ? AND state = 'current'
                """,
                (continuity_id, user_profile_id, character_id),
            ).fetchone()
            return (
                self._relationship_interpretation_from_row(row)
                if row is not None
                else None
            )

        return await self._read(operation)

    async def append_canonical_memory_event(
        self, event: CanonicalMemoryEvent
    ) -> None:
        await self.append_canonical_memory_events((event,))

    async def append_canonical_memory_events(
        self, events: tuple[CanonicalMemoryEvent, ...]
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            for event in events:
                self._append_canonical_memory_event_in_transaction(connection, event)

        await self._write(operation)

    async def append_captured_memory_events(
        self, events: tuple[CanonicalMemoryEvent, ...]
    ) -> tuple[str, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[str, ...]:
            single_keys: set[
                tuple[str, str, str, MemoryKind, str, frozenset[str]]
            ] = set()
            for event in events:
                if event.cardinality is not MemoryCardinality.SINGLE:
                    continue
                key = (
                    event.conversation_id,
                    event.branch_id,
                    event.subject_id,
                    event.kind,
                    event.slot,
                    event.known_by_character_ids,
                )
                if key in single_keys:
                    raise ValidationError(
                        "one message cannot capture multiple values for one single slot"
                    )
                single_keys.add(key)

            persisted_ids: list[str] = []
            for event in events:
                persisted_id = self._append_captured_memory_event_in_transaction(
                    connection, event
                )
                if persisted_id is not None:
                    persisted_ids.append(persisted_id)
            return tuple(persisted_ids)

        return await self._write(operation)

    async def append_memory_approval_decision(
        self, decision: MemoryApprovalDecision
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            target_row = connection.execute(
                "SELECT conversation_id FROM canonical_memory_events WHERE id = ?",
                (decision.target_event_id,),
            ).fetchone()
            if target_row is None:
                raise ValidationError("判断対象の正史記憶が見つかりません。")
            conversation_id = str(target_row["conversation_id"])
            source = self._require_message(connection, decision.source_message_id)
            if source.conversation_id != conversation_id:
                raise ValidationError("記憶判断の出典が別の会話に属しています。")
            events, decisions = self._load_canonical_memory_stream(
                connection, conversation_id
            )
            duplicate = next((item for item in decisions if item.id == decision.id), None)
            if duplicate is not None:
                if duplicate == decision:
                    return
                raise ValidationError("同じIDの記憶判断が既に存在します。")
            self._validate_memory_ledger(events, (*decisions, decision))
            connection.execute(
                """
                INSERT INTO canonical_memory_decisions(
                    id, conversation_id, target_event_id, state,
                    source_message_id, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    conversation_id,
                    decision.target_event_id,
                    decision.state.value,
                    decision.source_message_id,
                    _utc_iso(decision.recorded_at),
                ),
            )

        await self._write(operation)

    async def decide_canonical_memory(
        self,
        conversation_id: str,
        branch_id: str,
        target_event_id: str,
        state: MemoryApprovalState,
        decision_id: str,
        recorded_at: datetime,
    ) -> MemoryApprovalDecision:
        allowed_states = {
            MemoryApprovalState.CONFIRMED,
            MemoryApprovalState.REJECTED,
            MemoryApprovalState.UNDONE,
        }
        if state not in allowed_states:
            raise ValidationError("unsupported memory decision state")
        if recorded_at.tzinfo is None:
            raise ValidationError("memory decision timestamp must be timezone-aware")

        def operation(connection: sqlite3.Connection) -> MemoryApprovalDecision:
            self._require_conversation(connection, conversation_id)
            branch = connection.execute(
                "SELECT 1 FROM branches WHERE id = ? AND conversation_id = ?",
                (branch_id, conversation_id),
            ).fetchone()
            if branch is None:
                raise ValidationError("memory decision branch is not in the conversation")
            branch_lineage = self._branch_lineage(
                connection, conversation_id, branch_id
            )
            visible_source_ids = {
                message.id for message in self._message_path(connection, branch_id)
            }
            events, decisions = self._load_canonical_memory_stream(
                connection, conversation_id
            )
            target = next(
                (item for item in events if item.id == target_event_id), None
            )
            if target is None:
                raise ValidationError("memory decision target does not exist")
            if (
                target.branch_id not in branch_lineage
                or target.source_message_id not in visible_source_ids
            ):
                raise ValidationError(
                    "memory decision target is outside the selected branch path"
                )

            duplicate = next(
                (item for item in decisions if item.id == decision_id), None
            )
            if duplicate is not None:
                same_command = (
                    duplicate.target_event_id == target.id
                    and duplicate.state is state
                    and duplicate.source_message_id == target.source_message_id
                )
                target_decisions = tuple(
                    item for item in decisions if item.target_event_id == target.id
                )
                if same_command and target_decisions[-1].id == duplicate.id:
                    return duplicate
                raise ValidationError("memory decision command is stale or conflicting")

            if state is MemoryApprovalState.UNDONE:
                decision_states = {item.id: item.approval for item in events}
                for item in decisions:
                    decision_states[item.target_event_id] = item.state
                visible_events = tuple(
                    item
                    for item in events
                    if item.branch_id in branch_lineage
                    and item.source_message_id in visible_source_ids
                )
                accepted_events = tuple(
                    item
                    for item in visible_events
                    if decision_states[item.id]
                    in {
                        MemoryApprovalState.AUTO_SAVED,
                        MemoryApprovalState.CONFIRMED,
                    }
                )
                superseded_ids = transitive_superseded_event_ids(
                    accepted_events, visible_events
                )
                if target.id in superseded_ids:
                    raise ValidationError(
                        "memory decision target is no longer active"
                    )

            decision = MemoryApprovalDecision(
                id=decision_id,
                target_event_id=target.id,
                state=state,
                source_message_id=target.source_message_id,
                recorded_at=max(recorded_at, target.recorded_at),
            )
            self._validate_memory_ledger(events, (*decisions, decision))
            connection.execute(
                """
                INSERT INTO canonical_memory_decisions(
                    id, conversation_id, target_event_id, state,
                    source_message_id, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    conversation_id,
                    decision.target_event_id,
                    decision.state.value,
                    decision.source_message_id,
                    _utc_iso(decision.recorded_at),
                ),
            )
            return decision

        return await self._write(operation)

    async def project_canonical_memory(
        self,
        conversation_id: str,
        branch_id: str,
        speaker_character_id: str,
        current_source_message_id: str | None = None,
        include_historical: bool = False,
    ) -> tuple[CanonicalMemoryFact, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[CanonicalMemoryFact, ...]:
            self._require_conversation(connection, conversation_id)
            speaker = connection.execute(
                "SELECT 1 FROM characters WHERE id = ?", (speaker_character_id,)
            ).fetchone()
            if speaker is None:
                raise ValidationError("発言キャラクターが見つかりません。")
            if current_source_message_id is not None:
                source = self._require_message(connection, current_source_message_id)
                if source.conversation_id != conversation_id:
                    raise ValidationError("現在応答の出典が別の会話に属しています。")
            branch_lineage = self._branch_lineage(connection, conversation_id, branch_id)
            visible_source_message_ids = frozenset(
                message.id for message in self._message_path(connection, branch_id)
            )
            events, decisions = self._load_canonical_memory_stream(
                connection, conversation_id
            )
            ledger = self._validate_memory_ledger(events, decisions)
            return ledger.project(
                MemoryProjectionQuery(
                    conversation_id=conversation_id,
                    branch_lineage=branch_lineage,
                    speaker_character_id=speaker_character_id,
                    current_source_message_id=current_source_message_id,
                    include_historical=include_historical,
                    visible_source_message_ids=visible_source_message_ids,
                )
            )

        return await self._read(operation)

    async def list_canonical_memory_review_items(
        self, conversation_id: str, branch_id: str
    ) -> tuple[CanonicalMemoryReviewItem, ...]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[CanonicalMemoryReviewItem, ...]:
            self._require_conversation(connection, conversation_id)
            branch = connection.execute(
                "SELECT 1 FROM branches WHERE id = ? AND conversation_id = ?",
                (branch_id, conversation_id),
            ).fetchone()
            if branch is None:
                raise ValidationError("memory review branch is not in the conversation")
            branch_lineage = self._branch_lineage(
                connection, conversation_id, branch_id
            )
            visible_source_message_ids = frozenset(
                message.id for message in self._message_path(connection, branch_id)
            )
            events, decisions = self._load_canonical_memory_stream(
                connection, conversation_id
            )
            return self._validate_memory_ledger(events, decisions).review(
                conversation_id,
                branch_lineage,
                visible_source_message_ids,
            )

        return await self._read(operation)

    async def remember_explicit_memory(
        self,
        request_id: str,
        conversation_id: str,
        branch_id: str,
        source_message_id: str,
        expected_character_id: str,
        value: str,
        recorded_at: datetime,
    ) -> ExplicitMemoryEvent:
        normalized = value.strip()
        if not request_id.strip() or not expected_character_id.strip():
            raise ValidationError("明示記憶の操作情報が不足しています。")
        if not normalized or len(normalized) > MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS:
            raise ValidationError("明示記憶は1文字以上200文字以内で入力してください。")

        def operation(connection: sqlite3.Connection) -> ExplicitMemoryEvent:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                "SELECT * FROM explicit_memory_events WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._explicit_memory_event_from_row(existing_row)
                expected = (
                    conversation_id,
                    branch_id,
                    source_message_id,
                    expected_character_id,
                    normalized,
                )
                actual = (
                    existing.conversation_id,
                    existing.branch_id,
                    existing.source_message_id,
                    existing.character_id,
                    existing.value,
                )
                if actual != expected:
                    raise ValidationError("同じ明示記憶操作を別の内容へ再利用できません。")
                return existing

            conversation = self._require_conversation(connection, conversation_id)
            if conversation.active_branch_id != branch_id:
                raise ValidationError("表示中の会話分岐が変わりました。")
            settings = self._load_conversation_group_settings(
                connection, conversation_id
            )
            if settings.enabled:
                raise ValidationError("グループ会話では明示記憶を保存できません。")
            character_row = connection.execute(
                "SELECT character_id FROM character_versions WHERE id = ?",
                (conversation.character_version_id,),
            ).fetchone()
            if (
                character_row is None
                or str(character_row["character_id"]) != expected_character_id
            ):
                raise ValidationError("選択中のキャラクターが変わりました。")
            source = self._require_message(connection, source_message_id)
            if (
                source.conversation_id != conversation_id
                or source.role is not MessageRole.USER
                or source.state is not MessageState.COMPLETED
            ):
                raise ValidationError("完了済みの利用者発言だけを記憶できます。")
            visible_ids = {item.id for item in self._message_path(connection, branch_id)}
            if source.id not in visible_ids:
                raise ValidationError("表示中の分岐にない発言は記憶できません。")

            event = ExplicitMemoryEvent(
                id=str(uuid4()),
                request_id=request_id,
                conversation_id=conversation_id,
                branch_id=branch_id,
                source_message_id=source_message_id,
                character_id=expected_character_id,
                value=normalized,
                recorded_at=recorded_at,
            )
            connection.execute(
                """
                INSERT INTO explicit_memory_events(
                    id, request_id, conversation_id, branch_id,
                    source_message_id, character_id, value, recorded_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.request_id,
                    event.conversation_id,
                    event.branch_id,
                    event.source_message_id,
                    event.character_id,
                    event.value,
                    _utc_iso(event.recorded_at),
                ),
            )
            return event

        return await self._write(operation)

    async def undo_explicit_memory(
        self,
        decision_id: str,
        conversation_id: str,
        branch_id: str,
        target_event_id: str,
        expected_character_id: str,
        recorded_at: datetime,
    ) -> ExplicitMemoryDecision:
        if not decision_id.strip() or not target_event_id.strip():
            raise ValidationError("明示記憶のUndo情報が不足しています。")

        def operation(connection: sqlite3.Connection) -> ExplicitMemoryDecision:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                "SELECT * FROM explicit_memory_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._explicit_memory_decision_from_row(existing_row)
                if (
                    existing.conversation_id != conversation_id
                    or existing.target_event_id != target_event_id
                ):
                    raise ValidationError("同じUndo操作を別の記憶へ再利用できません。")
                return existing

            conversation = self._require_conversation(connection, conversation_id)
            if conversation.active_branch_id != branch_id:
                raise ValidationError("表示中の会話分岐が変わりました。")
            settings = self._load_conversation_group_settings(
                connection, conversation_id
            )
            if settings.enabled:
                raise ValidationError("グループ会話では明示記憶を変更できません。")
            character_row = connection.execute(
                "SELECT character_id FROM character_versions WHERE id = ?",
                (conversation.character_version_id,),
            ).fetchone()
            if (
                character_row is None
                or str(character_row["character_id"]) != expected_character_id
            ):
                raise ValidationError("選択中のキャラクターが変わりました。")
            event_row = connection.execute(
                "SELECT * FROM explicit_memory_events WHERE id = ?",
                (target_event_id,),
            ).fetchone()
            if event_row is None:
                raise ValidationError("明示記憶が見つかりません。")
            event = self._explicit_memory_event_from_row(event_row)
            visible_ids = {item.id for item in self._message_path(connection, branch_id)}
            if (
                event.conversation_id != conversation_id
                or event.character_id != expected_character_id
                or event.branch_id not in self._branch_lineage(
                    connection, conversation_id, branch_id
                )
                or event.source_message_id not in visible_ids
            ):
                raise ValidationError("現在の会話から操作できない明示記憶です。")

            prior = connection.execute(
                """
                SELECT * FROM explicit_memory_decisions
                WHERE target_event_id = ?
                """,
                (target_event_id,),
            ).fetchone()
            if prior is not None:
                raise ValidationError("明示記憶はすでに元に戻されています。")
            decision = ExplicitMemoryDecision(
                id=decision_id,
                conversation_id=conversation_id,
                target_event_id=target_event_id,
                recorded_at=recorded_at,
            )
            connection.execute(
                """
                INSERT INTO explicit_memory_decisions(
                    id, conversation_id, target_event_id, state, recorded_at
                ) VALUES(?, ?, ?, 'undone', ?)
                """,
                (
                    decision.id,
                    decision.conversation_id,
                    decision.target_event_id,
                    _utc_iso(decision.recorded_at),
                ),
            )
            return decision

        return await self._write(operation)

    async def project_explicit_memory(
        self,
        conversation_id: str,
        branch_id: str,
        character_id: str,
    ) -> tuple[ExplicitMemoryReviewItem, ...]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[ExplicitMemoryReviewItem, ...]:
            self._require_conversation(connection, conversation_id)
            lineage = self._branch_lineage(connection, conversation_id, branch_id)
            visible_ids = frozenset(
                item.id for item in self._message_path(connection, branch_id)
            )
            events, decisions = self._load_explicit_memory_stream(
                connection, conversation_id
            )
            return ExplicitMemoryLedger(events, decisions).project(
                ExplicitMemoryProjectionQuery(
                    conversation_id,
                    lineage,
                    character_id,
                    visible_ids,
                )
            )

        return await self._read(operation)

    async def start_send(self, conversation_id: str, content: str) -> RunSession:
        if not content.strip():
            raise ValidationError("メッセージを入力してください。")

        def operation(connection: sqlite3.Connection) -> RunSession:
            conversation = self._require_conversation(connection, conversation_id)
            branch = self._require_branch(connection, conversation.active_branch_id)
            return self._insert_run_session(
                connection=connection,
                conversation=conversation,
                branch_id=branch.id,
                parent_message_id=branch.head_message_id,
                user_content=content.strip(),
                source_message_id=None,
            )

        return await self._write(operation)

    async def start_rewrite(
        self,
        conversation_id: str,
        source_message_id: str,
        content: str,
    ) -> RunSession:
        if not content.strip():
            raise ValidationError("書き直したメッセージを入力してください。")

        def operation(connection: sqlite3.Connection) -> RunSession:
            conversation = self._require_conversation(connection, conversation_id)
            source = self._require_message(connection, source_message_id)
            if source.conversation_id != conversation_id or source.role is not MessageRole.USER:
                raise ValidationError("利用者の発言だけを書き直せます。")
            branch_id = str(uuid4())
            now = _now()
            connection.execute(
                """
                INSERT INTO branches(
                    id, conversation_id, parent_branch_id, forked_from_message_id, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
                    conversation_id,
                    conversation.active_branch_id,
                    source_message_id,
                    now,
                ),
            )
            session = self._insert_run_session(
                connection=connection,
                conversation=conversation,
                branch_id=branch_id,
                parent_message_id=source.parent_message_id,
                user_content=content.strip(),
                source_message_id=source_message_id,
            )
            connection.execute(
                "UPDATE conversations SET active_branch_id = ?, updated_at = ? WHERE id = ?",
                (branch_id, now, conversation_id),
            )
            return session

        return await self._write(operation)

    async def start_regenerate(
        self,
        conversation_id: str,
        source_message_id: str,
    ) -> RunSession:
        def operation(connection: sqlite3.Connection) -> RunSession:
            conversation = self._require_conversation(connection, conversation_id)
            source = self._require_message(connection, source_message_id)
            if source.conversation_id != conversation_id or source.role is not MessageRole.ASSISTANT:
                raise ValidationError("AIの発言だけを再生成できます。")
            if source.parent_message_id is None:
                raise ValidationError("再生成元の利用者発言がありません。")
            user_message = self._require_message(connection, source.parent_message_id)
            branch_id = str(uuid4())
            now = _now()
            connection.execute(
                """
                INSERT INTO branches(
                    id, conversation_id, parent_branch_id, forked_from_message_id, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
                    conversation_id,
                    conversation.active_branch_id,
                    source_message_id,
                    now,
                ),
            )
            session = self._insert_run_session(
                connection=connection,
                conversation=conversation,
                branch_id=branch_id,
                parent_message_id=user_message.id,
                user_content=None,
                source_message_id=source_message_id,
                existing_user=user_message,
            )
            connection.execute(
                "UPDATE conversations SET active_branch_id = ?, updated_at = ? WHERE id = ?",
                (branch_id, now, conversation_id),
            )
            return session

        return await self._write(operation)

    async def context_to_message(self, message_id: str) -> list[Message]:
        def operation(connection: sqlite3.Connection) -> list[Message]:
            rows = connection.execute(
                """
                WITH RECURSIVE path AS (
                    SELECT *, 0 AS depth FROM messages WHERE id = ?
                    UNION ALL
                    SELECT parent.*, path.depth + 1
                    FROM messages parent JOIN path ON path.parent_message_id = parent.id
                )
                SELECT * FROM path ORDER BY depth DESC
                """,
                (message_id,),
            ).fetchall()
            return [self._message_from_row(row) for row in rows]

        return await self._read(operation)

    async def prepare_context_summary(
        self,
        conversation_id: str,
        branch_id: str,
        source_message_ids: tuple[str, ...],
        source_hash: str,
        settings_hash: str,
        model: str,
        prompt_version: str,
    ) -> ContextSummaryPreparation:
        if not source_message_ids:
            raise ValidationError("要約対象の発言がありません。")

        def operation(connection: sqlite3.Connection) -> ContextSummaryPreparation:
            conversation = self._require_conversation(connection, conversation_id)
            branch = self._require_branch(connection, branch_id)
            if branch.conversation_id != conversation.id:
                raise ValidationError("別の会話の分岐へ要約を保存できません。")
            placeholders = ",".join("?" for _ in source_message_ids)
            rows = connection.execute(
                f"SELECT id, conversation_id FROM messages WHERE id IN ({placeholders})",
                source_message_ids,
            ).fetchall()
            if len(rows) != len(source_message_ids) or any(
                str(row["conversation_id"]) != conversation.id for row in rows
            ):
                raise ValidationError("別の会話の発言を要約できません。")
            branch_path = self._message_path(connection, branch_id)
            branch_prefix = tuple(
                message.id for message in branch_path[: len(source_message_ids)]
            )
            if branch_prefix != source_message_ids:
                raise ValidationError(
                    "現在の分岐に属する古い連続区間だけを要約できます。"
                )
            existing = connection.execute(
                """
                SELECT * FROM context_summaries
                WHERE conversation_id = ? AND branch_id = ? AND source_hash = ?
                  AND settings_hash = ? AND prompt_version = ? AND state = 'completed'
                  AND source_message_ids_json = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    conversation_id,
                    branch_id,
                    source_hash,
                    settings_hash,
                    prompt_version,
                    json.dumps(source_message_ids, separators=(",", ":")),
                ),
            ).fetchone()
            if existing is not None:
                return ContextSummaryPreparation(
                    self._context_summary_from_row(existing), False
                )
            summary_id = str(uuid4())
            created_at = _now()
            connection.execute(
                """
                INSERT INTO context_summaries(
                    id, conversation_id, branch_id, source_message_ids_json,
                    source_hash, settings_hash, model, prompt_version, content,
                    state, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '', 'pending', ?)
                """,
                (
                    summary_id,
                    conversation_id,
                    branch_id,
                    json.dumps(source_message_ids, separators=(",", ":")),
                    source_hash,
                    settings_hash,
                    model,
                    prompt_version,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM context_summaries WHERE id = ?", (summary_id,)
            ).fetchone()
            assert row is not None
            return ContextSummaryPreparation(self._context_summary_from_row(row), True)

        return await self._write(operation)

    async def mark_context_summary_running(self, summary_id: str) -> ContextSummary:
        def operation(connection: sqlite3.Connection) -> ContextSummary:
            cursor = connection.execute(
                "UPDATE context_summaries SET state = 'running' WHERE id = ? AND state = 'pending'",
                (summary_id,),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("要約試行を開始できません。")
            row = connection.execute(
                "SELECT * FROM context_summaries WHERE id = ?", (summary_id,)
            ).fetchone()
            assert row is not None
            return self._context_summary_from_row(row)

        return await self._write(operation)

    async def finish_context_summary(
        self,
        summary_id: str,
        content: str,
        state: ContextSummaryState,
        error_code: str | None = None,
    ) -> ContextSummary:
        if state not in {ContextSummaryState.COMPLETED, ContextSummaryState.FAILED}:
            raise ValidationError("要約を終端状態にできません。")

        def operation(connection: sqlite3.Connection) -> ContextSummary:
            cursor = connection.execute(
                """
                UPDATE context_summaries
                SET content = ?, state = ?, error_code = ?, completed_at = ?
                WHERE id = ? AND state IN ('pending', 'running')
                """,
                (content, state.value, error_code, _now(), summary_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("要約試行は既に終了しています。")
            row = connection.execute(
                "SELECT * FROM context_summaries WHERE id = ?", (summary_id,)
            ).fetchone()
            assert row is not None
            return self._context_summary_from_row(row)

        return await self._write(operation)

    async def list_context_summaries(
        self, conversation_id: str
    ) -> list[ContextSummary]:
        def operation(connection: sqlite3.Connection) -> list[ContextSummary]:
            rows = connection.execute(
                """
                SELECT * FROM context_summaries
                WHERE conversation_id = ? ORDER BY created_at
                """,
                (conversation_id,),
            ).fetchall()
            return [self._context_summary_from_row(row) for row in rows]

        return await self._read(operation)

    async def checkpoint_response(self, message_id: str, content: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                """
                UPDATE messages SET content = ?, state = 'streaming'
                WHERE id = ? AND state IN ('pending', 'streaming')
                """,
                (content, message_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("生成中ではない発言を更新しようとしました。")
            connection.execute(
                "UPDATE runs SET state = 'running', started_at = COALESCE(started_at, ?) WHERE response_message_id = ?",
                (_now(), message_id),
            )

        await self._write(operation)

    async def finish_response(
        self,
        session: RunSession,
        content: str,
        state: MessageState,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        total_duration_ns: int | None = None,
        generation_duration_ns: int | None = None,
        response_duration_ms: int | None = None,
        error_code: str | None = None,
    ) -> Message:
        terminal_states = {
            MessageState.COMPLETED: RunState.COMPLETED,
            MessageState.CANCELLED: RunState.CANCELLED,
            MessageState.FAILED: RunState.FAILED,
        }
        if state not in terminal_states:
            raise ValidationError("応答を終端状態にできません。")

        def operation(connection: sqlite3.Connection) -> Message:
            now = _now()
            cursor = connection.execute(
                """
                UPDATE messages SET content = ?, state = ?, completed_at = ?
                WHERE id = ? AND state IN ('pending', 'streaming')
                """,
                (content, state.value, now, session.assistant_message.id),
            )
            if cursor.rowcount != 1:
                raise PersistenceError("応答は既に終了しています。")
            connection.execute(
                """
                UPDATE runs
                SET state = ?, started_at = COALESCE(started_at, ?), completed_at = ?,
                    prompt_tokens = ?, output_tokens = ?, total_duration_ns = ?,
                    generation_duration_ns = ?, response_duration_ms = ?, error_code = ?
                WHERE id = ?
                """,
                (
                    terminal_states[state].value,
                    now,
                    now,
                    prompt_tokens,
                    output_tokens,
                    total_duration_ns,
                    generation_duration_ns,
                    response_duration_ms,
                    error_code,
                    session.run.id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (session.assistant_message.id,)
            ).fetchone()
            assert row is not None
            return self._message_from_row(row)

        return await self._write(operation)

    async def finish_turn_batch(
        self, session: RunSession, draft: TurnBatchDraft
    ) -> TurnBatch:
        def operation(connection: sqlite3.Connection) -> TurnBatch:
            connection.execute("BEGIN IMMEDIATE")
            now = _now()
            batch_id = str(uuid4())
            branch = connection.execute(
                """
                SELECT 1 FROM branches
                WHERE id = ? AND conversation_id = ? AND head_message_id = ?
                """,
                (
                    session.branch_id,
                    session.run.conversation_id,
                    session.assistant_message.id,
                ),
            ).fetchone()
            if branch is None:
                raise ValidationError(
                    "turn batch branch does not own the response message"
                )
            cast_rows = connection.execute(
                """
                SELECT character_id FROM conversation_cast_members
                WHERE conversation_id = ? ORDER BY position
                """,
                (session.run.conversation_id,),
            ).fetchall()
            registered_character_ids = tuple(
                str(row["character_id"]) for row in cast_rows
            )
            if draft.formal_character_ids != registered_character_ids:
                raise ValidationError(
                    "turn batch formal characters do not match the registered cast"
                )
            group_settings = self._load_conversation_group_settings(
                connection, session.run.conversation_id
            )
            if (
                not group_settings.enabled
                or draft.mode is not group_settings.mode
                or draft.spotlight_character_id
                != group_settings.spotlight_character_id
            ):
                raise ValidationError(
                    "turn batch mode does not match current group settings"
                )
            fallback_content = "\n\n".join(
                f"{segment.display_name.strip()}: {segment.content.strip()}"
                for segment in draft.segments
            )
            message_cursor = connection.execute(
                """
                UPDATE messages SET content = ?, state = 'completed', completed_at = ?
                WHERE id = ? AND conversation_id = ?
                  AND state IN ('pending', 'streaming')
                """,
                (
                    fallback_content,
                    now,
                    session.assistant_message.id,
                    session.run.conversation_id,
                ),
            )
            if message_cursor.rowcount != 1:
                raise PersistenceError("turn batch response is already terminal")
            run_state = (
                RunState.COMPLETED
                if draft.state is TurnBatchState.COMPLETED
                else RunState.FAILED
            )
            run_cursor = connection.execute(
                """
                UPDATE runs
                SET state = ?, started_at = COALESCE(started_at, ?), completed_at = ?,
                    prompt_tokens = ?, output_tokens = ?, total_duration_ns = ?,
                    generation_duration_ns = ?, response_duration_ms = ?, error_code = ?
                WHERE id = ? AND conversation_id = ?
                  AND request_message_id = ? AND response_message_id = ?
                  AND state IN ('pending', 'running')
                """,
                (
                    run_state.value,
                    now,
                    now,
                    draft.prompt_tokens,
                    draft.output_tokens,
                    draft.total_duration_ns,
                    draft.generation_duration_ns,
                    draft.response_duration_ms,
                    draft.error_code.strip() if draft.error_code is not None else None,
                    session.run.id,
                    session.run.conversation_id,
                    session.user_message.id,
                    session.assistant_message.id,
                ),
            )
            if run_cursor.rowcount != 1:
                raise PersistenceError("turn batch run is already terminal")
            connection.execute(
                """
                INSERT INTO turn_batches(
                    id, conversation_id, branch_id, source_message_id,
                    response_message_id, run_id, model, mode,
                    formal_character_ids_json, guest_ids_json, prompt_version,
                    state, repair_state, error_code, created_at,
                    spotlight_character_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    session.run.conversation_id,
                    session.branch_id,
                    session.user_message.id,
                    session.assistant_message.id,
                    session.run.id,
                    session.run.model,
                    draft.mode.value,
                    json.dumps(draft.formal_character_ids, ensure_ascii=False),
                    json.dumps(draft.guest_ids, ensure_ascii=False),
                    draft.prompt_version.strip(),
                    draft.state.value,
                    draft.repair_state.value,
                    draft.error_code.strip() if draft.error_code is not None else None,
                    now,
                    draft.spotlight_character_id,
                ),
            )
            for position, segment in enumerate(draft.segments):
                connection.execute(
                    """
                    INSERT INTO turn_segments(
                        id, turn_batch_id, position, speaker_kind, speaker_id,
                        display_name, content
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        batch_id,
                        position,
                        segment.speaker_kind.value,
                        segment.speaker_id,
                        segment.display_name.strip(),
                        segment.content.strip(),
                    ),
                )
            return self._load_turn_batch(connection, batch_id)

        return await self._write(operation)

    async def start_turn_batch_regenerate(
        self,
        conversation_id: str,
        source_response_message_id: str,
        expected_active_branch_id: str,
    ) -> RunSession:
        def operation(connection: sqlite3.Connection) -> RunSession:
            conversation = self._require_conversation(connection, conversation_id)
            if conversation.active_branch_id != expected_active_branch_id:
                raise ValidationError("active branch changed before regeneration")

            batch_row = connection.execute(
                "SELECT * FROM turn_batches WHERE response_message_id = ?",
                (source_response_message_id,),
            ).fetchone()
            if batch_row is None:
                raise ValidationError("turn batch was not found for the response")
            if str(batch_row["conversation_id"]) != conversation_id:
                raise ValidationError("turn batch belongs to another conversation")

            source_response = self._require_message(
                connection, source_response_message_id
            )
            source_user = self._require_message(
                connection, str(batch_row["source_message_id"])
            )
            if (
                source_response.conversation_id != conversation_id
                or source_response.role is not MessageRole.ASSISTANT
                or source_response.state is not MessageState.COMPLETED
                or source_response.parent_message_id != source_user.id
                or source_user.conversation_id != conversation_id
                or source_user.role is not MessageRole.USER
                or source_user.state is not MessageState.COMPLETED
            ):
                raise ValidationError("turn batch source messages are inconsistent")

            source_branch = self._require_branch(
                connection, str(batch_row["branch_id"])
            )
            if source_branch.conversation_id != conversation_id:
                raise ValidationError("turn batch branch belongs to another conversation")

            branch_id = str(uuid4())
            now = _now()
            connection.execute(
                """
                INSERT INTO branches(
                    id, conversation_id, parent_branch_id,
                    forked_from_message_id, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
                    conversation_id,
                    source_branch.id,
                    source_response.id,
                    now,
                ),
            )
            session = self._insert_run_session(
                connection=connection,
                conversation=conversation,
                branch_id=branch_id,
                parent_message_id=source_user.id,
                user_content=None,
                source_message_id=source_response.id,
                existing_user=source_user,
            )
            cursor = connection.execute(
                """
                UPDATE conversations SET active_branch_id = ?, updated_at = ?
                WHERE id = ? AND active_branch_id = ?
                """,
                (branch_id, now, conversation_id, expected_active_branch_id),
            )
            if cursor.rowcount != 1:
                raise ValidationError("active branch changed before regeneration")
            return session

        return await self._write(operation)

    async def get_turn_batch_for_response(
        self, response_message_id: str
    ) -> TurnBatch:
        def operation(connection: sqlite3.Connection) -> TurnBatch:
            row = connection.execute(
                "SELECT id FROM turn_batches WHERE response_message_id = ?",
                (response_message_id,),
            ).fetchone()
            if row is None:
                raise ValidationError("turn batch was not found for the response")
            return self._load_turn_batch(connection, str(row["id"]))

        return await self._read(operation)

    async def list_turn_batches_for_responses(
        self, response_message_ids: tuple[str, ...]
    ) -> tuple[TurnBatch, ...]:
        if not response_message_ids:
            return ()

        def operation(connection: sqlite3.Connection) -> tuple[TurnBatch, ...]:
            placeholders = ",".join("?" for _ in response_message_ids)
            batch_rows = connection.execute(
                f"SELECT * FROM turn_batches WHERE response_message_id IN ({placeholders})",
                response_message_ids,
            ).fetchall()
            if not batch_rows:
                return ()
            batch_ids = tuple(str(row["id"]) for row in batch_rows)
            batch_placeholders = ",".join("?" for _ in batch_ids)
            segment_rows = connection.execute(
                f"SELECT * FROM turn_segments WHERE turn_batch_id IN ({batch_placeholders}) "
                "ORDER BY turn_batch_id, position",
                batch_ids,
            ).fetchall()
            segments_by_batch: dict[str, list[sqlite3.Row]] = {
                batch_id: [] for batch_id in batch_ids
            }
            for segment_row in segment_rows:
                segments_by_batch[str(segment_row["turn_batch_id"])].append(
                    segment_row
                )
            return tuple(
                self._turn_batch_from_row(
                    row, tuple(segments_by_batch[str(row["id"])])
                )
                for row in batch_rows
            )

        return await self._read(operation)

    async def save_telemetry_metrics(
        self, run_id: str, metrics: tuple[TelemetryMetric, ...]
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            captured_at = _now()
            for metric in metrics:
                connection.execute(
                    """
                    INSERT INTO telemetry_samples(
                        id, run_id, metric_name, value, unit, source,
                        unavailable_reason, captured_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        run_id,
                        metric.name,
                        metric.value,
                        metric.unit,
                        metric.source,
                        metric.unavailable_reason,
                        captured_at,
                    ),
                )

        await self._write(operation)

    async def save_document(
        self, document: DocumentRecord, chunks: tuple[DocumentChunk, ...]
    ) -> DocumentRecord:
        def operation(connection: sqlite3.Connection) -> DocumentRecord:
            existing = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?",
                (document.content_hash,),
            ).fetchone()
            if existing is not None:
                return self._document_from_row(existing)
            connection.execute(
                """
                INSERT INTO documents(
                    id, title, media_type, content_text, content_hash, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.title,
                    document.media_type,
                    document.content,
                    document.content_hash,
                    document.created_at.isoformat(),
                ),
            )
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO chunks(
                        id, document_id, ordinal, content_text,
                        start_offset, end_offset
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.content,
                        chunk.start_offset,
                        chunk.end_offset,
                    ),
                )
            return document

        return await self._write(operation)

    async def search_chunks(
        self,
        query: str,
        top_k: int,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]:
        def operation(connection: sqlite3.Connection) -> list[RagSearchResult]:
            rows = connection.execute(
                """
                SELECT chunks.*, documents.title AS document_title
                FROM chunks JOIN documents ON documents.id = chunks.document_id
                ORDER BY documents.created_at, chunks.ordinal
                """
            ).fetchall()
            normalized = query.casefold()
            terms = list(dict.fromkeys([normalized, *normalized.split()]))
            scored: list[RagSearchResult] = []
            allowed_ids = set(document_ids) if document_ids is not None else None
            for row in rows:
                if allowed_ids is not None and str(row["document_id"]) not in allowed_ids:
                    continue
                content = str(row["content_text"])
                folded = content.casefold()
                score = float(sum(folded.count(term) for term in terms if term))
                if score <= 0:
                    continue
                scored.append(
                    RagSearchResult(
                        content=content,
                        citation=RagCitation(
                            document_id=str(row["document_id"]),
                            document_title=str(row["document_title"]),
                            chunk_id=str(row["id"]),
                            start_offset=int(row["start_offset"]),
                            end_offset=int(row["end_offset"]),
                        ),
                        score=score,
                    )
                )
            scored.sort(
                key=lambda result: (
                    -result.score,
                    result.citation.document_title,
                    result.citation.start_offset,
                )
            )
            return scored[:top_k]

        return await self._read(operation)

    async def list_documents(self) -> list[DocumentRecord]:
        def operation(connection: sqlite3.Connection) -> list[DocumentRecord]:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY title COLLATE NOCASE, created_at"
            ).fetchall()
            return [self._document_from_row(row) for row in rows]

        return await self._read(operation)

    async def set_conversation_documents(
        self, conversation_id: str, document_ids: tuple[str, ...]
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            self._require_conversation(connection, conversation_id)
            for document_id in document_ids:
                found = connection.execute(
                    "SELECT id FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if found is None:
                    raise ValidationError("選択した参照資料が見つかりません。")
            connection.execute(
                "DELETE FROM conversation_documents WHERE conversation_id = ?",
                (conversation_id,),
            )
            selected_at = _now()
            for document_id in document_ids:
                connection.execute(
                    """
                    INSERT INTO conversation_documents(
                        conversation_id, document_id, selected_at
                    ) VALUES(?, ?, ?)
                    """,
                    (conversation_id, document_id, selected_at),
                )

        await self._write(operation)

    async def list_conversation_documents(
        self, conversation_id: str
    ) -> list[DocumentRecord]:
        def operation(connection: sqlite3.Connection) -> list[DocumentRecord]:
            self._require_conversation(connection, conversation_id)
            rows = connection.execute(
                """
                SELECT documents.* FROM documents
                JOIN conversation_documents
                  ON conversation_documents.document_id = documents.id
                WHERE conversation_documents.conversation_id = ?
                ORDER BY conversation_documents.selected_at, documents.title COLLATE NOCASE
                """,
                (conversation_id,),
            ).fetchall()
            return [self._document_from_row(row) for row in rows]

        return await self._read(operation)

    async def save_run_rag_usage(
        self,
        run_id: str,
        selected_document_count: int,
        results: tuple[RagSearchResult, ...],
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            run = connection.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise ValidationError("参照元を関連付ける実行記録が見つかりません。")
            connection.execute(
                """
                INSERT INTO run_rag_usage(run_id, selected_document_count)
                VALUES(?, ?)
                """,
                (run_id, selected_document_count),
            )
            for rank, result in enumerate(results):
                connection.execute(
                    """
                    INSERT INTO run_citations(run_id, chunk_id, rank, score)
                    VALUES(?, ?, ?, ?)
                    """,
                    (run_id, result.citation.chunk_id, rank, result.score),
                )

        await self._write(operation)

    async def list_message_rag_usage(
        self, message_ids: list[str]
    ) -> dict[str, MessageRagUsage]:
        if not message_ids:
            return {}

        def operation(
            connection: sqlite3.Connection,
        ) -> dict[str, MessageRagUsage]:
            placeholders = ",".join("?" for _ in message_ids)
            rows = connection.execute(
                f"""
                SELECT runs.response_message_id,
                       run_rag_usage.selected_document_count,
                       documents.id AS document_id,
                       documents.title AS document_title, chunks.id AS chunk_id,
                       chunks.start_offset, chunks.end_offset, chunks.content_text
                FROM runs
                JOIN run_rag_usage ON run_rag_usage.run_id = runs.id
                LEFT JOIN run_citations ON run_citations.run_id = runs.id
                LEFT JOIN chunks ON chunks.id = run_citations.chunk_id
                LEFT JOIN documents ON documents.id = chunks.document_id
                WHERE runs.response_message_id IN ({placeholders})
                ORDER BY runs.response_message_id, run_citations.rank
                """,
                tuple(message_ids),
            ).fetchall()
            grouped: dict[str, list[MessageCitation]] = {}
            selected_counts: dict[str, int] = {}
            for row in rows:
                message_id = str(row["response_message_id"])
                selected_counts[message_id] = int(row["selected_document_count"])
                if row["chunk_id"] is None:
                    continue
                grouped.setdefault(message_id, []).append(
                    MessageCitation(
                        message_id=message_id,
                        document_id=str(row["document_id"]),
                        document_title=str(row["document_title"]),
                        chunk_id=str(row["chunk_id"]),
                        start_offset=int(row["start_offset"]),
                        end_offset=int(row["end_offset"]),
                        content=str(row["content_text"]),
                    )
                )
            return {
                message_id: MessageRagUsage(
                    message_id=message_id,
                    selected_document_count=selected_count,
                    citations=tuple(grouped.get(message_id, [])),
                )
                for message_id, selected_count in selected_counts.items()
            }

        return await self._read(operation)

    async def get_latest_telemetry(
        self, conversation_id: str | None = None
    ) -> LatestTelemetry | None:
        def operation(connection: sqlite3.Connection) -> LatestTelemetry | None:
            parameters: tuple[object, ...] = ()
            where = "WHERE state = 'completed'"
            if conversation_id is not None:
                where += " AND conversation_id = ?"
                parameters = (conversation_id,)
            row = connection.execute(
                f"SELECT * FROM runs {where} ORDER BY completed_at DESC LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            metric_rows = connection.execute(
                """
                SELECT metric_name, value, unit, source, unavailable_reason
                FROM telemetry_samples WHERE run_id = ? ORDER BY captured_at, rowid
                """,
                (row["id"],),
            ).fetchall()
            metrics = tuple(
                TelemetryMetric(
                    name=str(metric["metric_name"]),
                    value=(float(metric["value"]) if metric["value"] is not None else None),
                    unit=str(metric["unit"]),
                    source=str(metric["source"]),
                    unavailable_reason=(
                        str(metric["unavailable_reason"])
                        if metric["unavailable_reason"] is not None
                        else None
                    ),
                )
                for metric in metric_rows
            )
            return LatestTelemetry(self._run_from_row(row), metrics)

        return await self._read(operation)

    async def log_event(
        self,
        level: str,
        event_type: str,
        details: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO app_events(id, level, event_type, run_id, details_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    level,
                    event_type,
                    run_id,
                    json.dumps(details, ensure_ascii=False, sort_keys=True),
                    _now(),
                ),
            )

        await self._write(operation)

    def _insert_run_session(
        self,
        connection: sqlite3.Connection,
        conversation: Conversation,
        branch_id: str,
        parent_message_id: str | None,
        user_content: str | None,
        source_message_id: str | None,
        existing_user: Message | None = None,
    ) -> RunSession:
        now = _now()
        if existing_user is None:
            user = Message(
                id=str(uuid4()),
                conversation_id=conversation.id,
                parent_message_id=parent_message_id,
                source_message_id=source_message_id,
                role=MessageRole.USER,
                content=user_content or "",
                state=MessageState.COMPLETED,
                created_at=datetime.fromisoformat(now),
                completed_at=datetime.fromisoformat(now),
            )
            connection.execute(
                """
                INSERT INTO messages(
                    id, conversation_id, parent_message_id, source_message_id,
                    role, content, state, created_at, completed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.id,
                    user.conversation_id,
                    user.parent_message_id,
                    user.source_message_id,
                    user.role.value,
                    user.content,
                    user.state.value,
                    now,
                    now,
                ),
            )
        else:
            user = existing_user
        assistant = Message(
            id=str(uuid4()),
            conversation_id=conversation.id,
            parent_message_id=user.id,
            source_message_id=source_message_id if existing_user is not None else None,
            role=MessageRole.ASSISTANT,
            content="",
            state=MessageState.PENDING,
            created_at=datetime.fromisoformat(now),
        )
        connection.execute(
            """
            INSERT INTO messages(
                id, conversation_id, parent_message_id, source_message_id,
                role, content, state, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assistant.id,
                assistant.conversation_id,
                assistant.parent_message_id,
                assistant.source_message_id,
                assistant.role.value,
                assistant.content,
                assistant.state.value,
                now,
            ),
        )
        profile = self._profile_from_row(
            connection.execute(
                "SELECT * FROM model_profiles WHERE id = ?",
                (conversation.model_profile_id,),
            ).fetchone()
        )
        run = RunRecord(
            id=str(uuid4()),
            conversation_id=conversation.id,
            request_message_id=user.id,
            response_message_id=assistant.id,
            character_version_id=conversation.character_version_id,
            provider=profile.provider,
            model=profile.model_name,
            parameters=profile.parameters,
            state=RunState.PENDING,
        )
        connection.execute(
            """
            INSERT INTO runs(
                id, conversation_id, request_message_id, response_message_id,
                character_version_id, provider, model, parameters_json, state
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.conversation_id,
                run.request_message_id,
                run.response_message_id,
                run.character_version_id,
                run.provider,
                run.model,
                json.dumps(run.parameters, ensure_ascii=False, sort_keys=True),
                run.state.value,
            ),
        )
        connection.execute(
            "UPDATE branches SET head_message_id = ? WHERE id = ?",
            (assistant.id, branch_id),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation.id),
        )
        return RunSession(user, assistant, run, branch_id)

    def _append_captured_memory_event_in_transaction(
        self, connection: sqlite3.Connection, event: CanonicalMemoryEvent
    ) -> str | None:
        if event.supersedes_event_id is not None:
            raise ValidationError(
                "captured memory must not choose its own replacement target"
            )
        self._require_conversation(connection, event.conversation_id)
        branch_lineage = self._branch_lineage(
            connection, event.conversation_id, event.branch_id
        )
        message_path = self._message_path(connection, event.branch_id)
        message_positions = {
            message.id: index for index, message in enumerate(message_path)
        }
        if event.source_message_id not in message_positions:
            raise ValidationError("captured memory source is not in the branch")
        existing_events, decisions = self._load_canonical_memory_stream(
            connection, event.conversation_id
        )
        duplicate = next(
            (item for item in existing_events if item.id == event.id), None
        )
        if duplicate is not None:
            normalized = replace(
                event,
                supersedes_event_id=duplicate.supersedes_event_id,
                recorded_at=duplicate.recorded_at,
            )
            if duplicate == normalized:
                return duplicate.id
            raise ValidationError("canonical memory event id already exists")

        if event.cardinality is MemoryCardinality.MULTIPLE:
            self._append_canonical_memory_event_in_transaction(connection, event)
            return event.id

        matching = tuple(
            item
            for item in existing_events
            if item.branch_id in branch_lineage
            and item.source_message_id in message_positions
            and item.subject_id == event.subject_id
            and item.kind is event.kind
            and item.slot == event.slot
            and item.cardinality is MemoryCardinality.SINGLE
        )
        states = {item.id: item.approval for item in existing_events}
        for decision in decisions:
            states[decision.target_event_id] = decision.state
        unresolved = tuple(
            item
            for item in matching
            if states[item.id] is MemoryApprovalState.PENDING_CONFIRMATION
        )
        if unresolved:
            raise ValidationError(
                "single-slot memory has an unresolved confirmation candidate"
            )
        accepted = tuple(
            item
            for item in matching
            if states[item.id]
            in {MemoryApprovalState.AUTO_SAVED, MemoryApprovalState.CONFIRMED}
        )
        superseded_ids = transitive_superseded_event_ids(accepted, matching)
        active = tuple(item for item in accepted if item.id not in superseded_ids)
        if len(active) > 1:
            raise PersistenceError("single-slot memory has multiple active values")

        replacement_target = active[0] if active else None
        if replacement_target is not None:
            incoming_position = message_positions.get(event.source_message_id)
            target_position = message_positions.get(
                replacement_target.source_message_id
            )
            if incoming_position is None or target_position is None:
                raise ValidationError(
                    "single-slot source is not in the selected branch"
                )
            if incoming_position < target_position:
                return None
            if incoming_position == target_position:
                raise ValidationError(
                    "one source message cannot set two single-slot values"
                )
            if (
                replacement_target.known_by_character_ids
                != event.known_by_character_ids
            ):
                raise ValidationError(
                    "single-slot knowledge scope changes require explicit confirmation"
                )
            if replacement_target.value == event.value:
                return None
            event = replace(
                event,
                supersedes_event_id=replacement_target.id,
                recorded_at=max(event.recorded_at, replacement_target.recorded_at),
            )
        self._append_canonical_memory_event_in_transaction(connection, event)
        return event.id

    def _append_canonical_memory_event_in_transaction(
        self, connection: sqlite3.Connection, event: CanonicalMemoryEvent
    ) -> None:
        self._require_conversation(connection, event.conversation_id)
        branch_lineage = self._branch_lineage(
            connection, event.conversation_id, event.branch_id
        )
        source = self._require_message(connection, event.source_message_id)
        if source.conversation_id != event.conversation_id:
            raise ValidationError("memory source belongs to another conversation")
        branch_message_ids = {
            message.id for message in self._message_path(connection, event.branch_id)
        }
        if event.source_message_id not in branch_message_ids:
            raise ValidationError("memory source is not in the selected branch")

        existing_events, decisions = self._load_canonical_memory_stream(
            connection, event.conversation_id
        )
        duplicate = next(
            (item for item in existing_events if item.id == event.id), None
        )
        if duplicate is not None:
            if duplicate == event:
                return
            raise ValidationError("canonical memory event id already exists")
        if event.supersedes_event_id is not None:
            target = next(
                (
                    item
                    for item in existing_events
                    if item.id == event.supersedes_event_id
                ),
                None,
            )
            if target is not None and (
                target.branch_id not in branch_lineage
                or target.source_message_id not in branch_message_ids
            ):
                raise ValidationError(
                    "memory outside the selected branch path cannot be superseded"
                )
        for character_id in event.known_by_character_ids:
            exists = connection.execute(
                "SELECT 1 FROM characters WHERE id = ?", (character_id,)
            ).fetchone()
            if exists is None:
                raise ValidationError("knowledge scope character does not exist")
        self._validate_memory_ledger((*existing_events, event), decisions)

        connection.execute(
            """
            INSERT INTO canonical_memory_events(
                id, conversation_id, branch_id, subject_id, kind, slot, value,
                cardinality, approval, knowledge_count, source_message_id,
                supersedes_event_id, effective_at, recorded_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.conversation_id,
                event.branch_id,
                event.subject_id,
                event.kind.value,
                event.slot,
                event.value,
                event.cardinality.value,
                event.approval.value,
                len(event.known_by_character_ids),
                event.source_message_id,
                event.supersedes_event_id,
                _utc_iso(event.effective_at),
                _utc_iso(event.recorded_at),
            ),
        )
        connection.executemany(
            """
            INSERT INTO canonical_memory_event_knowledge(event_id, character_id)
            VALUES(?, ?)
            """,
            tuple(
                (event.id, character_id)
                for character_id in sorted(event.known_by_character_ids)
            ),
        )

    def _message_path(self, connection: sqlite3.Connection, branch_id: str) -> list[Message]:
        rows = connection.execute(
            """
            WITH RECURSIVE path AS (
                SELECT messages.*, 0 AS depth
                FROM messages JOIN branches ON branches.head_message_id = messages.id
                WHERE branches.id = ?
                UNION ALL
                SELECT parent.*, path.depth + 1
                FROM messages parent JOIN path ON path.parent_message_id = parent.id
            )
            SELECT * FROM path ORDER BY depth DESC
            """,
            (branch_id,),
        ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def _require_conversation(
        self, connection: sqlite3.Connection, conversation_id: str
    ) -> Conversation:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ? AND archived_at IS NULL",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ConversationNotFound("会話が見つかりません。")
        return self._conversation_from_row(row)

    def _require_branch(self, connection: sqlite3.Connection, branch_id: str) -> BranchInfo:
        row = connection.execute("SELECT * FROM branches WHERE id = ?", (branch_id,)).fetchone()
        if row is None:
            raise PersistenceError("会話の続きが見つかりません。")
        return self._branch_from_row(row)

    def _require_message(self, connection: sqlite3.Connection, message_id: str) -> Message:
        row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            raise ValidationError("発言が見つかりません。")
        return self._message_from_row(row)

    def _branch_lineage(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
        branch_id: str,
    ) -> tuple[str, ...]:
        lineage: list[str] = []
        visited: set[str] = set()
        current_id: str | None = branch_id
        while current_id is not None:
            if current_id in visited:
                raise PersistenceError("会話の分岐関係が循環しています。")
            visited.add(current_id)
            branch = self._require_branch(connection, current_id)
            if branch.conversation_id != conversation_id:
                raise ValidationError("会話に属さない分岐です。")
            lineage.append(branch.id)
            current_id = branch.parent_branch_id
        lineage.reverse()
        return tuple(lineage)

    def _load_canonical_memory_stream(
        self, connection: sqlite3.Connection, conversation_id: str
    ) -> tuple[tuple[CanonicalMemoryEvent, ...], tuple[MemoryApprovalDecision, ...]]:
        event_rows = connection.execute(
            """
            SELECT * FROM canonical_memory_events
            WHERE conversation_id = ? ORDER BY recorded_at, id
            """,
            (conversation_id,),
        ).fetchall()
        knowledge_rows = connection.execute(
            """
            SELECT knowledge.event_id, knowledge.character_id
            FROM canonical_memory_event_knowledge knowledge
            JOIN canonical_memory_events event ON event.id = knowledge.event_id
            WHERE event.conversation_id = ?
            ORDER BY knowledge.event_id, knowledge.character_id
            """,
            (conversation_id,),
        ).fetchall()
        knowledge: dict[str, set[str]] = {}
        for row in knowledge_rows:
            knowledge.setdefault(str(row["event_id"]), set()).add(
                str(row["character_id"])
            )
        for row in event_rows:
            event_id = str(row["id"])
            if len(knowledge.get(event_id, set())) != int(row["knowledge_count"]):
                raise PersistenceError("正史記憶の知識範囲が壊れています。")
        events = tuple(
            CanonicalMemoryEvent(
                id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                branch_id=str(row["branch_id"]),
                subject_id=str(row["subject_id"]),
                kind=MemoryKind(str(row["kind"])),
                slot=str(row["slot"]),
                value=str(row["value"]),
                cardinality=MemoryCardinality(str(row["cardinality"])),
                approval=MemoryApprovalState(str(row["approval"])),
                source_message_id=str(row["source_message_id"]),
                known_by_character_ids=frozenset(knowledge.get(str(row["id"]), set())),
                supersedes_event_id=(
                    str(row["supersedes_event_id"])
                    if row["supersedes_event_id"]
                    else None
                ),
                effective_at=datetime.fromisoformat(str(row["effective_at"])),
                recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
            )
            for row in event_rows
        )
        decision_rows = connection.execute(
            """
            SELECT decision.*
            FROM canonical_memory_decisions decision
            JOIN canonical_memory_events event ON event.id = decision.target_event_id
            WHERE event.conversation_id = ?
            ORDER BY decision.sequence
            """,
            (conversation_id,),
        ).fetchall()
        decisions = tuple(
            MemoryApprovalDecision(
                id=str(row["id"]),
                target_event_id=str(row["target_event_id"]),
                state=MemoryApprovalState(str(row["state"])),
                source_message_id=str(row["source_message_id"]),
                recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
            )
            for row in decision_rows
        )
        return events, decisions

    def _load_explicit_memory_stream(
        self, connection: sqlite3.Connection, conversation_id: str
    ) -> tuple[
        tuple[ExplicitMemoryEvent, ...],
        tuple[ExplicitMemoryDecision, ...],
    ]:
        event_rows = connection.execute(
            """
            SELECT * FROM explicit_memory_events
            WHERE conversation_id = ? ORDER BY recorded_at, id
            """,
            (conversation_id,),
        ).fetchall()
        decision_rows = connection.execute(
            """
            SELECT * FROM explicit_memory_decisions
            WHERE conversation_id = ? ORDER BY sequence
            """,
            (conversation_id,),
        ).fetchall()
        return (
            tuple(self._explicit_memory_event_from_row(row) for row in event_rows),
            tuple(
                self._explicit_memory_decision_from_row(row)
                for row in decision_rows
            ),
        )

    @staticmethod
    def _validate_memory_ledger(
        events: tuple[CanonicalMemoryEvent, ...],
        decisions: tuple[MemoryApprovalDecision, ...],
    ) -> CanonicalMemoryLedger:
        try:
            return CanonicalMemoryLedger(events, decisions)
        except ValueError as error:
            raise ValidationError(str(error)) from error

    def _require_translation(
        self, connection: sqlite3.Connection, translation_id: str
    ) -> Translation:
        row = connection.execute(
            "SELECT * FROM message_translations WHERE id = ?", (translation_id,)
        ).fetchone()
        if row is None:
            raise ValidationError("翻訳結果が見つかりません。")
        return self._translation_from_row(row)

    def _load_turn_batch(
        self, connection: sqlite3.Connection, batch_id: str
    ) -> TurnBatch:
        row = connection.execute(
            "SELECT * FROM turn_batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise ValidationError("turn batch was not found")
        segment_rows = connection.execute(
            "SELECT * FROM turn_segments WHERE turn_batch_id = ? ORDER BY position",
            (batch_id,),
        ).fetchall()
        return self._turn_batch_from_row(row, tuple(segment_rows))

    def _turn_batch_from_row(
        self, row: sqlite3.Row, segment_rows: tuple[sqlite3.Row, ...]
    ) -> TurnBatch:
        batch_id = str(row["id"])
        segments = tuple(
            TurnSegment(
                id=str(segment_row["id"]),
                turn_batch_id=batch_id,
                position=int(segment_row["position"]),
                speaker_kind=TurnSpeakerKind(str(segment_row["speaker_kind"])),
                speaker_id=(
                    str(segment_row["speaker_id"])
                    if segment_row["speaker_id"] is not None
                    else None
                ),
                display_name=str(segment_row["display_name"]),
                content=str(segment_row["content"]),
            )
            for segment_row in segment_rows
        )
        return TurnBatch(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            branch_id=str(row["branch_id"]),
            source_message_id=str(row["source_message_id"]),
            response_message_id=str(row["response_message_id"]),
            run_id=str(row["run_id"]),
            model=str(row["model"]),
            mode=TurnMode(str(row["mode"])),
            formal_character_ids=self._json_string_tuple(
                str(row["formal_character_ids_json"])
            ),
            guest_ids=self._json_string_tuple(str(row["guest_ids_json"])),
            prompt_version=str(row["prompt_version"]),
            state=TurnBatchState(str(row["state"])),
            repair_state=TurnRepairState(str(row["repair_state"])),
            error_code=(
                str(row["error_code"]) if row["error_code"] is not None else None
            ),
            spotlight_character_id=(
                str(row["spotlight_character_id"])
                if row["spotlight_character_id"] is not None
                else None
            ),
            segments=segments,
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _json_string_tuple(value: str) -> tuple[str, ...]:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or any(
            not isinstance(item, str) for item in parsed
        ):
            raise PersistenceError("stored turn batch ids are invalid")
        return tuple(parsed)

    @staticmethod
    def _load_conversation_cast(
        connection: sqlite3.Connection, conversation_id: str
    ) -> ConversationCast:
        rows = connection.execute(
            """
            SELECT ccm.character_id, ccm.character_version_id,
                   c.display_name, ccm.position
            FROM conversation_cast_members ccm
            JOIN characters c ON c.id = ccm.character_id
            WHERE ccm.conversation_id = ?
            ORDER BY ccm.position
            """,
            (conversation_id,),
        ).fetchall()
        if not rows:
            raise PersistenceError("conversation formal cast is empty")
        return ConversationCast(
            conversation_id=conversation_id,
            members=tuple(
                FormalCastMember(
                    character_id=str(row["character_id"]),
                    character_version_id=str(row["character_version_id"]),
                    display_name=str(row["display_name"]),
                    position=int(row["position"]),
                )
                for row in rows
            ),
        )

    def _replace_conversation_cast(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
        character_version_ids: tuple[str, ...],
        now: str,
    ) -> ConversationCast:
        placeholders = ",".join("?" for _ in character_version_ids)
        rows = connection.execute(
            f"""
            SELECT cv.id, cv.character_id, c.display_name
            FROM character_versions cv
            JOIN characters c ON c.id = cv.character_id
            WHERE cv.id IN ({placeholders}) AND c.archived_at IS NULL
            """,
            character_version_ids,
        ).fetchall()
        by_version_id = {str(row["id"]): row for row in rows}
        if len(by_version_id) != len(character_version_ids):
            raise ValidationError("formal cast character version was not found")
        ordered_rows = [
            by_version_id[version_id] for version_id in character_version_ids
        ]
        stable_ids = tuple(str(row["character_id"]) for row in ordered_rows)
        if len(stable_ids) != len(set(stable_ids)):
            raise ValidationError(
                "formal cast cannot contain two versions of the same character"
            )
        connection.execute(
            "DELETE FROM conversation_cast_members WHERE conversation_id = ?",
            (conversation_id,),
        )
        for position, row in enumerate(ordered_rows):
            connection.execute(
                """
                INSERT INTO conversation_cast_members(
                    conversation_id, character_id, character_version_id,
                    position, added_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    str(row["character_id"]),
                    str(row["id"]),
                    position,
                    now,
                ),
            )
        connection.execute(
            """
            UPDATE conversations
            SET character_version_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (character_version_ids[0], now, conversation_id),
        )
        return self._load_conversation_cast(connection, conversation_id)

    @staticmethod
    def _load_conversation_group_settings(
        connection: sqlite3.Connection, conversation_id: str
    ) -> ConversationGroupSettings:
        row = connection.execute(
            """
            SELECT * FROM conversation_group_settings
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise PersistenceError("conversation group settings were not found")
        try:
            mode = TurnMode(str(row["mode"]))
        except ValueError as error:
            raise PersistenceError("stored conversation group mode is invalid") from error
        return ConversationGroupSettings(
            conversation_id=conversation_id,
            enabled=bool(row["enabled"]),
            mode=mode,
            spotlight_character_id=(
                str(row["spotlight_character_id"])
                if row["spotlight_character_id"] is not None
                else None
            ),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    def _validate_profile_event_actor(
        self,
        connection: sqlite3.Connection,
        event: ProfileEvent,
        actor: LedgerActor,
    ) -> None:
        if not event.item_kind.strip() or not event.item_name.strip() or not event.value.strip():
            raise ValidationError("Profile項目は空にできません。")
        if len(set(event.known_by_character_ids)) != len(event.known_by_character_ids):
            raise ValidationError("Profile共有先が重複しています。")
        if event.scope in {ProfileScope.PROFILE_ONLY, ProfileScope.CONTINUITY}:
            if event.known_by_character_ids:
                raise ValidationError("この共有範囲は個別の共有先を持ちません。")
        elif not event.known_by_character_ids:
            raise ValidationError("選択共有には共有先が必要です。")
        if actor is LedgerActor.USER:
            if (
                event.origin is not ProfileOrigin.USER_ASSERTED
                or event.approval is not ProfileApproval.CONFIRMED
            ):
                raise ValidationError("利用者Profileの作成元と判断が一致しません。")
        elif actor is LedgerActor.AI:
            if event.origin is ProfileOrigin.AI_AUTO_SAVED:
                if (
                    event.approval is not ProfileApproval.AUTO_SAVED
                    or event.scope is not ProfileScope.PROFILE_ONLY
                ):
                    raise ValidationError(
                        "AIの自動追記は本人管理の低リスク項目だけです。"
                    )
            elif event.origin is ProfileOrigin.AI_PROPOSED:
                if event.approval is not ProfileApproval.PENDING_CONFIRMATION:
                    raise ValidationError("AI提案は確認待ちにする必要があります。")
            else:
                raise ValidationError("AIは利用者確定Profileを作成できません。")
        else:
            raise ValidationError("Profileイベントの作成者が不正です。")
        if event.supersedes_event_id is not None:
            previous = connection.execute(
                """
                SELECT user_profile_id, item_kind, item_name, origin, scope
                FROM profile_events WHERE id = ?
                """,
                (event.supersedes_event_id,),
            ).fetchone()
            if previous is None:
                raise ValidationError("更新元のProfile項目が見つかりません。")
            if (
                str(previous["user_profile_id"]) != event.user_profile_id
                or str(previous["item_kind"]) != event.item_kind
                or str(previous["item_name"]) != event.item_name
            ):
                raise ValidationError("別のProfile項目を更新元にできません。")
            if actor is not LedgerActor.USER and (
                str(previous["origin"]) == ProfileOrigin.USER_ASSERTED.value
                or (
                    str(previous["scope"]) == ProfileScope.PROFILE_ONLY.value
                    and event.scope is not ProfileScope.PROFILE_ONLY
                )
            ):
                raise ValidationError(
                    "AIは利用者確定値の上書きや共有範囲拡大を実行できません。"
                )

    def _validate_profile_source(
        self, connection: sqlite3.Connection, event: ProfileEvent
    ) -> None:
        profile = connection.execute(
            "SELECT id FROM user_profiles WHERE id = ?", (event.user_profile_id,)
        ).fetchone()
        if profile is None:
            raise ValidationError("Profileが見つかりません。")
        source_values = (
            event.source_conversation_id,
            event.source_branch_id,
            event.source_message_id,
        )
        if all(value is None for value in source_values):
            if event.manual_operation_id is None:
                raise ValidationError("Profile項目には手動操作IDまたは出典が必要です。")
            return
        if any(value is None for value in source_values):
            raise ValidationError("Profileの会話出典が不完全です。")
        assert event.source_conversation_id is not None
        assert event.source_branch_id is not None
        assert event.source_message_id is not None
        continuity = connection.execute(
            """
            SELECT continuity.user_profile_id
            FROM conversations conversation
            JOIN continuities continuity ON continuity.id = conversation.continuity_id
            WHERE conversation.id = ?
            """,
            (event.source_conversation_id,),
        ).fetchone()
        if continuity is None or str(continuity["user_profile_id"]) != event.user_profile_id:
            raise ValidationError("Profile出典が別世界線に属しています。")
        visible = {
            item.id for item in self._message_path(connection, event.source_branch_id)
        }
        if event.source_message_id not in visible:
            raise ValidationError("Profile出典が指定分岐から到達できません。")

    def _validate_relationship_source(
        self, connection: sqlite3.Connection, event: RelationshipEvent
    ) -> None:
        ledger = connection.execute(
            "SELECT 1 FROM continuities WHERE id = ? AND user_profile_id = ?",
            (event.continuity_id, event.user_profile_id),
        ).fetchone()
        if ledger is None:
            raise ValidationError("関係台帳の世界線とProfileが一致しません。")
        character = connection.execute(
            "SELECT id FROM characters WHERE id = ? AND archived_at IS NULL",
            (event.character_id,),
        ).fetchone()
        if character is None:
            raise ValidationError("関係台帳のキャラクターが見つかりません。")
        if len(set(event.known_by_character_ids)) != len(
            event.known_by_character_ids
        ):
            raise ValidationError("関係イベントの知識範囲が重複しています。")
        source_values = (
            event.source_conversation_id,
            event.source_branch_id,
            event.source_message_id,
        )
        if all(value is None for value in source_values):
            if event.evidence_start is not None or event.evidence_end is not None:
                raise ValidationError("手動関係イベントは本文範囲を持てません。")
            return
        if any(value is None for value in source_values):
            raise ValidationError("関係イベントの会話出典が不完全です。")
        assert event.source_conversation_id is not None
        assert event.source_branch_id is not None
        assert event.source_message_id is not None
        conversation = connection.execute(
            "SELECT continuity_id FROM conversations WHERE id = ?",
            (event.source_conversation_id,),
        ).fetchone()
        if conversation is None or str(conversation["continuity_id"]) != event.continuity_id:
            raise ValidationError("関係イベントの出典が別世界線に属しています。")
        source = next(
            (
                item
                for item in self._message_path(connection, event.source_branch_id)
                if item.id == event.source_message_id
            ),
            None,
        )
        if source is None:
            raise ValidationError("関係イベントの出典が指定分岐から到達できません。")
        if (
            event.evidence_start is None
            or event.evidence_end is None
            or event.evidence_start < 0
            or event.evidence_end <= event.evidence_start
            or event.evidence_end > len(source.content)
        ):
            raise ValidationError("関係イベントの根拠範囲が不正です。")

    @staticmethod
    def _profile_scope_is_visible(
        event: ProfileEvent, requested_character_ids: set[str]
    ) -> bool:
        if event.scope is ProfileScope.PROFILE_ONLY:
            return not requested_character_ids
        if event.scope is ProfileScope.CONTINUITY:
            return True
        return bool(requested_character_ids) and requested_character_ids.issubset(
            event.known_by_character_ids
        )

    @staticmethod
    def _profile_event_state(
        connection: sqlite3.Connection,
        event_id: str,
        initial: ProfileApproval,
    ) -> tuple[ProfileApproval, ProfileUsageState]:
        approval = initial
        usage = ProfileUsageState.ACTIVE
        rows = connection.execute(
            "SELECT state FROM profile_decisions WHERE target_event_id = ? ORDER BY sequence",
            (event_id,),
        ).fetchall()
        for row in rows:
            state = str(row["state"])
            if state in {"confirmed", "rejected", "undone"}:
                approval = ProfileApproval(state)
            elif state == "disabled":
                usage = ProfileUsageState.DISABLED
            elif state == "active":
                usage = ProfileUsageState.ACTIVE
        return approval, usage

    @staticmethod
    def _relationship_event_approval(
        connection: sqlite3.Connection,
        event_id: str,
        initial: RelationshipApproval,
    ) -> RelationshipApproval:
        row = connection.execute(
            """
            SELECT state FROM relationship_decisions
            WHERE target_event_id = ? ORDER BY sequence DESC LIMIT 1
            """,
            (event_id,),
        ).fetchone()
        return initial if row is None else RelationshipApproval(str(row["state"]))

    def _profile_event_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ProfileEvent:
        scope_rows = connection.execute(
            """
            SELECT character_id FROM profile_event_scopes
            WHERE event_id = ? ORDER BY character_id
            """,
            (row["id"],),
        ).fetchall()
        scopes = tuple(str(item["character_id"]) for item in scope_rows)
        if len(scopes) != int(row["scope_count"]):
            raise PersistenceError("Profile共有範囲が壊れています。")
        return ProfileEvent(
            id=str(row["id"]),
            user_profile_id=str(row["user_profile_id"]),
            item_kind=str(row["item_kind"]),
            item_name=str(row["item_name"]),
            value=str(row["value"]),
            origin=ProfileOrigin(str(row["origin"])),
            approval=ProfileApproval(str(row["approval"])),
            scope=ProfileScope(str(row["scope"])),
            known_by_character_ids=scopes,
            source_conversation_id=(
                str(row["source_conversation_id"])
                if row["source_conversation_id"] is not None
                else None
            ),
            source_branch_id=(
                str(row["source_branch_id"])
                if row["source_branch_id"] is not None
                else None
            ),
            source_message_id=(
                str(row["source_message_id"])
                if row["source_message_id"] is not None
                else None
            ),
            manual_operation_id=(
                str(row["manual_operation_id"])
                if row["manual_operation_id"] is not None
                else None
            ),
            supersedes_event_id=(
                str(row["supersedes_event_id"])
                if row["supersedes_event_id"] is not None
                else None
            ),
            effective_at=datetime.fromisoformat(str(row["effective_at"])),
            recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
        )

    def _relationship_event_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> RelationshipEvent:
        scope_rows = connection.execute(
            """
            SELECT character_id FROM relationship_event_knowledge
            WHERE event_id = ? ORDER BY character_id
            """,
            (row["id"],),
        ).fetchall()
        knowledge = tuple(str(item["character_id"]) for item in scope_rows)
        if len(knowledge) != int(row["knowledge_count"]):
            raise PersistenceError("関係イベントの知識範囲が壊れています。")
        return RelationshipEvent(
            id=str(row["id"]),
            continuity_id=str(row["continuity_id"]),
            user_profile_id=str(row["user_profile_id"]),
            character_id=str(row["character_id"]),
            source_conversation_id=(
                str(row["source_conversation_id"])
                if row["source_conversation_id"] is not None
                else None
            ),
            source_branch_id=(
                str(row["source_branch_id"])
                if row["source_branch_id"] is not None
                else None
            ),
            source_message_id=(
                str(row["source_message_id"])
                if row["source_message_id"] is not None
                else None
            ),
            meaning=RelationshipMeaning(str(row["meaning"])),
            severity=RelationshipSeverity(str(row["severity"])),
            evidence_context=EvidenceContext(str(row["evidence_context"])),
            evidence_start=(
                int(row["evidence_start"])
                if row["evidence_start"] is not None
                else None
            ),
            evidence_end=(
                int(row["evidence_end"]) if row["evidence_end"] is not None else None
            ),
            reason=str(row["reason"]),
            approval=RelationshipApproval(str(row["approval"])),
            policy_version=str(row["policy_version"]),
            known_by_character_ids=knowledge,
            relationship_definition_id=(
                str(row["relationship_definition_id"])
                if row["relationship_definition_id"] is not None
                else None
            ),
            assignment_state=(
                RelationshipAssignmentState(str(row["assignment_state"]))
                if row["assignment_state"] is not None
                else None
            ),
            role=str(row["role"]) if row["role"] is not None else None,
            recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
        )

    @classmethod
    def _relationship_definition_from_row(
        cls, row: sqlite3.Row
    ) -> RelationshipDefinition:
        return RelationshipDefinition(
            id=str(row["id"]),
            category=str(row["category"]),
            group_name=str(row["group_name"]),
            display_name=str(row["display_name"]),
            direction=RelationshipDirection(str(row["direction"])),
            role_a=str(row["role_a"]) if row["role_a"] is not None else None,
            role_b=str(row["role_b"]) if row["role_b"] is not None else None,
            caution_tags=cls._string_tuple_from_json(
                str(row["caution_tags_json"]), "関係定義の注意タグ"
            ),
            archived_at=_parse_time(row["archived_at"]),
        )

    @classmethod
    def _relationship_interpretation_from_row(
        cls, row: sqlite3.Row
    ) -> RelationshipInterpretation:
        return RelationshipInterpretation(
            id=str(row["id"]),
            continuity_id=str(row["continuity_id"]),
            user_profile_id=str(row["user_profile_id"]),
            character_id=str(row["character_id"]),
            character_version_id=str(row["character_version_id"]),
            relationship_definition_ids=cls._string_tuple_from_json(
                str(row["relationship_definition_ids_json"]), "関係定義"
            ),
            summary=str(row["summary"]),
            evidence_event_ids=cls._string_tuple_from_json(
                str(row["evidence_event_ids_json"]), "関係解釈の根拠"
            ),
            state=RelationshipInterpretationState(str(row["state"])),
            generated_at=datetime.fromisoformat(str(row["generated_at"])),
        )

    @staticmethod
    def _string_tuple_from_json(value: str, label: str) -> tuple[str, ...]:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise PersistenceError(f"{label}が壊れています。")
        return tuple(parsed)

    @staticmethod
    def _continuity_from_row(row: sqlite3.Row) -> Continuity:
        return Continuity(
            id=str(row["id"]),
            display_name=str(row["display_name"]),
            user_profile_id=str(row["user_profile_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            archived_at=_parse_time(row["archived_at"]),
        )

    @staticmethod
    def _user_profile_from_row(row: sqlite3.Row) -> UserProfile:
        return UserProfile(
            id=str(row["id"]),
            display_name=str(row["display_name"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _profile_purge_receipt_from_row(row: sqlite3.Row) -> ProfilePurgeReceipt:
        return ProfilePurgeReceipt(
            request_id=str(row["request_id"]),
            user_profile_id=str(row["user_profile_id"]),
            deleted_event_count=int(row["deleted_event_count"]),
            deleted_relationship_event_count=int(
                row["deleted_relationship_event_count"]
            ),
            completed_at=datetime.fromisoformat(str(row["completed_at"])),
        )

    @staticmethod
    def _character_from_row(row: sqlite3.Row) -> CharacterVersion:
        return CharacterVersion(
            id=str(row["id"]),
            character_id=str(row["character_id"]),
            display_name=str(row["display_name"]),
            version=int(row["version"]),
            system_prompt=str(row["system_prompt"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _profile_from_row(row: sqlite3.Row | None) -> ModelProfile:
        if row is None:
            raise PersistenceError("モデル設定が見つかりません。")
        parameters = json.loads(str(row["parameters_json"]))
        if not isinstance(parameters, dict):
            raise PersistenceError("モデル設定が壊れています。")
        return ModelProfile(
            id=str(row["id"]),
            provider=str(row["provider"]),
            model_name=str(row["model_name"]),
            parameters=parameters,
        )

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        active_branch_id = row["active_branch_id"]
        if active_branch_id is None:
            raise PersistenceError("会話の続きが設定されていません。")
        return Conversation(
            id=str(row["id"]),
            title=str(row["title"]),
            active_branch_id=str(active_branch_id),
            character_version_id=str(row["character_version_id"]),
            model_profile_id=str(row["model_profile_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            archived_at=_parse_time(row["archived_at"]),
            auto_translate=bool(row["auto_translate"]),
            continuity_id=str(row["continuity_id"]),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        return Message(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            parent_message_id=str(row["parent_message_id"]) if row["parent_message_id"] else None,
            source_message_id=str(row["source_message_id"]) if row["source_message_id"] else None,
            role=MessageRole(str(row["role"])),
            content=str(row["content"]),
            state=MessageState(str(row["state"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            completed_at=_parse_time(row["completed_at"]),
        )

    @staticmethod
    def _translation_from_row(row: sqlite3.Row) -> Translation:
        return Translation(
            id=str(row["id"]),
            message_id=str(row["message_id"]),
            source_hash=str(row["source_hash"]),
            target_language=str(row["target_language"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            content=str(row["content"]),
            state=TranslationState(str(row["state"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            completed_at=_parse_time(row["completed_at"]),
            error_code=str(row["error_code"]) if row["error_code"] else None,
            reused_from_id=(
                str(row["reused_from_id"]) if row["reused_from_id"] else None
            ),
        )

    @staticmethod
    def _context_summary_from_row(row: sqlite3.Row) -> ContextSummary:
        source_message_ids = json.loads(str(row["source_message_ids_json"]))
        if not isinstance(source_message_ids, list) or not all(
            isinstance(value, str) for value in source_message_ids
        ):
            raise PersistenceError("要約の対象発言が壊れています。")
        return ContextSummary(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            branch_id=str(row["branch_id"]),
            source_message_ids=tuple(source_message_ids),
            source_hash=str(row["source_hash"]),
            settings_hash=str(row["settings_hash"]),
            model=str(row["model"]),
            prompt_version=str(row["prompt_version"]),
            content=str(row["content"]),
            state=ContextSummaryState(str(row["state"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            completed_at=_parse_time(row["completed_at"]),
            error_code=str(row["error_code"]) if row["error_code"] else None,
        )

    @staticmethod
    def _explicit_memory_event_from_row(row: sqlite3.Row) -> ExplicitMemoryEvent:
        return ExplicitMemoryEvent(
            id=str(row["id"]),
            request_id=str(row["request_id"]),
            conversation_id=str(row["conversation_id"]),
            branch_id=str(row["branch_id"]),
            source_message_id=str(row["source_message_id"]),
            character_id=str(row["character_id"]),
            value=str(row["value"]),
            recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
        )

    @staticmethod
    def _explicit_memory_decision_from_row(
        row: sqlite3.Row,
    ) -> ExplicitMemoryDecision:
        if str(row["state"]) != "undone":
            raise PersistenceError("明示記憶の判断状態が壊れています。")
        return ExplicitMemoryDecision(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            target_event_id=str(row["target_event_id"]),
            recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
        )

    @staticmethod
    def _scheduled_job_from_row(row: sqlite3.Row) -> ScheduledJob:
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise PersistenceError("定期ジョブの入力が壊れています。")
        return ScheduledJob(
            id=str(row["id"]),
            handler_name=str(row["handler_name"]),
            interval_seconds=int(row["interval_seconds"]),
            first_due_at=datetime.fromisoformat(str(row["first_due_at"])),
            payload=cast(dict[str, object], payload),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _job_run_from_row(row: sqlite3.Row) -> JobRun:
        return JobRun(
            id=str(row["id"]),
            job_id=str(row["job_id"]),
            scheduled_for=datetime.fromisoformat(str(row["scheduled_for"])),
            attempt=int(row["attempt"]),
            state=JobRunState(str(row["state"])),
            retry_of_run_id=(
                str(row["retry_of_run_id"]) if row["retry_of_run_id"] else None
            ),
            failure_reason=(
                str(row["failure_reason"]) if row["failure_reason"] else None
            ),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            started_at=_parse_time(row["started_at"]),
            completed_at=_parse_time(row["completed_at"]),
        )

    @staticmethod
    def _agent_run_from_row(row: sqlite3.Row) -> AgentRun:
        allowed_tools_value = json.loads(str(row["allowed_tools_json"]))
        if not isinstance(allowed_tools_value, list) or not all(
            isinstance(value, str) for value in allowed_tools_value
        ):
            raise PersistenceError("Agent runの許可ツールが壊れています。")
        return AgentRun(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            objective=str(row["objective"]),
            allowed_tools=tuple(allowed_tools_value),
            limits=AgentExecutionLimits(
                max_cost_units=int(row["max_cost_units"]),
                max_steps=int(row["max_steps"]),
                max_duration_seconds=float(row["max_duration_seconds"]),
            ),
            state=AgentRunState(str(row["state"])),
            failure_reason=(
                str(row["failure_reason"]) if row["failure_reason"] else None
            ),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            started_at=_parse_time(row["started_at"]),
            completed_at=_parse_time(row["completed_at"]),
        )

    @staticmethod
    def _agent_step_from_row(row: sqlite3.Row) -> AgentStep:
        arguments_value = json.loads(str(row["arguments_json"]))
        if not isinstance(arguments_value, dict):
            raise PersistenceError("Agent stepの引数が壊れています。")
        arguments = {str(key): value for key, value in arguments_value.items()}
        return AgentStep(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            ordinal=int(row["ordinal"]),
            tool_name=str(row["tool_name"]),
            arguments=arguments,
            action_hash=str(row["action_hash"]),
            data_classification=(
                DataClassification(str(row["data_classification"]))
                if row["data_classification"]
                else None
            ),
            effect=(
                AgentToolEffect(str(row["effect"])) if row["effect"] else None
            ),
            destination=(
                Locality(str(row["destination"])) if row["destination"] else None
            ),
            cost_class=(
                CostClass(str(row["cost_class"])) if row["cost_class"] else None
            ),
            cost_units=int(row["cost_units"]) if row["cost_units"] else None,
            state=AgentStepState(str(row["state"])),
            result_size_bytes=(
                int(row["result_size_bytes"])
                if row["result_size_bytes"] is not None
                else None
            ),
            result_sha256=(
                str(row["result_sha256"]) if row["result_sha256"] else None
            ),
            restore_token=(
                str(row["restore_token"]) if row["restore_token"] else None
            ),
            failure_reason=(
                str(row["failure_reason"]) if row["failure_reason"] else None
            ),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            started_at=_parse_time(row["started_at"]),
            completed_at=_parse_time(row["completed_at"]),
        )

    @staticmethod
    def _computer_use_run_from_row(row: sqlite3.Row) -> ComputerUseRun:
        return ComputerUseRun(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            objective=str(row["objective"]),
            observation_id=str(row["observation_id"]),
            plan_hash=str(row["plan_hash"]),
            limits=ComputerUseLimits(
                max_actions=int(row["max_actions"]),
                max_duration_seconds=float(row["max_duration_seconds"]),
                approval_timeout_seconds=float(row["approval_timeout_seconds"]),
            ),
            planned_action_count=int(row["planned_action_count"]),
            state=ComputerUseRunState(str(row["state"])),
            failure_reason=(
                str(row["failure_reason"]) if row["failure_reason"] else None
            ),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            started_at=_parse_time(row["started_at"]),
            completed_at=_parse_time(row["completed_at"]),
        )

    @staticmethod
    def _computer_action_from_row(row: sqlite3.Row) -> ComputerActionAudit:
        return ComputerActionAudit(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            ordinal=int(row["ordinal"]),
            request=ComputerActionRequest(
                id=str(row["request_id"]),
                action_type=ComputerActionType(str(row["action_type"])),
                target_profile_id=str(row["target_profile_id"]),
                text=str(row["input_text"]) if row["input_text"] is not None else None,
            ),
            state=ComputerActionState(str(row["state"])),
            failure_reason=(
                str(row["failure_reason"]) if row["failure_reason"] else None
            ),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            started_at=_parse_time(row["started_at"]),
            completed_at=_parse_time(row["completed_at"]),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        parameters = json.loads(str(row["parameters_json"]))
        if not isinstance(parameters, dict):
            raise PersistenceError("実行記録が壊れています。")
        return RunRecord(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            request_message_id=str(row["request_message_id"]),
            response_message_id=str(row["response_message_id"]),
            character_version_id=str(row["character_version_id"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            parameters=parameters,
            state=RunState(str(row["state"])),
            started_at=_parse_time(row["started_at"]),
            completed_at=_parse_time(row["completed_at"]),
            prompt_tokens=(int(row["prompt_tokens"]) if row["prompt_tokens"] is not None else None),
            output_tokens=(int(row["output_tokens"]) if row["output_tokens"] is not None else None),
            total_duration_ns=(
                int(row["total_duration_ns"]) if row["total_duration_ns"] is not None else None
            ),
            generation_duration_ns=(
                int(row["generation_duration_ns"])
                if row["generation_duration_ns"] is not None
                else None
            ),
            response_duration_ms=(
                int(row["response_duration_ms"])
                if row["response_duration_ms"] is not None
                else None
            ),
            error_code=str(row["error_code"]) if row["error_code"] else None,
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=str(row["id"]),
            title=str(row["title"]),
            media_type=str(row["media_type"]),
            content=str(row["content_text"]),
            content_hash=str(row["content_hash"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _branch_from_row(row: sqlite3.Row) -> BranchInfo:
        return BranchInfo(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            parent_branch_id=str(row["parent_branch_id"]) if row["parent_branch_id"] else None,
            forked_from_message_id=(
                str(row["forked_from_message_id"]) if row["forked_from_message_id"] else None
            ),
            head_message_id=str(row["head_message_id"]) if row["head_message_id"] else None,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            hidden_at=_parse_time(row["hidden_at"]),
        )
