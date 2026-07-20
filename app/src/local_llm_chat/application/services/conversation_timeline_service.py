from __future__ import annotations

from dataclasses import dataclass

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.group_turns import TurnBatch, TurnSegment
from local_llm_chat.domain.models import Message
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MessageRole


@dataclass(frozen=True, slots=True)
class ConversationTimelineItem:
    message: Message
    turn_batch: TurnBatch | None = None
    segment: TurnSegment | None = None
    is_batch_end: bool = True


class ConversationTimelineService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository

    async def list_items(
        self, conversation_id: str
    ) -> tuple[ConversationTimelineItem, ...]:
        messages = await self._repository.list_active_messages(conversation_id)
        response_ids = tuple(
            message.id
            for message in messages
            if message.role is MessageRole.ASSISTANT
        )
        batches = await self._repository.list_turn_batches_for_responses(response_ids)
        batches_by_response = {batch.response_message_id: batch for batch in batches}
        items: list[ConversationTimelineItem] = []
        for message in messages:
            batch = batches_by_response.get(message.id)
            if batch is None:
                items.append(ConversationTimelineItem(message))
                continue
            if batch.conversation_id != conversation_id:
                raise ValidationError("turn batch belongs to another conversation")
            if not batch.segments:
                items.append(ConversationTimelineItem(message))
                continue
            last_position = len(batch.segments) - 1
            items.extend(
                ConversationTimelineItem(
                    message=message,
                    turn_batch=batch,
                    segment=segment,
                    is_batch_end=position == last_position,
                )
                for position, segment in enumerate(batch.segments)
            )
        return tuple(items)
