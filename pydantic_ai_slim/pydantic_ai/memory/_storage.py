from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic_ai.messages import ModelMessage

from ._types import OMRecord


@runtime_checkable
class OMStorage(Protocol):
    """Protocol for observational memory persistence."""

    async def load_record(self, thread_id: str) -> OMRecord | None:
        """Load the observation record for a thread, or None if not found."""
        ...

    async def save_record(self, record: OMRecord) -> None:
        """Save or update the observation record."""
        ...

    async def load_messages(self, thread_id: str) -> list[ModelMessage]:
        """Load the full message history for a thread."""
        ...

    async def save_messages(self, thread_id: str, messages: Sequence[ModelMessage]) -> None:
        """Save/replace the message history for a thread."""
        ...


class InMemoryOMStorage:
    """In-memory implementation of OMStorage for testing."""

    def __init__(self) -> None:
        self._records: dict[str, OMRecord] = {}
        self._messages: dict[str, list[ModelMessage]] = {}

    async def load_record(self, thread_id: str) -> OMRecord | None:
        return self._records.get(thread_id)

    async def save_record(self, record: OMRecord) -> None:
        self._records[record.thread_id] = record

    async def load_messages(self, thread_id: str) -> list[ModelMessage]:
        return list(self._messages.get(thread_id, []))

    async def save_messages(self, thread_id: str, messages: Sequence[ModelMessage]) -> None:
        self._messages[thread_id] = list(messages)
