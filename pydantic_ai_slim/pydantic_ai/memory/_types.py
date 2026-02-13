from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from pydantic_ai.messages import ModelMessage


@dataclass
class OMRecord:
    """Persistent observation state for a single thread."""

    thread_id: str
    active_observations: str = ''
    observation_token_count: int = 0
    last_observed_at: datetime | None = None
    generation_count: int = 0
    current_task: str | None = None
    suggested_continuation: str | None = None


@dataclass
class OMContext:
    """Per-run state returned by `ObservationalMemory.prepare()`.

    Carries the loaded history and observation record for a single agent run,
    plus the ContextVar token for cleanup in `commit()`.
    """

    thread_id: str
    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    record: OMRecord | None = None
    _context_token: Token[OMContext | None] | None = field(default=None, repr=False)

    def set_context_var(self, context_var: ContextVar[OMContext | None]) -> None:
        """Set this context as the active OMContext in the given ContextVar."""
        self._context_token = context_var.set(self)

    def reset_context_var(self, context_var: ContextVar[OMContext | None]) -> None:
        """Reset the ContextVar to its previous value."""
        if self._context_token is not None:
            context_var.reset(self._context_token)


class ObserverOutput(BaseModel):
    """Structured output from the Observer agent."""

    observations: str
    current_task: str | None = None
    suggested_continuation: str | None = None


class ReflectorOutput(BaseModel):
    """Structured output from the Reflector agent."""

    compressed_observations: str
