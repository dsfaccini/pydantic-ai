"""ObservationalMemory: three-agent compression system for unlimited conversation length."""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from opentelemetry.trace import NoOpTracer, Tracer

from pydantic_ai import models
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from ._observer import OBSERVER_MODEL_SETTINGS, OBSERVER_SYSTEM_PROMPT, build_observer_prompt
from ._reflector import REFLECTOR_MODEL_SETTINGS, REFLECTOR_SYSTEM_PROMPT, build_reflector_prompt
from ._storage import OMStorage
from ._token_counter import TokenCounter, count_message_tokens, default_token_counter
from ._types import ObserverOutput, OMContext, OMRecord, ReflectorOutput

_CURRENT_OM_CONTEXT: ContextVar[OMContext | None] = ContextVar('pydantic_ai.om_context', default=None)

_NOOP_TRACER = NoOpTracer()


@dataclass(frozen=True)
class ObservationalMemory:
    """Stateless observational memory manager.

    Provides a history processor that injects observations into the agent's context,
    and `prepare()`/`commit()` methods that manage observation lifecycle per-thread.
    """

    storage: OMStorage
    observer_model: models.Model | models.KnownModelName | str
    reflector_model: models.Model | models.KnownModelName | str | None = None
    observation_threshold: int = 30000
    reflection_threshold: int = 40000
    observe_on: Literal['commit', 'prepare'] = 'commit'
    token_counter: TokenCounter = default_token_counter
    tracer: Tracer | None = None

    @property
    def _tracer(self) -> Tracer:
        return self.tracer or _NOOP_TRACER

    async def prepare(self, thread_id: str) -> OMContext:
        """Load thread state and set up the ContextVar for the history processor.

        If `observe_on='prepare'`, triggers observation/reflection before returning.
        """
        with self._tracer.start_as_current_span(
            'om.prepare',
            attributes={
                'pydantic_ai.memory.thread_id': thread_id,
                'pydantic_ai.memory.observe_on': self.observe_on,
                'logfire.msg': f'om.prepare {thread_id}',
            },
        ) as span:
            record = await self.storage.load_record(thread_id)
            messages = await self.storage.load_messages(thread_id)

            span.set_attribute('pydantic_ai.memory.has_record', record is not None)
            span.set_attribute('pydantic_ai.memory.message_count', len(messages))
            span.set_attribute(
                'pydantic_ai.memory.observation_tokens',
                record.observation_token_count if record else 0,
            )

            if self.observe_on == 'prepare' and messages:
                record = await self._maybe_observe_and_reflect(record, messages, thread_id)

            ctx = OMContext(
                thread_id=thread_id,
                messages=messages,
                record=record,
            )
            ctx.set_context_var(_CURRENT_OM_CONTEXT)
            return ctx

    async def commit(self, ctx: OMContext, messages: Sequence[ModelMessage]) -> None:
        """Save messages, optionally trigger observation/reflection, and clean up ContextVar."""
        tokens_before = ctx.record.observation_token_count if ctx.record else 0

        with self._tracer.start_as_current_span(
            'om.commit',
            attributes={
                'pydantic_ai.memory.thread_id': ctx.thread_id,
                'pydantic_ai.memory.message_count': len(messages),
                'pydantic_ai.memory.observation_tokens_before': tokens_before,
                'logfire.msg': f'om.commit {ctx.thread_id}',
            },
        ) as span:
            try:
                await self.storage.save_messages(ctx.thread_id, messages)

                observer_triggered = False
                reflector_triggered = False

                if self.observe_on == 'commit' and messages:
                    record_before = ctx.record
                    record_after = await self._maybe_observe_and_reflect(ctx.record, list(messages), ctx.thread_id)

                    observer_triggered = record_after.last_observed_at != (
                        record_before.last_observed_at if record_before else None
                    )
                    reflector_triggered = record_after.generation_count > (
                        (record_before.generation_count if record_before else 0) + (1 if observer_triggered else 0)
                    )
                    tokens_after = record_after.observation_token_count
                else:
                    tokens_after = tokens_before

                span.set_attribute('pydantic_ai.memory.observer_triggered', observer_triggered)
                span.set_attribute('pydantic_ai.memory.reflector_triggered', reflector_triggered)
                span.set_attribute('pydantic_ai.memory.observation_tokens_after', tokens_after)

                if observer_triggered:
                    suffix = ' [reflected]' if reflector_triggered else ' [observed]'
                    span.set_attribute('logfire.msg', f'om.commit {ctx.thread_id}{suffix}')
            finally:
                ctx.reset_context_var(_CURRENT_OM_CONTEXT)

    def processor(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """History processor that injects observations and filters old messages.

        Registered at agent construction: `Agent(..., history_processors=[om.processor])`
        """
        ctx = _CURRENT_OM_CONTEXT.get()
        if ctx is None or ctx.record is None:
            return messages

        record = ctx.record
        if not record.active_observations:
            return messages

        messages_before = len(messages)
        with self._tracer.start_as_current_span(
            'om.processor',
            attributes={
                'pydantic_ai.memory.thread_id': ctx.thread_id,
                'pydantic_ai.memory.has_observations': True,
                'pydantic_ai.memory.messages_before': messages_before,
            },
        ) as span:
            observation_text = (
                f'=== MEMORY (Observations from previous interactions) ===\n\n{record.active_observations}'
            )
            if record.current_task:
                observation_text += f'\n\nCurrent task: {record.current_task}'
            observation_text += '\n\n=== END MEMORY ===\n\nUse these observations as context. Do not mention them explicitly unless asked.'

            observation_part = SystemPromptPart(content=observation_text)

            if record.last_observed_at is not None:
                messages = [m for m in messages if _message_after(m, record.last_observed_at)]

            if record.suggested_continuation:
                continuation_part = SystemPromptPart(content=f'Suggested continuation: {record.suggested_continuation}')
                result = [
                    ModelRequest(parts=[observation_part, continuation_part]),
                    *messages,
                ]
            else:
                result = [ModelRequest(parts=[observation_part]), *messages]

            span.set_attribute('pydantic_ai.memory.messages_after', len(result))
            span.set_attribute('pydantic_ai.memory.messages_filtered', messages_before - len(messages))
            span.set_attribute('logfire.msg', 'om.processor')
            return result

    async def _maybe_observe_and_reflect(
        self,
        record: OMRecord | None,
        messages: list[ModelMessage],
        thread_id: str,
    ) -> OMRecord:
        """Run observation and reflection if thresholds are exceeded."""
        if record is None:
            record = OMRecord(thread_id=thread_id)

        unobserved = _messages_after(messages, record.last_observed_at)
        unobserved_tokens = count_message_tokens(unobserved, self.token_counter)

        if unobserved_tokens >= self.observation_threshold:
            record = await self._run_observer(record, unobserved)

        if record.observation_token_count >= self.reflection_threshold:
            record = await self._run_reflector(record)

        return record

    async def _run_observer(self, record: OMRecord, unobserved: list[ModelMessage]) -> OMRecord:
        """Run the observer agent to extract observations from unobserved messages."""
        observer = Agent(
            self.observer_model,
            name='om.observer',
            system_prompt=OBSERVER_SYSTEM_PROMPT,
            output_type=ObserverOutput,
            model_settings=OBSERVER_MODEL_SETTINGS,
        )

        prompt = build_observer_prompt(
            existing_observations=record.active_observations or None,
            messages_to_observe=unobserved,
        )

        result = await observer.run(prompt)
        output = result.output

        new_observations = output.observations
        if record.active_observations:
            combined = f'{record.active_observations}\n\n{new_observations}'
        else:
            combined = new_observations

        observed_at = _max_message_timestamp(unobserved)
        updated = replace(
            record,
            active_observations=combined,
            observation_token_count=self.token_counter(combined),
            last_observed_at=observed_at,
            generation_count=record.generation_count + 1,
            current_task=output.current_task or record.current_task,
            suggested_continuation=output.suggested_continuation,
        )
        await self.storage.save_record(updated)
        return updated

    async def _run_reflector(self, record: OMRecord) -> OMRecord:
        """Run the reflector agent to compress observations."""
        reflector_model = self.reflector_model or self.observer_model
        reflector = Agent(
            reflector_model,
            name='om.reflector',
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            output_type=ReflectorOutput,
            model_settings=REFLECTOR_MODEL_SETTINGS,
        )

        prompt = build_reflector_prompt(record.active_observations)
        result = await reflector.run(prompt)
        output = result.output

        updated = replace(
            record,
            active_observations=output.compressed_observations,
            observation_token_count=self.token_counter(output.compressed_observations),
            generation_count=record.generation_count + 1,
        )
        await self.storage.save_record(updated)
        return updated


def _max_message_timestamp(messages: list[ModelMessage]) -> datetime:
    """Return the latest timestamp from a list of messages, falling back to now."""
    timestamps = [m.timestamp for m in messages if isinstance(m, ModelRequest) and m.timestamp is not None]
    return max(timestamps) if timestamps else datetime.now()


def _message_after(msg: ModelMessage, cutoff: datetime) -> bool:
    """Check if a message was sent after the cutoff time."""
    if isinstance(msg, ModelRequest):
        return msg.timestamp is None or msg.timestamp > cutoff
    return True


def _messages_after(messages: list[ModelMessage], cutoff: datetime | None) -> list[ModelMessage]:
    """Filter messages to only those after the cutoff time."""
    if cutoff is None:
        return messages
    return [m for m in messages if _message_after(m, cutoff)]
