"""Tests for observational memory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.memory import InMemoryOMStorage, ObservationalMemory, OMContext, OMRecord
from pydantic_ai.memory._token_counter import count_message_tokens, default_token_counter, messages_to_text
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = [pytest.mark.anyio]


# --- Token counter tests ---


def test_default_token_counter():
    assert default_token_counter('') == 0
    assert default_token_counter('hello') == 1
    assert default_token_counter('hello world, this is a test') == 6


def test_messages_to_text():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[TextPart(content='Hi there')]),
    ]
    text = messages_to_text(messages)
    assert 'Hello' in text
    assert 'Hi there' in text


def test_count_message_tokens():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello world')]),
    ]
    count = count_message_tokens(messages)
    assert count == len('Hello world') // 4


# --- InMemoryOMStorage tests ---


async def test_storage_record_crud():
    storage = InMemoryOMStorage()
    assert await storage.load_record('t1') is None

    record = OMRecord(thread_id='t1', active_observations='obs1')
    await storage.save_record(record)
    loaded = await storage.load_record('t1')
    assert loaded is not None
    assert loaded.active_observations == 'obs1'

    record.active_observations = 'obs2'
    await storage.save_record(record)
    loaded = await storage.load_record('t1')
    assert loaded is not None
    assert loaded.active_observations == 'obs2'


async def test_storage_messages_crud():
    storage = InMemoryOMStorage()
    assert await storage.load_messages('t1') == []

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[TextPart(content='Hi')]),
    ]
    await storage.save_messages('t1', messages)
    loaded = await storage.load_messages('t1')
    assert len(loaded) == 2

    loaded2 = await storage.load_messages('t1')
    assert loaded is not loaded2


# --- OMContext tests ---


async def test_om_context_basic():
    ctx = OMContext(thread_id='t1')
    assert ctx.thread_id == 't1'
    assert ctx.messages == []
    assert ctx.record is None


# --- History processor tests ---


def _make_function_model(received_messages: list[ModelMessage]) -> FunctionModel:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received_messages.clear()
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Response')])

    async def stream_fn(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        received_messages.clear()
        received_messages.extend(messages)
        yield 'Response'

    return FunctionModel(model_fn, stream_function=stream_fn)


async def test_processor_noop_without_context():
    """When no OMContext is set, the processor is a no-op."""
    storage = InMemoryOMStorage()
    om = ObservationalMemory(
        storage=storage,
        observer_model='test',
        observation_threshold=100,
    )

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
    ]
    result = om.processor(messages)
    assert result is messages


async def test_processor_noop_without_observations():
    """When OMContext exists but has no observations, processor is a no-op."""
    storage = InMemoryOMStorage()
    om = ObservationalMemory(
        storage=storage,
        observer_model='test',
        observation_threshold=100,
    )

    ctx = await om.prepare('t1')
    try:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ]
        result = om.processor(messages)
        assert result is messages
    finally:
        await om.commit(ctx, [])


async def test_processor_injects_observations():
    """When OMContext has observations, processor injects them and filters old messages."""
    storage = InMemoryOMStorage()
    now = datetime.now(tz=timezone.utc)
    record = OMRecord(
        thread_id='t1',
        active_observations='* User likes Python\n* User is working on a project',
        observation_token_count=50,
        last_observed_at=now,
        current_task='Building a web app',
        suggested_continuation='Ask about the tech stack',
    )
    await storage.save_record(record)

    om = ObservationalMemory(
        storage=storage,
        observer_model='test',
        observation_threshold=100_000,
    )

    ctx = await om.prepare('t1')
    try:
        # Create messages: one old (before observation), one new (after observation)
        old_msg = ModelRequest(
            parts=[UserPromptPart(content='Old message')],
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        new_msg = ModelRequest(
            parts=[UserPromptPart(content='New message')],
            timestamp=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )

        result = om.processor([old_msg, new_msg])

        # Old message should be filtered out
        # Result should have: [observation_request, new_msg]
        assert len(result) == 2

        # First message should be the observation injection
        assert isinstance(result[0], ModelRequest)
        obs_parts = result[0].parts
        assert len(obs_parts) == 2
        assert isinstance(obs_parts[0], SystemPromptPart)
        assert '* User likes Python' in obs_parts[0].content
        assert 'Building a web app' in obs_parts[0].content
        assert isinstance(obs_parts[1], SystemPromptPart)
        assert 'Ask about the tech stack' in obs_parts[1].content

        # Second message should be the new message
        assert result[1] is new_msg
    finally:
        await om.commit(ctx, [])


# --- Integration tests with FunctionModel ---


async def test_observational_memory_integration_eager():
    """Test full cycle: prepare -> run -> commit with eager observation.

    Uses a low threshold so the observer triggers on commit.
    """
    received_messages: list[ModelMessage] = []
    function_model = _make_function_model(received_messages)

    # The observer also uses a FunctionModel that returns structured output
    observer_responses: list[str] = []

    def observer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        import json

        response = json.dumps(
            {
                'observations': '* 🔴 (14:30) User said hello\n* 🟡 (14:30) Assistant greeted back',
                'current_task': 'Greeting exchange',
                'suggested_continuation': None,
            }
        )
        observer_responses.append(response)
        return ModelResponse(parts=[TextPart(content=response)])

    observer_fn_model = FunctionModel(observer_model_fn)

    storage = InMemoryOMStorage()
    om = ObservationalMemory(
        storage=storage,
        observer_model=observer_fn_model,
        observation_threshold=1,
        reflection_threshold=100_000,
        observe_on='commit',
    )

    agent = Agent(function_model, history_processors=[om.processor])

    # Turn 1
    ctx = await om.prepare(thread_id='t1')
    result = await agent.run('Hello!', message_history=ctx.messages)
    await om.commit(ctx, result.all_messages())

    # Verify observer was called (threshold is 1 token)
    assert len(observer_responses) == 1

    # Verify observations were stored
    record = await storage.load_record('t1')
    assert record is not None
    assert '🔴' in record.active_observations
    assert record.current_task == 'Greeting exchange'

    # Turn 2: prepare should load observations
    ctx2 = await om.prepare(thread_id='t1')
    assert ctx2.record is not None
    assert ctx2.record.active_observations != ''


async def test_observational_memory_integration_lazy():
    """Test lazy mode: observation happens on prepare, not commit."""
    received_messages: list[ModelMessage] = []
    function_model = _make_function_model(received_messages)

    observer_called = False

    def observer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal observer_called
        import json

        observer_called = True
        response = json.dumps(
            {
                'observations': '* 🔴 (14:30) User said hi',
                'current_task': None,
                'suggested_continuation': None,
            }
        )
        return ModelResponse(parts=[TextPart(content=response)])

    observer_fn_model = FunctionModel(observer_model_fn)

    storage = InMemoryOMStorage()
    om = ObservationalMemory(
        storage=storage,
        observer_model=observer_fn_model,
        observation_threshold=1,
        reflection_threshold=100_000,
        observe_on='prepare',
    )

    agent = Agent(function_model, history_processors=[om.processor])

    # Turn 1: no prior messages, so observer won't trigger
    ctx = await om.prepare(thread_id='t1')
    result = await agent.run('Hi!', message_history=ctx.messages)
    await om.commit(ctx, result.all_messages())
    assert not observer_called

    # Turn 2: now there are messages, prepare should trigger observer
    await om.prepare(thread_id='t1')
    assert observer_called


async def test_reflection_triggers():
    """Test that reflection triggers when observation tokens exceed threshold."""
    observer_call_count = 0
    reflector_called = False

    def observer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal observer_call_count
        import json

        observer_call_count += 1
        # Return a large observation to trigger reflection
        long_obs = '* 🔴 observation ' * 200
        response = json.dumps(
            {
                'observations': long_obs,
                'current_task': None,
                'suggested_continuation': None,
            }
        )
        return ModelResponse(parts=[TextPart(content=response)])

    def reflector_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal reflector_called
        import json

        reflector_called = True
        response = json.dumps(
            {
                'compressed_observations': '* 🔴 Compressed summary of observations',
            }
        )
        return ModelResponse(parts=[TextPart(content=response)])

    received: list[ModelMessage] = []
    main_model = _make_function_model(received)

    storage = InMemoryOMStorage()
    om = ObservationalMemory(
        storage=storage,
        observer_model=FunctionModel(observer_model_fn),
        reflector_model=FunctionModel(reflector_model_fn),
        observation_threshold=1,
        reflection_threshold=50,
        observe_on='commit',
    )

    agent = Agent(main_model, history_processors=[om.processor])

    ctx = await om.prepare(thread_id='t1')
    result = await agent.run('Hello', message_history=ctx.messages)
    await om.commit(ctx, result.all_messages())

    assert observer_call_count == 1
    assert reflector_called

    record = await storage.load_record('t1')
    assert record is not None
    assert 'Compressed summary' in record.active_observations


async def test_concurrent_threads_isolation():
    """Two concurrent prepare/commit flows on different threads don't interfere."""
    call_log: list[str] = []

    def observer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        import json

        response = json.dumps(
            {
                'observations': '* obs',
                'current_task': None,
                'suggested_continuation': None,
            }
        )
        return ModelResponse(parts=[TextPart(content=response)])

    received: list[ModelMessage] = []
    main_model = _make_function_model(received)

    storage = InMemoryOMStorage()
    om = ObservationalMemory(
        storage=storage,
        observer_model=FunctionModel(observer_model_fn),
        observation_threshold=1,
        reflection_threshold=100_000,
        observe_on='commit',
    )

    agent = Agent(main_model, history_processors=[om.processor])

    async def run_thread(thread_id: str):
        ctx = await om.prepare(thread_id=thread_id)
        call_log.append(f'{thread_id}:prepare')
        result = await agent.run(f'Hello from {thread_id}', message_history=ctx.messages)
        call_log.append(f'{thread_id}:run')
        await om.commit(ctx, result.all_messages())
        call_log.append(f'{thread_id}:commit')

    await asyncio.gather(run_thread('t1'), run_thread('t2'))

    # Both threads should have completed
    assert 't1:commit' in call_log
    assert 't2:commit' in call_log

    # Each thread should have its own record
    r1 = await storage.load_record('t1')
    r2 = await storage.load_record('t2')
    assert r1 is not None
    assert r2 is not None
    assert r1.thread_id == 't1'
    assert r2.thread_id == 't2'
