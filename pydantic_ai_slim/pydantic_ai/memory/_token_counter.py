from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

TokenCounter = Callable[[str], int]


def default_token_counter(text: str) -> int:
    """Estimate token count as len(text) // 4."""
    return len(text) // 4


def _part_to_text(part: ModelResponsePart | SystemPromptPart | UserPromptPart | ToolReturnPart) -> str:
    if isinstance(part, (SystemPromptPart, TextPart, ThinkingPart)):
        return part.content
    if isinstance(part, UserPromptPart):
        if isinstance(part.content, str):
            return part.content
        return ' '.join(str(c) for c in part.content)
    if isinstance(part, ToolCallPart):
        return f'{part.tool_name}({part.args})'
    if isinstance(part, ToolReturnPart):
        return str(part.content)
    return ''


def messages_to_text(messages: Sequence[ModelMessage]) -> str:
    """Serialize a list of ModelMessages to plain text for token counting or observer input."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                text = _part_to_text(part)  # pyright: ignore[reportArgumentType]
                if text:
                    lines.append(text)
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                text = _part_to_text(part)
                if text:
                    lines.append(text)
    return '\n'.join(lines)


def count_message_tokens(
    messages: Sequence[ModelMessage],
    counter: TokenCounter = default_token_counter,
) -> int:
    """Count tokens across a sequence of messages."""
    return counter(messages_to_text(messages))
