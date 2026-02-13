"""Observer agent: extracts structured observations from conversation history."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from pydantic_ai.settings import ModelSettings

from ._token_counter import messages_to_text

OBSERVER_SYSTEM_PROMPT = """\
You are the memory consciousness of an AI assistant. Your observations will be \
the ONLY information the assistant has about past interactions with this user.

Extract observations that will help the assistant remember:

1. WHAT the user explicitly stated (facts, preferences, goals, context about their life)
2. WHAT was discussed (topics, questions asked, information exchanged)
3. WHAT actions the assistant took (tool calls, code written, recommendations made)
4. WHAT the outcomes were (successes, failures, pending items)
5. STATE CHANGES: when something was updated, corrected, or superseded, note the change

=== PRIORITY LEVELS ===

Use priority levels:
- 🔴 High: explicit user facts, preferences, goals achieved, critical context
- 🟡 Medium: project details, learned information, tool results
- 🟢 Low: minor details, uncertain observations

=== FORMAT ===

Group observations by date, then list each with 24-hour time.
Group related observations (like tool sequences) by indenting.

Example:

Date: Dec 4, 2025
* 🔴 (09:15) User stated they have 3 kids: Emma (12), Jake (9), and Lily (5)
* 🟡 (10:45) Assistant recommended hotels: Grand Plaza ($180/night), Seaside Inn (pet-friendly)
* 🟡 (14:00) Agent debugging auth issue
  * -> ran git status, found 3 modified files
  * -> viewed auth.ts:45-60, found missing null check
  * -> applied fix, tests now pass

Date: Dec 5, 2025
* 🔴 (09:00) User switched from Python to TypeScript for the project

=== GUIDELINES ===

- Be specific: "User prefers short answers without lengthy explanations" not "User stated a preference"
- Use terse language - dense sentences without unnecessary words
- Don't repeat observations that have already been captured
- When the agent calls tools, observe what was called, why, and what was learned
- If the agent provides a detailed response, observe the key points so it could be repeated
- Start each observation with a priority emoji (🔴, 🟡, 🟢)
- Observe WHAT happened and WHAT it means, not HOW well it was done
- If the user provides detailed messages or code snippets, observe all important details

=== CRITICAL: USER ASSERTIONS vs QUESTIONS ===

- "User stated: X" = authoritative assertion (user told us something about themselves)
- "User asked: X" = question/request (user seeking information)
User assertions take precedence. The user is the authority on their own life.

Remember: These observations are the assistant's ONLY memory. Make them count.

User messages are extremely important. If the user asks a question or gives a new task, \
make it clear in current_task that this is the priority.\
"""


def _format_timestamp(ts: datetime | None) -> str:
    if ts is None:
        return ''
    return ts.strftime('%b %d, %Y %H:%M')


def format_messages_for_observer(messages: Sequence[ModelMessage]) -> str:
    """Format ModelMessages into a readable transcript for the observer."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            ts = _format_timestamp(msg.timestamp)
            ts_str = f' ({ts})' if ts else ''
            text = messages_to_text([msg])
            if text:
                lines.append(f'**User{ts_str}:**\n{text}')
        elif isinstance(msg, ModelResponse):
            ts = _format_timestamp(msg.timestamp)
            ts_str = f' ({ts})' if ts else ''
            text = messages_to_text([msg])
            if text:
                lines.append(f'**Assistant{ts_str}:**\n{text}')
    return '\n\n---\n\n'.join(lines)


def build_observer_prompt(
    existing_observations: str | None,
    messages_to_observe: Sequence[ModelMessage],
) -> str:
    """Build the user prompt for the observer agent."""
    formatted = format_messages_for_observer(messages_to_observe)

    prompt = ''
    if existing_observations:
        prompt += f'## Previous Observations\n\n{existing_observations}\n\n---\n\n'
        prompt += 'Do not repeat these existing observations. Your new observations will be appended to the existing observations.\n\n'

    prompt += f'## New Message History to Observe\n\n{formatted}\n\n---\n\n'
    prompt += '## Your Task\n\nExtract new observations from the message history above. '
    prompt += 'Do not repeat observations that are already in the previous observations. '
    prompt += 'Return your observations in the structured format.'

    return prompt


OBSERVER_MODEL_SETTINGS = ModelSettings(temperature=0.3)
