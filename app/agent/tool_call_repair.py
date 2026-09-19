"""Repair tool calls that a model leaks as PLAIN TEXT instead of the structured
`tool_calls` channel.

Some Ollama models (e.g. qwen2:7b) intermittently emit a tool call in the
Hermes/Qwen text template — ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
— inside ``message.content`` rather than in the structured ``message.tool_calls``
field. When that happens ``create_agent`` sees an AIMessage with no ``tool_calls``,
treats it as the final answer, and returns the raw template as the answer (never
running the tool). This middleware detects that leak, parses it, and rewrites the
AIMessage to carry a real tool call so the agent loop actually invokes the tool.

No-op for every normal turn (a proper structured tool call, or a plain final
answer with no leaked template), so it does not change behaviour for models/docs
that never leak.
"""

import json
import re

from langchain.agents.middleware import ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage

# <tool_call>{...}</tool_call>  OR  (tool_call){...}</tool_call>  (opener varies by
# how Ollama surfaces the leaked template). Non-greedy JSON body, DOTALL for newlines.
_TOOLCALL_TEXT_RE = re.compile(
    r"(?:<tool_call>|\(tool_call\))\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def extract_leaked_tool_calls(text: str) -> list[dict]:
    """Parse leaked ``<tool_call>{...}</tool_call>`` blobs into LangChain tool calls.

    Returns a list of ``{"name", "args", "id", "type"}`` dicts (empty if none).
    Blobs that are not valid JSON, or lack a ``name``, are skipped (never raises).
    """
    calls = []
    for i, m in enumerate(_TOOLCALL_TEXT_RE.finditer(text or "")):
        try:
            obj = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters", {})
        if not isinstance(args, dict):
            args = {}
        calls.append(
            {"name": name, "args": args, "id": f"repair_{i}", "type": "tool_call"}
        )
    return calls


@wrap_model_call
def repair_leaked_tool_calls(request, handler):
    """Rewrite a text-leaked tool call into a real structured tool call."""
    response = handler(request)

    # Locate the AIMessage in the response (ModelResponse.result[0]) or the message
    # itself if a bare AIMessage was returned.
    ai = None
    if isinstance(response, AIMessage):
        ai = response
    elif getattr(response, "result", None):
        ai = response.result[0]

    if ai is None or getattr(ai, "tool_calls", None):
        return response  # already has structured tool calls, or nothing to inspect

    content = ai.content if isinstance(ai.content, str) else str(ai.content)
    calls = extract_leaked_tool_calls(content)
    if not calls:
        return response  # normal answer, no leak — no-op

    repaired = ai.model_copy(update={"tool_calls": calls, "content": ""})
    if isinstance(response, AIMessage):
        return repaired
    return ModelResponse(
        result=[repaired],
        structured_response=getattr(response, "structured_response", None),
    )
