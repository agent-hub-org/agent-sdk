from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime, UTC
from typing import Annotated, Any, Callable, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SubAgentInput(BaseModel):
    query: str
    entities: list[str] = []
    workspace_context: str = ""   # plain-text rendering of reads_from keys — NOT the full workspace
    user_profile: dict[str, Any] = {}


class SubAgentOutput(BaseModel):
    agent_name: str
    findings: str
    structured: dict[str, Any] = {}
    confidence: float = 1.0
    cached: bool = False
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_input: SubAgentInput
    findings: str
    tool_calls_count: int


logger = logging.getLogger("agent_sdk.sub_agents.base")


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None


@dataclass
class SubAgent:
    name: str
    model_id: str
    tools: list[str]
    system_prompt: str
    output_schema: type[SubAgentOutput]
    reads_from: list[str]
    writes_to: str
    cache_ttl: int | None = None
    cache_key_fn: Callable[[SubAgentInput], str] | None = None
    _graph: Any = dc_field(default=None, init=False, repr=False)

    def cache_key(self, inp: SubAgentInput) -> str:
        if self.cache_key_fn:
            return self.cache_key_fn(inp)
        entity_hash = hashlib.sha256(
            json.dumps(sorted(inp.entities), sort_keys=True).encode()
        ).hexdigest()[:16]
        now = datetime.now(UTC)
        bucket = (
            now.strftime("%Y-%m-%d-%H")
            if self.cache_ttl and self.cache_ttl <= 1800
            else now.strftime("%Y-%m-%d")
        )
        return f"sub_agent:{self.name}:{entity_hash}:{bucket}"

    def compile(self, tools_map: dict[str, Any], llm_factory: Callable[[str], Any]) -> Any:
        from langchain_core.messages import SystemMessage, HumanMessage
        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode

        llm = llm_factory(self.model_id)
        bound_tools = [tools_map[t] for t in self.tools if t in tools_map]
        bound_llm = llm.bind_tools(bound_tools) if bound_tools else llm

        def sa_call(state: SubAgentState) -> dict:
            msgs = list(state["messages"])
            response = bound_llm.invoke(msgs)
            return {
                "messages": [response],
                "tool_calls_count": state.get("tool_calls_count", 0),
            }

        def should_continue(state: SubAgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "finalize"

        def sa_finalize(state: SubAgentState) -> dict:
            last = state["messages"][-1]
            content = getattr(last, "content", "") or ""
            return {"findings": content}

        g = StateGraph(SubAgentState)
        g.add_node("sa_call", sa_call)
        g.add_node("sa_finalize", sa_finalize)
        g.add_edge(START, "sa_call")

        if bound_tools:
            g.add_node("tools", ToolNode(bound_tools))
            g.add_conditional_edges("sa_call", should_continue, {"tools": "tools", "finalize": "sa_finalize"})
            g.add_edge("tools", "sa_call")
        else:
            g.add_conditional_edges("sa_call", should_continue, {"tools": "sa_finalize", "finalize": "sa_finalize"})

        g.add_edge("sa_finalize", END)
        self._graph = g.compile()
        return self._graph

    def _parse_output(self, state: dict) -> SubAgentOutput:
        findings = state.get("findings", "")
        structured = _extract_json(findings) or {}
        return self.output_schema(
            agent_name=self.name,
            findings=findings,
            structured=structured,
            confidence=1.0 if findings else 0.0,
        )

    async def run(
        self,
        inp: SubAgentInput,
        workspace_store: Any,
        workspace_id: str,
        sub_agent_cache: Any,
    ) -> SubAgentOutput:
        from langchain_core.messages import SystemMessage, HumanMessage

        if self._graph is None:
            raise RuntimeError(
                f"SubAgent '{self.name}' must be compiled before calling run(). "
                "Call compile(tools_map, llm_factory) first."
            )

        if self.cache_ttl is not None:
            ck = self.cache_key(inp)
            cached = await sub_agent_cache.get(ck)
            if cached:
                out = self.output_schema.model_validate(cached)
                out.cached = True
                await workspace_store.write(workspace_id, self.writes_to, out.model_dump())
                logger.debug("SubAgent %s: cache hit (key=%s)", self.name, ck)
                return out
        else:
            ck = None

        initial_state: SubAgentState = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._build_prompt(inp)),
            ],
            "agent_input": inp,
            "findings": "",
            "tool_calls_count": 0,
        }
        result = await self._graph.ainvoke(initial_state)
        out = self._parse_output(result)

        await workspace_store.write(workspace_id, self.writes_to, out.model_dump())

        if self.cache_ttl is not None and ck is not None:
            await sub_agent_cache.set(ck, out.model_dump(), ttl=self.cache_ttl)

        return out

    def _build_prompt(self, inp: SubAgentInput) -> str:
        parts = [f"Query: {inp.query}"]
        if inp.entities:
            parts.append(f"Entities: {', '.join(inp.entities)}")
        if inp.workspace_context:
            parts.append(f"Context from prior agents:\n{inp.workspace_context}")
        return "\n\n".join(parts)
