"""Financial query orchestration: combined classify-and-plan LLM call."""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_sdk.agents.llm_utils import invoke_with_retry
from agent_sdk.financial.phase_helpers import get_phase_llm

logger = logging.getLogger("agent_sdk.financial.orchestrator")

_JSON_FENCE_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)


def fix_json_control_chars(s: str) -> str:
    """Escape literal \\n/\\r/\\t inside JSON string values emitted by LLMs."""
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            out.append(ch)
            escape_next = False
        elif ch == '\\':
            out.append(ch)
            escape_next = True
        elif ch == '"':
            out.append(ch)
            in_string = not in_string
        elif in_string and ch == '\n':
            out.append('\\n')
        elif in_string and ch == '\r':
            out.append('\\r')
        elif in_string and ch == '\t':
            out.append('\\t')
        else:
            out.append(ch)
    return ''.join(out)


def extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM text output."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(fix_json_control_chars(text))
        except (json.JSONDecodeError, TypeError):
            pass

    for match in _JSON_FENCE_PATTERN.findall(text):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            try:
                return json.loads(fix_json_control_chars(match.strip()))
            except json.JSONDecodeError:
                continue

    candidates: list[dict] = []
    brace_depth = 0
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                for attempt in (text[start:i + 1], fix_json_control_chars(text[start:i + 1])):
                    try:
                        obj = json.loads(attempt)
                        if isinstance(obj, dict):
                            candidates.append(obj)
                        break
                    except json.JSONDecodeError:
                        pass
                start = None

    if candidates:
        scored = []
        for obj in candidates:
            string_keys = sum(1 for v in obj.values() if isinstance(v, str) and len(v) > 10)
            scored.append((len(obj) + string_keys * 2, obj))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    logger.warning(
        "extract_json: all parsing strategies failed (text length=%d, first 500 chars: '%s')",
        len(text), text[:500],
    )
    return None


def normalize_classification(raw: dict) -> dict:
    """Remap common LLM field-name variations to QueryClassification fields."""
    normalized = dict(raw)

    for alias in ("type", "classification", "category"):
        if alias in normalized and "query_type" not in normalized:
            normalized["query_type"] = normalized.pop(alias)

    for alias in ("reason", "explanation"):
        if alias in normalized and "reasoning" not in normalized:
            normalized["reasoning"] = normalized.pop(alias)

    _phase_bool_map = {
        "requires_regime_assessment": "regime_assessment",
        "requires_causal_analysis": "causal_analysis",
        "requires_sector_analysis": "sector_analysis",
        "requires_company_analysis": "company_analysis",
        "requires_risk_assessment": "risk_assessment",
    }
    has_legacy_bools = any(k in normalized for k in _phase_bool_map)
    has_phases_key = "phases" in normalized or "reasoning_phases" in normalized

    if has_legacy_bools and not has_phases_key:
        _registry_order = [
            "regime_assessment", "causal_analysis",
            "sector_analysis", "company_analysis", "risk_assessment",
        ]
        normalized["phases"] = [
            phase for phase in _registry_order
            if normalized.get(f"requires_{phase}", False)
        ]

    if "reasoning_phases" in normalized and "phases" not in normalized:
        normalized["phases"] = normalized.pop("reasoning_phases")

    valid_keys = {"query_type", "entities", "phases", "reasoning"}
    return {k: v for k, v in normalized.items() if k in valid_keys}


def _strip_context_block(text: str) -> str:
    marker = "[/CONTEXT]"
    if marker in text:
        return text[text.find(marker) + len(marker):].strip()
    return text.strip()


async def financial_orchestrate(agent, state) -> dict:
    """Classify the query and build a tool-specific plan in a single LLM call."""
    from agent_sdk.financial.prompts import FINANCIAL_ORCHESTRATE_COMBINED_PROMPT
    from agent_sdk.financial.schemas import QueryClassification, QueryType
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY

    logger.info("Classifying query and building plan for financial reasoning pipeline")
    llm = get_phase_llm(agent, state)

    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    clean_query = _strip_context_block(user_query)

    recent_context: list[str] = []
    for msg in state.messages[:-1]:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
            recent_context.append(f"Assistant: {msg.content}")
        elif isinstance(msg, HumanMessage):
            content = _strip_context_block(msg.content)
            if content:
                recent_context.append(f"User: {content}")
    recent_context = recent_context[-4:]

    if recent_context:
        user_content = f"Recent conversation:\n{chr(10).join(recent_context)}\n\nQuery: {clean_query}"
    else:
        user_content = f"Query: {clean_query}"

    _valid_phases = set(PHASE_REGISTRY.keys()) | {"comparative_analysis"}
    _default_qc = QueryClassification(
        query_type=QueryType.DATA_RETRIEVAL,
        phases=["company_analysis"],
        reasoning="Classification failed — running minimal pipeline",
    )

    qc = _default_qc
    plan_text = ""
    try:
        response = await invoke_with_retry(llm, [
            SystemMessage(content=FINANCIAL_ORCHESTRATE_COMBINED_PROMPT),
            HumanMessage(content=user_content),
        ])
        combined = extract_json(response.content)
        if combined:
            plan_obj = combined.pop("plan", "")
            plan_text = "\n".join(str(s) for s in plan_obj).strip() if isinstance(plan_obj, list) else str(plan_obj).strip()
            try:
                qc = QueryClassification(**normalize_classification(combined))
            except Exception:
                logger.warning("Could not build QueryClassification from combined response — using default")
        else:
            logger.warning("Combined orchestrate response missing JSON — using default classification")
    except Exception:
        logger.exception("financial_orchestrate: combined LLM call failed — using defaults")

    phases: list[str] = []
    if qc.query_type == QueryType.DATA_RETRIEVAL:
        phases = ["company_analysis"]
    elif qc.query_type == QueryType.COMPARATIVE:
        phases = ["comparative_analysis"]
    elif qc.phases:
        phases = [p for p in qc.phases if p in _valid_phases]
    if phases:
        phases.append("synthesis")

    entities = list(qc.entities) if hasattr(qc, "entities") and qc.entities else []
    logger.info("financial_orchestrate: type=%s phases=%s entities=%s plan=%d chars",
                qc.query_type.value, phases, entities, len(plan_text))

    return {
        "scratchpad": plan_text if plan_text else None,
        "phases_to_run": phases,
        "current_phase": phases[0] if phases else "done",
        "query_type": qc.query_type.value,
        "entities": entities,
        "iteration": state.iteration + 1,
    }
