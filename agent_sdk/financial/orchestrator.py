"""Financial query orchestration: template-router LLM call."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid as _uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_sdk.agents.llm_utils import invoke_with_retry

logger = logging.getLogger("agent_sdk.financial.orchestrator")

_VALID_TEMPLATES = {
    "educational", "company_snapshot", "price_and_charts", "news_query",
    "fundamentals", "valuation", "general_analysis", "sector_overview",
    "macro_query", "risk_deep_dive", "portfolio_review", "comparative",
    "buy_decision",
}

_ORCHESTRATOR_SYSTEM = """\
You are a financial query router. Classify the user's query into exactly one routing template.

Templates:
- educational: purely conceptual questions (what is X, how does Y work)
- company_snapshot: simple company description questions
- price_and_charts: price level, technical chart, 52-week range questions
- news_query: recent news or developments for a company
- fundamentals: financial statements, revenue, earnings data
- valuation: fair value, DCF, overvalued/undervalued questions
- general_analysis: ambiguous "tell me about X" questions
- sector_overview: sector-level outlook without a specific company
- macro_query: macro topics (interest rates, FII flows, market regime) without a company
- risk_deep_dive: risk, downside, stress-test questions for a company
- portfolio_review: portfolio fit, holdings analysis
- comparative: comparing two or more companies
- buy_decision: buy/sell/invest recommendation requests

Output ONLY valid JSON (no markdown fence):
{"template_name": "<one of the above>", "entities": ["<ticker_or_company>", ...], "as_of_date": "<YYYY-MM-DD or null>"}
"""


def _get_orchestrator_llm(agent):
    from agent_sdk.llm_services.model_registry import get_llm
    return get_llm("azure/gpt-5.4-mini")

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
    """Route the query to a template and assign a workspace_id."""
    messages = state["messages"] if isinstance(state, dict) else state.messages
    last_human = next(
        (m for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        None,
    )
    query = last_human.content if last_human else ""

    llm = _get_orchestrator_llm(agent)
    response = await invoke_with_retry(
        llm,
        [SystemMessage(content=_ORCHESTRATOR_SYSTEM), HumanMessage(content=query)],
    )

    raw = extract_json(response.content) or {}
    template_name = raw.get("template_name", "general_analysis")
    if template_name not in _VALID_TEMPLATES:
        template_name = "general_analysis"

    entities = raw.get("entities") or []
    as_of_date = raw.get("as_of_date")

    workspace_id = hashlib.sha256(
        f"{id(state)}:{_uuid.uuid4()}".encode()
    ).hexdigest()[:32]

    logger.info(
        "financial_orchestrate: template=%s entities=%s workspace_id=%s",
        template_name, entities, workspace_id,
    )

    return {
        "current_template": template_name,
        "entities": entities,
        "as_of_date": as_of_date,
        "workspace_id": workspace_id,
    }
