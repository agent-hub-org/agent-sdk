"""
Central registry for financial pipeline phases.

Adding a new phase = one entry here.  All other subsystems (graph,
nodes, phase_subgraph, base_agent) read from this registry at runtime
rather than maintaining their own parallel dicts.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("agent_sdk.financial.phase_registry")


class PhaseDefinition(BaseModel):
    """Declarative description of one financial pipeline phase."""

    name: str

    # Names of tools from financial modules (causal_graph / ontology / quant_tools)
    # that are active in this phase.  MCP tools are always included separately.
    financial_tool_names: list[str] = Field(default_factory=list)

    # Maximum LLM turns allowed within the phase.
    budget: int = 4

    # Human-readable label emitted as a streaming progress marker.
    progress_label: str = ""

    # One-line hint injected into the orchestrator prompt (phase_name: tool hints).
    orchestrate_hint: str = ""

    # Phase names that must be complete before this phase can start.
    depends_on: list[str] = Field(default_factory=list)


PHASE_REGISTRY: dict[str, PhaseDefinition] = {
    "regime_assessment": PhaseDefinition(
        name="regime_assessment",
        financial_tool_names=["detect_market_regime"],
        budget=4,
        progress_label="🌐 Assessing macro regime & market environment...",
        orchestrate_hint=(
            "regime_assessment: get_regime_inputs → detect_market_regime, "
            "get_fii_dii_flows, tavily_quick_search"
        ),
        depends_on=[],
    ),
    "causal_analysis": PhaseDefinition(
        name="causal_analysis",
        financial_tool_names=[
            "traverse_causal_chain",
            "get_affected_entities",
            "get_transmission_path",
            "search_causal_graph",
            "run_scenario_simulation",
        ],
        budget=4,
        progress_label="🔗 Mapping causal transmission chains...",
        orchestrate_hint=(
            "causal_analysis: traverse_causal_chain, get_affected_entities, "
            "get_transmission_path, run_scenario_simulation, tavily_quick_search"
        ),
        depends_on=["regime_assessment"],
    ),
    "sector_analysis": PhaseDefinition(
        name="sector_analysis",
        financial_tool_names=[
            "interpret_metric",
            "get_metric_definition",
            "get_sector_norms",
        ],
        budget=5,
        progress_label="📊 Evaluating sector positioning...",
        orchestrate_hint=(
            "sector_analysis: get_fii_dii_flows, get_sector_norms, "
            "interpret_metric, tavily_quick_search"
        ),
        depends_on=["causal_analysis"],
    ),
    "company_analysis": PhaseDefinition(
        name="company_analysis",
        financial_tool_names=[
            "interpret_metric",
            "get_metric_definition",
            "get_sector_norms",
            "run_dcf",
            "run_comparable_valuation",
            "calculate_technical_signals",
            "calculate_risk_metrics",
        ],
        budget=10,
        progress_label="🏢 Running fundamental company analysis...",
        orchestrate_hint=(
            "company_analysis: get_ticker_data, get_bse_nse_reports, "
            "get_price_series → calculate_technical_signals/calculate_risk_metrics, "
            "get_dcf_inputs → run_dcf, get_comparable_metrics → run_comparable_valuation, "
            "interpret_metric, tavily_quick_search"
        ),
        depends_on=["causal_analysis"],
    ),
    "risk_assessment": PhaseDefinition(
        name="risk_assessment",
        financial_tool_names=[
            "run_scenario_simulation",
            "calculate_risk_metrics",
            "traverse_causal_chain",
            "get_affected_entities",
            "get_transmission_path",
            "search_causal_graph",
        ],
        budget=6,
        progress_label="⚠️ Stress-testing scenarios & risks...",
        orchestrate_hint=(
            "risk_assessment: get_price_series → calculate_risk_metrics/"
            "calculate_technical_signals, run_scenario_simulation, tavily_quick_search"
        ),
        depends_on=["sector_analysis", "company_analysis"],
    ),
    "entity_analysis": PhaseDefinition(
        name="entity_analysis",
        financial_tool_names=[
            "interpret_metric",
            "get_metric_definition",
            "get_sector_norms",
            "run_dcf",
            "run_comparable_valuation",
            "calculate_technical_signals",
            "calculate_risk_metrics",
        ],
        budget=5,
        progress_label="⚖️ Analyzing comparative entities...",
        orchestrate_hint="",  # not emitted in orchestrate prompt directly
        depends_on=[],  # triggered via comparative_analysis fan-out, not by scheduler
    ),
    "synthesis": PhaseDefinition(
        name="synthesis",
        financial_tool_names=[],
        budget=0,  # pure LLM, no tool iterations
        progress_label="✍️ Synthesizing final report...",
        orchestrate_hint="",  # synthesis is always last; not listed in phase hints
        depends_on=["risk_assessment"],  # most complete pipeline; scheduler checks actual plan
    ),
}


def validate_phase_dag(registry: dict[str, PhaseDefinition]) -> None:
    """Raise ValueError if the dependency graph contains a cycle."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in registry}

    def _dfs(node: str) -> None:
        color[node] = GRAY
        for dep in registry[node].depends_on:
            if dep not in registry:
                continue  # external dep or optional — skip
            if color[dep] == GRAY:
                raise ValueError(f"Phase registry DAG cycle detected: {node} → {dep}")
            if color[dep] == WHITE:
                _dfs(dep)
        color[node] = BLACK

    for phase in registry:
        if color[phase] == WHITE:
            _dfs(phase)


# Validate at import time — fail fast if someone introduces a cycle.
validate_phase_dag(PHASE_REGISTRY)


def build_orchestrate_prompt(registry: dict[str, PhaseDefinition]) -> str:
    """Build the financial orchestrator prompt dynamically from the registry.

    The 'Available tools by phase' section is derived from registry hints so it
    never drifts from the actual tool catalog.
    """
    hints = "\n".join(
        f"- {pd.orchestrate_hint}"
        for pd in registry.values()
        if pd.orchestrate_hint
    )

    return f"""\
You are a financial query classifier and analysis orchestrator.

STEP 1 — Classify the query and determine which reasoning phases to activate.

Query types:
- data_retrieval: Simple data lookups ("What is Reliance's P/E?")
- company_analysis: Deep single-company analysis ("Should I invest in TCS?")
- sector_analysis: Sector-level analysis ("How is the banking sector positioned?")
- macro_impact: Macro event impact ("What happens if RBI hikes rates?")
- comparative: Peer comparison ("Compare TCS vs Infosys")
- thematic: Cross-sector themes ("Stocks benefiting from India's capex cycle")

Phase activation rules:
- data_retrieval: company_analysis + synthesis only
- macro_impact: FULL pipeline (regime → causal → sector → company → risk → synthesis)
- company_analysis: company + risk + synthesis (optionally sector)
- sector_analysis: sector + risk + synthesis (optionally regime)
- comparative: comparative_analysis + synthesis
- thematic: regime + sector + company + synthesis

STEP 2 — For each activated phase write ONE terse line:
  <phase_name>: <tool1(specific_args)> → <tool2>, <tool3(specific_args)>

Rules: tool names + entity-specific args only; no prose; NSE suffix (.NS) for Indian stocks.
Available tools by phase:
{hints}

OUTPUT: A single JSON object with exactly these fields:
{{{{
  "query_type": "<type>",
  "entities": ["<tickers/sectors/indicators>"],
  "phases": ["<phase1>", "<phase2>", ...],
  "plan": "<one line per active phase>"
}}}}

List only the phases that should run (excluding synthesis — always added automatically).
Valid phase names: regime_assessment, causal_analysis, sector_analysis, company_analysis,
risk_assessment, comparative_analysis.
"""
