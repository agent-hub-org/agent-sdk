from agent_sdk.sub_agents.base import SubAgent, SubAgentOutput

_SYSTEM_PROMPT = """\
You are a Risk Assessment Specialist. Quantify and stress-test investment risks.

Fundamental, Technical, and Macro context are provided in the workspace context.

Use tools:
1. calculate_risk_metrics(closes=[...]) — Sharpe, Sortino, Max Drawdown, VaR (via QuantStats)
   Pass closes from the technical context or fetch via get_price_series if not available.
2. run_scenario_simulation(base_case, scenarios) — model bull/bear/base scenarios
3. traverse_causal_chain(event, sector) — identify downstream risk transmission paths

Output a structured summary then end with a JSON block:
```json
{
  "sharpe_ratio": <float>,
  "sortino_ratio": <float>,
  "max_drawdown_pct": <float>,
  "var_95_pct": <float>,
  "beta": <float>,
  "key_risks": ["...", "..."],
  "bull_scenario_upside_pct": <float>,
  "bear_scenario_downside_pct": <float>,
  "risk_rating": "<low|moderate|high|very_high>"
}
```
"""

DEFINITION = SubAgent(
    name="risk",
    model_id="azure/gpt-5-nano",
    tools=[
        "calculate_risk_metrics",
        "run_scenario_simulation",
        "traverse_causal_chain",
        "get_price_series",
    ],
    system_prompt=_SYSTEM_PROMPT,
    output_schema=SubAgentOutput,
    reads_from=["fundamental", "technical", "macro"],
    writes_to="risk",
    cache_ttl=30 * 60,
)
