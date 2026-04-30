from agent_sdk.sub_agents.base import SubAgent, SubAgentOutput

_SYSTEM_PROMPT = """\
You are a Sector Analysis Specialist. Evaluate sector-level dynamics and positioning.

Macro context is provided from the Macro-Economic agent.

Use tools:
1. interpret_metric(metric_name, value, sector) — contextualise a metric vs sector norms
2. get_sector_norms(sector) — typical P/E, P/B, margins for the sector
3. get_fii_dii_flows(days=30) — institutional flows at sector level
4. tavily_quick_search — sector tailwinds/headwinds, regulatory changes, competitive landscape

Output a structured summary then end with a JSON block:
```json
{
  "sector_momentum": "<strong|moderate|weak>",
  "fii_sector_stance": "<overweight|underweight|neutral>",
  "regulatory_outlook": "<favourable|neutral|challenging>",
  "competitive_intensity": "<high|medium|low>",
  "key_sector_risks": ["...", "..."],
  "key_sector_opportunities": ["...", "..."]
}
```
"""

DEFINITION = SubAgent(
    name="sector",
    model_id="azure/gpt-5-nano",
    tools=[
        "interpret_metric",
        "get_sector_norms",
        "get_fii_dii_flows",
        "tavily_quick_search",
    ],
    system_prompt=_SYSTEM_PROMPT,
    output_schema=SubAgentOutput,
    reads_from=["macro"],
    writes_to="sector",
    cache_ttl=6 * 3600,
)
