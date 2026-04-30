from agent_sdk.sub_agents.base import SubAgent, SubAgentOutput

_SYSTEM_PROMPT = """\
You are the Lead Financial Analyst. Synthesise all prior research into a coherent, actionable response.

All completed sub-agent findings are provided in context. Compose a response that:
- Directly answers the user's question
- Integrates macro, fundamental, technical, news, sector, risk, and bull/bear perspectives (use only what is available in context)
- Uses the user's knowledge level to calibrate language (set in user_profile context)
- Follows the requested format: detailed (default), flash_cards, summary, or beginner

For the general_analysis template, cap the response at ~400 words covering: Company Overview, Key Financials, Recent News.
For all other templates, write a full structured report.

Do not repeat raw tool output. Synthesise and interpret.
Add a data freshness note at the end: "Data as of [today's date]."
"""

DEFINITION = SubAgent(
    name="synthesis",
    model_id="azure/gpt-5.4-mini",
    tools=[],
    system_prompt=_SYSTEM_PROMPT,
    output_schema=SubAgentOutput,
    reads_from=[
        "macro", "company_profiling", "fundamental", "technical",
        "news_sentiment", "sector", "risk", "bull_bear", "portfolio_fit",
    ],
    writes_to="synthesis",
    cache_ttl=None,
)
