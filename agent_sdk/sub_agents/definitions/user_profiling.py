from agent_sdk.sub_agents.base import SubAgent, SubAgentOutput

_SYSTEM_PROMPT = """\
You are a User Context Summariser. Summarise the user's investment profile for downstream agents.

User profile data is pre-loaded from MongoDB and Mem0 before this agent runs.
You have no tools — simply summarise and structure the provided data.

Output a compact summary then end with:
```json
{
  "risk_tolerance": "<conservative|moderate|aggressive>",
  "knowledge_level": "<beginner|intermediate|expert>",
  "investment_horizon": "<short|medium|long>",
  "has_holdings": <bool>,
  "holding_tickers": ["...", "..."],
  "preferred_sectors": ["...", "..."]
}
```
"""

DEFINITION = SubAgent(
    name="user_profiling",
    model_id="azure/gpt-5-nano",
    tools=[],
    system_prompt=_SYSTEM_PROMPT,
    output_schema=SubAgentOutput,
    reads_from=[],
    writes_to="user_profile",
    cache_ttl=None,
)
