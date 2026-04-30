"""Central registry: imports all sub-agent definitions and exposes SUB_AGENT_REGISTRY."""
from __future__ import annotations

from agent_sdk.sub_agents.base import SubAgent

_DEFINITION_MODULES = [
    "agent_sdk.sub_agents.definitions.macro",
    "agent_sdk.sub_agents.definitions.company_profiling",
    "agent_sdk.sub_agents.definitions.fundamental",
    "agent_sdk.sub_agents.definitions.technical",
    "agent_sdk.sub_agents.definitions.news_sentiment",
    "agent_sdk.sub_agents.definitions.sector",
    "agent_sdk.sub_agents.definitions.risk",
    "agent_sdk.sub_agents.definitions.bull_bear",
    "agent_sdk.sub_agents.definitions.portfolio_fit",
    "agent_sdk.sub_agents.definitions.user_profiling",
]

_POST_PROCESS_MODULES = [
    "agent_sdk.sub_agents.definitions.synthesis",
    "agent_sdk.sub_agents.definitions.compliance",
    "agent_sdk.sub_agents.definitions.jargon_simplifier",
]


def _load_registry() -> dict[str, SubAgent]:
    import importlib
    registry: dict[str, SubAgent] = {}
    for mod_path in _DEFINITION_MODULES + _POST_PROCESS_MODULES:
        mod = importlib.import_module(mod_path)
        agent: SubAgent = mod.DEFINITION
        registry[agent.name] = agent
    return registry


SUB_AGENT_REGISTRY: dict[str, SubAgent] = _load_registry()
