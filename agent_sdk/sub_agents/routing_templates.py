from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentSpec:
    name: str
    condition: Callable[[Any], bool] | None = None  # None = always run


@dataclass
class RoutingTemplate:
    name: str
    required_agents: list[AgentSpec]       # analysis + strategy agents dispatched via Send
    post_process: list[AgentSpec]          # sequential terminal nodes (synthesis always first)


def _has_holdings(state: Any) -> bool:
    return bool((state or {}).get("has_holdings", False))


def _is_beginner(state: Any) -> bool:
    return (state or {}).get("knowledge_level", "expert") in {"beginner", "intermediate"}


ROUTING_TEMPLATES: dict[str, RoutingTemplate] = {
    "educational": RoutingTemplate(
        name="educational",
        required_agents=[],
        post_process=[],
    ),
    "company_snapshot": RoutingTemplate(
        name="company_snapshot",
        required_agents=[AgentSpec("company_profiling")],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "price_and_charts": RoutingTemplate(
        name="price_and_charts",
        required_agents=[AgentSpec("company_profiling"), AgentSpec("technical")],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "news_query": RoutingTemplate(
        name="news_query",
        required_agents=[AgentSpec("company_profiling"), AgentSpec("news_sentiment")],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "fundamentals": RoutingTemplate(
        name="fundamentals",
        required_agents=[AgentSpec("company_profiling"), AgentSpec("fundamental")],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "valuation": RoutingTemplate(
        name="valuation",
        required_agents=[
            AgentSpec("macro"),
            AgentSpec("company_profiling"),
            AgentSpec("fundamental"),
            AgentSpec("technical"),
            AgentSpec("news_sentiment"),
            AgentSpec("sector"),
            AgentSpec("bull_bear"),
        ],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "general_analysis": RoutingTemplate(
        name="general_analysis",
        required_agents=[
            AgentSpec("company_profiling"),
            AgentSpec("fundamental"),
            AgentSpec("news_sentiment"),
        ],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "sector_overview": RoutingTemplate(
        name="sector_overview",
        required_agents=[
            AgentSpec("macro"),
            AgentSpec("sector"),
        ],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "macro_query": RoutingTemplate(
        name="macro_query",
        required_agents=[AgentSpec("macro")],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "risk_deep_dive": RoutingTemplate(
        name="risk_deep_dive",
        required_agents=[
            AgentSpec("company_profiling"),
            AgentSpec("fundamental"),
            AgentSpec("technical"),
            AgentSpec("news_sentiment"),
            AgentSpec("macro"),
            AgentSpec("risk"),
            AgentSpec("bull_bear"),
        ],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "portfolio_review": RoutingTemplate(
        name="portfolio_review",
        required_agents=[
            AgentSpec("company_profiling"),
            AgentSpec("fundamental"),
            AgentSpec("technical"),
            AgentSpec("risk"),
            AgentSpec("portfolio_fit", condition=_has_holdings),
        ],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "comparative": RoutingTemplate(
        name="comparative",
        required_agents=[
            AgentSpec("macro"),
            AgentSpec("company_profiling"),
            AgentSpec("fundamental"),
            AgentSpec("technical"),
            AgentSpec("news_sentiment"),
            AgentSpec("sector"),
            AgentSpec("bull_bear"),
        ],
        post_process=[AgentSpec("synthesis"), AgentSpec("compliance")],
    ),
    "buy_decision": RoutingTemplate(
        name="buy_decision",
        required_agents=[
            AgentSpec("macro"),
            AgentSpec("company_profiling"),
            AgentSpec("news_sentiment"),
            AgentSpec("fundamental"),
            AgentSpec("technical"),
            AgentSpec("sector"),
            AgentSpec("risk"),
            AgentSpec("bull_bear"),
            AgentSpec("portfolio_fit", condition=_has_holdings),
        ],
        post_process=[
            AgentSpec("synthesis"),
            AgentSpec("compliance"),
            AgentSpec("jargon_simplifier", condition=_is_beginner),
        ],
    ),
}
