from agent_sdk.sub_agents.routing_templates import ROUTING_TEMPLATES, RoutingTemplate, AgentSpec


def test_all_13_templates_defined():
    expected = {
        "educational", "company_snapshot", "price_and_charts", "news_query",
        "fundamentals", "valuation", "general_analysis", "sector_overview",
        "macro_query", "risk_deep_dive", "portfolio_review", "comparative",
        "buy_decision",
    }
    assert set(ROUTING_TEMPLATES.keys()) == expected


def test_educational_has_no_agents():
    t = ROUTING_TEMPLATES["educational"]
    assert t.required_agents == []


def test_buy_decision_contains_all_analysis_agents():
    t = ROUTING_TEMPLATES["buy_decision"]
    required = {s.name for s in t.required_agents}
    assert "macro" in required
    assert "company_profiling" in required
    assert "fundamental" in required
    assert "technical" in required
    assert "news_sentiment" in required
    assert "sector" in required
    assert "risk" in required
    assert "bull_bear" in required


def test_macro_query_has_only_macro():
    t = ROUTING_TEMPLATES["macro_query"]
    names = [s.name for s in t.required_agents]
    assert names == ["macro"]


def test_company_snapshot_has_only_company_profiling():
    t = ROUTING_TEMPLATES["company_snapshot"]
    names = [s.name for s in t.required_agents]
    assert names == ["company_profiling"]


def test_portfolio_fit_is_conditional_in_buy_decision():
    t = ROUTING_TEMPLATES["buy_decision"]
    conditional_names = [s.name for s in t.required_agents if s.condition is not None]
    assert "portfolio_fit" in conditional_names


def test_jargon_simplifier_in_post_process_buy_decision():
    t = ROUTING_TEMPLATES["buy_decision"]
    assert "jargon_simplifier" in [s.name for s in t.post_process]


def test_compliance_always_in_post_process():
    for name, t in ROUTING_TEMPLATES.items():
        if name == "educational":
            continue
        post_names = [s.name for s in t.post_process]
        assert "compliance" in post_names, f"{name} missing compliance"
