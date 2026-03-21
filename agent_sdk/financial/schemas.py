"""
Typed output schemas for each phase of the financial reasoning pipeline.

Each schema is a Pydantic model that captures the structured findings
from one reasoning phase, making them available to subsequent phases.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    """Determines which pipeline phases to activate."""
    DATA_RETRIEVAL = "data_retrieval"          # Simple lookup — skip to data fetch
    COMPANY_ANALYSIS = "company_analysis"      # Single-company deep dive
    SECTOR_ANALYSIS = "sector_analysis"        # Sector-level analysis
    MACRO_IMPACT = "macro_impact"              # Full pipeline — macro event impact
    COMPARATIVE = "comparative"                # Peer comparison
    THEMATIC = "thematic"                      # Cross-sector thematic analysis


class QueryClassification(BaseModel):
    """Output of the query classifier — determines pipeline routing."""
    query_type: QueryType
    entities: list[str] = Field(default_factory=list, description="Tickers, sectors, or macro indicators mentioned")
    requires_regime_assessment: bool = False
    requires_causal_analysis: bool = False
    requires_sector_analysis: bool = False
    requires_company_analysis: bool = False
    requires_risk_assessment: bool = True
    reasoning: str = Field(default="", description="Brief explanation of classification")


# ---------------------------------------------------------------------------
# Regime Assessment
# ---------------------------------------------------------------------------

class MonetaryRegime(str, Enum):
    TIGHTENING = "tightening"
    NEUTRAL = "neutral"
    EASING = "easing"
    CRISIS = "crisis"


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class CyclePosition(str, Enum):
    EARLY_EXPANSION = "early_expansion"
    MID_EXPANSION = "mid_expansion"
    LATE_EXPANSION = "late_expansion"
    PEAK = "peak"
    EARLY_CONTRACTION = "early_contraction"
    RECESSION = "recession"
    RECOVERY = "recovery"


class RegimeContext(BaseModel):
    """Structured output from the regime assessment phase."""
    monetary_regime: MonetaryRegime = MonetaryRegime.NEUTRAL
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    cycle_position: CyclePosition = CyclePosition.MID_EXPANSION

    # Key indicators
    repo_rate: Optional[float] = None
    cpi_yoy: Optional[float] = None
    iip_yoy: Optional[float] = None
    pmi_manufacturing: Optional[float] = None
    pmi_services: Optional[float] = None
    usd_inr: Optional[float] = None
    crude_brent: Optional[float] = None
    india_vix: Optional[float] = None
    nifty_pe: Optional[float] = None
    fii_net_flow_30d_cr: Optional[float] = None
    dii_net_flow_30d_cr: Optional[float] = None
    yield_10y_gsec: Optional[float] = None
    credit_growth_yoy: Optional[float] = None

    # Qualitative assessment
    key_risks: list[str] = Field(default_factory=list)
    key_tailwinds: list[str] = Field(default_factory=list)
    regime_summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Causal Analysis
# ---------------------------------------------------------------------------

class CausalLink(BaseModel):
    """A single link in a causal chain."""
    source: str
    target: str
    direction: str = Field(description="positive or negative")
    magnitude: str = Field(description="weak, moderate, or strong")
    time_lag: str = Field(description="immediate, 1-2Q, 2-4Q, 1-2Y")
    confidence: str = Field(description="well-established, theoretical, regime-dependent")
    mechanism: str = Field(default="", description="How the causal link works")


class CausalChain(BaseModel):
    """A traced causal chain from trigger event to affected entities."""
    trigger_event: str
    links: list[CausalLink] = Field(default_factory=list)
    affected_sectors: list[str] = Field(default_factory=list)
    affected_companies: list[str] = Field(default_factory=list)
    net_impact_summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CausalAnalysisResult(BaseModel):
    """Full output of the causal analysis phase."""
    chains: list[CausalChain] = Field(default_factory=list)
    primary_transmission_mechanisms: list[str] = Field(default_factory=list)
    second_order_effects: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Sector Analysis
# ---------------------------------------------------------------------------

class SectorMetrics(BaseModel):
    """Key metrics for a sector."""
    sector: str
    median_pe: Optional[float] = None
    median_pb: Optional[float] = None
    median_roe: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    margin_trend: str = ""  # expanding, stable, compressing
    fii_stance: str = ""    # overweight, neutral, underweight
    relative_strength: Optional[float] = None


class SectorFindings(BaseModel):
    """Structured output from the sector analysis phase."""
    sectors_analyzed: list[SectorMetrics] = Field(default_factory=list)
    sector_rotation_signals: list[str] = Field(default_factory=list)
    top_sectors: list[str] = Field(default_factory=list)
    bottom_sectors: list[str] = Field(default_factory=list)
    sector_narrative: str = ""
    regime_alignment: str = Field(default="", description="How sector dynamics align with the current regime")


# ---------------------------------------------------------------------------
# Company Analysis
# ---------------------------------------------------------------------------

class ValuationMetrics(BaseModel):
    """Company valuation snapshot."""
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    pb: Optional[float] = None
    ev_ebitda: Optional[float] = None
    dividend_yield: Optional[float] = None
    mcap_cr: Optional[float] = None


class FundamentalMetrics(BaseModel):
    """Company fundamental snapshot."""
    revenue_cr: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    ebitda_margin: Optional[float] = None
    pat_cr: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None
    promoter_holding: Optional[float] = None
    promoter_pledge: Optional[float] = None
    fcf_cr: Optional[float] = None


class CompanyAnalysis(BaseModel):
    """Structured output from the company analysis phase."""
    ticker: str = ""
    company_name: str = ""
    sector: str = ""

    valuation: ValuationMetrics = Field(default_factory=ValuationMetrics)
    fundamentals: FundamentalMetrics = Field(default_factory=FundamentalMetrics)

    # Interpretations (filled by ontology lookups)
    valuation_assessment: str = ""  # cheap, fair, expensive, extremely_expensive
    quality_assessment: str = ""    # high, moderate, low
    growth_assessment: str = ""     # high_growth, moderate_growth, stable, declining

    # DCF / intrinsic value (filled by quant tools)
    intrinsic_value: Optional[float] = None
    intrinsic_value_upside_pct: Optional[float] = None
    dcf_assumptions: str = ""

    # Peer comparison
    peer_tickers: list[str] = Field(default_factory=list)
    peer_ranking: str = ""

    # Thesis
    bull_case: str = ""
    bear_case: str = ""
    key_catalysts: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)

    narrative: str = ""


# ---------------------------------------------------------------------------
# Risk Assessment
# ---------------------------------------------------------------------------

class ScenarioResult(BaseModel):
    """Result of a single scenario simulation."""
    scenario_name: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    impact_description: str = ""
    estimated_impact_pct: Optional[float] = None


class RiskAssessment(BaseModel):
    """Structured output from the risk assessment phase."""
    scenarios: list[ScenarioResult] = Field(default_factory=list)
    key_risk_factors: list[str] = Field(default_factory=list)
    risk_reward_summary: str = ""
    max_drawdown_estimate_pct: Optional[float] = None
    confidence_in_thesis: float = Field(default=0.5, ge=0.0, le=1.0)
    validation_warnings: list[str] = Field(default_factory=list, description="Flags from symbolic validators")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class SynthesisReport(BaseModel):
    """Final synthesized output combining all reasoning phases."""
    executive_summary: str = ""
    regime_context_summary: str = ""
    causal_analysis_summary: str = ""
    sector_analysis_summary: str = ""
    company_analysis_summary: str = ""
    risk_assessment_summary: str = ""

    recommendation: str = ""  # strong_buy, buy, hold, sell, strong_sell, or descriptive
    conviction_level: str = ""  # high, moderate, low
    time_horizon: str = ""      # short_term, medium_term, long_term

    key_insights: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    full_report: str = Field(default="", description="Full narrative report for the user")
