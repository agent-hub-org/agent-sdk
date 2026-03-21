"""
Financial Reasoning Brain — structured multi-dimensional analysis for Indian markets.

Modules:
- schemas: Typed output schemas for each reasoning phase
- causal_graph: NetworkX-based financial causal knowledge graph
- ontology: Financial metric definitions, sector norms, interpretive thresholds
- quant_tools: DCF, comparable valuation, scenario simulation, technical analysis
- validators: Symbolic validation layer (accounting identities, logical consistency)
- prompts: Phase-specific system prompts for the cognitive pipeline
"""

from agent_sdk.financial.schemas import (
    QueryClassification,
    RegimeContext,
    CausalChain,
    SectorFindings,
    CompanyAnalysis,
    RiskAssessment,
    SynthesisReport,
)

__all__ = [
    "QueryClassification",
    "RegimeContext",
    "CausalChain",
    "SectorFindings",
    "CompanyAnalysis",
    "RiskAssessment",
    "SynthesisReport",
]
