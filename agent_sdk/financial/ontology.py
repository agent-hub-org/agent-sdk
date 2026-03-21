"""
Financial Ontology — interpretive context for financial metrics.

Provides metric definitions, sector-specific norms, interpretive thresholds,
and Indian market specifics. Exposed as LangChain StructuredTools.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger("agent_sdk.financial.ontology")


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

class MetricDefinition(BaseModel):
    """Definition and interpretation guide for a financial metric."""
    name: str
    full_name: str
    formula: str
    what_it_measures: str
    higher_is: str  # "better", "worse", or "neutral"
    limitations: list[str] = Field(default_factory=list)
    subtypes: list[str] = Field(default_factory=list)
    indian_market_notes: str = ""


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "PE": MetricDefinition(
        name="PE", full_name="Price-to-Earnings Ratio",
        formula="Market Price / Earnings Per Share",
        what_it_measures="How much investors pay per rupee of earnings",
        higher_is="neutral",
        subtypes=["trailing_PE (TTM)", "forward_PE (consensus estimates)"],
        limitations=[
            "Meaningless for loss-making companies",
            "Can be distorted by one-time items",
            "Doesn't account for debt levels",
            "Growth companies legitimately trade at higher PE",
        ],
        indian_market_notes="Indian market historically trades at PE premium to other EMs due to growth premium and SIP flows",
    ),
    "PB": MetricDefinition(
        name="PB", full_name="Price-to-Book Ratio",
        formula="Market Price / Book Value Per Share",
        what_it_measures="How much investors pay relative to net asset value",
        higher_is="neutral",
        subtypes=["adjusted_PB (excludes intangibles)"],
        limitations=[
            "Book value can be understated for asset-light companies",
            "Not meaningful for tech/services companies",
            "Historical cost accounting may not reflect true asset value",
        ],
        indian_market_notes="Most relevant for banking (PB of 1-3x), metals, and capital-intensive sectors",
    ),
    "EV_EBITDA": MetricDefinition(
        name="EV_EBITDA", full_name="Enterprise Value to EBITDA",
        formula="(Market Cap + Debt - Cash) / EBITDA",
        what_it_measures="Capital-structure-neutral valuation — what an acquirer would pay for operating earnings",
        higher_is="neutral",
        subtypes=["trailing EV/EBITDA", "forward EV/EBITDA"],
        limitations=[
            "Ignores capex requirements (high-capex businesses look cheaper)",
            "EBITDA can be manipulated via lease accounting",
            "Not suitable for financial companies",
        ],
        indian_market_notes="Preferred by institutional investors for cross-border comparison",
    ),
    "ROE": MetricDefinition(
        name="ROE", full_name="Return on Equity",
        formula="Net Income / Shareholders' Equity",
        what_it_measures="How efficiently a company generates profits from shareholders' capital",
        higher_is="better",
        limitations=[
            "Can be artificially inflated by high leverage",
            "Buybacks reduce equity, inflating ROE",
            "One-time items can distort",
        ],
        indian_market_notes="Quality threshold in India: ROE > 15% is good, > 20% is excellent. Banks: ROE > 14% is good.",
    ),
    "ROCE": MetricDefinition(
        name="ROCE", full_name="Return on Capital Employed",
        formula="EBIT / (Total Assets - Current Liabilities)",
        what_it_measures="How efficiently ALL capital (equity + debt) generates returns",
        higher_is="better",
        limitations=[
            "Capital-intensive businesses naturally have lower ROCE",
            "Can be distorted by large cash balances",
        ],
        indian_market_notes="Better than ROE for comparing across capital structures. > 15% is good for most Indian companies.",
    ),
    "DEBT_EQUITY": MetricDefinition(
        name="DEBT_EQUITY", full_name="Debt-to-Equity Ratio",
        formula="Total Debt / Shareholders' Equity",
        what_it_measures="Financial leverage — how much debt finances the business",
        higher_is="worse",
        limitations=[
            "Banks and NBFCs naturally have high D/E — use different metrics",
            "Off-balance-sheet liabilities not captured",
        ],
        indian_market_notes="< 0.5 is conservative, 0.5-1.0 is moderate, > 1.0 is leveraged (ex-financials). Indian cos historically prefer lower leverage.",
    ),
    "DIVIDEND_YIELD": MetricDefinition(
        name="DIVIDEND_YIELD", full_name="Dividend Yield",
        formula="Annual Dividend Per Share / Market Price",
        what_it_measures="Cash return to shareholders as percentage of price",
        higher_is="better",
        limitations=[
            "High yield can signal price decline rather than generous payout",
            "Doesn't capture buybacks",
            "Dividend can be cut",
        ],
        indian_market_notes="Indian market avg yield is 1.2-1.5%. PSUs typically yield higher (3-5%). ITC is a notable high-yield private company.",
    ),
    "PROMOTER_HOLDING": MetricDefinition(
        name="PROMOTER_HOLDING", full_name="Promoter Shareholding %",
        formula="Shares held by promoters / Total shares outstanding",
        what_it_measures="Skin-in-the-game of the founding/controlling group",
        higher_is="better",
        limitations=[
            "Very high holding (>75%) can mean low float and governance risk",
            "MNC subsidiaries naturally have high promoter holding",
        ],
        indian_market_notes="Unique to Indian markets. > 50% is generally positive. Declining trend is a red flag. Pledge % is critical — high pledge (>20% of holding) is dangerous.",
    ),
    "PROMOTER_PLEDGE": MetricDefinition(
        name="PROMOTER_PLEDGE", full_name="Promoter Shares Pledged %",
        formula="Pledged promoter shares / Total promoter shares",
        what_it_measures="Proportion of promoter shares used as loan collateral",
        higher_is="worse",
        limitations=["May not capture pledges through complex holding structures"],
        indian_market_notes="CRITICAL risk indicator in India. > 10% is a yellow flag, > 20% is a red flag, > 40% is a serious risk of forced selling cascade.",
    ),
    "FCF": MetricDefinition(
        name="FCF", full_name="Free Cash Flow",
        formula="Operating Cash Flow - Capital Expenditure",
        what_it_measures="Cash available after maintaining/expanding the asset base",
        higher_is="better",
        limitations=[
            "Capex timing can create volatility",
            "Growth companies may legitimately have negative FCF",
        ],
    ),
    "INTEREST_COVERAGE": MetricDefinition(
        name="INTEREST_COVERAGE", full_name="Interest Coverage Ratio",
        formula="EBIT / Interest Expense",
        what_it_measures="Ability to service debt from operating profits",
        higher_is="better",
        limitations=["Doesn't capture principal repayment obligations"],
        indian_market_notes="> 3x is comfortable, < 1.5x is concerning. In rate hiking cycles, watch this closely.",
    ),
    "NIM": MetricDefinition(
        name="NIM", full_name="Net Interest Margin",
        formula="(Interest Income - Interest Expense) / Average Interest-Earning Assets",
        what_it_measures="Core profitability of lending businesses",
        higher_is="better",
        limitations=["Only applicable to banks and NBFCs"],
        indian_market_notes="Indian private banks: 3.5-4.5% is good. PSU banks: 2.5-3.5%. NBFCs: 5-10% depending on segment.",
    ),
    "GNPA": MetricDefinition(
        name="GNPA", full_name="Gross Non-Performing Assets %",
        formula="Gross NPAs / Gross Advances",
        what_it_measures="Asset quality — proportion of loans in default",
        higher_is="worse",
        limitations=["Restructured loans may not be classified as NPA"],
        indian_market_notes="Private banks: < 2% is excellent, 2-4% is acceptable. PSU banks: historically 5-15%, improving trend.",
    ),
}


# ---------------------------------------------------------------------------
# Sector Norms — valuation ranges by sector for Indian markets
# ---------------------------------------------------------------------------

class SectorNorms(BaseModel):
    """Typical valuation and fundamental ranges for a sector."""
    sector: str
    pe_range: tuple[float, float] = (0, 0)   # (low, high) — typical trailing PE range
    pe_median: float = 0
    pb_range: tuple[float, float] = (0, 0)
    pb_median: float = 0
    ev_ebitda_range: tuple[float, float] = (0, 0)
    roe_typical: tuple[float, float] = (0, 0)
    debt_equity_typical: tuple[float, float] = (0, 0)
    key_metrics: list[str] = Field(default_factory=list, description="Most important metrics for this sector")
    notes: str = ""


SECTOR_NORMS: dict[str, SectorNorms] = {
    "banking": SectorNorms(
        sector="Banking",
        pe_range=(8, 20), pe_median=13.5,
        pb_range=(0.8, 3.5), pb_median=1.8,
        roe_typical=(10, 18),
        debt_equity_typical=(6, 12),  # banks naturally have high leverage
        key_metrics=["PB", "ROE", "NIM", "GNPA", "CASA_RATIO"],
        notes="PE is less meaningful for banks — use PB and ROE. NIM expansion/compression is the key driver. Watch credit costs.",
    ),
    "nbfc": SectorNorms(
        sector="NBFC",
        pe_range=(12, 35), pe_median=22,
        pb_range=(2, 8), pb_median=4,
        roe_typical=(15, 25),
        key_metrics=["PB", "ROE", "NIM", "GNPA", "AUM_GROWTH"],
        notes="Premium NBFCs (Bajaj Finance) trade at significant premium to banks. AUM growth and asset quality are key.",
    ),
    "it_services": SectorNorms(
        sector="IT Services",
        pe_range=(18, 35), pe_median=25,
        pb_range=(5, 15), pb_median=8,
        ev_ebitda_range=(12, 25),
        roe_typical=(25, 40),
        debt_equity_typical=(0, 0.3),
        key_metrics=["PE", "EV_EBITDA", "ROE", "REVENUE_GROWTH", "ATTRITION"],
        notes="Asset-light, high ROE businesses. PE of 25 is median — not expensive for quality names. USD/INR is a key earnings driver. Watch deal TCV and attrition.",
    ),
    "pharma": SectorNorms(
        sector="Pharma",
        pe_range=(15, 35), pe_median=25,
        pb_range=(2, 8), pb_median=4,
        ev_ebitda_range=(10, 22),
        roe_typical=(12, 22),
        debt_equity_typical=(0, 0.5),
        key_metrics=["PE", "EV_EBITDA", "ROE", "ANDA_PIPELINE", "US_REVENUE_SHARE"],
        notes="US generics business drives valuation. ANDA pipeline, FDA observations, and para-IV filings are key catalysts.",
    ),
    "auto": SectorNorms(
        sector="Auto",
        pe_range=(15, 30), pe_median=22,
        pb_range=(3, 8), pb_median=5,
        ev_ebitda_range=(8, 18),
        roe_typical=(12, 22),
        debt_equity_typical=(0, 0.5),
        key_metrics=["PE", "EV_EBITDA", "ROE", "VOLUME_GROWTH", "ASP_TREND"],
        notes="Cyclical sector — PE expands at cycle bottom. Monthly volume data is a leading indicator. EV transition is the structural theme.",
    ),
    "fmcg": SectorNorms(
        sector="FMCG",
        pe_range=(35, 70), pe_median=50,
        pb_range=(10, 30), pb_median=18,
        ev_ebitda_range=(25, 50),
        roe_typical=(25, 60),
        debt_equity_typical=(0, 0.3),
        key_metrics=["PE", "ROE", "VOLUME_GROWTH", "RURAL_MIX"],
        notes="FMCG commands premium multiples in India due to structural consumption story. Volume growth > price-led growth is preferred. Watch rural recovery.",
    ),
    "metals": SectorNorms(
        sector="Metals & Mining",
        pe_range=(5, 15), pe_median=8,
        pb_range=(0.5, 2.5), pb_median=1.2,
        ev_ebitda_range=(3, 8),
        roe_typical=(8, 20),
        debt_equity_typical=(0.5, 1.5),
        key_metrics=["EV_EBITDA", "PB", "EBITDA_PER_TON"],
        notes="Deeply cyclical — buy on high PE (earnings trough), sell on low PE (earnings peak). China demand is the primary driver. EV/EBITDA and EBITDA/ton are preferred metrics.",
    ),
    "realty": SectorNorms(
        sector="Real Estate",
        pe_range=(15, 40), pe_median=25,
        pb_range=(2, 6), pb_median=3.5,
        roe_typical=(8, 18),
        debt_equity_typical=(0.3, 1.0),
        key_metrics=["PB", "PE", "PRESALES", "LAUNCH_PIPELINE", "NET_DEBT"],
        notes="Presales momentum and launch pipeline matter more than trailing earnings. Net debt is a critical risk metric. Interest rate sensitive.",
    ),
    "cement": SectorNorms(
        sector="Cement",
        pe_range=(20, 40), pe_median=30,
        ev_ebitda_range=(10, 20),
        roe_typical=(10, 20),
        key_metrics=["EV_EBITDA", "EV_PER_TON", "EBITDA_PER_TON", "CAPACITY_UTILIZATION"],
        notes="EV/ton is the key valuation metric for M&A and replacement cost comparison. Capacity utilization drives pricing power.",
    ),
    "infra": SectorNorms(
        sector="Infrastructure",
        pe_range=(15, 30), pe_median=22,
        pb_range=(2, 6), pb_median=3.5,
        roe_typical=(12, 20),
        key_metrics=["PE", "ORDER_BOOK", "ORDER_INFLOW", "BTB_RATIO"],
        notes="Order book-to-billing ratio (BTB) and order inflow growth are key. Govt capex budget is the primary driver.",
    ),
    "oil_gas": SectorNorms(
        sector="Oil & Gas",
        pe_range=(6, 15), pe_median=10,
        pb_range=(0.5, 2), pb_median=1.2,
        ev_ebitda_range=(3, 8),
        key_metrics=["PE", "EV_EBITDA", "GRM", "CRUDE_REALIZATION"],
        notes="Differentiate upstream (ONGC — crude realization) vs downstream/OMC (BPCL — GRM and marketing margins). Govt fuel pricing policy is a key overhang for OMCs.",
    ),
    "power": SectorNorms(
        sector="Power",
        pe_range=(10, 20), pe_median=14,
        pb_range=(1, 3), pb_median=1.8,
        roe_typical=(10, 16),
        debt_equity_typical=(1, 3),
        key_metrics=["PE", "PB", "ROE", "PLF", "TARIFF_TRAJECTORY"],
        notes="Regulated returns for utilities. PLF (Plant Load Factor) and tariff revisions are key. Renewables trade at premium multiples.",
    ),
    "telecom": SectorNorms(
        sector="Telecom",
        pe_range=(30, 80), pe_median=50,
        ev_ebitda_range=(6, 12),
        key_metrics=["EV_EBITDA", "ARPU", "SUBSCRIBER_ADDITIONS"],
        notes="ARPU (Average Revenue Per User) growth is the primary driver. High capex industry — EV/EBITDA preferred over PE. India is a 3-player market.",
    ),
    "insurance": SectorNorms(
        sector="Insurance",
        pe_range=(40, 80), pe_median=60,
        key_metrics=["EMBEDDED_VALUE", "VNB_MARGIN", "APE_GROWTH"],
        notes="Use EV (Embedded Value) multiples, not PE. VNB margin (Value of New Business margin) measures profitability of new policies. APE (Annualized Premium Equivalent) measures growth.",
    ),
    "chemicals": SectorNorms(
        sector="Chemicals",
        pe_range=(20, 45), pe_median=30,
        ev_ebitda_range=(12, 25),
        roe_typical=(15, 25),
        key_metrics=["PE", "EV_EBITDA", "ROE", "CAPEX_PIPELINE"],
        notes="China+1 theme is structural. Capex pipeline and commissioning timelines drive re-ratings. Watch crude/feedstock costs.",
    ),
    "defence": SectorNorms(
        sector="Defence",
        pe_range=(25, 50), pe_median=35,
        key_metrics=["PE", "ORDER_BOOK", "ORDER_INFLOW", "REVENUE_VISIBILITY"],
        notes="Long order book cycles (3-7 years). Order inflow from MoD is the key catalyst. Make in India policy is structural tailwind.",
    ),
}


# ---------------------------------------------------------------------------
# Interpretation Functions
# ---------------------------------------------------------------------------

def interpret_metric(metric: str, value: float, sector: str | None = None,
                     market: str = "india") -> dict:
    """
    Interpret a financial metric value in context.
    Returns assessment (cheap/fair/expensive for valuation, or good/moderate/poor for quality)
    with sector-specific context.
    """
    metric_upper = metric.upper().replace("/", "_").replace("-", "_")
    metric_def = METRIC_DEFINITIONS.get(metric_upper)

    result = {
        "metric": metric,
        "value": value,
        "sector": sector,
        "market": market,
    }

    if metric_def:
        result["definition"] = metric_def.full_name
        result["formula"] = metric_def.formula
        result["what_it_measures"] = metric_def.what_it_measures
        result["limitations"] = metric_def.limitations
        if metric_def.indian_market_notes:
            result["indian_market_context"] = metric_def.indian_market_notes

    # Sector-specific interpretation
    sector_key = _normalize_sector(sector) if sector else None
    norms = SECTOR_NORMS.get(sector_key) if sector_key else None

    if norms:
        result["sector_norms"] = {"sector": norms.sector}

        if metric_upper == "PE" and norms.pe_range != (0, 0):
            low, high = norms.pe_range
            median = norms.pe_median
            result["sector_norms"].update({
                "typical_range": f"{low}-{high}",
                "median": median,
            })
            if value < low:
                result["assessment"] = "cheap"
                result["interpretation"] = f"PE of {value} is below the typical {norms.sector} range ({low}-{high}). May indicate undervaluation or earnings concerns."
            elif value < median:
                result["assessment"] = "below_median"
                result["interpretation"] = f"PE of {value} is below the {norms.sector} median of {median}. Moderately attractive if fundamentals are intact."
            elif value <= high:
                result["assessment"] = "fair_to_elevated"
                result["interpretation"] = f"PE of {value} is above the {norms.sector} median ({median}) but within typical range. Growth premium may be justified."
            else:
                result["assessment"] = "expensive"
                result["interpretation"] = f"PE of {value} is above the typical {norms.sector} range ({low}-{high}). Requires strong growth to justify."

        elif metric_upper == "PB" and norms.pb_range != (0, 0):
            low, high = norms.pb_range
            median = norms.pb_median
            result["sector_norms"].update({"typical_range": f"{low}-{high}", "median": median})
            if value < low:
                result["assessment"] = "cheap"
                result["interpretation"] = f"PB of {value} is below typical {norms.sector} range. May indicate deep value or fundamental deterioration."
            elif value <= high:
                result["assessment"] = "fair"
                result["interpretation"] = f"PB of {value} is within the typical {norms.sector} range ({low}-{high})."
            else:
                result["assessment"] = "expensive"
                result["interpretation"] = f"PB of {value} is above the typical {norms.sector} range."

        elif metric_upper == "ROE" and norms.roe_typical != (0, 0):
            low, high = norms.roe_typical
            result["sector_norms"].update({"typical_range": f"{low}%-{high}%"})
            if value < low:
                result["assessment"] = "poor"
                result["interpretation"] = f"ROE of {value}% is below the typical {norms.sector} range ({low}%-{high}%). Indicates poor capital efficiency."
            elif value <= high:
                result["assessment"] = "good"
                result["interpretation"] = f"ROE of {value}% is within the typical {norms.sector} range."
            else:
                result["assessment"] = "excellent"
                result["interpretation"] = f"ROE of {value}% is above the typical {norms.sector} range. Superior capital efficiency."

        elif metric_upper in ("EV_EBITDA", "EVEBITDA") and norms.ev_ebitda_range != (0, 0):
            low, high = norms.ev_ebitda_range
            result["sector_norms"].update({"typical_range": f"{low}-{high}"})
            if value < low:
                result["assessment"] = "cheap"
                result["interpretation"] = f"EV/EBITDA of {value} is below typical {norms.sector} range ({low}-{high})."
            elif value <= high:
                result["assessment"] = "fair"
                result["interpretation"] = f"EV/EBITDA of {value} is within the typical {norms.sector} range."
            else:
                result["assessment"] = "expensive"
                result["interpretation"] = f"EV/EBITDA of {value} is above the typical {norms.sector} range."

        if norms.notes:
            result["sector_notes"] = norms.notes
        result["sector_key_metrics"] = norms.key_metrics
    else:
        # Generic interpretation without sector context
        if metric_upper == "PE":
            if value < 10:
                result["assessment"] = "low"
                result["interpretation"] = "Low PE — may indicate value, cyclicality, or earnings risk."
            elif value < 20:
                result["assessment"] = "moderate"
                result["interpretation"] = "Moderate PE — reasonable for stable businesses."
            elif value < 35:
                result["assessment"] = "elevated"
                result["interpretation"] = "Elevated PE — growth premium expected."
            else:
                result["assessment"] = "high"
                result["interpretation"] = "High PE — requires strong and sustained earnings growth to justify."
            result["note"] = "Sector context not provided — interpretation is generic. Provide sector for more accurate assessment."

    return result


def get_metric_definition(metric: str) -> dict:
    """Return the full definition for a financial metric."""
    metric_upper = metric.upper().replace("/", "_").replace("-", "_")
    defn = METRIC_DEFINITIONS.get(metric_upper)
    if defn:
        return defn.model_dump()
    return {"error": f"Metric '{metric}' not found. Available: {list(METRIC_DEFINITIONS.keys())}"}


def get_sector_norms(sector: str) -> dict:
    """Return valuation and fundamental norms for a sector."""
    key = _normalize_sector(sector)
    norms = SECTOR_NORMS.get(key)
    if norms:
        return norms.model_dump()
    return {"error": f"Sector '{sector}' not found. Available: {list(SECTOR_NORMS.keys())}"}


def list_sectors() -> list[str]:
    """Return all sectors with defined norms."""
    return list(SECTOR_NORMS.keys())


def _normalize_sector(sector: str) -> str | None:
    """Normalize a sector name to match SECTOR_NORMS keys."""
    if not sector:
        return None
    s = sector.lower().strip()
    # Direct match
    if s in SECTOR_NORMS:
        return s
    # Common aliases
    aliases = {
        "bank": "banking", "banks": "banking", "private banks": "banking", "psu banks": "banking",
        "it": "it_services", "technology": "it_services", "tech": "it_services", "software": "it_services",
        "pharmaceutical": "pharma", "pharmaceuticals": "pharma",
        "automobile": "auto", "automobiles": "auto", "automotive": "auto",
        "consumer": "fmcg", "consumer staples": "fmcg", "consumer goods": "fmcg",
        "metal": "metals", "mining": "metals", "steel": "metals",
        "real estate": "realty", "property": "realty", "housing": "realty",
        "infrastructure": "infra", "construction": "infra",
        "oil": "oil_gas", "oil and gas": "oil_gas", "energy": "oil_gas", "petroleum": "oil_gas",
        "chemical": "chemicals",
        "insurance": "insurance", "life insurance": "insurance",
        "cement": "cement",
        "power": "power", "utilities": "power", "electricity": "power",
        "telecom": "telecom", "telecommunications": "telecom",
        "defence": "defence", "defense": "defence",
    }
    return aliases.get(s)


# ---------------------------------------------------------------------------
# LangChain StructuredTools
# ---------------------------------------------------------------------------

class InterpretMetricInput(BaseModel):
    metric: str = Field(description="Metric name (e.g., 'PE', 'PB', 'ROE', 'EV_EBITDA', 'DEBT_EQUITY')")
    value: float = Field(description="The metric value to interpret")
    sector: Optional[str] = Field(default=None, description="Sector for context (e.g., 'banking', 'it_services', 'auto')")
    market: str = Field(default="india", description="Market context")


class GetMetricDefinitionInput(BaseModel):
    metric: str = Field(description="Metric name to look up")


class GetSectorNormsInput(BaseModel):
    sector: str = Field(description="Sector name (e.g., 'banking', 'it_services', 'pharma', 'auto', 'fmcg')")


def get_ontology_tools() -> list[StructuredTool]:
    """Return LangChain StructuredTools for financial ontology operations."""
    return [
        StructuredTool.from_function(
            func=lambda **kwargs: interpret_metric(**kwargs),
            name="interpret_metric",
            description=(
                "Interpret a financial metric value in sector and market context. "
                "Returns whether the value is cheap/fair/expensive (for valuation metrics) "
                "or good/moderate/poor (for quality metrics) with sector-specific context. "
                "Example: interpret_metric(metric='PE', value=25, sector='banking')"
            ),
            args_schema=InterpretMetricInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: get_metric_definition(**kwargs),
            name="get_metric_definition",
            description=(
                "Get the full definition of a financial metric including formula, "
                "what it measures, limitations, and Indian market notes. "
                "Example: get_metric_definition(metric='ROE')"
            ),
            args_schema=GetMetricDefinitionInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: get_sector_norms(**kwargs),
            name="get_sector_norms",
            description=(
                "Get typical valuation ranges and key metrics for an Indian market sector. "
                "Example: get_sector_norms(sector='banking')"
            ),
            args_schema=GetSectorNormsInput,
        ),
    ]
