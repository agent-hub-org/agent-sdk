"""
Financial Causal Knowledge Graph — NetworkX-based directed graph of causal
relationships in the Indian financial system.

Nodes: macro indicators, sectors, companies, financial concepts.
Edges: causal relationships with direction, magnitude, time lag, and confidence.

Exposed as LangChain StructuredTools for use in the cognitive pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger("agent_sdk.financial.causal_graph")


# ---------------------------------------------------------------------------
# Node & Edge Types
# ---------------------------------------------------------------------------

class CausalNode(BaseModel):
    """A node in the causal graph."""
    id: str
    label: str
    category: str  # macro_indicator, sector, company, concept, commodity, currency, policy
    description: str = ""
    market: str = "india"


class CausalEdge(BaseModel):
    """A directed causal edge."""
    source: str
    target: str
    direction: str = "positive"    # positive or negative
    magnitude: str = "moderate"    # weak, moderate, strong
    time_lag: str = "1-2Q"         # immediate, 1-2Q, 2-4Q, 1-2Y
    confidence: str = "well-established"  # well-established, theoretical, regime-dependent
    mechanism: str = ""            # how the causal link works


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def _build_graph() -> nx.DiGraph:
    """Build the Indian financial markets causal knowledge graph."""
    G = nx.DiGraph()

    # ===== NODES =====

    nodes = [
        # --- Macro Indicators ---
        CausalNode(id="repo_rate", label="RBI Repo Rate", category="policy",
                   description="RBI's benchmark lending rate"),
        CausalNode(id="reverse_repo", label="RBI Reverse Repo Rate", category="policy",
                   description="Rate at which RBI borrows from banks"),
        CausalNode(id="crr", label="CRR", category="policy",
                   description="Cash Reserve Ratio — fraction of deposits banks must hold with RBI"),
        CausalNode(id="cpi", label="CPI Inflation", category="macro_indicator",
                   description="Consumer Price Index YoY change"),
        CausalNode(id="wpi", label="WPI Inflation", category="macro_indicator",
                   description="Wholesale Price Index YoY change"),
        CausalNode(id="iip", label="IIP Growth", category="macro_indicator",
                   description="Index of Industrial Production YoY"),
        CausalNode(id="pmi_mfg", label="PMI Manufacturing", category="macro_indicator",
                   description="Purchasing Managers Index — Manufacturing"),
        CausalNode(id="pmi_svc", label="PMI Services", category="macro_indicator",
                   description="Purchasing Managers Index — Services"),
        CausalNode(id="gdp_growth", label="GDP Growth", category="macro_indicator",
                   description="India real GDP growth rate"),
        CausalNode(id="fiscal_deficit", label="Fiscal Deficit", category="macro_indicator",
                   description="Government fiscal deficit as % of GDP"),
        CausalNode(id="cad", label="Current Account Deficit", category="macro_indicator",
                   description="India's current account deficit as % of GDP"),
        CausalNode(id="gsec_10y", label="10Y G-Sec Yield", category="macro_indicator",
                   description="India 10-year government bond yield"),
        CausalNode(id="credit_growth", label="Bank Credit Growth", category="macro_indicator",
                   description="YoY growth in bank credit to commercial sector"),
        CausalNode(id="india_vix", label="India VIX", category="macro_indicator",
                   description="Volatility index derived from Nifty 50 options"),

        # --- Global / Commodity ---
        CausalNode(id="crude_oil", label="Brent Crude Oil", category="commodity",
                   description="Brent crude oil price USD/barrel"),
        CausalNode(id="gold", label="Gold Price", category="commodity",
                   description="International gold price"),
        CausalNode(id="usd_inr", label="USD/INR", category="currency",
                   description="US Dollar to Indian Rupee exchange rate"),
        CausalNode(id="dxy", label="US Dollar Index (DXY)", category="currency",
                   description="Broad USD strength index"),
        CausalNode(id="us_fed_rate", label="US Fed Funds Rate", category="policy",
                   description="US Federal Reserve benchmark rate"),
        CausalNode(id="us_10y", label="US 10Y Treasury Yield", category="macro_indicator",
                   description="US 10-year Treasury yield"),
        CausalNode(id="china_pmi", label="China PMI", category="macro_indicator",
                   description="China manufacturing PMI"),
        CausalNode(id="global_risk_appetite", label="Global Risk Appetite", category="concept",
                   description="Aggregate risk-on/risk-off sentiment"),

        # --- Flow Indicators ---
        CausalNode(id="fii_flows", label="FII Net Flows", category="macro_indicator",
                   description="Foreign Institutional Investor net equity flows"),
        CausalNode(id="dii_flows", label="DII Net Flows", category="macro_indicator",
                   description="Domestic Institutional Investor net equity flows"),
        CausalNode(id="sip_flows", label="SIP Monthly Flows", category="macro_indicator",
                   description="Systematic Investment Plan monthly inflows"),
        CausalNode(id="mf_equity_flows", label="MF Equity Flows", category="macro_indicator",
                   description="Mutual fund net equity flows"),

        # --- Transmission Concepts ---
        CausalNode(id="lending_rates", label="Bank Lending Rates", category="concept",
                   description="MCLR / external benchmark lending rates"),
        CausalNode(id="deposit_rates", label="Bank Deposit Rates", category="concept",
                   description="Fixed deposit and savings rates"),
        CausalNode(id="liquidity", label="Banking System Liquidity", category="concept",
                   description="Surplus/deficit liquidity in the banking system"),
        CausalNode(id="rupee_depreciation", label="Rupee Depreciation Pressure", category="concept"),
        CausalNode(id="rupee_appreciation", label="Rupee Appreciation Pressure", category="concept"),
        CausalNode(id="input_cost_pressure", label="Input Cost Pressure", category="concept"),
        CausalNode(id="consumer_demand", label="Consumer Demand", category="concept"),
        CausalNode(id="capex_cycle", label="Corporate Capex Cycle", category="concept"),
        CausalNode(id="govt_capex", label="Government Capex", category="concept"),
        CausalNode(id="export_competitiveness", label="Export Competitiveness", category="concept"),
        CausalNode(id="import_bill", label="India Import Bill", category="concept"),
        CausalNode(id="transportation_cost", label="Transportation Cost", category="concept"),
        CausalNode(id="food_inflation", label="Food Inflation", category="concept"),
        CausalNode(id="rural_demand", label="Rural Demand", category="concept"),
        CausalNode(id="urban_demand", label="Urban Demand", category="concept"),
        CausalNode(id="auto_loan_rates", label="Auto Loan Rates", category="concept"),
        CausalNode(id="home_loan_rates", label="Home Loan Rates", category="concept"),
        CausalNode(id="corporate_bond_spreads", label="Corporate Bond Spreads", category="concept"),
        CausalNode(id="npa_pressure", label="NPA / Asset Quality Pressure", category="concept"),
        CausalNode(id="margin_pressure", label="Margin Pressure (General)", category="concept"),

        # --- Sectors ---
        CausalNode(id="sector_banking", label="Banking Sector", category="sector"),
        CausalNode(id="sector_nbfc", label="NBFC Sector", category="sector"),
        CausalNode(id="sector_it", label="IT Services Sector", category="sector"),
        CausalNode(id="sector_pharma", label="Pharma Sector", category="sector"),
        CausalNode(id="sector_auto", label="Auto Sector", category="sector"),
        CausalNode(id="sector_auto_ancillary", label="Auto Ancillary Sector", category="sector"),
        CausalNode(id="sector_fmcg", label="FMCG Sector", category="sector"),
        CausalNode(id="sector_metals", label="Metals & Mining Sector", category="sector"),
        CausalNode(id="sector_realty", label="Real Estate Sector", category="sector"),
        CausalNode(id="sector_infra", label="Infrastructure Sector", category="sector"),
        CausalNode(id="sector_cement", label="Cement Sector", category="sector"),
        CausalNode(id="sector_chemicals", label="Chemicals Sector", category="sector"),
        CausalNode(id="sector_oil_gas", label="Oil & Gas Sector", category="sector"),
        CausalNode(id="sector_omc", label="Oil Marketing Companies", category="sector"),
        CausalNode(id="sector_upstream_oil", label="Upstream Oil & Gas", category="sector"),
        CausalNode(id="sector_power", label="Power Sector", category="sector"),
        CausalNode(id="sector_telecom", label="Telecom Sector", category="sector"),
        CausalNode(id="sector_insurance", label="Insurance Sector", category="sector"),
        CausalNode(id="sector_defence", label="Defence Sector", category="sector"),
        CausalNode(id="sector_textiles", label="Textiles Sector", category="sector"),
        CausalNode(id="sector_sugar", label="Sugar Sector", category="sector"),
        CausalNode(id="sector_aviation", label="Aviation Sector", category="sector"),
        CausalNode(id="sector_hotels", label="Hotels & Tourism Sector", category="sector"),
        CausalNode(id="sector_media", label="Media & Entertainment Sector", category="sector"),

        # --- Key Companies ---
        CausalNode(id="RELIANCE", label="Reliance Industries", category="company"),
        CausalNode(id="TCS", label="Tata Consultancy Services", category="company"),
        CausalNode(id="INFY", label="Infosys", category="company"),
        CausalNode(id="HDFCBANK", label="HDFC Bank", category="company"),
        CausalNode(id="ICICIBANK", label="ICICI Bank", category="company"),
        CausalNode(id="SBIN", label="State Bank of India", category="company"),
        CausalNode(id="KOTAKBANK", label="Kotak Mahindra Bank", category="company"),
        CausalNode(id="AXISBANK", label="Axis Bank", category="company"),
        CausalNode(id="BAJFINANCE", label="Bajaj Finance", category="company"),
        CausalNode(id="HINDUNILVR", label="Hindustan Unilever", category="company"),
        CausalNode(id="ITC", label="ITC Limited", category="company"),
        CausalNode(id="MARUTI", label="Maruti Suzuki", category="company"),
        CausalNode(id="TATAMOTORS", label="Tata Motors", category="company"),
        CausalNode(id="M_M", label="Mahindra & Mahindra", category="company"),
        CausalNode(id="SUNPHARMA", label="Sun Pharma", category="company"),
        CausalNode(id="DRREDDY", label="Dr. Reddy's Labs", category="company"),
        CausalNode(id="CIPLA", label="Cipla", category="company"),
        CausalNode(id="BHARTIARTL", label="Bharti Airtel", category="company"),
        CausalNode(id="TATASTEEL", label="Tata Steel", category="company"),
        CausalNode(id="HINDALCO", label="Hindalco Industries", category="company"),
        CausalNode(id="JSWSTEEL", label="JSW Steel", category="company"),
        CausalNode(id="ULTRACEMCO", label="UltraTech Cement", category="company"),
        CausalNode(id="SHREECEM", label="Shree Cement", category="company"),
        CausalNode(id="AMBUJACEM", label="Ambuja Cements", category="company"),
        CausalNode(id="BPCL", label="Bharat Petroleum", category="company"),
        CausalNode(id="HPCL", label="Hindustan Petroleum", category="company"),
        CausalNode(id="IOC", label="Indian Oil Corporation", category="company"),
        CausalNode(id="ONGC", label="Oil & Natural Gas Corp", category="company"),
        CausalNode(id="OILIND", label="Oil India", category="company"),
        CausalNode(id="NTPC", label="NTPC Limited", category="company"),
        CausalNode(id="POWERGRID", label="Power Grid Corp", category="company"),
        CausalNode(id="LT", label="Larsen & Toubro", category="company"),
        CausalNode(id="GODREJCP", label="Godrej Consumer Products", category="company"),
        CausalNode(id="DABUR", label="Dabur India", category="company"),
        CausalNode(id="BRITANNIA", label="Britannia Industries", category="company"),
        CausalNode(id="NESTLEIND", label="Nestle India", category="company"),
        CausalNode(id="ASIANPAINT", label="Asian Paints", category="company"),
        CausalNode(id="PIDILITIND", label="Pidilite Industries", category="company"),
        CausalNode(id="DLF", label="DLF Limited", category="company"),
        CausalNode(id="GODREJPROP", label="Godrej Properties", category="company"),
        CausalNode(id="HDFCLIFE", label="HDFC Life Insurance", category="company"),
        CausalNode(id="SBILIFE", label="SBI Life Insurance", category="company"),
        CausalNode(id="HAL", label="Hindustan Aeronautics", category="company"),
        CausalNode(id="BEL", label="Bharat Electronics", category="company"),
        CausalNode(id="WIPRO", label="Wipro", category="company"),
        CausalNode(id="HCLTECH", label="HCL Technologies", category="company"),
        CausalNode(id="TECHM", label="Tech Mahindra", category="company"),
        CausalNode(id="BAJAJ_AUTO", label="Bajaj Auto", category="company"),
        CausalNode(id="HEROMOTOCO", label="Hero MotoCorp", category="company"),
        CausalNode(id="EICHERMOT", label="Eicher Motors", category="company"),
        CausalNode(id="TITAN", label="Titan Company", category="company"),
        CausalNode(id="INDIGO", label="InterGlobe Aviation (IndiGo)", category="company"),
        CausalNode(id="PIIND", label="PI Industries", category="company"),
        CausalNode(id="SRF", label="SRF Limited", category="company"),
        CausalNode(id="ADANIENT", label="Adani Enterprises", category="company"),
        CausalNode(id="ADANIPORTS", label="Adani Ports", category="company"),
        CausalNode(id="ADANIGREEN", label="Adani Green Energy", category="company"),
    ]

    for node in nodes:
        G.add_node(node.id, **node.model_dump())

    # ===== EDGES =====
    edges = [
        # --- RBI Repo Rate Transmission ---
        CausalEdge(source="repo_rate", target="lending_rates", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Banks adjust MCLR/EBLR in line with repo rate changes"),
        CausalEdge(source="repo_rate", target="deposit_rates", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Banks adjust FD rates with a lag after repo changes"),
        CausalEdge(source="repo_rate", target="gsec_10y", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Bond yields move with policy rate expectations"),
        CausalEdge(source="repo_rate", target="liquidity", direction="negative", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Higher repo rate tightens banking system liquidity"),
        CausalEdge(source="repo_rate", target="corporate_bond_spreads", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Higher rates increase corporate borrowing costs and risk premiums"),

        # --- Lending Rate Transmission ---
        CausalEdge(source="lending_rates", target="auto_loan_rates", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Auto loan EMIs directly tied to benchmark lending rates"),
        CausalEdge(source="lending_rates", target="home_loan_rates", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Home loan EMIs directly tied to EBLR"),
        CausalEdge(source="lending_rates", target="consumer_demand", direction="negative", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Higher borrowing costs reduce consumer spending on credit-driven purchases"),
        CausalEdge(source="lending_rates", target="capex_cycle", direction="negative", magnitude="moderate",
                   time_lag="2-4Q", confidence="well-established",
                   mechanism="Higher borrowing costs delay corporate capex decisions"),
        CausalEdge(source="lending_rates", target="sector_nbfc", direction="negative", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="NBFCs face higher cost of funds, compressing NIMs"),

        # --- Auto Loan → Auto Sector Chain ---
        CausalEdge(source="auto_loan_rates", target="sector_auto", direction="negative", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Higher auto loan EMIs reduce unit sales, especially in entry-level segment"),
        CausalEdge(source="sector_auto", target="sector_auto_ancillary", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Auto OEM production directly drives ancillary demand"),
        CausalEdge(source="sector_auto", target="MARUTI", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Market leader in passenger vehicles — most exposed to auto cycle"),
        CausalEdge(source="sector_auto", target="TATAMOTORS", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_auto", target="M_M", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_auto", target="BAJAJ_AUTO", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_auto", target="HEROMOTOCO", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Two-wheeler demand highly sensitive to rural demand and financing costs"),
        CausalEdge(source="sector_auto", target="EICHERMOT", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),

        # --- Home Loan → Real Estate Chain ---
        CausalEdge(source="home_loan_rates", target="sector_realty", direction="negative", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Higher EMIs reduce housing affordability, damping sales volumes"),
        CausalEdge(source="sector_realty", target="sector_cement", direction="positive", magnitude="moderate",
                   time_lag="2-4Q", confidence="well-established",
                   mechanism="Housing construction drives cement demand"),
        CausalEdge(source="sector_realty", target="DLF", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_realty", target="GODREJPROP", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_realty", target="ASIANPAINT", direction="positive", magnitude="moderate",
                   time_lag="2-4Q", confidence="well-established",
                   mechanism="New housing drives paint demand with a lag"),
        CausalEdge(source="sector_realty", target="PIDILITIND", direction="positive", magnitude="moderate",
                   time_lag="2-4Q", confidence="well-established",
                   mechanism="Construction and renovation drive adhesives demand"),

        # --- Crude Oil Transmission ---
        CausalEdge(source="crude_oil", target="sector_omc", direction="negative", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="OMCs face margin compression when crude rises faster than fuel price revisions"),
        CausalEdge(source="crude_oil", target="sector_upstream_oil", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Higher crude prices directly boost upstream realization"),
        CausalEdge(source="crude_oil", target="transportation_cost", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Diesel prices directly affect freight costs"),
        CausalEdge(source="crude_oil", target="import_bill", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="India imports ~85% of crude — largest import component"),
        CausalEdge(source="crude_oil", target="cad", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Oil import bill is the largest driver of India's CAD"),
        CausalEdge(source="crude_oil", target="sector_chemicals", direction="negative", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Crude derivatives are key feedstock for chemicals"),
        CausalEdge(source="crude_oil", target="sector_aviation", direction="negative", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="ATF is 35-40% of airline operating costs"),
        CausalEdge(source="crude_oil", target="INDIGO", direction="negative", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="ATF cost directly impacts IndiGo's operating margins"),
        CausalEdge(source="crude_oil", target="cpi", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Fuel and transportation cost pass-through to consumer prices"),

        CausalEdge(source="sector_omc", target="BPCL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_omc", target="HPCL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_omc", target="IOC", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_upstream_oil", target="ONGC", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_upstream_oil", target="OILIND", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Transportation → FMCG/Consumer ---
        CausalEdge(source="transportation_cost", target="sector_fmcg", direction="negative", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Distribution cost pressure compresses FMCG margins"),
        CausalEdge(source="transportation_cost", target="input_cost_pressure", direction="positive",
                   magnitude="moderate", time_lag="immediate", confidence="well-established"),
        CausalEdge(source="input_cost_pressure", target="margin_pressure", direction="positive",
                   magnitude="moderate", time_lag="1-2Q", confidence="well-established"),

        # --- CAD → Currency → FII Flows ---
        CausalEdge(source="cad", target="rupee_depreciation", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Wider CAD increases demand for USD, weakening INR"),
        CausalEdge(source="rupee_depreciation", target="usd_inr", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="rupee_depreciation", target="fii_flows", direction="negative", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Currency depreciation erodes USD returns for FIIs"),
        CausalEdge(source="rupee_depreciation", target="sector_it", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="IT companies earn in USD — weaker rupee boosts INR revenue"),
        CausalEdge(source="rupee_depreciation", target="sector_pharma", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Export-oriented pharma benefits from weaker rupee"),
        CausalEdge(source="rupee_depreciation", target="import_bill", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),

        # --- USD/INR → IT & Pharma ---
        CausalEdge(source="usd_inr", target="TCS", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="~95% USD revenue — every 1% INR depreciation adds ~40bps to margin"),
        CausalEdge(source="usd_inr", target="INFY", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="usd_inr", target="WIPRO", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="usd_inr", target="HCLTECH", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="usd_inr", target="TECHM", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),

        # --- US Fed Rate → Global Flows ---
        CausalEdge(source="us_fed_rate", target="us_10y", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="us_fed_rate", target="dxy", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Higher US rates strengthen USD"),
        CausalEdge(source="us_fed_rate", target="global_risk_appetite", direction="negative", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Higher US rates reduce EM risk appetite"),
        CausalEdge(source="us_10y", target="fii_flows", direction="negative", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Higher US yields make EM carry trades less attractive"),
        CausalEdge(source="dxy", target="usd_inr", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Stronger USD puts pressure on INR"),
        CausalEdge(source="global_risk_appetite", target="fii_flows", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Risk-on sentiment drives EM allocations"),

        # --- FII/DII Flows → Market ---
        CausalEdge(source="fii_flows", target="sector_banking", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Banks are FII favorites — high weight in indices"),
        CausalEdge(source="fii_flows", target="sector_it", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="dii_flows", target="sector_fmcg", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="DIIs and retail MFs favor stable FMCG names"),
        CausalEdge(source="sip_flows", target="mf_equity_flows", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="SIPs are the primary channel for retail MF equity flows"),
        CausalEdge(source="mf_equity_flows", target="dii_flows", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- CPI Inflation → RBI Response ---
        CausalEdge(source="cpi", target="repo_rate", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="RBI targets 4% CPI — persistent deviation triggers rate action"),
        CausalEdge(source="food_inflation", target="cpi", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Food has ~46% weight in India CPI basket"),
        CausalEdge(source="food_inflation", target="rural_demand", direction="negative", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="High food prices erode rural purchasing power for non-essentials"),

        # --- Rural/Urban Demand → Consumer Sectors ---
        CausalEdge(source="rural_demand", target="sector_fmcg", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Rural India is 35-40% of FMCG revenue"),
        CausalEdge(source="rural_demand", target="HINDUNILVR", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="HUL has deep rural penetration"),
        CausalEdge(source="rural_demand", target="DABUR", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Dabur has ~45% rural revenue mix"),
        CausalEdge(source="rural_demand", target="HEROMOTOCO", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Two-wheelers are strongly tied to rural demand"),
        CausalEdge(source="rural_demand", target="BRITANNIA", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established"),
        CausalEdge(source="urban_demand", target="sector_auto", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established"),
        CausalEdge(source="urban_demand", target="TITAN", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Titan's jewellery and watches are urban discretionary purchases"),
        CausalEdge(source="consumer_demand", target="rural_demand", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="consumer_demand", target="urban_demand", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),

        # --- Banking Sector Specifics ---
        CausalEdge(source="repo_rate", target="sector_banking", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="regime-dependent",
                   mechanism="Rate hikes can expand NIMs if lending rate pass-through is faster than deposit rate increase"),
        CausalEdge(source="credit_growth", target="sector_banking", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Loan book growth drives banking revenue"),
        CausalEdge(source="npa_pressure", target="sector_banking", direction="negative", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Rising NPAs require provisioning, hitting profitability"),
        CausalEdge(source="sector_banking", target="HDFCBANK", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_banking", target="ICICIBANK", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_banking", target="SBIN", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_banking", target="KOTAKBANK", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_banking", target="AXISBANK", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- NBFC Specifics ---
        CausalEdge(source="sector_nbfc", target="BAJFINANCE", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="liquidity", target="sector_nbfc", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="NBFCs depend on market borrowings — tight liquidity squeezes funding"),

        # --- IT Sector Specifics ---
        CausalEdge(source="us_fed_rate", target="sector_it", direction="negative", magnitude="moderate",
                   time_lag="2-4Q", confidence="well-established",
                   mechanism="US rate hikes reduce client IT budgets, impacting deal flow"),
        CausalEdge(source="sector_it", target="TCS", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_it", target="INFY", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_it", target="WIPRO", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_it", target="HCLTECH", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_it", target="TECHM", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Metals Sector ---
        CausalEdge(source="china_pmi", target="sector_metals", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="China is the marginal buyer — Chinese demand sets global metal prices"),
        CausalEdge(source="sector_metals", target="TATASTEEL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_metals", target="JSWSTEEL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_metals", target="HINDALCO", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="capex_cycle", target="sector_metals", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Infrastructure and construction capex drives domestic steel demand"),

        # --- Cement ---
        CausalEdge(source="sector_cement", target="ULTRACEMCO", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_cement", target="SHREECEM", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_cement", target="AMBUJACEM", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="capex_cycle", target="sector_cement", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Infrastructure projects are large cement consumers"),
        CausalEdge(source="govt_capex", target="sector_cement", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established"),

        # --- Government Capex → Infrastructure ---
        CausalEdge(source="govt_capex", target="sector_infra", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Govt infrastructure spending directly drives order books"),
        CausalEdge(source="govt_capex", target="capex_cycle", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Govt capex crowds in private capex"),
        CausalEdge(source="fiscal_deficit", target="govt_capex", direction="negative", magnitude="moderate",
                   time_lag="2-4Q", confidence="regime-dependent",
                   mechanism="Fiscal consolidation pressure can constrain capex allocation"),
        CausalEdge(source="sector_infra", target="LT", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="L&T is the bellwether for India's infra capex cycle"),

        # --- Defence ---
        CausalEdge(source="govt_capex", target="sector_defence", direction="positive", magnitude="moderate",
                   time_lag="2-4Q", confidence="well-established",
                   mechanism="Defence budget allocation part of government capex"),
        CausalEdge(source="sector_defence", target="HAL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_defence", target="BEL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Power Sector ---
        CausalEdge(source="gdp_growth", target="sector_power", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Economic growth drives power demand"),
        CausalEdge(source="sector_power", target="NTPC", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_power", target="POWERGRID", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Telecom ---
        CausalEdge(source="sector_telecom", target="BHARTIARTL", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_telecom", target="RELIANCE", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Jio is a major segment of Reliance"),

        # --- Gold ---
        CausalEdge(source="gold", target="TITAN", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="regime-dependent",
                   mechanism="Rising gold prices can boost or dampen jewellery demand depending on sentiment"),
        CausalEdge(source="global_risk_appetite", target="gold", direction="negative", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Gold is a risk-off asset — demand rises when risk appetite falls"),

        # --- Insurance ---
        CausalEdge(source="gsec_10y", target="sector_insurance", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Higher yields improve insurance investment returns"),
        CausalEdge(source="sector_insurance", target="HDFCLIFE", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_insurance", target="SBILIFE", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Pharma Specifics ---
        CausalEdge(source="sector_pharma", target="SUNPHARMA", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_pharma", target="DRREDDY", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_pharma", target="CIPLA", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- FMCG Company Links ---
        CausalEdge(source="sector_fmcg", target="HINDUNILVR", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_fmcg", target="ITC", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_fmcg", target="NESTLEIND", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_fmcg", target="GODREJCP", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_fmcg", target="DABUR", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_fmcg", target="BRITANNIA", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Chemicals ---
        CausalEdge(source="sector_chemicals", target="PIIND", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_chemicals", target="SRF", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),

        # --- Conglomerate (Reliance) ---
        CausalEdge(source="crude_oil", target="RELIANCE", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="regime-dependent",
                   mechanism="O2C segment benefits from higher crude but GRM is the key driver"),

        # --- Adani Group ---
        CausalEdge(source="sector_infra", target="ADANIPORTS", direction="positive", magnitude="strong",
                   time_lag="immediate", confidence="well-established"),
        CausalEdge(source="sector_power", target="ADANIGREEN", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established"),

        # --- CRR → Liquidity ---
        CausalEdge(source="crr", target="liquidity", direction="negative", magnitude="strong",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Higher CRR locks up more deposits with RBI, reducing lendable funds"),

        # --- GDP → Broad Market ---
        CausalEdge(source="gdp_growth", target="credit_growth", direction="positive", magnitude="moderate",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="Economic expansion drives corporate and retail credit demand"),
        CausalEdge(source="gdp_growth", target="consumer_demand", direction="positive", magnitude="strong",
                   time_lag="1-2Q", confidence="well-established",
                   mechanism="GDP growth reflects and drives consumer spending"),
        CausalEdge(source="iip", target="gdp_growth", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Industrial production is a major component of GDP"),
        CausalEdge(source="pmi_mfg", target="iip", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="PMI is a leading indicator for IIP"),
        CausalEdge(source="pmi_svc", target="gdp_growth", direction="positive", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="Services are ~55% of India GDP"),

        # --- India VIX ---
        CausalEdge(source="india_vix", target="fii_flows", direction="negative", magnitude="moderate",
                   time_lag="immediate", confidence="well-established",
                   mechanism="High volatility deters foreign inflows"),

        # --- Textiles / Export Competitiveness ---
        CausalEdge(source="export_competitiveness", target="sector_textiles", direction="positive",
                   magnitude="moderate", time_lag="1-2Q", confidence="well-established"),
        CausalEdge(source="rupee_depreciation", target="export_competitiveness", direction="positive",
                   magnitude="moderate", time_lag="1-2Q", confidence="well-established",
                   mechanism="Weaker rupee makes Indian exports cheaper globally"),
    ]

    for edge in edges:
        G.add_edge(
            edge.source, edge.target,
            direction=edge.direction,
            magnitude=edge.magnitude,
            time_lag=edge.time_lag,
            confidence=edge.confidence,
            mechanism=edge.mechanism,
        )

    logger.info("Causal graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Singleton Graph Instance
# ---------------------------------------------------------------------------

_GRAPH: nx.DiGraph | None = None


def get_graph() -> nx.DiGraph:
    """Return the singleton causal graph, building it on first access."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Query Functions (used by tools)
# ---------------------------------------------------------------------------

def traverse_causal_chain(source: str, depth: int = 3, direction_filter: str | None = None) -> list[dict]:
    """
    BFS traversal from a source node up to `depth` hops.
    Returns a list of paths with edge attributes.
    """
    G = get_graph()
    if source not in G:
        return [{"error": f"Node '{source}' not found in causal graph. Available nodes: {_suggest_nodes(source)}"}]

    results = []
    visited = set()
    queue = [(source, [source], 0)]

    while queue:
        current, path, d = queue.pop(0)
        if d >= depth:
            continue

        for neighbor in G.successors(current):
            if neighbor in visited and neighbor != source:
                continue

            edge_data = G.edges[current, neighbor]

            if direction_filter and edge_data.get("direction") != direction_filter:
                continue

            new_path = path + [neighbor]
            node_data = G.nodes.get(neighbor, {})
            results.append({
                "path": " → ".join(new_path),
                "depth": d + 1,
                "latest_link": {
                    "from": current,
                    "to": neighbor,
                    "to_label": node_data.get("label", neighbor),
                    "to_category": node_data.get("category", "unknown"),
                    **edge_data,
                },
            })
            visited.add(neighbor)
            queue.append((neighbor, new_path, d + 1))

    return results if results else [{"info": f"No outgoing causal links from '{source}' within depth {depth}"}]


def get_affected_entities(event: str, entity_types: list[str] | None = None, depth: int = 4) -> dict:
    """
    Given a trigger event (node ID), find all affected entities grouped by category.
    """
    G = get_graph()
    if event not in G:
        return {"error": f"Node '{event}' not found. Suggestions: {_suggest_nodes(event)}"}

    chain = traverse_causal_chain(event, depth=depth)
    affected: dict[str, list[dict]] = {}

    for link in chain:
        if "error" in link or "info" in link:
            continue
        cat = link["latest_link"]["to_category"]
        if entity_types and cat not in entity_types:
            continue
        affected.setdefault(cat, []).append({
            "id": link["latest_link"]["to"],
            "label": link["latest_link"]["to_label"],
            "direction": link["latest_link"]["direction"],
            "magnitude": link["latest_link"]["magnitude"],
            "time_lag": link["latest_link"]["time_lag"],
            "path": link["path"],
        })

    return {
        "trigger": event,
        "trigger_label": G.nodes[event].get("label", event),
        "affected_by_category": affected,
        "total_affected": sum(len(v) for v in affected.values()),
    }


def get_transmission_path(source: str, target: str) -> list[dict]:
    """
    Find all simple paths between source and target in the causal graph
    and return them with edge attributes.
    """
    G = get_graph()
    if source not in G:
        return [{"error": f"Source '{source}' not found. Suggestions: {_suggest_nodes(source)}"}]
    if target not in G:
        return [{"error": f"Target '{target}' not found. Suggestions: {_suggest_nodes(target)}"}]

    try:
        all_paths = list(nx.all_simple_paths(G, source, target, cutoff=6))
    except nx.NetworkXError:
        return [{"error": f"No path exists from '{source}' to '{target}'"}]

    if not all_paths:
        return [{"info": f"No causal path from '{source}' to '{target}' within 6 hops"}]

    results = []
    for path in all_paths[:10]:  # limit to 10 paths
        links = []
        for i in range(len(path) - 1):
            edge_data = G.edges[path[i], path[i + 1]]
            links.append({
                "from": path[i],
                "from_label": G.nodes[path[i]].get("label", path[i]),
                "to": path[i + 1],
                "to_label": G.nodes[path[i + 1]].get("label", path[i + 1]),
                **edge_data,
            })

        # Compute net direction
        net_negative = sum(1 for l in links if l["direction"] == "negative") % 2 == 1
        results.append({
            "path": " → ".join(path),
            "path_labels": " → ".join(G.nodes[n].get("label", n) for n in path),
            "hops": len(path) - 1,
            "net_direction": "negative" if net_negative else "positive",
            "links": links,
        })

    return results


def search_nodes(query: str, category: str | None = None) -> list[dict]:
    """Search for nodes in the causal graph by keyword."""
    G = get_graph()
    query_lower = query.lower()
    results = []

    for node_id, data in G.nodes(data=True):
        label = data.get("label", "").lower()
        desc = data.get("description", "").lower()
        cat = data.get("category", "")

        if category and cat != category:
            continue

        if query_lower in node_id.lower() or query_lower in label or query_lower in desc:
            results.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "category": cat,
                "description": data.get("description", ""),
                "outgoing_edges": G.out_degree(node_id),
                "incoming_edges": G.in_degree(node_id),
            })

    return results if results else [{"info": f"No nodes matching '{query}'"}]


def _suggest_nodes(query: str) -> list[str]:
    """Suggest similar node IDs for a failed lookup."""
    G = get_graph()
    query_lower = query.lower()
    suggestions = []
    for node_id in G.nodes:
        if query_lower in node_id.lower() or query_lower in G.nodes[node_id].get("label", "").lower():
            suggestions.append(node_id)
    return suggestions[:5]


# ---------------------------------------------------------------------------
# LangChain StructuredTools
# ---------------------------------------------------------------------------

class TraverseCausalChainInput(BaseModel):
    source: str = Field(description="Node ID to start traversal from (e.g., 'crude_oil', 'repo_rate', 'sector_banking')")
    depth: int = Field(default=3, description="Maximum traversal depth (1-6)")
    direction_filter: Optional[str] = Field(default=None, description="Filter edges by direction: 'positive' or 'negative'")


class GetAffectedEntitiesInput(BaseModel):
    event: str = Field(description="Trigger event node ID (e.g., 'crude_oil', 'repo_rate')")
    entity_types: Optional[list[str]] = Field(default=None, description="Filter by category: 'company', 'sector', 'concept', etc.")
    depth: int = Field(default=4, description="Maximum traversal depth")


class GetTransmissionPathInput(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")


class SearchNodesInput(BaseModel):
    query: str = Field(description="Search keyword")
    category: Optional[str] = Field(default=None, description="Filter by category: macro_indicator, sector, company, concept, commodity, currency, policy")


def get_causal_graph_tools() -> list[StructuredTool]:
    """Return LangChain StructuredTools for causal graph operations."""
    return [
        StructuredTool.from_function(
            func=lambda **kwargs: traverse_causal_chain(**kwargs),
            name="traverse_causal_chain",
            description=(
                "Traverse the financial causal knowledge graph from a source node. "
                "Returns all downstream causal effects with direction, magnitude, and time lag. "
                "Example: traverse_causal_chain(source='repo_rate', depth=3)"
            ),
            args_schema=TraverseCausalChainInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: get_affected_entities(**kwargs),
            name="get_affected_entities",
            description=(
                "Find all entities (sectors, companies) affected by a trigger event, "
                "grouped by category. Example: get_affected_entities(event='crude_oil', entity_types=['sector', 'company'])"
            ),
            args_schema=GetAffectedEntitiesInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: get_transmission_path(**kwargs),
            name="get_transmission_path",
            description=(
                "Find all causal transmission paths between two nodes in the financial system. "
                "Example: get_transmission_path(source='repo_rate', target='MARUTI')"
            ),
            args_schema=GetTransmissionPathInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: search_nodes(**kwargs),
            name="search_causal_graph",
            description=(
                "Search the causal knowledge graph for nodes by keyword. "
                "Example: search_causal_graph(query='banking') or search_causal_graph(query='oil', category='company')"
            ),
            args_schema=SearchNodesInput,
        ),
    ]
