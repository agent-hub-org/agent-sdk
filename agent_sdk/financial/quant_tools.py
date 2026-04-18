"""
Quantitative Reasoning Tools — computational substrate for financial analysis.

Provides DCF engine, comparable valuation, scenario simulation, factor
decomposition, technical signal calculation, and regime detection.
Exposed as LangChain StructuredTools.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("agent_sdk.financial.quant_tools")


# ---------------------------------------------------------------------------
# DCF Engine
# ---------------------------------------------------------------------------

class DCFInput(BaseModel):
    current_fcf_cr: float = Field(description="Current annual Free Cash Flow in crores")
    growth_rate_pct: float = Field(description="Expected FCF growth rate (%) for high-growth phase")
    high_growth_years: int | None = Field(default=5, description="Number of years of high growth")
    fade_rate_pct: float | None = Field(default=5.0, description="Growth rate (%) during fade period")
    fade_years: int | None = Field(default=5, description="Number of fade years")
    terminal_growth_pct: float | None = Field(default=4.0, description="Terminal perpetual growth rate (%)")
    discount_rate_pct: float | None = Field(default=12.0, description="Weighted average cost of capital (%)")
    net_debt_cr: float | None = Field(default=0, description="Net debt (debt - cash) in crores")
    shares_outstanding_cr: float | None = Field(default=1.0, description="Shares outstanding in crores")
    current_price: float | None = Field(default=None, description="Current market price for upside calculation")

    @field_validator("high_growth_years", "fade_rate_pct", "fade_years", "terminal_growth_pct", 
                     "discount_rate_pct", "net_debt_cr", "shares_outstanding_cr", mode="before")
    @classmethod
    def allow_none_for_numeric(cls, v: Any, info: Any) -> Any:
        if v is not None:
            return v
        return cls.model_fields[info.field_name].default


def run_dcf(**kwargs) -> dict:
    """
    Run a two-stage DCF valuation with sensitivity analysis.
    Returns intrinsic value per share and a sensitivity table.
    """
    inp = DCFInput(**kwargs)

    r = inp.discount_rate_pct / 100
    g_high = inp.growth_rate_pct / 100
    g_fade = inp.fade_rate_pct / 100
    g_terminal = inp.terminal_growth_pct / 100

    if r <= g_terminal:
        return {"error": "Discount rate must be greater than terminal growth rate"}

    # Phase 1: High growth
    pv_phase1 = 0.0
    fcf = inp.current_fcf_cr
    projected_fcfs = []
    for year in range(1, inp.high_growth_years + 1):
        fcf = fcf * (1 + g_high)
        pv = fcf / ((1 + r) ** year)
        pv_phase1 += pv
        projected_fcfs.append({"year": year, "fcf_cr": round(fcf, 2), "pv_cr": round(pv, 2)})

    # Phase 2: Fade
    pv_phase2 = 0.0
    for i in range(1, inp.fade_years + 1):
        year = inp.high_growth_years + i
        # Linear fade from high growth to terminal
        blend = i / inp.fade_years
        growth = g_high * (1 - blend) + g_fade * blend
        fcf = fcf * (1 + growth)
        pv = fcf / ((1 + r) ** year)
        pv_phase2 += pv
        projected_fcfs.append({"year": year, "fcf_cr": round(fcf, 2), "pv_cr": round(pv, 2), "phase": "fade"})

    # Terminal value
    terminal_fcf = fcf * (1 + g_terminal)
    terminal_value = terminal_fcf / (r - g_terminal)
    terminal_year = inp.high_growth_years + inp.fade_years
    pv_terminal = terminal_value / ((1 + r) ** terminal_year)

    # Enterprise value
    enterprise_value = pv_phase1 + pv_phase2 + pv_terminal

    # Equity value
    equity_value = enterprise_value - inp.net_debt_cr
    intrinsic_per_share = equity_value / inp.shares_outstanding_cr if inp.shares_outstanding_cr > 0 else 0

    # Sensitivity table: discount rate × growth rate
    sensitivity = []
    for dr_delta in [-2, -1, 0, 1, 2]:
        row = {}
        dr = inp.discount_rate_pct + dr_delta
        for gr_delta in [-3, -1.5, 0, 1.5, 3]:
            gr = inp.growth_rate_pct + gr_delta
            val = _quick_dcf(inp.current_fcf_cr, gr/100, inp.high_growth_years,
                             g_fade, inp.fade_years, g_terminal, dr/100,
                             inp.net_debt_cr, inp.shares_outstanding_cr)
            row[f"growth_{gr:.1f}%"] = round(val, 2)
        sensitivity.append({"discount_rate": f"{dr:.1f}%", "values": row})

    result = {
        "intrinsic_value_per_share": round(intrinsic_per_share, 2),
        "enterprise_value_cr": round(enterprise_value, 2),
        "equity_value_cr": round(equity_value, 2),
        "pv_high_growth_cr": round(pv_phase1, 2),
        "pv_fade_cr": round(pv_phase2, 2),
        "pv_terminal_cr": round(pv_terminal, 2),
        "terminal_value_share_pct": round(pv_terminal / enterprise_value * 100, 1) if enterprise_value else 0,
        "assumptions": {
            "fcf_cr": inp.current_fcf_cr,
            "growth_rate": f"{inp.growth_rate_pct}%",
            "high_growth_years": inp.high_growth_years,
            "fade_rate": f"{inp.fade_rate_pct}%",
            "terminal_growth": f"{inp.terminal_growth_pct}%",
            "discount_rate": f"{inp.discount_rate_pct}%",
            "net_debt_cr": inp.net_debt_cr,
        },
        "sensitivity_table": sensitivity,
        "projected_fcfs": projected_fcfs,
    }

    if inp.current_price is not None and inp.current_price > 0:
        upside = (intrinsic_per_share - inp.current_price) / inp.current_price * 100
        result["current_price"] = inp.current_price
        result["upside_pct"] = round(upside, 2)
        result["margin_of_safety"] = "adequate" if upside > 25 else "thin" if upside > 10 else "negative"

    return result


def _quick_dcf(fcf, g_high, high_years, g_fade, fade_years, g_terminal, r,
               net_debt, shares) -> float:
    """Quick DCF for sensitivity table — returns intrinsic value per share."""
    if r <= g_terminal:
        return 0.0
    total_pv = 0.0
    f = fcf
    for year in range(1, high_years + 1):
        f = f * (1 + g_high)
        total_pv += f / ((1 + r) ** year)
    for i in range(1, fade_years + 1):
        year = high_years + i
        blend = i / fade_years
        growth = g_high * (1 - blend) + g_fade * blend
        f = f * (1 + growth)
        total_pv += f / ((1 + r) ** year)
    terminal = f * (1 + g_terminal) / (r - g_terminal)
    total_pv += terminal / ((1 + r) ** (high_years + fade_years))
    equity = total_pv - net_debt
    return equity / shares if shares > 0 else 0


# ---------------------------------------------------------------------------
# Comparable Valuation
# ---------------------------------------------------------------------------

class ComparableInput(BaseModel):
    target_ticker: str = Field(description="Ticker of the company being valued")
    target_metrics: dict = Field(
        default_factory=dict,
        description="Dict with keys like 'pe', 'pb', 'ev_ebitda', 'roe', 'revenue_growth', 'ebitda_margin'",
    )
    peers: list[dict] = Field(
        default_factory=list,
        description="List of peer dicts, each with 'ticker' and same metric keys as target_metrics",
    )


def run_comparable_valuation(**kwargs) -> dict:
    """
    Compare a company against peers across multiple valuation and fundamental metrics.
    Returns relative positioning, percentile rankings, and implied valuations.
    """
    inp = ComparableInput(**kwargs)

    if not inp.target_metrics:
        return {
            "error": "target_metrics is required — provide a dict of valuation/fundamental metrics for the target company.",
            "hint": "Call get_ticker_data first to retrieve metrics (pe, pb, roe, etc.), then pass them here along with peer data.",
            "example": "run_comparable_valuation(target_ticker='<TICKER>', target_metrics={'pe': <val>, 'pb': <val>, 'roe': <val>}, peers=[{'ticker': '<PEER_TICKER>', 'pe': <val>, 'pb': <val>, 'roe': <val>}])",
        }
    if not inp.peers:
        return {"error": "At least one peer is required for comparison"}

    all_companies = [{"ticker": inp.target_ticker, **inp.target_metrics}] + inp.peers
    metrics = [k for k in inp.target_metrics.keys() if k != "ticker"]

    rankings = {}
    for metric in metrics:
        values = [(c.get("ticker", "?"), c.get(metric)) for c in all_companies if c.get(metric) is not None]
        if len(values) < 2:
            continue

        # Sort — for valuation metrics (PE, PB, EV/EBITDA), lower is "cheaper"
        # For quality metrics (ROE, margin, growth), higher is better
        higher_is_better = metric.lower() in ("roe", "roce", "revenue_growth", "ebitda_margin",
                                                "pat_growth", "fcf_yield", "dividend_yield",
                                                "interest_coverage", "promoter_holding")
        sorted_values = sorted(values, key=lambda x: x[1], reverse=higher_is_better)

        rank = next(i + 1 for i, (t, _) in enumerate(sorted_values) if t == inp.target_ticker)
        all_vals = [v for _, v in values]
        target_val = inp.target_metrics.get(metric)

        rankings[metric] = {
            "value": target_val,
            "rank": rank,
            "out_of": len(values),
            "peer_median": round(sorted(all_vals)[len(all_vals) // 2], 2),
            "peer_min": round(min(all_vals), 2),
            "peer_max": round(max(all_vals), 2),
            "percentile": round((1 - (rank - 1) / (len(values) - 1)) * 100, 1) if len(values) > 1 else 50,
            "ranking": [{"ticker": t, "value": round(v, 2)} for t, v in sorted_values],
        }

    # Overall assessment
    valuation_metrics = [m for m in ("pe", "pb", "ev_ebitda") if m in rankings]
    quality_metrics = [m for m in ("roe", "roce", "ebitda_margin") if m in rankings]

    val_percentiles = [rankings[m]["percentile"] for m in valuation_metrics]
    qual_percentiles = [rankings[m]["percentile"] for m in quality_metrics]

    avg_val = sum(val_percentiles) / len(val_percentiles) if val_percentiles else 50
    avg_qual = sum(qual_percentiles) / len(qual_percentiles) if qual_percentiles else 50

    if avg_val > 70 and avg_qual > 60:
        overall = "Attractively valued with strong fundamentals relative to peers"
    elif avg_val > 50:
        overall = "Reasonably valued relative to peers"
    elif avg_qual > 70:
        overall = "Premium valuation but justified by superior fundamentals"
    else:
        overall = "Expensive relative to peers without clear fundamental justification"

    return {
        "target": inp.target_ticker,
        "peer_count": len(inp.peers),
        "metric_rankings": rankings,
        "summary": {
            "valuation_percentile": round(avg_val, 1),
            "quality_percentile": round(avg_qual, 1),
            "overall_assessment": overall,
        },
    }


# ---------------------------------------------------------------------------
# Scenario Simulator
# ---------------------------------------------------------------------------

class ScenarioInput(BaseModel):
    scenario_name: str = Field(description="Name of the scenario")
    variable_changes: dict[str, float] = Field(
        default_factory=dict,
        description="Dict of causal graph node IDs to percentage changes. E.g., {'crude_oil': 20, 'repo_rate': 0.5}"
    )
    target_entities: list[str] = Field(
        default_factory=list,
        description="Specific entities to estimate impact on. If empty, uses full causal graph."
    )


def run_scenario_simulation(**kwargs) -> dict:
    """
    Simulate a macro scenario by propagating variable changes through
    the causal knowledge graph. Estimates impact on sectors and companies.
    """
    from agent_sdk.financial.causal_graph import get_graph

    inp = ScenarioInput(**kwargs)
    if not inp.variable_changes:
        return {
            "error": "variable_changes is required — provide a dict mapping causal graph node IDs to % changes.",
            "example": "{'crude_oil': 20, 'repo_rate': 0.5, 'india_vix': 15}",
            "hint": "Call traverse_causal_chain or search_causal_graph first to find valid node IDs.",
        }
    G = get_graph()

    impacts: dict[str, dict] = {}

    magnitude_multipliers = {"weak": 0.3, "moderate": 0.6, "strong": 0.9}
    time_lag_factors = {"immediate": 1.0, "1-2Q": 0.85, "2-4Q": 0.7, "1-2Y": 0.5}

    for source_node, change_pct in inp.variable_changes.items():
        if source_node not in G:
            impacts[source_node] = {"error": f"Node '{source_node}' not found in causal graph"}
            continue

        # BFS propagation with decay
        visited = set()
        queue = [(source_node, change_pct, 0)]

        while queue:
            current, current_impact, depth = queue.pop(0)
            if depth >= 4 or current in visited:
                continue
            visited.add(current)

            for neighbor in G.successors(current):
                edge = G.edges[current, neighbor]
                direction_sign = -1 if edge.get("direction") == "negative" else 1
                mag = magnitude_multipliers.get(edge.get("magnitude", "moderate"), 0.6)
                time_factor = time_lag_factors.get(edge.get("time_lag", "1-2Q"), 0.7)

                propagated_impact = current_impact * direction_sign * mag * time_factor

                node_data = G.nodes.get(neighbor, {})
                cat = node_data.get("category", "unknown")

                # Only record sectors and companies, or explicit targets
                if cat in ("sector", "company") or neighbor in inp.target_entities:
                    key = neighbor
                    if key in impacts:
                        impacts[key]["estimated_impact_pct"] += propagated_impact
                        impacts[key]["contributing_paths"].append({
                            "from": current,
                            "impact_contribution": round(propagated_impact, 3),
                            "edge": {k: v for k, v in edge.items()},
                        })
                    else:
                        impacts[key] = {
                            "entity": neighbor,
                            "label": node_data.get("label", neighbor),
                            "category": cat,
                            "estimated_impact_pct": propagated_impact,
                            "contributing_paths": [{
                                "from": current,
                                "impact_contribution": round(propagated_impact, 3),
                                "edge": {k: v for k, v in edge.items()},
                            }],
                        }

                queue.append((neighbor, propagated_impact, depth + 1))

    # Filter to target entities if specified
    if inp.target_entities:
        impacts = {k: v for k, v in impacts.items() if k in inp.target_entities or "error" in v}

    # Round and sort by absolute impact
    for v in impacts.values():
        if "estimated_impact_pct" in v:
            v["estimated_impact_pct"] = round(v["estimated_impact_pct"], 3)

    sorted_impacts = dict(sorted(
        impacts.items(),
        key=lambda x: abs(x[1].get("estimated_impact_pct", 0)),
        reverse=True,
    ))

    # Categorize
    beneficiaries = {k: v for k, v in sorted_impacts.items() if v.get("estimated_impact_pct", 0) > 0.5}
    losers = {k: v for k, v in sorted_impacts.items() if v.get("estimated_impact_pct", 0) < -0.5}

    return {
        "scenario": inp.scenario_name,
        "input_changes": inp.variable_changes,
        "all_impacts": sorted_impacts,
        "beneficiaries": beneficiaries,
        "negatively_impacted": losers,
        "total_entities_affected": len(sorted_impacts),
    }


# ---------------------------------------------------------------------------
# Technical Signal Calculator
# ---------------------------------------------------------------------------

class TechnicalSignalInput(BaseModel):
    prices: list[float] = Field(description="List of closing prices (most recent last)")
    volumes: list[float] | None = Field(default_factory=list, description="List of volumes (same length as prices)")

    @field_validator("volumes", mode="before")
    @classmethod
    def allow_none_for_list(cls, v: Any) -> Any:
        return v if v is not None else []


def calculate_technical_signals(**kwargs) -> dict:
    """
    Calculate common technical indicators from price data.
    Returns RSI, MACD, moving averages, and support/resistance levels.
    """
    inp = TechnicalSignalInput(**kwargs)
    prices = inp.prices
    n = len(prices)

    if n < 26:
        return {"error": f"Need at least 26 data points for technical analysis, got {n}"}

    result = {}

    # Moving Averages
    if n >= 20:
        result["sma_20"] = round(sum(prices[-20:]) / 20, 2)
    if n >= 50:
        result["sma_50"] = round(sum(prices[-50:]) / 50, 2)
    if n >= 200:
        result["sma_200"] = round(sum(prices[-200:]) / 200, 2)

    current_price = prices[-1]
    result["current_price"] = current_price

    # Price vs MAs
    ma_signals = []
    if "sma_20" in result:
        if current_price > result["sma_20"]:
            ma_signals.append("Above 20-DMA (short-term bullish)")
        else:
            ma_signals.append("Below 20-DMA (short-term bearish)")
    if "sma_50" in result and "sma_200" in result:
        if result["sma_50"] > result["sma_200"]:
            ma_signals.append("Golden cross (50-DMA > 200-DMA — bullish)")
        else:
            ma_signals.append("Death cross (50-DMA < 200-DMA — bearish)")
    result["ma_signals"] = ma_signals

    # RSI (14-period)
    period = 14
    if n >= period + 1:
        gains, losses = [], []
        for i in range(n - period, n):
            change = prices[i] - prices[i - 1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        result["rsi_14"] = round(rsi, 2)
        if rsi > 70:
            result["rsi_signal"] = "Overbought (>70) — potential reversal/consolidation"
        elif rsi < 30:
            result["rsi_signal"] = "Oversold (<30) — potential bounce"
        else:
            result["rsi_signal"] = "Neutral"

    # MACD (12, 26, 9)
    if n >= 26:
        ema_12 = _ema(prices, 12)
        ema_26 = _ema(prices, 26)
        macd_line = ema_12 - ema_26

        # Signal line (9-period EMA of MACD) — simplified
        result["macd"] = round(macd_line, 2)
        result["macd_signal"] = "Bullish" if macd_line > 0 else "Bearish"

    # Support / Resistance (simple pivot-based)
    recent = prices[-20:]
    high_20 = max(recent)
    low_20 = min(recent)
    result["resistance_20d"] = round(high_20, 2)
    result["support_20d"] = round(low_20, 2)

    if n >= 52:
        result["52w_high"] = round(max(prices[-252:]) if n >= 252 else max(prices), 2)
        result["52w_low"] = round(min(prices[-252:]) if n >= 252 else min(prices), 2)
        pct_from_high = (current_price - result["52w_high"]) / result["52w_high"] * 100
        result["pct_from_52w_high"] = round(pct_from_high, 2)

    # Volatility (20-day)
    if n >= 21:
        returns = [math.log(prices[i] / prices[i-1]) for i in range(n-20, n) if prices[i-1] > 0]
        if returns:
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            daily_vol = math.sqrt(variance)
            annualized_vol = daily_vol * math.sqrt(252)
            result["volatility_20d_annualized"] = round(annualized_vol * 100, 2)

    # Volume analysis
    if inp.volumes and len(inp.volumes) >= 20:
        avg_vol_20 = sum(inp.volumes[-20:]) / 20
        current_vol = inp.volumes[-1]
        result["avg_volume_20d"] = round(avg_vol_20, 0)
        result["volume_ratio"] = round(current_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0
        if result["volume_ratio"] > 1.5:
            result["volume_signal"] = "High volume — confirms trend"
        elif result["volume_ratio"] < 0.5:
            result["volume_signal"] = "Low volume — weak conviction"
        else:
            result["volume_signal"] = "Normal volume"

    return result


def _ema(data: list[float], period: int) -> float:
    """Calculate Exponential Moving Average."""
    k = 2 / (period + 1)
    ema = sum(data[:period]) / period  # SMA as seed
    for price in data[period:]:
        ema = price * k + ema * (1 - k)
    return ema


# ---------------------------------------------------------------------------
# Risk Metrics Calculator
# ---------------------------------------------------------------------------

class RiskMetricsInput(BaseModel):
    prices: list[float] = Field(description="List of daily closing prices (most recent last, minimum 30)")
    risk_free_rate_pct: float | None = Field(default=6.5, description="Annual risk-free rate in % (default: RBI repo rate proxy 6.5%)")
    trading_days: int | None = Field(default=252, description="Trading days per year for annualization (252 for India)")

    @field_validator("risk_free_rate_pct", "trading_days", mode="before")
    @classmethod
    def allow_none_for_numeric(cls, v: Any, info: Any) -> Any:
        if v is not None:
            return v
        return cls.model_fields[info.field_name].default


def calculate_risk_metrics(**kwargs) -> dict:
    """
    Compute risk-adjusted return metrics from price series.
    Returns Sharpe ratio, Sortino ratio, Max Drawdown, annualized volatility,
    and VaR (95% historical) for a given price series.
    """
    inp = RiskMetricsInput(**kwargs)
    prices = inp.prices
    n = len(prices)

    if n < 30:
        return {"error": f"Need at least 30 data points for risk metrics, got {n}"}

    # Daily log returns
    returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, n) if prices[i - 1] > 0]
    if len(returns) < 2:
        return {"error": "Insufficient valid returns to compute risk metrics"}

    # Annualize
    ann_factor = inp.trading_days
    mean_daily = sum(returns) / len(returns)
    ann_return = mean_daily * ann_factor

    variance = sum((r - mean_daily) ** 2 for r in returns) / len(returns)
    daily_vol = math.sqrt(variance)
    ann_vol = daily_vol * math.sqrt(ann_factor)

    rf_daily = inp.risk_free_rate_pct / 100 / ann_factor

    # Sharpe Ratio
    sharpe = (ann_return - inp.risk_free_rate_pct / 100) / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio — downside deviation only
    downside_returns = [r - rf_daily for r in returns if (r - rf_daily) < 0]
    if downside_returns:
        downside_variance = sum(r ** 2 for r in downside_returns) / len(returns)
        downside_dev = math.sqrt(downside_variance) * math.sqrt(ann_factor)
        sortino = (ann_return - inp.risk_free_rate_pct / 100) / downside_dev if downside_dev > 0 else 0.0
    else:
        sortino = float("inf")

    # Maximum Drawdown
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    # VaR (95% historical)
    sorted_returns = sorted(returns)
    var_95_idx = max(0, int(len(sorted_returns) * 0.05) - 1)
    var_95_daily = -sorted_returns[var_95_idx]
    var_95_ann = var_95_daily * math.sqrt(ann_factor)

    # Calmar Ratio (annualized return / max drawdown — useful for momentum strategies)
    calmar = abs(ann_return / max_drawdown) if max_drawdown != 0 else float("inf")

    return {
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3) if sortino != float("inf") else "∞ (no downside returns)",
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "var_95_daily_pct": round(var_95_daily * 100, 2),
        "var_95_annualized_pct": round(var_95_ann * 100, 2),
        "calmar_ratio": round(calmar, 3) if calmar != float("inf") else "∞ (no drawdown)",
        "data_points": n,
        "risk_free_rate_pct": inp.risk_free_rate_pct,
        "interpretation": _interpret_risk_metrics(sharpe, max_drawdown * 100, ann_vol * 100),
    }


def _interpret_risk_metrics(sharpe: float, max_dd_pct: float, vol_pct: float) -> str:
    parts = []
    if sharpe > 1.5:
        parts.append("Excellent risk-adjusted returns (Sharpe > 1.5)")
    elif sharpe > 1.0:
        parts.append("Good risk-adjusted returns (Sharpe 1.0-1.5)")
    elif sharpe > 0.5:
        parts.append("Acceptable risk-adjusted returns (Sharpe 0.5-1.0)")
    elif sharpe > 0:
        parts.append("Weak risk-adjusted returns (Sharpe 0-0.5)")
    else:
        parts.append("Negative risk-adjusted returns (Sharpe < 0)")

    if abs(max_dd_pct) > 50:
        parts.append("severe drawdown (>50% — high risk)")
    elif abs(max_dd_pct) > 30:
        parts.append("significant drawdown (30-50%)")
    elif abs(max_dd_pct) > 15:
        parts.append("moderate drawdown (15-30%)")
    else:
        parts.append("contained drawdown (<15%)")

    if vol_pct > 40:
        parts.append("very high volatility (>40% ann)")
    elif vol_pct > 25:
        parts.append("high volatility (25-40% ann)")
    elif vol_pct > 15:
        parts.append("moderate volatility (15-25% ann)")
    else:
        parts.append("low volatility (<15% ann)")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Correlation / Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetectorInput(BaseModel):
    nifty_pe: Optional[float] = Field(default=None, description="Current Nifty 50 trailing PE")
    india_vix: Optional[float] = Field(default=None, description="Current India VIX level")
    gsec_10y: Optional[float] = Field(default=None, description="10Y G-Sec yield (%)")
    repo_rate: Optional[float] = Field(default=None, description="RBI repo rate (%)")
    cpi_yoy: Optional[float] = Field(default=None, description="CPI inflation YoY (%)")
    credit_growth: Optional[float] = Field(default=None, description="Bank credit growth YoY (%)")
    fii_net_30d: Optional[float] = Field(default=None, description="FII net flows last 30 days (crores)")
    usd_inr: Optional[float] = Field(default=None, description="USD/INR exchange rate")
    crude_brent: Optional[float] = Field(default=None, description="Brent crude USD/barrel")


def detect_regime(**kwargs) -> dict:
    """
    Deterministic regime classifier based on quantitative indicators.
    Provides a ground-truth regime assessment that the LLM reasons FROM.
    """
    inp = RegimeDetectorInput(**kwargs)
    signals = {}
    regime_scores = {"bull": 0, "bear": 0, "volatile": 0, "sideways": 0}
    monetary_scores = {"tightening": 0, "easing": 0, "neutral": 0}
    cycle_signals = []

    # Nifty PE
    if inp.nifty_pe is not None:
        if inp.nifty_pe > 24:
            signals["valuation"] = f"Nifty PE at {inp.nifty_pe} — expensive (above 24)"
            regime_scores["bear"] += 1
        elif inp.nifty_pe > 20:
            signals["valuation"] = f"Nifty PE at {inp.nifty_pe} — fair to slightly elevated"
            regime_scores["sideways"] += 1
        elif inp.nifty_pe > 16:
            signals["valuation"] = f"Nifty PE at {inp.nifty_pe} — reasonable"
            regime_scores["bull"] += 1
        else:
            signals["valuation"] = f"Nifty PE at {inp.nifty_pe} — cheap (below 16)"
            regime_scores["bull"] += 2

    # India VIX
    if inp.india_vix is not None:
        if inp.india_vix > 25:
            signals["volatility"] = f"VIX at {inp.india_vix} — high fear"
            regime_scores["volatile"] += 2
        elif inp.india_vix > 18:
            signals["volatility"] = f"VIX at {inp.india_vix} — elevated"
            regime_scores["volatile"] += 1
        elif inp.india_vix > 12:
            signals["volatility"] = f"VIX at {inp.india_vix} — normal"
        else:
            signals["volatility"] = f"VIX at {inp.india_vix} — complacent (low)"
            regime_scores["bull"] += 1

    # CPI vs Repo Rate
    if inp.cpi_yoy is not None and inp.repo_rate is not None:
        real_rate = inp.repo_rate - inp.cpi_yoy
        signals["real_rate"] = f"Real rate: {real_rate:.1f}% (repo {inp.repo_rate}% - CPI {inp.cpi_yoy}%)"
        if real_rate > 2:
            monetary_scores["tightening"] += 2
            signals["monetary_stance"] = "Tight — high positive real rate"
        elif real_rate > 0:
            monetary_scores["neutral"] += 1
            signals["monetary_stance"] = "Neutral — mildly positive real rate"
        else:
            monetary_scores["easing"] += 1
            signals["monetary_stance"] = "Accommodative — negative real rate"

    if inp.cpi_yoy is not None:
        if inp.cpi_yoy > 6:
            signals["inflation"] = f"CPI at {inp.cpi_yoy}% — above RBI tolerance band"
            monetary_scores["tightening"] += 1
            cycle_signals.append("inflationary_pressure")
        elif inp.cpi_yoy > 4:
            signals["inflation"] = f"CPI at {inp.cpi_yoy}% — within target band but elevated"
        elif inp.cpi_yoy > 2:
            signals["inflation"] = f"CPI at {inp.cpi_yoy}% — benign"
            monetary_scores["easing"] += 1

    # Credit Growth
    if inp.credit_growth is not None:
        if inp.credit_growth > 15:
            signals["credit"] = f"Credit growth at {inp.credit_growth}% — strong expansion"
            cycle_signals.append("expansion")
            regime_scores["bull"] += 1
        elif inp.credit_growth > 10:
            signals["credit"] = f"Credit growth at {inp.credit_growth}% — healthy"
            cycle_signals.append("mid_expansion")
        elif inp.credit_growth > 5:
            signals["credit"] = f"Credit growth at {inp.credit_growth}% — moderate"
            cycle_signals.append("late_expansion")
        else:
            signals["credit"] = f"Credit growth at {inp.credit_growth}% — weak"
            cycle_signals.append("contraction")
            regime_scores["bear"] += 1

    # FII Flows
    if inp.fii_net_30d is not None:
        if inp.fii_net_30d > 10000:
            signals["fii"] = f"FII net +₹{inp.fii_net_30d:,.0f}cr (30d) — strong inflows"
            regime_scores["bull"] += 1
        elif inp.fii_net_30d > 0:
            signals["fii"] = f"FII net +₹{inp.fii_net_30d:,.0f}cr (30d) — mild inflows"
        elif inp.fii_net_30d > -10000:
            signals["fii"] = f"FII net ₹{inp.fii_net_30d:,.0f}cr (30d) — mild outflows"
            regime_scores["bear"] += 1
        else:
            signals["fii"] = f"FII net ₹{inp.fii_net_30d:,.0f}cr (30d) — heavy selling"
            regime_scores["bear"] += 2

    # Crude oil
    if inp.crude_brent is not None:
        if inp.crude_brent > 90:
            signals["crude"] = f"Brent at ${inp.crude_brent} — headwind for India"
            regime_scores["bear"] += 1
        elif inp.crude_brent > 75:
            signals["crude"] = f"Brent at ${inp.crude_brent} — manageable"
        else:
            signals["crude"] = f"Brent at ${inp.crude_brent} — tailwind for India"
            regime_scores["bull"] += 1

    # USD/INR
    if inp.usd_inr is not None:
        signals["currency"] = f"USD/INR at {inp.usd_inr}"

    # 10Y G-Sec
    if inp.gsec_10y is not None:
        if inp.gsec_10y > 7.5:
            signals["bond_yield"] = f"10Y G-Sec at {inp.gsec_10y}% — elevated, tight conditions"
            monetary_scores["tightening"] += 1
        elif inp.gsec_10y > 6.5:
            signals["bond_yield"] = f"10Y G-Sec at {inp.gsec_10y}% — neutral"
        else:
            signals["bond_yield"] = f"10Y G-Sec at {inp.gsec_10y}% — accommodative"
            monetary_scores["easing"] += 1

    # Determine regimes
    market_regime = max(regime_scores, key=regime_scores.get)
    if regime_scores[market_regime] == 0:
        market_regime = "sideways"

    monetary_regime = max(monetary_scores, key=monetary_scores.get)
    if monetary_scores[monetary_regime] == 0:
        monetary_regime = "neutral"

    # Cycle position estimate
    if "contraction" in cycle_signals:
        cycle = "contraction"
    elif "inflationary_pressure" in cycle_signals and "late_expansion" in cycle_signals:
        cycle = "late_expansion"
    elif "expansion" in cycle_signals:
        cycle = "early_expansion"
    elif "mid_expansion" in cycle_signals:
        cycle = "mid_expansion"
    else:
        cycle = "mid_expansion"

    confidence = min(len(signals) / 8, 1.0)  # More data points = higher confidence

    return {
        "market_regime": market_regime,
        "market_regime_scores": regime_scores,
        "monetary_regime": monetary_regime,
        "monetary_regime_scores": monetary_scores,
        "estimated_cycle_position": cycle,
        "individual_signals": signals,
        "data_points_available": len(signals),
        "confidence": round(confidence, 2),
    }


# ---------------------------------------------------------------------------
# LangChain StructuredTools
# ---------------------------------------------------------------------------

def get_quant_tools() -> list[StructuredTool]:
    """Return LangChain StructuredTools for quantitative analysis."""
    return [
        StructuredTool.from_function(
            func=lambda **kwargs: run_dcf(**kwargs),
            name="run_dcf",
            description=(
                "Run a two-stage Discounted Cash Flow (DCF) valuation. "
                "Takes growth, margin, and discount rate assumptions, returns intrinsic value "
                "per share with sensitivity table. "
                "Example: run_dcf(current_fcf_cr=5000, growth_rate_pct=15, discount_rate_pct=12, shares_outstanding_cr=600)"
            ),
            args_schema=DCFInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: run_comparable_valuation(**kwargs),
            name="run_comparable_valuation",
            description=(
                "Compare a company against peers across valuation and fundamental metrics. "
                "Returns relative positioning, percentile rankings, and overall assessment. "
                "Example: run_comparable_valuation(target_ticker='TCS', target_metrics={'pe': 28, 'roe': 45}, "
                "peers=[{'ticker': 'INFY', 'pe': 25, 'roe': 32}])"
            ),
            args_schema=ComparableInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: run_scenario_simulation(**kwargs),
            name="run_scenario_simulation",
            description=(
                "Simulate a macro scenario by propagating changes through the causal knowledge graph. "
                "Estimates impact on sectors and companies. "
                "Example: run_scenario_simulation(scenario_name='Crude shock', variable_changes={'crude_oil': 30})"
            ),
            args_schema=ScenarioInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: calculate_technical_signals(**kwargs),
            name="calculate_technical_signals",
            description=(
                "Calculate technical indicators (RSI, MACD, moving averages, support/resistance) "
                "from price data. Requires at least 26 data points. "
                "Example: calculate_technical_signals(prices=[100, 102, 101, ...], volumes=[1000000, ...])"
            ),
            args_schema=TechnicalSignalInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: detect_regime(**kwargs),
            name="detect_market_regime",
            description=(
                "Deterministic regime classifier using quantitative indicators. "
                "Classifies market regime (bull/bear/sideways/volatile), monetary regime "
                "(tightening/easing/neutral), and cycle position. "
                "Example: detect_market_regime(nifty_pe=22, india_vix=14, cpi_yoy=5.2, repo_rate=6.5)"
            ),
            args_schema=RegimeDetectorInput,
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: calculate_risk_metrics(**kwargs),
            name="calculate_risk_metrics",
            description=(
                "Compute risk-adjusted return metrics from a price series: "
                "Sharpe ratio, Sortino ratio, Max Drawdown, annualized volatility, and VaR (95% historical). "
                "Requires at least 30 daily closing prices. "
                "Example: calculate_risk_metrics(prices=[100, 102, 99, ...], risk_free_rate_pct=6.5)"
            ),
            args_schema=RiskMetricsInput,
        ),
    ]
