"""
Symbolic Validation Layer — deterministic validators that run after each
reasoning phase to catch LLM errors.

Checks:
- Accounting identity violations
- Logical consistency between stated conclusions and supporting data
- Confidence calibration (stated confidence vs evidence strength)
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("agent_sdk.financial.validators")


class ValidationResult(BaseModel):
    """Result of a validation check."""
    passed: bool
    check_name: str
    severity: str = "warning"  # info, warning, error
    message: str = ""
    details: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Accounting Identity Checker
# ---------------------------------------------------------------------------

def validate_accounting_identities(
    revenue: Optional[float] = None,
    cogs: Optional[float] = None,
    gross_profit: Optional[float] = None,
    ebitda: Optional[float] = None,
    ebitda_margin_pct: Optional[float] = None,
    ebit: Optional[float] = None,
    depreciation: Optional[float] = None,
    pat: Optional[float] = None,
    tax: Optional[float] = None,
    interest: Optional[float] = None,
    total_assets: Optional[float] = None,
    total_liabilities: Optional[float] = None,
    equity: Optional[float] = None,
    operating_cf: Optional[float] = None,
    investing_cf: Optional[float] = None,
    financing_cf: Optional[float] = None,
    net_change_cash: Optional[float] = None,
) -> list[ValidationResult]:
    """
    Check accounting identities. Returns list of validation results.
    Tolerance of 2% for rounding.
    """
    results = []
    tol = 0.02  # 2% tolerance

    # Revenue - COGS = Gross Profit
    if all(v is not None for v in [revenue, cogs, gross_profit]) and revenue != 0:
        expected = revenue - cogs
        diff_pct = abs(expected - gross_profit) / abs(revenue)
        results.append(ValidationResult(
            passed=diff_pct <= tol,
            check_name="gross_profit_identity",
            severity="error" if diff_pct > tol else "info",
            message=f"Revenue ({revenue}) - COGS ({cogs}) = {expected}, reported Gross Profit = {gross_profit}",
            details={"diff_pct": round(diff_pct * 100, 2)},
        ))

    # EBITDA margin consistency
    if all(v is not None for v in [revenue, ebitda, ebitda_margin_pct]) and revenue != 0:
        computed_margin = ebitda / revenue * 100
        diff = abs(computed_margin - ebitda_margin_pct)
        results.append(ValidationResult(
            passed=diff <= 1.0,  # 1 percentage point tolerance
            check_name="ebitda_margin_consistency",
            severity="warning" if diff > 1.0 else "info",
            message=f"Computed EBITDA margin: {computed_margin:.1f}%, stated: {ebitda_margin_pct:.1f}%",
            details={"diff_pp": round(diff, 2)},
        ))

    # EBITDA - Depreciation = EBIT
    if all(v is not None for v in [ebitda, depreciation, ebit]) and ebitda != 0:
        expected = ebitda - depreciation
        diff_pct = abs(expected - ebit) / abs(ebitda)
        results.append(ValidationResult(
            passed=diff_pct <= tol,
            check_name="ebit_identity",
            severity="error" if diff_pct > tol else "info",
            message=f"EBITDA ({ebitda}) - D&A ({depreciation}) = {expected}, reported EBIT = {ebit}",
            details={"diff_pct": round(diff_pct * 100, 2)},
        ))

    # Balance sheet: Assets = Liabilities + Equity
    if all(v is not None for v in [total_assets, total_liabilities, equity]) and total_assets != 0:
        expected = total_liabilities + equity
        diff_pct = abs(expected - total_assets) / abs(total_assets)
        results.append(ValidationResult(
            passed=diff_pct <= tol,
            check_name="balance_sheet_identity",
            severity="error" if diff_pct > tol else "info",
            message=f"Liabilities ({total_liabilities}) + Equity ({equity}) = {expected}, Total Assets = {total_assets}",
            details={"diff_pct": round(diff_pct * 100, 2)},
        ))

    # Cash flow: OCF + ICF + FCF ≈ Net change in cash
    if all(v is not None for v in [operating_cf, investing_cf, financing_cf, net_change_cash]):
        expected = operating_cf + investing_cf + financing_cf
        diff = abs(expected - net_change_cash)
        denominator = max(abs(net_change_cash), abs(operating_cf), 1)
        diff_pct = diff / denominator
        results.append(ValidationResult(
            passed=diff_pct <= tol,
            check_name="cash_flow_reconciliation",
            severity="error" if diff_pct > tol else "info",
            message=f"OCF ({operating_cf}) + ICF ({investing_cf}) + FCF ({financing_cf}) = {expected}, Net change = {net_change_cash}",
            details={"diff_pct": round(diff_pct * 100, 2)},
        ))

    return results


# ---------------------------------------------------------------------------
# Logical Consistency Checker
# ---------------------------------------------------------------------------

def validate_logical_consistency(
    revenue_growth_pct: Optional[float] = None,
    margin_change_bps: Optional[float] = None,
    earnings_growth_pct: Optional[float] = None,
    valuation_assessment: Optional[str] = None,
    recommendation: Optional[str] = None,
    pe: Optional[float] = None,
    sector_pe_median: Optional[float] = None,
    roe: Optional[float] = None,
    debt_to_equity: Optional[float] = None,
    interest_coverage: Optional[float] = None,
) -> list[ValidationResult]:
    """
    Check for logical contradictions in analysis conclusions.
    """
    results = []

    # Earnings growth vs revenue growth + margin change
    if all(v is not None for v in [revenue_growth_pct, margin_change_bps, earnings_growth_pct]):
        # Rough check: if revenue is flat and margins compress, earnings can't grow strongly
        if revenue_growth_pct < 2 and margin_change_bps < -200 and earnings_growth_pct > 10:
            results.append(ValidationResult(
                passed=False,
                check_name="earnings_growth_consistency",
                severity="error",
                message=(
                    f"Inconsistency: Revenue growth {revenue_growth_pct}% with margin compression "
                    f"of {margin_change_bps}bps, but earnings growth projected at {earnings_growth_pct}%. "
                    "Earnings cannot grow materially when revenue is flat and margins are compressing."
                ),
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                check_name="earnings_growth_consistency",
                message="Earnings growth is directionally consistent with revenue and margin assumptions.",
            ))

    # Valuation assessment vs PE relative to sector
    if all(v is not None for v in [valuation_assessment, pe, sector_pe_median]):
        val_lower = valuation_assessment.lower()
        pe_premium = (pe - sector_pe_median) / sector_pe_median * 100

        if "cheap" in val_lower and pe_premium > 30:
            results.append(ValidationResult(
                passed=False,
                check_name="valuation_label_consistency",
                severity="warning",
                message=(
                    f"Labeled as '{valuation_assessment}' but PE ({pe}) is {pe_premium:.0f}% above "
                    f"sector median ({sector_pe_median}). Reconsider valuation assessment."
                ),
            ))
        elif "expensive" in val_lower and pe_premium < -20:
            results.append(ValidationResult(
                passed=False,
                check_name="valuation_label_consistency",
                severity="warning",
                message=(
                    f"Labeled as '{valuation_assessment}' but PE ({pe}) is {abs(pe_premium):.0f}% below "
                    f"sector median ({sector_pe_median}). May be unjustified."
                ),
            ))

    # Recommendation vs valuation
    if valuation_assessment and recommendation:
        val_lower = valuation_assessment.lower()
        rec_lower = recommendation.lower()
        if "expensive" in val_lower and ("strong_buy" in rec_lower or "buy" == rec_lower):
            results.append(ValidationResult(
                passed=False,
                check_name="recommendation_valuation_alignment",
                severity="warning",
                message=f"Recommending '{recommendation}' but valuation assessed as '{valuation_assessment}'. Justify the premium.",
            ))
        if "cheap" in val_lower and "sell" in rec_lower:
            results.append(ValidationResult(
                passed=False,
                check_name="recommendation_valuation_alignment",
                severity="warning",
                message=f"Recommending '{recommendation}' but valuation assessed as '{valuation_assessment}'. Explain value trap concern.",
            ))

    # Debt quality red flags
    if debt_to_equity is not None and interest_coverage is not None:
        if debt_to_equity > 2 and interest_coverage < 1.5:
            results.append(ValidationResult(
                passed=False,
                check_name="debt_quality_flag",
                severity="error",
                message=(
                    f"High leverage (D/E: {debt_to_equity}) with weak interest coverage ({interest_coverage}x). "
                    "Significant debt servicing risk — must be addressed in analysis."
                ),
            ))

    return results


# ---------------------------------------------------------------------------
# Confidence Calibration
# ---------------------------------------------------------------------------

def validate_confidence(
    stated_confidence: float,
    data_points_available: int = 0,
    contradictions_found: int = 0,
    data_recency_days: int = 0,
    assumptions_count: int = 0,
) -> ValidationResult:
    """
    Check that stated confidence aligns with evidence quality.
    """
    # Compute evidence-based confidence bound
    max_confidence = 1.0

    # Penalize for few data points
    if data_points_available < 3:
        max_confidence *= 0.5
    elif data_points_available < 6:
        max_confidence *= 0.7
    elif data_points_available < 10:
        max_confidence *= 0.85

    # Penalize for contradictions
    max_confidence *= max(0.3, 1 - contradictions_found * 0.15)

    # Penalize for stale data
    if data_recency_days > 90:
        max_confidence *= 0.6
    elif data_recency_days > 30:
        max_confidence *= 0.8

    # Penalize for many assumptions
    if assumptions_count > 5:
        max_confidence *= 0.7
    elif assumptions_count > 3:
        max_confidence *= 0.85

    max_confidence = round(max_confidence, 2)

    if stated_confidence > max_confidence + 0.1:
        return ValidationResult(
            passed=False,
            check_name="confidence_calibration",
            severity="warning",
            message=(
                f"Stated confidence ({stated_confidence:.0%}) exceeds evidence-supported maximum "
                f"({max_confidence:.0%}). Data points: {data_points_available}, "
                f"contradictions: {contradictions_found}, data age: {data_recency_days}d, "
                f"assumptions: {assumptions_count}."
            ),
            details={
                "stated": stated_confidence,
                "max_supported": max_confidence,
                "gap": round(stated_confidence - max_confidence, 2),
            },
        )

    return ValidationResult(
        passed=True,
        check_name="confidence_calibration",
        message=f"Confidence ({stated_confidence:.0%}) is within evidence-supported range (max {max_confidence:.0%}).",
        details={"stated": stated_confidence, "max_supported": max_confidence},
    )


# ---------------------------------------------------------------------------
# Aggregate Validator
# ---------------------------------------------------------------------------

def run_all_validations(
    # Accounting
    revenue: Optional[float] = None,
    cogs: Optional[float] = None,
    gross_profit: Optional[float] = None,
    ebitda: Optional[float] = None,
    ebitda_margin_pct: Optional[float] = None,
    ebit: Optional[float] = None,
    depreciation: Optional[float] = None,
    total_assets: Optional[float] = None,
    total_liabilities: Optional[float] = None,
    equity: Optional[float] = None,
    # Logical
    revenue_growth_pct: Optional[float] = None,
    margin_change_bps: Optional[float] = None,
    earnings_growth_pct: Optional[float] = None,
    valuation_assessment: Optional[str] = None,
    recommendation: Optional[str] = None,
    pe: Optional[float] = None,
    sector_pe_median: Optional[float] = None,
    roe: Optional[float] = None,
    debt_to_equity: Optional[float] = None,
    interest_coverage: Optional[float] = None,
    # Confidence
    stated_confidence: Optional[float] = None,
    data_points_available: int = 0,
) -> dict:
    """Run all validators and return a summary."""
    all_results: list[ValidationResult] = []

    all_results.extend(validate_accounting_identities(
        revenue=revenue, cogs=cogs, gross_profit=gross_profit,
        ebitda=ebitda, ebitda_margin_pct=ebitda_margin_pct,
        ebit=ebit, depreciation=depreciation,
        total_assets=total_assets, total_liabilities=total_liabilities, equity=equity,
    ))

    all_results.extend(validate_logical_consistency(
        revenue_growth_pct=revenue_growth_pct, margin_change_bps=margin_change_bps,
        earnings_growth_pct=earnings_growth_pct, valuation_assessment=valuation_assessment,
        recommendation=recommendation, pe=pe, sector_pe_median=sector_pe_median,
        roe=roe, debt_to_equity=debt_to_equity, interest_coverage=interest_coverage,
    ))

    if stated_confidence is not None:
        all_results.append(validate_confidence(
            stated_confidence=stated_confidence,
            data_points_available=data_points_available,
        ))

    errors = [r for r in all_results if not r.passed and r.severity == "error"]
    warnings = [r for r in all_results if not r.passed and r.severity == "warning"]
    passed = [r for r in all_results if r.passed]

    return {
        "total_checks": len(all_results),
        "passed": len(passed),
        "warnings": len(warnings),
        "errors": len(errors),
        "all_passed": len(errors) == 0 and len(warnings) == 0,
        "error_messages": [r.message for r in errors],
        "warning_messages": [r.message for r in warnings],
        "details": [r.model_dump() for r in all_results],
    }
