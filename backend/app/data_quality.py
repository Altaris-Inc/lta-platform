"""
data_quality.py — Comprehensive DQ checks for LTA loan tapes.

Implements:
1. Type Inference
2. Missing / Null Detection
3. Column Name / Type Overrides
4. Domain Validation (numeric + date ranges)
5. Near-Constant Detection
6. Outlier Detection (IQR)
7. Boolean Token Mapping
8. Numeric Parsing
9. Date Convention Detection
10. Column Drop Rules
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

NUMERIC_THRESHOLD = 0.70
DATE_THRESHOLD = 0.70
BOOL_THRESHOLD = 0.70
DATE_LIKENESS_THRESHOLD = 0.30

NULL_TOKENS = {"null", "none", "n/a", "na", "nan", "nat", "-", "--", ""}

EXCEL_ERROR_PATTERN = re.compile(
    r"^#(value!|n/a|ref!|div/0!|name[?]|num!|null!|error!)", re.I
)

TRUE_TOKENS = {"1", "true", "t", "y", "yes"}
FALSE_TOKENS = {"0", "false", "f", "n", "no"}

NEAR_CONSTANT_DOMINANCE = 0.70
NEAR_CONSTANT_REL_TOL_PCT = 0.01
NEAR_CONSTANT_CV_TOL = 0.01

IQR_K = 2.0
IQR_MIN_VALUES = 5

ID_LIKE_PATTERN = re.compile(r"(?:^|[_\s])(id|key|code|num|number|ref|uuid|guid|registration|serial|account|crn|ssn|ein|tin)(?:[_\s]|$)", re.I)
FORCE_STRING_COLS: set = set()

DATE_REGEX = re.compile(
    r"^\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4}$|"
    r"^\d{4}\d{2}\d{2}$|"
    r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"
)

DOMAIN_RULES: dict = {
    "fico_origination": {"type": "numeric_range", "min": 300, "max": 900, "inclusive": True},
    "fico_current":     {"type": "numeric_range", "min": 300, "max": 900, "inclusive": True},
    "current_balance":  {"type": "numeric_range", "min": 0, "max": None, "inclusive": True},
    "original_balance": {"type": "numeric_range", "min": 0, "max": None, "inclusive": True},
    "interest_rate":    {"type": "numeric_range", "min": 0, "max": 100, "inclusive": True},
    "dti":              {"type": "numeric_range", "min": 0, "max": 200, "inclusive": True},
    "ltv":              {"type": "numeric_range", "min": 0, "max": 300, "inclusive": True},
    "original_term":    {"type": "numeric_range", "min": 0, "max": 600, "inclusive": True},
    "remaining_term":   {"type": "numeric_range", "min": 0, "max": 600, "inclusive": True},
    "origination_date": {"type": "date_range", "min": "1980-01-01", "max": "2100-12-31"},
    "maturity_date":    {"type": "date_range", "min": "1980-01-01", "max": "2100-12-31"},
}

DOMAIN_PATTERN_RULES = [
    {"pattern": re.compile(r"fico|credit.?score", re.I),
     "type": "numeric_range", "min": 300, "max": 900},
    {"pattern": re.compile(r"(^|_)(city|ltv|loan.?to.?value)", re.I),
     "type": "numeric_range", "min": 0, "max": 300},
    {"pattern": re.compile(r"rate|interest|coupon|apr|apy|wac", re.I),
     "type": "numeric_range", "min": 0, "max": 100},
    {"pattern": re.compile(r"balance|upb|principal|amount|exposure", re.I),
     "type": "numeric_range", "min": 0, "max": None},
    {"pattern": re.compile(r"(^|_)(term|months)", re.I),
     "type": "numeric_range", "min": 0, "max": 600},
    {"pattern": re.compile(r"origination|fund|booking|open.?date|incorporation|inception|issue.?date", re.I),
     "type": "date_range", "min": "1980-01-01", "max": "2100-12-31"},
    {"pattern": re.compile(r"maturity|expiry|exp.?date|settlement|closure", re.I),
     "type": "date_range", "min": "1980-01-01", "max": "2100-12-31"},
    {"pattern": re.compile(r"date", re.I),
     "type": "date_range", "min": "1980-01-01", "max": "2100-12-31"},
]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _parse_numeric(v) -> Optional[float]:
    """Parse value to float, stripping currency symbols, commas, %."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in NULL_TOKENS:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]
    s = re.sub(r"[$£€¥%,\s]", "", s)
    if "%" in str(v):
        try:
            return (-1 if negative else 1) * float(s) / 100
        except ValueError:
            return None
    try:
        return (-1 if negative else 1) * float(s)
    except ValueError:
        return None


def _is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        import pandas as _pd
        if _pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in NULL_TOKENS:
        return True
    if EXCEL_ERROR_PATTERN.match(s):
        return True
    return False


def _parse_date(v) -> Optional[datetime]:
    """Try to parse a value as a date. Returns None if not parseable."""
    if _is_null(v):
        return None
    s = str(v).strip()
    if "00/01/1900" in s or "1900-01-00" in s:
        return datetime(1900, 1, 1)
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
                "%Y%m%d", "%d.%m.%Y", "%m.%d.%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _looks_date_like(s: str) -> bool:
    return bool(DATE_REGEX.match(str(s).strip()))


def _get_domain_rule(field_key: str, col_name: str) -> Optional[dict]:
    """Get domain rule: exact field match first, then pattern match on col name."""
    if field_key in DOMAIN_RULES:
        return DOMAIN_RULES[field_key]
    col_lower = col_name.lower()
    for pr in DOMAIN_PATTERN_RULES:
        if pr["pattern"].search(col_lower):
            return pr
    return None


# ═══════════════════════════════════════════════════════════════
# 1. TYPE INFERENCE
# ═══════════════════════════════════════════════════════════════

def infer_type(series: pd.Series, col_name: str = "") -> str:
    """Infer column type: boolean, datetime, numeric, string."""
    if ID_LIKE_PATTERN.search(col_name) or col_name in FORCE_STRING_COLS:
        return "string"

    non_null = series.dropna()
    non_null = non_null[non_null.apply(lambda v: not _is_null(v))]
    n = len(non_null)
    if n == 0:
        return "string"

    # Force datetime if any cell contains 00/01/1900
    if non_null.astype(str).str.contains("00/01/1900", na=False).any():
        return "datetime"

    # Boolean check
    bool_vals = non_null.astype(str).str.strip().str.lower()
    bool_match = bool_vals.isin(TRUE_TOKENS | FALSE_TOKENS).sum()
    if bool_match / n >= BOOL_THRESHOLD:
        return "boolean"

    # Date check
    date_parsed = non_null.apply(_parse_date)
    if date_parsed.notna().sum() / n >= DATE_THRESHOLD:
        return "datetime"

    # Date-likeness promotion
    date_like = non_null.astype(str).apply(_looks_date_like).sum()
    if date_like / n >= DATE_LIKENESS_THRESHOLD:
        return "datetime"

    # Numeric check
    num_parsed = non_null.apply(_parse_numeric)
    if num_parsed.notna().sum() / n >= NUMERIC_THRESHOLD:
        return "numeric"

    return "string"


# ═══════════════════════════════════════════════════════════════
# 2. MISSING / NULL DETECTION
# ═══════════════════════════════════════════════════════════════

def check_missing(series: pd.Series) -> dict:
    """Return missing count, empty count, null token count, missing pct."""
    n = len(series)
    if n == 0:
        return {"total": 0, "missing": 0, "missing_pct": 0.0}

    empty = series.apply(lambda v: v is None or (
        isinstance(v, float) and math.isnan(v))).sum()
    null_token = series.apply(
        lambda v: str(v).strip().lower() in NULL_TOKENS and not _is_null(v) or False
    ).sum()
    # Recount properly
    total_missing = series.apply(_is_null).sum()
    return {
        "total": n,
        "missing": int(total_missing),
        "empty": int(empty),
        "null_token": int(null_token),
        "missing_pct": round(total_missing / n * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════
# 3 + 4. DOMAIN VALIDATION
# ═══════════════════════════════════════════════════════════════

def check_domain(series: pd.Series, field_key: str, col_name: str,
                 inferred_type: str) -> dict:
    """Check values against domain rules. Returns violation count and sample."""
    rule = _get_domain_rule(field_key, col_name)
    if not rule:
        return {"violations": 0, "violation_pct": 0.0, "samples": []}

    n = len(series)
    violations = []

    if rule["type"] == "numeric_range":
        vals = series.apply(_parse_numeric)
        non_null = vals.dropna()
        mn = rule.get("min")
        mx = rule.get("max")
        for idx, v in non_null.items():
            bad = False
            if mn is not None and v < mn:
                bad = True
            if mx is not None and v > mx:
                bad = True
            if bad:
                violations.append(str(series.at[idx]))
        return {
            "violations": len(violations),
            "violation_pct": round(len(violations) / n * 100, 2) if n else 0.0,
            "samples": violations[:5],
        }

    elif rule["type"] == "date_range":
        mn_dt = datetime.strptime(rule["min"], "%Y-%m-%d") if rule.get("min") else None
        mx_dt = datetime.strptime(rule["max"], "%Y-%m-%d") if rule.get("max") else None
        for idx, v in series.items():
            dt = _parse_date(v)
            if dt is None:
                continue
            bad = False
            if mn_dt and dt < mn_dt:
                bad = True
            if mx_dt and dt > mx_dt:
                bad = True
            if bad:
                violations.append(str(v))
        return {
            "violations": len(violations),
            "violation_pct": round(len(violations) / n * 100, 2) if n else 0.0,
            "samples": violations[:5],
        }

    return {"violations": 0, "violation_pct": 0.0, "samples": []}


# ═══════════════════════════════════════════════════════════════
# 5. NEAR-CONSTANT DETECTION
# ═══════════════════════════════════════════════════════════════

def check_near_constant(series: pd.Series) -> dict:
    """Detect near-constant columns via dominance, range tightness, or CV."""
    non_null = series.dropna()
    non_null = non_null[non_null.apply(lambda v: not _is_null(v))]
    n = len(non_null)
    if n == 0:
        return {"is_near_constant": False, "reason": None}

    # Dominance check
    top_val = non_null.astype(str).value_counts()
    if len(top_val) > 0:
        dominance = top_val.iloc[0] / n
        if dominance >= NEAR_CONSTANT_DOMINANCE:
            return {
                "is_near_constant": True,
                "reason": f"Top value '{top_val.index[0]}' dominates {dominance:.0%}",
                "dominance": round(float(dominance), 4),
            }

    # Numeric tightness checks
    nums = non_null.apply(_parse_numeric).dropna()
    if len(nums) >= 2:
        median = float(nums.median())
        rng = float(nums.max() - nums.min())
        if median != 0 and rng / abs(median) <= NEAR_CONSTANT_REL_TOL_PCT:
            return {
                "is_near_constant": True,
                "reason": f"Range {rng:.4f} is ≤1% of median {median:.4f}",
                "dominance": None,
            }
        mean = float(nums.mean())
        std = float(nums.std())
        if mean != 0 and std / abs(mean) <= NEAR_CONSTANT_CV_TOL:
            return {
                "is_near_constant": True,
                "reason": f"CV={std/abs(mean):.4f} ≤ 0.01",
                "dominance": None,
            }

    return {"is_near_constant": False, "reason": None}


# ═══════════════════════════════════════════════════════════════
# 6. OUTLIER DETECTION (IQR)
# ═══════════════════════════════════════════════════════════════

def check_outliers(series: pd.Series) -> dict:
    """IQR-based outlier detection. Requires ≥5 non-null values and IQR > 0."""
    nums = series.apply(_parse_numeric).dropna()
    if len(nums) < IQR_MIN_VALUES:
        return {"outliers": 0, "outlier_pct": 0.0, "low": None, "high": None}

    q1 = float(nums.quantile(0.25))
    q3 = float(nums.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return {"outliers": 0, "outlier_pct": 0.0, "low": None, "high": None}

    low = q1 - IQR_K * iqr
    high = q3 + IQR_K * iqr
    outlier_mask = (nums < low) | (nums > high)
    outlier_count = int(outlier_mask.sum())

    return {
        "outliers": outlier_count,
        "outlier_pct": round(outlier_count / len(nums) * 100, 2),
        "low": round(low, 4),
        "high": round(high, 4),
    }


# ═══════════════════════════════════════════════════════════════
# 9. DATE CONVENTION DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_date_convention(series: pd.Series) -> Optional[str]:
    """Detect EU (DD/MM/YYYY) vs US (MM/DD/YYYY) date format."""
    eu_count = us_count = 0
    for v in series.dropna():
        s = str(v).strip()
        m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})$", s)
        if not m:
            continue
        d1, d2 = int(m.group(1)), int(m.group(2))
        if d1 > 12:
            eu_count += 1
        elif d2 > 12:
            us_count += 1

    total = eu_count + us_count
    if total == 0:
        return None
    eu_rate = eu_count / total
    us_rate = us_count / total
    if eu_rate - us_rate >= 0.03 or eu_count > us_count:
        return "EU (DD/MM/YYYY)"
    if us_rate - eu_rate >= 0.03 or us_count > eu_count:
        return "US (MM/DD/YYYY)"
    return "ambiguous"


# ═══════════════════════════════════════════════════════════════
# MAIN: RUN ALL DQ CHECKS
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# KNOWN DERIVED COLUMNS AND THEIR SOURCE DEPENDENCIES
# ═══════════════════════════════════════════════════════════════

DERIVED_COLUMN_NAMES = {
    "period_balance", "period_principal", "period_interest",
    "period_payment", "delinquency_state", "performance_status",
    "cumulative_loss", "prepayment_flag",
}

DERIVED_SOURCE_DEPS = {
    "period_balance": ["current_balance"],
    "period_principal": ["current_balance", "original_balance"],
    "period_interest": ["interest_rate", "current_balance"],
    "delinquency_state": ["loan_status", "days_past_due"],
}

DEFAULT_DPD_THRESHOLD = 120
PERFORMING_DPD_THRESHOLD = 1
HIGH_CARDINALITY_THRESHOLD = 13


# ═══════════════════════════════════════════════════════════════
# N1. STRUCTURAL / SCHEMA CHECKS
# ═══════════════════════════════════════════════════════════════

def check_structural(df: pd.DataFrame, mapping: dict) -> list:
    """
    N1 structural checks:
    - Required columns missing
    - All-blank mapped columns
    - Output column collision
    - Missing source column for derivation
    """
    flags = []
    mapped_cols = set(mapping.values())

    # Required columns
    for required_field in ["loan_id", "snapshot_date"]:
        if required_field not in mapping:
            flags.append({
                "flag": "MISSING_REQUIRED_COLUMN",
                "field": required_field,
                "column": None,
                "detail": f"Required field '{required_field}' is not mapped",
                "severity": "error",
            })

    # All-blank mapped columns
    for field_key, col_name in mapping.items():
        if col_name not in df.columns:
            continue
        series = df[col_name]
        non_null = series.dropna()
        non_null_nonempty = non_null[non_null.astype(str).str.strip() != ""]
        if len(non_null_nonempty) == 0:
            flags.append({
                "flag": "ALL_BLANK_COLUMN",
                "field": field_key,
                "column": col_name,
                "detail": f"Column '{col_name}' is entirely blank or null",
                "severity": "error",
            })

    # Output column collision
    for derived_col in DERIVED_COLUMN_NAMES:
        if derived_col in df.columns and derived_col not in mapped_cols:
            flags.append({
                "flag": "OUTPUT_COLUMN_COLLISION",
                "field": None,
                "column": derived_col,
                "detail": f"Derived column '{derived_col}' already exists in tape",
                "severity": "warning",
            })

    # Missing source column for derivation
    for derived_col, sources in DERIVED_SOURCE_DEPS.items():
        if derived_col in df.columns:
            continue
        for src_field in sources:
            if src_field not in mapping:
                flags.append({
                    "flag": "MISSING_SOURCE_COLUMN",
                    "field": src_field,
                    "column": None,
                    "detail": f"Field '{src_field}' needed to derive '{derived_col}' is not mapped",
                    "severity": "warning",
                })

    return flags


# ═══════════════════════════════════════════════════════════════
# N3. BOOLEAN EXTENDED CHECKS
# ═══════════════════════════════════════════════════════════════

BOOL_GROUP_YN = {"y", "n", "yes", "no"}
BOOL_GROUP_10 = {"1", "0"}
BOOL_GROUP_TF = {"true", "false", "t", "f"}


def check_boolean_extended(series: pd.Series) -> dict:
    """
    N3 boolean extended checks:
    - Mixed boolean encoding (Y/N + 1/0 + True/False simultaneously)
    - Boolean column has nulls
    """
    non_null = series.dropna()
    non_null = non_null[~non_null.apply(_is_null)]
    vals = non_null.astype(str).str.strip().str.lower()

    has_null = series.apply(_is_null).any()

    groups_present = []
    if vals.isin(BOOL_GROUP_YN).any():
        groups_present.append("Y/N")
    if vals.isin(BOOL_GROUP_10).any():
        groups_present.append("1/0")
    if vals.isin(BOOL_GROUP_TF).any():
        groups_present.append("True/False")

    mixed_encoding = len(groups_present) > 1

    return {
        "mixed_encoding": mixed_encoding,
        "encoding_groups": groups_present,
        "has_null": bool(has_null),
    }


# ═══════════════════════════════════════════════════════════════
# N4. LOAN PERFORMANCE DOMAIN CHECKS
# ═══════════════════════════════════════════════════════════════

def check_loan_performance(
    df: pd.DataFrame,
    mapping: dict,
    default_threshold: int = DEFAULT_DPD_THRESHOLD,
    performing_threshold: int = PERFORMING_DPD_THRESHOLD,
) -> list:
    """
    N4 loan performance checks (only runs if days_past_due is mapped):
    - DPD negative values
    - Default threshold breach (DPD >= 120)
    - Non-performing loan flag (DPD >= 1)
    """
    flags = []

    dpd_col = mapping.get("days_past_due")
    if not dpd_col or dpd_col not in df.columns:
        return flags

    dpd_series = df[dpd_col].apply(_parse_numeric)
    valid = dpd_series.dropna()
    n = len(df)

    # Negative DPD
    neg_mask = valid < 0
    neg_count = int(neg_mask.sum())
    if neg_count > 0:
        flags.append({
            "flag": "NEGATIVE_DPD",
            "field": "days_past_due",
            "column": dpd_col,
            "count": neg_count,
            "pct": round(neg_count / n * 100, 2),
            "detail": f"{neg_count} rows have days_past_due < 0",
            "severity": "error",
        })

    # Default threshold breach
    default_mask = valid >= default_threshold
    default_count = int(default_mask.sum())
    if default_count > 0:
        flags.append({
            "flag": "DEFAULT_THRESHOLD_BREACH",
            "field": "days_past_due",
            "column": dpd_col,
            "count": default_count,
            "pct": round(default_count / n * 100, 2),
            "detail": f"{default_count} rows have days_past_due >= {default_threshold}",
            "severity": "warning",
        })

    # Non-performing loans
    non_perf_mask = valid >= performing_threshold
    non_perf_count = int(non_perf_mask.sum())
    if non_perf_count > 0:
        flags.append({
            "flag": "NON_PERFORMING_LOAN",
            "field": "days_past_due",
            "column": dpd_col,
            "count": non_perf_count,
            "pct": round(non_perf_count / n * 100, 2),
            "detail": f"{non_perf_count} rows have days_past_due >= {performing_threshold}",
            "severity": "info",
        })

    return flags


# ═══════════════════════════════════════════════════════════════
# N5. PANEL STRUCTURE CHECKS
# ═══════════════════════════════════════════════════════════════

def check_panel_structure(df: pd.DataFrame, mapping: dict) -> list:
    """
    N5 panel structure checks:
    - Duplicate panel key (loan_id + snapshot_date)
    - loan_status entirely missing
    - days_past_due missing with/without fallback
    - delinquency_state missing
    - High cardinality for stratification
    """
    flags = []

    loan_id_col = mapping.get("loan_id")
    snapshot_col = mapping.get("snapshot_date")
    dpd_col = mapping.get("days_past_due")
    mpd_col = mapping.get("months_past_due")
    status_col = mapping.get("loan_status")
    delinq_col = mapping.get("delinquency_state")

    # Duplicate panel key
    if loan_id_col and snapshot_col and             loan_id_col in df.columns and snapshot_col in df.columns:
        dupes = df.duplicated(subset=[loan_id_col, snapshot_col]).sum()
        if dupes > 0:
            flags.append({
                "flag": "DUPLICATE_PANEL_KEY",
                "field": "loan_id + snapshot_date",
                "column": f"{loan_id_col}, {snapshot_col}",
                "count": int(dupes),
                "pct": round(dupes / len(df) * 100, 2),
                "detail": f"{dupes} duplicate (loan_id, snapshot_date) pairs",
                "severity": "error",
            })

    # loan_status entirely missing
    if not status_col or status_col not in df.columns:
        flags.append({
            "flag": "MISSING_LOAN_STATUS",
            "field": "loan_status",
            "column": None,
            "count": None,
            "pct": None,
            "detail": "loan_status field is not mapped",
            "severity": "warning",
        })

    # days_past_due missing
    if not dpd_col or dpd_col not in df.columns:
        if mpd_col and mpd_col in df.columns:
            flags.append({
                "flag": "MISSING_DPD_FALLBACK_AVAILABLE",
                "field": "days_past_due",
                "column": None,
                "count": None,
                "pct": None,
                "detail": f"days_past_due not mapped but months_past_due ({mpd_col}) available — DPD = MPD × 30",
                "severity": "warning",
            })
        else:
            flags.append({
                "flag": "MISSING_DPD_NO_FALLBACK",
                "field": "days_past_due",
                "column": None,
                "count": None,
                "pct": None,
                "detail": "days_past_due not mapped and no months_past_due fallback available",
                "severity": "error",
            })

    # delinquency_state missing
    if not delinq_col or delinq_col not in df.columns:
        synthesis_note = f"loan_status ({status_col}) available for synthesis"             if status_col and status_col in df.columns else "no synthesis source available"
        flags.append({
            "flag": "MISSING_DELINQUENCY_STATE",
            "field": "delinquency_state",
            "column": None,
            "count": None,
            "pct": None,
            "detail": f"delinquency_state not mapped — {synthesis_note}",
            "severity": "warning",
        })

    # High cardinality for stratification
    for field_key, col_name in mapping.items():
        if col_name not in df.columns:
            continue
        inferred = infer_type(df[col_name], col_name)
        if inferred in ("string", "boolean") and not ID_LIKE_PATTERN.search(col_name):
            n_distinct = df[col_name].nunique(dropna=True)
            if n_distinct > HIGH_CARDINALITY_THRESHOLD:
                flags.append({
                    "flag": "HIGH_CARDINALITY_FOR_STRAT",
                    "field": field_key,
                    "column": col_name,
                    "count": int(n_distinct),
                    "pct": None,
                    "detail": f"{n_distinct} distinct values — too many for cross-tab stratification",
                    "severity": "info",
                })

    return flags



def run_dq_checks(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Run all DQ checks on mapped and unmapped columns.

    Args:
        df: The full loan tape DataFrame
        mapping: dict of {field_key: column_name}

    Returns:
        dict with summary stats, per-column results, and unmapped column results
    """
    results = []
    total_cells = 0
    total_missing = 0
    total_violations = 0
    total_outliers = 0

    mapped_cols = set(mapping.values())

    for field_key, col_name in mapping.items():
        if col_name not in df.columns:
            continue

        series = df[col_name]
        n = len(series)
        total_cells += n

        # Type inference
        inferred_type = infer_type(series, col_name)

        # Missing check
        missing_info = check_missing(series)
        total_missing += missing_info["missing"]

        # Domain validation
        domain_info = check_domain(series, field_key, col_name, inferred_type)
        total_violations += domain_info["violations"]

        # Near-constant
        near_const = check_near_constant(series)

        # Outliers (numeric only)
        outlier_info = {"outliers": 0, "outlier_pct": 0.0, "low": None, "high": None}
        if inferred_type == "numeric":
            outlier_info = check_outliers(series)
            total_outliers += outlier_info["outliers"]

        # Date convention (datetime only)
        date_convention = None
        if inferred_type == "datetime":
            date_convention = detect_date_convention(series)

        # Boolean extended checks
        bool_ext = {"mixed_encoding": False, "encoding_groups": [], "has_null": False}
        if inferred_type == "boolean":
            bool_ext = check_boolean_extended(series)

        # Build issue list for this column
        issues = []
        if missing_info["missing_pct"] > 10:
            issues.append(f"{missing_info['missing_pct']:.1f}% missing values")
        if domain_info["violations"] > 0:
            issues.append(f"{domain_info['violations']} out-of-range values")
        if near_const["is_near_constant"]:
            issues.append(f"Near-constant: {near_const['reason']}")
        if outlier_info["outliers"] > 0:
            issues.append(f"{outlier_info['outliers']} outliers detected")
        if bool_ext["mixed_encoding"]:
            issues.append(f"Mixed boolean encoding: {', '.join(bool_ext['encoding_groups'])}")
        if bool_ext["has_null"]:
            issues.append("Boolean column has nulls — unsafe for int coercion")

        # Overall status
        if any("missing" in i or "out-of-range" in i for i in issues):
            status = "⚠️ Warning"
        elif issues:
            status = "ℹ️ Info"
        else:
            status = "✅ OK"

        results.append({
            "field_key": field_key,
            "column": col_name,
            "row_count": n,
            "inferred_type": inferred_type,
            "missing": missing_info["missing"],
            "missing_pct": missing_info["missing_pct"],
            "domain_violations": domain_info["violations"],
            "domain_violation_pct": domain_info["violation_pct"],
            "domain_samples": domain_info["samples"],
            "near_constant": near_const["is_near_constant"],
            "near_constant_reason": near_const.get("reason"),
            "outliers": outlier_info["outliers"],
            "outlier_pct": outlier_info["outlier_pct"],
            "outlier_low": outlier_info["low"],
            "outlier_high": outlier_info["high"],
            "date_convention": date_convention,
            "bool_mixed_encoding": bool_ext["mixed_encoding"],
            "bool_encoding_groups": bool_ext["encoding_groups"],
            "bool_has_null": bool_ext["has_null"],
            "issues": issues,
            "status": status,
        })

    # ── Unmapped columns ──
    unmapped_results = []
    for col_name in df.columns:
        if col_name in mapped_cols:
            continue

        series = df[col_name]
        n = len(series)

        inferred_type = infer_type(series, col_name)
        missing_info = check_missing(series)
        domain_info = check_domain(series, "", col_name, inferred_type)
        near_const = check_near_constant(series)

        outlier_info = {"outliers": 0, "outlier_pct": 0.0, "low": None, "high": None}
        if inferred_type == "numeric":
            outlier_info = check_outliers(series)

        date_convention = None
        if inferred_type == "datetime":
            date_convention = detect_date_convention(series)

        issues = []
        if missing_info["missing_pct"] > 10:
            issues.append(f"{missing_info['missing_pct']:.1f}% missing values")
        if domain_info["violations"] > 0:
            issues.append(f"{domain_info['violations']} out-of-range values")
        if near_const["is_near_constant"]:
            issues.append(f"Near-constant: {near_const['reason']}")
        if outlier_info["outliers"] > 0:
            issues.append(f"{outlier_info['outliers']} outliers detected")

        status = "⚠️ Warning" if any("missing" in i or "out-of-range" in i for i in issues)             else "ℹ️ Info" if issues else "✅ OK"

        unmapped_results.append({
            "column": col_name,
            "row_count": n,
            "inferred_type": inferred_type,
            "missing": missing_info["missing"],
            "missing_pct": missing_info["missing_pct"],
            "domain_violations": domain_info["violations"],
            "domain_violation_pct": domain_info["violation_pct"],
            "domain_samples": domain_info["samples"],
            "near_constant": near_const["is_near_constant"],
            "near_constant_reason": near_const.get("reason"),
            "outliers": outlier_info["outliers"],
            "outlier_pct": outlier_info["outlier_pct"],
            "outlier_low": outlier_info["low"],
            "outlier_high": outlier_info["high"],
            "date_convention": date_convention,
            "issues": issues,
            "status": status,
        })

    # Summary
    completeness = ((total_cells - total_missing) / total_cells * 100
                    if total_cells > 0 else 0)
    grade = "A" if completeness >= 95 else "B" if completeness >= 85 else \
            "C" if completeness >= 70 else "D"

    warning_cols = [r["column"] for r in results if r["status"] == "⚠️ Warning"]
    ok_cols = [r["column"] for r in results if r["status"] == "✅ OK"]

    # Run new checks
    structural_flags = check_structural(df, mapping)
    performance_flags = check_loan_performance(df, mapping)
    panel_flags = check_panel_structure(df, mapping)
    all_extra_flags = structural_flags + performance_flags + panel_flags

    error_flags = [f for f in all_extra_flags if f["severity"] == "error"]
    warning_flags = [f for f in all_extra_flags if f["severity"] == "warning"]

    return {
        "summary": {
            "total_columns_checked": len(results),
            "total_rows": len(df),
            "completeness": round(completeness, 2),
            "grade": grade,
            "total_missing": total_missing,
            "total_violations": total_violations,
            "total_outliers": total_outliers,
            "warning_count": len(warning_cols),
            "ok_count": len(ok_cols),
            "unmapped_count": len(unmapped_results),
            "structural_errors": len(error_flags),
            "structural_warnings": len(warning_flags),
        },
        "columns": results,
        "unmapped": unmapped_results,
        "structural": structural_flags,
        "performance": performance_flags,
        "panel": panel_flags,
    }
