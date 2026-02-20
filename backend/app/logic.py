"""
Loan Tape Analyzer — Core Logic Module

All pure functions: column matching, analysis, validation, regression.
No UI dependencies — fully testable with pytest.
"""
import re
import math
import numpy as np
import pandas as pd
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# STANDARD FIELDS — 47 ABS fields with regex patterns
# ═══════════════════════════════════════════════════════════════

STD_FIELDS = {
    "loan_id":              {"label": "Loan ID",       "patterns": [r"loan.?id", r"account.?(id|num|no)", r"^id$"]},
    "original_balance":     {"label": "Orig Bal",      "patterns": [r"orig.?(bal|amount|principal)", r"loan.?amount", r"funded"]},
    "current_balance":      {"label": "Curr Bal",      "patterns": [r"curr.?bal", r"current.?(bal|amount)", r"^upb$", r"outstanding"]},
    "interest_rate":        {"label": "Rate",           "patterns": [r"interest.?rate", r"^rate$", r"coupon", r"^apr$"]},
    "original_term":        {"label": "Orig Term",     "patterns": [r"orig.?term", r"^term$", r"loan.?term"]},
    "remaining_term":       {"label": "Rem Term",      "patterns": [r"remain.?term", r"rem.?term"]},
    "monthly_payment":      {"label": "Mo Pmt",        "patterns": [r"monthly.?pay", r"installment", r"^pmt$"]},
    "fico_origination":     {"label": "FICO Orig",     "patterns": [r"fico.?orig", r"orig.?fico", r"orig.?score"]},
    "fico_current":         {"label": "FICO Curr",     "patterns": [r"fico.?curr", r"curr.?fico", r"fico$", r"credit.?score$"]},
    "dti":                  {"label": "DTI",            "patterns": [r"^dti", r"debt.?to.?income", r"dti.?back"]},
    "loan_status":          {"label": "Status",         "patterns": [r"loan.?status", r"^status$"]},
    "dpd":                  {"label": "DPD",            "patterns": [r"days.?past", r"^dpd$"]},
    "dpd_bucket":           {"label": "DPD Bucket",    "patterns": [r"dpd.?bucket"]},
    "times_30dpd":          {"label": "30DPD Ct",      "patterns": [r"times.?30"]},
    "times_60dpd":          {"label": "60DPD Ct",      "patterns": [r"times.?60"]},
    "times_90dpd":          {"label": "90DPD Ct",      "patterns": [r"times.?90"]},
    "origination_date":     {"label": "Orig Date",     "patterns": [r"orig.?date", r"origination", r"issue.?date"]},
    "loan_purpose":         {"label": "Purpose",        "patterns": [r"purpose"]},
    "state":                {"label": "State",          "patterns": [r"state", r"borrower.?state"]},
    "annual_income":        {"label": "Annual Inc",    "patterns": [r"annual.?income", r"gross.?income"]},
    "monthly_income":       {"label": "Mo Inc",        "patterns": [r"monthly.?income"]},
    "employment_status":    {"label": "Empl",           "patterns": [r"employ.?status"]},
    "employment_length":    {"label": "Empl Yrs",      "patterns": [r"employ.?length"]},
    "grade":                {"label": "Grade",          "patterns": [r"^grade$", r"^sub.?grade$"]},
    "origination_channel":  {"label": "Channel",        "patterns": [r"channel", r"orig.?channel"]},
    "income_verification":  {"label": "Inc Verif",     "patterns": [r"income.?verif", r"verif.?status"]},
    "housing_status":       {"label": "Housing",        "patterns": [r"housing", r"home.?own"]},
    "open_accounts":        {"label": "Open Accts",    "patterns": [r"open.?(acc|credit|lines)"]},
    "revolving_utilization":{"label": "Rev Util",      "patterns": [r"revolv.?util"]},
    "total_paid_principal":  {"label": "Princ Paid",   "patterns": [r"total.?princ.?paid", r"principal.?paid"]},
    "total_paid_interest":   {"label": "Int Paid",     "patterns": [r"total.?int.?paid", r"interest.?paid"]},
    "origination_fee":      {"label": "Orig Fee",      "patterns": [r"orig.?fee"]},
    "late_fees":            {"label": "Late Fees",     "patterns": [r"late.?fee"]},
    "recoveries":           {"label": "Recoveries",    "patterns": [r"recover"]},
    "net_loss":             {"label": "Net Loss",      "patterns": [r"net.?loss", r"write.?off"]},
    "months_on_book":       {"label": "MOB",            "patterns": [r"months.?on.?book", r"loan.?age", r"^mob$"]},
    "vintage":              {"label": "Vintage",        "patterns": [r"vintage", r"cohort", r"orig.?year"]},
    "co_borrower":          {"label": "Co-Borr",       "patterns": [r"co.?borrow", r"joint"]},
    "hardship":             {"label": "Hardship",       "patterns": [r"hardship", r"forbear"]},
    "modification":         {"label": "Mod",            "patterns": [r"modif", r"tdr"]},
    "pd_score":             {"label": "PD",             "patterns": [r"^pd", r"prob.?default", r"pd.?model"]},
    "lgd":                  {"label": "LGD",            "patterns": [r"^lgd", r"loss.?given"]},
    "expected_loss":        {"label": "Exp Loss",      "patterns": [r"expected.?loss", r"^el$"]},
    "pool_id":              {"label": "Pool",           "patterns": [r"pool", r"trust"]},
    "servicer":             {"label": "Servicer",       "patterns": [r"servicer"]},
    "investor":             {"label": "Investor",       "patterns": [r"investor"]},
    "autopay":              {"label": "Autopay",        "patterns": [r"auto.?pay"]},
}


# ═══════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════

def parse_numeric(v) -> Optional[float]:
    """Parse a value to float, stripping $, %, commas, whitespace. Returns None if not numeric."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    s = re.sub(r'[$,%\s]', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def format_currency(v) -> str:
    if v is None:
        return "—"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:.1f}K"
    return f"${v:.0f}"


def format_pct(v) -> str:
    return f"{(v or 0):.1f}%"


def format_rate(v) -> str:
    return f"{(v or 0):.2f}%"


def format_score(v) -> str:
    return str(round(v or 0))


# ═══════════════════════════════════════════════════════════════
# COLUMN MATCHING
# ═══════════════════════════════════════════════════════════════

def rule_match(df: pd.DataFrame, fields: Optional[dict] = None) -> dict:
    """
    Match DataFrame columns to standard fields using regex patterns + value heuristics.
    Returns dict: {field_key: column_name}
    """
    flds = fields or STD_FIELDS
    hdrs = list(df.columns)
    rows = df.head(20)

    # Track best match: col -> (field_key, score)
    col_best = {}  # column -> (field_key, score)
    field_best = {}  # field_key -> (column, score)

    for h in hdrs:
        for fk, fdef in flds.items():
            score = 0
            for pat in fdef["patterns"]:
                if re.search(pat, h.strip(), re.IGNORECASE):
                    score += 50
                    break

            if score > 0 and len(rows) > 0:
                vals = rows[h].dropna()
                vals = vals[vals.astype(str).str.strip() != ""]
                if len(vals) > 0:
                    nums = vals.apply(parse_numeric).dropna()
                    nr = len(nums) / len(vals) if len(vals) > 0 else 0
                    if nr > 0.8 and len(nums) > 0:
                        av = nums.mean()
                        if 300 < av < 900:
                            score += 15
                        elif 0 < av < 35:
                            score += 10
                        elif 100 < av < 1e7:
                            score += 5
                    # State abbreviation heuristic
                    state_matches = vals.astype(str).str.strip().str.match(r'^[A-Z]{2}$')
                    if state_matches.sum() > len(vals) * 0.5:
                        score += 20

            if score >= 50:
                if fk not in field_best or score > field_best[fk][1]:
                    # Remove previous assignment for this field
                    if fk in field_best:
                        old_col = field_best[fk][0]
                        if old_col in col_best and col_best[old_col][0] == fk:
                            del col_best[old_col]
                    # Check if column already assigned to another field with higher score
                    if h not in col_best or score > col_best[h][1]:
                        if h in col_best:
                            old_fk = col_best[h][0]
                            if old_fk in field_best:
                                del field_best[old_fk]
                        col_best[h] = (fk, score)
                        field_best[fk] = (h, score)

    return {fk: col for fk, (col, _) in field_best.items()}


def score_template(template_mapping: dict, headers: list) -> float:
    """Score how well a template matches incoming headers. Returns 0.0-1.0."""
    if not template_mapping:
        return 0.0
    total = len(template_mapping)
    hits = sum(1 for col in template_mapping.values() if col in headers)
    return hits / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

def get_numeric(df: pd.DataFrame, mp: dict, field: str) -> pd.Series:
    """Get parsed numeric values for a mapped field."""
    if field not in mp or mp[field] not in df.columns:
        return pd.Series(dtype=float)
    return df[mp[field]].reset_index(drop=True).apply(parse_numeric)


def get_string(df: pd.DataFrame, mp: dict, field: str) -> pd.Series:
    """Get string values for a mapped field."""
    if field not in mp or mp[field] not in df.columns:
        return pd.Series(dtype=str)
    return df[mp[field]].reset_index(drop=True).fillna("").astype(str).str.strip()


def bucket(df: pd.DataFrame, mp: dict, field: str, buckets: list, total_bal: float) -> list:
    """Create balance-weighted distribution buckets."""
    vals = get_numeric(df, mp, field)
    bals = get_numeric(df, mp, "current_balance")
    results = []
    for b in buckets:
        if vals.empty:
            results.append({
                "name": b["label"], "count": 0, "balance": 0,
                "pct": 0, "field": field, "min": b["min"], "max": b["max"]
            })
        else:
            mask = (vals >= b["min"]) & (vals <= b["max"]) & vals.notna()
            count = mask.sum()
            bal = bals.loc[mask].fillna(0).sum() if not bals.empty else 0
            pct = (bal / total_bal * 100) if total_bal > 0 else 0
            results.append({
                "name": b["label"], "count": int(count), "balance": bal,
                "pct": pct, "field": field, "min": b["min"], "max": b["max"]
            })
    return results


def group_by(df: pd.DataFrame, mp: dict, field: str, total_bal: float) -> list:
    """Group by a categorical field, return sorted by balance."""
    vals = get_string(df, mp, field)
    bals = get_numeric(df, mp, "current_balance").fillna(0)
    if vals.empty:
        return []
    grouped = pd.DataFrame({"val": vals.replace("", "Unknown"), "bal": bals})
    agg = grouped.groupby("val").agg(count=("bal", "size"), balance=("bal", "sum")).reset_index()
    agg["pct"] = agg["balance"] / total_bal * 100 if total_bal > 0 else 0
    agg = agg.sort_values("balance", ascending=False)
    return [
        {"name": row["val"], "count": int(row["count"]), "balance": row["balance"],
         "pct": row["pct"], "field": field}
        for _, row in agg.iterrows()
    ]


def analyze(df: pd.DataFrame, mp: dict) -> dict:
    """Full pool analysis. Returns dict with all metrics, distributions, groupings."""
    df = df.reset_index(drop=True)
    bals = get_numeric(df, mp, "current_balance").dropna()
    obs = get_numeric(df, mp, "original_balance").dropna()
    tb = bals.sum()
    tob = obs.sum()

    def wa(field):
        """Weighted average by current balance."""
        v = get_numeric(df, mp, field)
        b = get_numeric(df, mp, "current_balance")
        mask = v.notna() & b.notna()
        if mask.sum() == 0 or tb == 0:
            return 0.0
        return (v[mask] * b[mask]).sum() / b[mask].sum()

    rates = get_numeric(df, mp, "interest_rate").dropna()
    mobs = get_numeric(df, mp, "months_on_book").dropna()

    # Distributions
    fico_buckets = [
        {"label": "<580", "min": 0, "max": 579}, {"label": "580-619", "min": 580, "max": 619},
        {"label": "620-659", "min": 620, "max": 659}, {"label": "660-699", "min": 660, "max": 699},
        {"label": "700-739", "min": 700, "max": 739}, {"label": "740-779", "min": 740, "max": 779},
        {"label": "780+", "min": 780, "max": 999},
    ]
    rate_buckets = [
        {"label": "0-6%", "min": 0, "max": 6}, {"label": "6-9%", "min": 6.01, "max": 9},
        {"label": "9-12%", "min": 9.01, "max": 12}, {"label": "12-18%", "min": 12.01, "max": 18},
        {"label": "18-24%", "min": 18.01, "max": 24}, {"label": "24%+", "min": 24.01, "max": 99},
    ]
    dti_buckets = [
        {"label": "0-20%", "min": 0, "max": 20}, {"label": "20-30%", "min": 20.01, "max": 30},
        {"label": "30-40%", "min": 30.01, "max": 40}, {"label": "40-50%", "min": 40.01, "max": 50},
        {"label": "50%+", "min": 50.01, "max": 200},
    ]
    term_buckets = [
        {"label": "12-24", "min": 12, "max": 24}, {"label": "36", "min": 25, "max": 36},
        {"label": "48", "min": 37, "max": 48}, {"label": "60", "min": 49, "max": 60},
        {"label": "72-84", "min": 61, "max": 84},
    ]

    fico_dist = bucket(df, mp, "fico_origination", fico_buckets, tb)
    rate_dist = bucket(df, mp, "interest_rate", rate_buckets, tb)
    dti_dist = bucket(df, mp, "dti", dti_buckets, tb)
    term_dist = bucket(df, mp, "original_term", term_buckets, tb)

    # Groupings
    geo = group_by(df, mp, "state", tb)[:20]
    stat = group_by(df, mp, "loan_status", tb)
    purp = group_by(df, mp, "loan_purpose", tb)
    grad = group_by(df, mp, "grade", tb)
    chan = group_by(df, mp, "origination_channel", tb)
    veri = group_by(df, mp, "income_verification", tb)
    hous = group_by(df, mp, "housing_status", tb)
    vint = group_by(df, mp, "vintage", tb)

    # Concentration
    hhi = sum((g["pct"] / 100) ** 2 for g in geo)

    # Delinquency
    def dq_pct(pattern):
        if tb == 0:
            return 0.0
        return sum(s["balance"] for s in stat if re.search(pattern, s["name"], re.IGNORECASE)) / tb * 100

    co_bal = sum(s["balance"] for s in stat if re.search(r"charge|default|loss", s["name"], re.IGNORECASE))
    nl = get_numeric(df, mp, "net_loss").fillna(0).sum()
    rec = get_numeric(df, mp, "recoveries").fillna(0).sum()

    return {
        "N": len(df), "tb": tb, "tob": tob,
        "avg": tb / len(bals) if len(bals) > 0 else 0,
        "wa_rate": wa("interest_rate"),
        "wa_fico_orig": wa("fico_origination"),
        "wa_fico_curr": wa("fico_current"),
        "wa_dti": wa("dti"),
        "wa_mob": float(mobs.mean()) if len(mobs) > 0 else 0,
        "min_rate": float(rates.min()) if len(rates) > 0 else 0,
        "max_rate": float(rates.max()) if len(rates) > 0 else 0,
        "fico_dist": fico_dist, "rate_dist": rate_dist,
        "dti_dist": dti_dist, "term_dist": term_dist,
        "geo": geo, "stat": stat, "purp": purp, "grad": grad,
        "chan": chan, "veri": veri, "hous": hous, "vint": vint,
        "hhi": hhi, "top_state_conc": geo[0]["pct"] if geo else 0,
        "gross_loss_rate": (co_bal / tob * 100) if tob > 0 else 0,
        "net_loss_rate": (nl / tob * 100) if tob > 0 else 0,
        "net_loss": nl, "recoveries": rec,
        "pool_factor": (tb / tob * 100) if tob > 0 else 0,
        "dq30": dq_pct(r"30"),
        "dq60": dq_pct(r"60"),
        "dq90": dq_pct(r"90|120|default|charge|bankrupt"),
    }


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

VALIDATION_RULES = {
    "current_balance":  {"min": 0, "max": 1e8},
    "original_balance": {"min": 0, "max": 1e8},
    "interest_rate":    {"min": 0, "max": 35},
    "fico_origination": {"min": 300, "max": 900},
    "fico_current":     {"min": 300, "max": 900},
    "dti":              {"min": 0, "max": 200},
}

def validate(df: pd.DataFrame, mp: dict) -> dict:
    """Data quality validation. Returns completeness, missing count, OOR count, issues."""
    tc = mc = oor = 0
    issues = []

    for field, col in mp.items():
        if col not in df.columns:
            continue
        tc += len(df)
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        mc += int(missing.sum())

        if field in VALIDATION_RULES:
            rule = VALIDATION_RULES[field]
            vals = df[col].apply(parse_numeric)
            out_of_range = vals.notna() & ((vals < rule["min"]) | (vals > rule["max"]))
            oor += int(out_of_range.sum())
            for idx in out_of_range[out_of_range].index[:20 - len(issues)]:
                if len(issues) < 20:
                    issues.append(f"{field}={df.at[idx, col]} row {idx + 1}")

    comp = ((tc - mc) / tc * 100) if tc > 0 else 0
    return {"completeness": comp, "missing_count": mc, "oor_count": oor, "issues": issues}


# ═══════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════

def calc_regression(x: np.ndarray, y: np.ndarray) -> Optional[dict]:
    """Simple OLS regression. Returns slope, intercept, R², r, line endpoints."""
    if len(x) < 3:
        return None

    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()

    denom = n * sxx - sx * sx
    if denom == 0:
        return None

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    y_mean = sy / n
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = ((y - (slope * x + intercept)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r = math.sqrt(abs(r2)) * (1 if slope >= 0 else -1)

    return {
        "slope": float(slope), "intercept": float(intercept),
        "r2": float(r2), "r": float(r), "n": n,
        "x_min": float(x.min()), "x_max": float(x.max()),
    }


def calc_multi_regression(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> Optional[dict]:
    """Multiple OLS regression with 2 predictors. Returns b0, b1, b2, R², adj R²."""
    if len(x1) < 5:
        return None

    n = len(x1)
    X = np.column_stack([np.ones(n), x1, x2])

    try:
        # Normal equation: b = (X'X)^-1 X'y
        XtX = X.T @ X
        Xty = X.T @ y
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return None

    y_pred = X @ coef
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 3) if n > 3 else r2

    return {
        "b0": float(coef[0]), "b1": float(coef[1]), "b2": float(coef[2]),
        "r2": float(r2), "adj_r2": float(adj_r2), "n": n,
    }


# ═══════════════════════════════════════════════════════════════
# AI AUTO-MATCH (Anthropic Claude API)
# ═══════════════════════════════════════════════════════════════

def ai_match(df: pd.DataFrame, fields: Optional[dict] = None, api_key: Optional[str] = None) -> Optional[dict]:
    """
    Use Claude to match CSV columns to standard ABS fields.
    Returns dict: {field_key: column_name} or None on failure.
    Requires ANTHROPIC_API_KEY env var or api_key parameter.
    """
    import os
    import json as _json

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None

    flds = fields or STD_FIELDS
    hdrs = list(df.columns)

    # Build sample rows for context
    sample = df.head(5).to_dict(orient="records")
    sample_str = ""
    for i, row in enumerate(sample):
        sample_str += f"Row {i+1}: {row}\n"

    field_list = "\n".join(f"  {k}: {v['label']}" for k, v in flds.items())

    prompt = f"""You are an ABS loan tape analyst. Match CSV columns to standard fields.

CSV columns: {hdrs}

Sample data (first 5 rows):
{sample_str}

Standard fields to match:
{field_list}

For each standard field, determine which CSV column (if any) maps to it.
Only include fields where you are confident about the match.
Do NOT map a CSV column to more than one field.

Respond ONLY with a JSON object mapping field_key to column_name.
Example: {{"loan_id": "Loan_ID", "current_balance": "Balance", "interest_rate": "Rate"}}
No explanation, no markdown, just the JSON object."""

    try:
        import httpx
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"].strip()

        # Strip markdown fences if present
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        mapping = _json.loads(text)

        # Validate: only keep valid field_key -> column pairs
        clean = {}
        used_cols = set()
        for fk, col in mapping.items():
            if fk in flds and col in hdrs and col not in used_cols:
                clean[fk] = col
                used_cols.add(col)

        return clean if clean else None

    except Exception:
        return None
