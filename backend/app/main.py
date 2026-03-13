"""
Loan Tape Analyzer — FastAPI Backend

Run: uvicorn app.main:app --reload
"""
import io
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from contextlib import asynccontextmanager

from app.db import get_db, init_db
from app.models import User, Tape, Template, CustomField
from app.schemas import (
    UserCreate, UserOut, TapeOut, MappingUpdate,
    TemplateCreate, TemplateOut, CustomFieldCreate, CustomFieldOut,
    AnalysisOut, ValidationOut, RegressionRequest, RegressionOut,
)
from app.logic import (
    STD_FIELDS, FIELD_TIERS, CANONICAL_FIELDS, EXTENDED_FIELDS, OPTIONAL_FIELDS, LONGITUDINAL_FIELDS,
    rule_match, score_template, analyze, validate,
    detect_tape_type, process_longitudinal,
    parse_numeric, calc_regression, calc_multi_regression,
    ai_rank_candidates,
)


# ── App lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title="Loan Tape Analyzer API",
    version="1.0.0",
    description="ABS loan tape cracking — upload, map, analyze, stratify.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ──
async def get_current_user(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> User:
    result = await db.execute(select(User).where(User.api_key == x_api_key))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(401, "Invalid API key")
    return user


# ── Tape data storage (disk + in-memory cache) ──
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_tape_cache: dict[str, pd.DataFrame] = {}


def _save_tape_df(tape_id: str, df: pd.DataFrame):
    """Save tape data to disk and cache."""
    path = os.path.join(UPLOAD_DIR, f"{tape_id}.csv")
    df.to_csv(path, index=False)
    _tape_cache[tape_id] = df


def _get_tape_df(tape_id: str) -> pd.DataFrame:
    """Load tape data from cache or disk."""
    if tape_id in _tape_cache:
        return _tape_cache[tape_id]
    # Try loading from disk
    path = os.path.join(UPLOAD_DIR, f"{tape_id}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        _tape_cache[tape_id] = df
        return df
    raise HTTPException(404, "Tape data not found. Re-upload the file.")


def _delete_tape_df(tape_id: str):
    """Remove tape data from cache and disk."""
    _tape_cache.pop(tape_id, None)
    path = os.path.join(UPLOAD_DIR, f"{tape_id}.csv")
    if os.path.exists(path):
        os.remove(path)


# ═══════════════════════════════════════════════════════════════
# USERS
# ═══════════════════════════════════════════════════════════════

@app.post("/api/users", response_model=UserOut, tags=["Users"])
async def create_user(body: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user. Returns API key for authentication."""
    user = User(name=body.name, email=body.email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ═══════════════════════════════════════════════════════════════
# TAPES
# ═══════════════════════════════════════════════════════════════

@app.post("/api/tapes", response_model=TapeOut, tags=["Tapes"])
def _looks_like_headers(vals: list) -> bool:
    """
    Heuristic check: a header row should have mostly short string labels,
    not dates, long sentences, or purely numeric values.
    """
    import re
    date_pat = re.compile(r'^\d{1,4}[\/\-\.]\d{1,2}([\/\-\.]\d{1,4})?$')
    score = 0
    for v in vals:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", ""):
            continue
        # Looks bad (data-like)
        if date_pat.match(s):
            score -= 2
        elif s.replace(".", "", 1).replace("-", "", 1).isdigit():
            score -= 1
        elif len(s) > 40:
            score -= 2
        elif any(word in s.lower() for word in ["as of", "report", "total", "summary", "sheet"]):
            score -= 3
        # Looks good (header-like)
        elif re.match(r'^[a-zA-Z][a-zA-Z0-9_\s]{1,30}$', s):
            score += 1
    return score > 0


def _read_tape(content: bytes, filename: str = "") -> pd.DataFrame:
    """
    Read a CSV or Excel file, auto-detecting the header row (rows 0-3).
    Uses heuristic first, then AI fallback for ambiguous cases.
    """
    import os
    is_excel = filename.lower().endswith((".xlsx", ".xls"))
    openai_key = os.getenv("OPENAI_API_KEY")

    candidates = []
    for header_row in range(4):
        try:
            if is_excel:
                df = pd.read_excel(io.BytesIO(content), header=header_row)
            else:
                df = pd.read_csv(io.BytesIO(content), header=header_row)
            candidates.append((header_row, df))
        except Exception:
            continue

    if not candidates:
        # Last resort fallback
        if is_excel:
            return pd.read_excel(io.BytesIO(content))
        return pd.read_csv(io.BytesIO(content))

    # Try heuristic first
    for header_row, df in candidates:
        row_vals = list(df.columns)
        if _looks_like_headers(row_vals):
            print(f"Header row detected at row {header_row} (heuristic)")
            return df

    # Fallback to row 0
    print("Header detection inconclusive — defaulting to row 0")
    return candidates[0][1]


async def upload_tape(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a CSV loan tape. Auto-matches columns, detects type, processes longitudinal."""
    content = await file.read()
    df = _read_tape(content, file.filename)
    hdrs = list(df.columns)

    # Try template auto-detect
    result = await db.execute(
        select(Template).where(Template.user_id == user.id)
    )
    templates = result.scalars().all()

    best_tpl, best_score = None, 0
    for tpl in templates:
        sc = score_template(tpl.mapping, hdrs)
        if sc > best_score:
            best_score = sc
            best_tpl = tpl

    if best_tpl and best_score >= 0.8:
        mapping = {k: v for k, v in best_tpl.mapping.items() if v in hdrs}
    else:
        cf_result = await db.execute(
            select(CustomField).where(CustomField.user_id == user.id)
        )
        custom = cf_result.scalars().all()
        fields = dict(STD_FIELDS)
        for cf in custom:
            fields[cf.key] = {"label": cf.label, "patterns": cf.patterns}
        mapping = rule_match(df, fields)

    # Detect tape type
    tape_info = detect_tape_type(df, mapping)
    analysis_df = df

    # Process longitudinal tapes
    processing_log = None
    if tape_info["type"] == "longitudinal":
        latest, full_ts, log = process_longitudinal(df, mapping)
        analysis_df = latest       # use latest snapshot for pool-level metrics
        processing_log = log
        # Save full time-series (all rows + derived columns) as the tape data
        # so exports and previews return every row, not just latest per loan
        tape_df_to_save = full_ts
    else:
        tape_df_to_save = df

    # Run analysis on latest snapshot (for longitudinal) or full df (for static)
    an = analyze(analysis_df, mapping)
    vl = validate(analysis_df, mapping)

    # Store tape type info in analysis
    an["tape_type"] = tape_info
    if processing_log:
        an["processing_log"] = processing_log

    tape = Tape(
        user_id=user.id, filename=file.filename,
        row_count=len(tape_df_to_save), col_count=len(hdrs),
        headers=list(tape_df_to_save.columns),  # includes derived columns
        mapping=mapping,
        analysis=an, validation=vl,
    )
    db.add(tape)
    await db.commit()
    await db.refresh(tape)

    _save_tape_df(tape.id, tape_df_to_save)  # save full rows including derived cols
    return tape


@app.get("/api/tapes", response_model=list[TapeOut], tags=["Tapes"])
async def list_tapes(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List all tapes for the current user."""
    result = await db.execute(
        select(Tape).where(Tape.user_id == user.id).order_by(Tape.created_at.desc())
    )
    return result.scalars().all()


@app.get("/api/tapes/{tape_id}", response_model=TapeOut, tags=["Tapes"])
async def get_tape(
    tape_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get tape metadata and mapping."""
    result = await db.execute(
        select(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    tape = result.scalar_one_or_none()
    if not tape:
        raise HTTPException(404, "Tape not found")
    return tape


@app.delete("/api/tapes/{tape_id}", tags=["Tapes"])
async def delete_tape(
    tape_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a tape."""
    await db.execute(
        delete(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    await db.commit()
    _delete_tape_df(tape_id)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════
# MAPPING
# ═══════════════════════════════════════════════════════════════

@app.put("/api/tapes/{tape_id}/mapping", response_model=TapeOut, tags=["Mapping"])
async def update_mapping(
    tape_id: str, body: MappingUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update column mapping for a tape. Re-runs analysis."""
    result = await db.execute(
        select(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    tape = result.scalar_one_or_none()
    if not tape:
        raise HTTPException(404, "Tape not found")

    df = _get_tape_df(tape_id)
    tape.mapping = body.mapping
    tape.analysis = analyze(df, body.mapping)
    tape.validation = validate(df, body.mapping)
    await db.commit()
    await db.refresh(tape)
    return tape


@app.post("/api/tapes/{tape_id}/automatch", response_model=TapeOut, tags=["Mapping"])
async def auto_match(
    tape_id: str,
    mode: str = Query("rule", description="Matching mode: 'rule' or 'ai'"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Re-run auto-matching on a tape. mode='rule' for regex, mode='ai' for Claude AI."""
    result = await db.execute(
        select(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    tape = result.scalar_one_or_none()
    if not tape:
        raise HTTPException(404, "Tape not found")

    df = _get_tape_df(tape_id)

    cf_result = await db.execute(
        select(CustomField).where(CustomField.user_id == user.id)
    )
    custom = cf_result.scalars().all()
    fields = dict(STD_FIELDS)
    for cf in custom:
        fields[cf.key] = {"label": cf.label, "patterns": cf.patterns}

    if mode == "ai":
        from app.logic import ai_match as _ai_match
        ai_mapping = _ai_match(df, fields)
        if ai_mapping:
            # Merge: AI results take priority, fill gaps with rule-based
            rule_mapping = rule_match(df, fields)
            merged = dict(rule_mapping)
            merged.update(ai_mapping)
            tape.mapping = merged
        else:
            raise HTTPException(400, "AI matching failed. Set OPENAI_API_KEY or ANTHROPIC_API_KEY env var.")
    else:
        tape.mapping = rule_match(df, fields)

    tape.analysis = analyze(df, tape.mapping)
    tape.validation = validate(df, tape.mapping)
    await db.commit()
    await db.refresh(tape)
    return tape


# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/tapes/{tape_id}/analysis", tags=["Analysis"])
async def get_analysis(
    tape_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get cached analysis results for a tape."""
    result = await db.execute(
        select(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    tape = result.scalar_one_or_none()
    if not tape:
        raise HTTPException(404, "Tape not found")
    if not tape.analysis:
        raise HTTPException(400, "No analysis available. Update mapping first.")
    return {"tape_id": tape_id, **tape.analysis}


@app.get("/api/tapes/{tape_id}/validation", tags=["Analysis"])
async def get_validation(
    tape_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get data quality validation results."""
    result = await db.execute(
        select(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    tape = result.scalar_one_or_none()
    if not tape:
        raise HTTPException(404, "Tape not found")
    return {"tape_id": tape_id, **(tape.validation or {})}


# ═══════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════

@app.post("/api/tapes/{tape_id}/regression", response_model=RegressionOut, tags=["Analysis"])
async def run_regression(
    tape_id: str, body: RegressionRequest,
    user: User = Depends(get_current_user),
):
    """Run OLS regression on two or three columns."""
    df = _get_tape_df(tape_id)

    x = df[body.x_column].apply(parse_numeric).dropna()
    y = df[body.y_column].apply(parse_numeric).dropna()
    mask = x.index.intersection(y.index)
    x, y = x[mask].values, y[mask].values

    if body.z_column:
        z = df[body.z_column].apply(parse_numeric).dropna()
        mask3 = np.intersect1d(mask, z.index)
        reg = calc_multi_regression(
            x[np.isin(mask, mask3)], z[mask3].values, y[np.isin(mask, mask3)]
        )
        if not reg:
            raise HTTPException(400, "Regression failed — need ≥5 valid data points")
        return RegressionOut(n=reg["n"], b0=reg["b0"], b1=reg["b1"], b2=reg["b2"],
                             r2=reg["r2"], adj_r2=reg["adj_r2"])
    else:
        reg = calc_regression(x, y)
        if not reg:
            raise HTTPException(400, "Regression failed — need ≥3 valid data points")
        return RegressionOut(n=reg["n"], slope=reg["slope"], intercept=reg["intercept"],
                             r2=reg["r2"], r=reg["r"])


# ═══════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════

@app.get("/api/tapes/{tape_id}/export", tags=["Export"])
async def export_csv(
    tape_id: str,
    filter_col: str = Query(None),
    filter_val: str = Query(None),
    user: User = Depends(get_current_user),
):
    """Export tape data as CSV. Optionally filter by column value."""
    df = _get_tape_df(tape_id)
    if filter_col and filter_val:
        df = df[df[filter_col].astype(str) == filter_val]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=export_{tape_id}.csv"},
    )


# ═══════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════

@app.post("/api/templates", response_model=TemplateOut, tags=["Templates"])
async def create_template(
    body: TemplateCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Save an originator template."""
    tpl = Template(user_id=user.id, name=body.name,
                   originator=body.originator, mapping=body.mapping)
    db.add(tpl)
    await db.commit()
    await db.refresh(tpl)
    return tpl


@app.get("/api/templates", response_model=list[TemplateOut], tags=["Templates"])
async def list_templates(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Template).where(Template.user_id == user.id).order_by(Template.created_at.desc())
    )
    return result.scalars().all()


@app.delete("/api/templates/{template_id}", tags=["Templates"])
async def delete_template(
    template_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await db.execute(
        delete(Template).where(Template.id == template_id, Template.user_id == user.id)
    )
    await db.commit()
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════
# CUSTOM FIELDS
# ═══════════════════════════════════════════════════════════════

@app.post("/api/fields", response_model=CustomFieldOut, tags=["Custom Fields"])
async def create_field(
    body: CustomFieldCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    cf = CustomField(user_id=user.id, key=body.key, label=body.label, patterns=body.patterns)
    db.add(cf)
    await db.commit()
    await db.refresh(cf)
    return cf


@app.get("/api/fields", response_model=list[CustomFieldOut], tags=["Custom Fields"])
async def list_fields(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(CustomField).where(CustomField.user_id == user.id)
    )
    return result.scalars().all()


@app.delete("/api/fields/{field_id}", tags=["Custom Fields"])
async def delete_field(
    field_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await db.execute(
        delete(CustomField).where(CustomField.id == field_id, CustomField.user_id == user.id)
    )
    await db.commit()
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/fields/standard", tags=["Custom Fields"])
async def list_standard_fields():
    """List all built-in standard fields with tier classification."""
    return {k: {"label": v["label"], "pattern_count": len(v["patterns"]),
                "tier": FIELD_TIERS.get(k, "optional")}
            for k, v in STD_FIELDS.items()}


@app.get("/api/tapes/{tape_id}/suggest/{field_key}", tags=["Mapping"])
async def suggest_field_mapping(
    tape_id: str,
    field_key: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Return AI-ranked top candidate columns for a specific standard field."""
    result = await db.execute(
        select(Tape).where(Tape.id == tape_id, Tape.user_id == user.id)
    )
    tape = result.scalar_one_or_none()
    if not tape:
        raise HTTPException(404, "Tape not found")

    df = _get_tape_df(tape_id)
    hdrs = list(df.columns)

    field_def = STD_FIELDS.get(field_key)
    if not field_def:
        raise HTTPException(404, f"Unknown field: {field_key}")

    import os
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return {"suggestions": [], "error": "No AI key configured"}

    # Exclude already-mapped columns as candidates
    current_mapping = tape.mapping or {}
    mapped_cols = set(current_mapping.values())
    candidates = [h for h in hdrs if h not in mapped_cols][:50]

    ranked = ai_rank_candidates(field_key, field_def, candidates, df, openai_key)

    return {
        "field_key": field_key,
        "suggestions": [{"col": col, "score": score} for score, col in ranked if score > 0]
    }
