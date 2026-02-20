"""
Loan Tape Analyzer — FastAPI Backend

Run: uvicorn app.main:app --reload
"""
import io
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
    STD_FIELDS, rule_match, score_template, analyze, validate,
    parse_numeric, calc_regression, calc_multi_regression,
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


# ── Tape data cache (in-memory per process — production should use Redis) ──
_tape_cache: dict[str, pd.DataFrame] = {}


def _get_tape_df(tape_id: str) -> pd.DataFrame:
    if tape_id not in _tape_cache:
        raise HTTPException(404, "Tape data not in memory. Re-upload the file.")
    return _tape_cache[tape_id]


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
async def upload_tape(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a CSV loan tape. Auto-matches columns and caches data."""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
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
        # Get user's custom fields
        cf_result = await db.execute(
            select(CustomField).where(CustomField.user_id == user.id)
        )
        custom = cf_result.scalars().all()
        fields = dict(STD_FIELDS)
        for cf in custom:
            fields[cf.key] = {"label": cf.label, "patterns": cf.patterns}
        mapping = rule_match(df, fields)

    # Run analysis
    an = analyze(df, mapping)
    vl = validate(df, mapping)

    tape = Tape(
        user_id=user.id, filename=file.filename,
        row_count=len(df), col_count=len(hdrs),
        headers=hdrs, mapping=mapping,
        analysis=an, validation=vl,
    )
    db.add(tape)
    await db.commit()
    await db.refresh(tape)

    _tape_cache[tape.id] = df
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
    _tape_cache.pop(tape_id, None)
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
            raise HTTPException(400, "AI matching failed. Check ANTHROPIC_API_KEY env var.")
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
    """List all 47 built-in standard fields."""
    return {k: {"label": v["label"], "pattern_count": len(v["patterns"])}
            for k, v in STD_FIELDS.items()}
