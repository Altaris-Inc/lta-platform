"""Pydantic schemas — request/response models for the API."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Users ──
class UserCreate(BaseModel):
    name: str
    email: str

class UserOut(BaseModel):
    id: str
    name: str
    email: str
    api_key: str
    created_at: datetime
    model_config = {"from_attributes": True}


# ── Tapes ──
class TapeOut(BaseModel):
    id: str
    filename: str
    row_count: int
    col_count: int
    headers: list[str]
    mapping: dict
    created_at: datetime
    model_config = {"from_attributes": True}

class MappingUpdate(BaseModel):
    mapping: dict[str, str]


# ── Templates ──
class TemplateCreate(BaseModel):
    name: str
    originator: str
    mapping: dict[str, str]

class TemplateOut(BaseModel):
    id: str
    name: str
    originator: str
    mapping: dict
    created_at: datetime
    model_config = {"from_attributes": True}


# ── Custom Fields ──
class CustomFieldCreate(BaseModel):
    key: str
    label: str
    patterns: list[str] = []

class CustomFieldOut(BaseModel):
    id: str
    key: str
    label: str
    patterns: list[str]
    created_at: datetime
    model_config = {"from_attributes": True}


# ── Analysis ──
class AnalysisOut(BaseModel):
    tape_id: str
    N: int
    tb: float
    tob: float
    avg: float
    wa_rate: float
    wa_fico_orig: float
    wa_fico_curr: float
    wa_dti: float
    wa_mob: float
    pool_factor: float
    dq30: float
    dq60: float
    dq90: float
    gross_loss_rate: float
    net_loss_rate: float
    net_loss: float
    recoveries: float
    hhi: float
    top_state_conc: float
    fico_dist: list[dict]
    rate_dist: list[dict]
    dti_dist: list[dict]
    term_dist: list[dict]
    geo: list[dict]
    stat: list[dict]
    purp: list[dict]
    grad: list[dict]


# ── Validation ──
class ValidationOut(BaseModel):
    tape_id: str
    completeness: float
    missing_count: int
    oor_count: int
    issues: list[str]


# ── Regression ──
class RegressionRequest(BaseModel):
    x_column: str
    y_column: str
    z_column: Optional[str] = None

class RegressionOut(BaseModel):
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r2: Optional[float] = None
    r: Optional[float] = None
    n: int = 0
    # Multi-reg fields
    b0: Optional[float] = None
    b1: Optional[float] = None
    b2: Optional[float] = None
    adj_r2: Optional[float] = None
