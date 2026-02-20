"""Database models — users, tapes, templates, custom fields, analyses."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Float, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db import Base


def gen_id():
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    api_key = Column(String(64), unique=True, nullable=False, default=lambda: uuid.uuid4().hex)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    tapes = relationship("Tape", back_populates="user", cascade="all, delete-orphan")
    templates = relationship("Template", back_populates="user", cascade="all, delete-orphan")
    custom_fields = relationship("CustomField", back_populates="user", cascade="all, delete-orphan")


class Tape(Base):
    __tablename__ = "tapes"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    headers = Column(JSON, default=list)          # list of column names
    mapping = Column(JSON, default=dict)           # {field_key: col_name}
    analysis = Column(JSON, nullable=True)         # cached analysis results
    validation = Column(JSON, nullable=True)       # cached validation results
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="tapes")


class Template(Base):
    __tablename__ = "templates"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    originator = Column(String(200), nullable=False)
    mapping = Column(JSON, nullable=False, default=dict)  # {field_key: col_name}
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="templates")


class CustomField(Base):
    __tablename__ = "custom_fields"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    key = Column(String(100), nullable=False)
    label = Column(String(200), nullable=False)
    patterns = Column(JSON, default=list)  # list of regex pattern strings
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user = relationship("User", back_populates="custom_fields")
