# LTA Platform — Architecture

```
┌─────────────────────────────────────────────────┐
│                   FRONTEND                       │
│              Streamlit (app.py)                   │
│   Upload · Mapping · Strats · Charts · Admin     │
│                                                   │
│   Calls backend API via requests                  │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/REST
┌──────────────────▼──────────────────────────────┐
│                   BACKEND                        │
│              FastAPI (main.py)                    │
│                                                   │
│  /api/tapes      — upload, list, get, delete     │
│  /api/analyze    — run analysis on a tape        │
│  /api/match      — column matching (rule + AI)   │
│  /api/templates  — CRUD originator templates     │
│  /api/fields     — CRUD custom fields            │
│  /api/validate   — data quality checks           │
│  /api/regression — OLS simple + multiple         │
│  /api/export     — CSV export of filtered data   │
│  /api/users      — auth (API keys)               │
│                                                   │
│  logic.py  — pure functions (no DB dependency)   │
│  models.py — SQLAlchemy ORM models               │
│  schemas.py — Pydantic request/response models   │
│  db.py     — database connection                 │
└──────────────────┬──────────────────────────────┘
                   │ SQL
┌──────────────────▼──────────────────────────────┐
│                 DATABASE                         │
│              PostgreSQL                           │
│                                                   │
│  users         — API keys, names                 │
│  tapes         — uploaded tape metadata          │
│  tape_data     — raw CSV stored as JSON/binary   │
│  mappings      — column mappings per tape        │
│  templates     — originator templates            │
│  custom_fields — user-defined fields             │
│  analyses      — cached analysis results         │
└─────────────────────────────────────────────────┘
```

## API-First Design
- Streamlit frontend is just ONE consumer of the API
- External tools (Python scripts, notebooks, CI pipelines) can call the same API
- Example: `curl -X POST /api/analyze -H "X-API-Key: ..." -F "file=@tape.csv"`
