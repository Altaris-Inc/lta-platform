# 📊 Loan Tape Analyzer (LTA)

ABS loan tape cracking platform. Upload a CSV loan tape, auto-map columns to standard fields, and get instant analytics — stratifications, concentration risk, data quality, regression, and drill-down to individual loans.

## Features

- **Auto Column Mapping** — rule-based + AI-powered (Claude) field matching
- **47 ABS Standard Fields** — FICO, DTI, LTV, rates, terms, balances, etc.
- **Pool Overview** — key metrics, WAC, WAM, WALA, delinquency rates
- **Tape Cracking** — status distribution, vintage analysis, seasoning
- **Stratifications** — FICO, rate, DTI, term, grade, geography + custom strats
- **Drill-Down** — click any bucket to see individual loans, CSV export
- **Charts & Regression** — scatter plots, OLS trendline, equation display, multi-chart
- **Data Quality** — completeness, out-of-range, grade scoring
- **Concentration Risk** — HHI, top exposures, geographic concentration
- **Templates** — save/load column mappings per originator
- **Multi-User** — API key auth, isolated data per user
- **REST API** — 19 endpoints for programmatic access

## Quick Start (Local)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app_api.py
```

Opens at http://localhost:8501

## Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env — set DB_PASSWORD
docker compose up -d --build
```

Opens at http://localhost:8501

## Deploy to Production

See **[DEPLOY.md](DEPLOY.md)** for full instructions covering DigitalOcean, Railway, Render, Fly.io.

## Architecture

```
Streamlit (8501) ──▶ FastAPI (8000) ──▶ PostgreSQL (5432)
```

## Tech Stack

Frontend: Streamlit, Plotly, Pandas | Backend: FastAPI, SQLAlchemy | DB: PostgreSQL/SQLite | AI: Claude API (optional)
