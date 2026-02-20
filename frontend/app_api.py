"""
Loan Tape Analyzer — Streamlit Frontend (API Mode)
Sidebar navigation, polished UI matching the React version.

Run backend first:  cd ../backend && uvicorn app.main:app --reload
Then:               streamlit run app_api.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from api_client import LTAClient

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Loan Tape Analyzer", page_icon="lta-logo.svg", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0B0E11; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #12161C; border-right: 1px solid #1E2530; }
    section[data-testid="stSidebar"] .stRadio label { color: #8494A7 !important; font-size: 13px; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] { color: #00D4AA !important; font-weight: 600; }
    /* Cards */
    .metric-card { background: #171C24; border-radius: 10px; padding: 14px 18px; border: 1px solid #1E2530; margin-bottom: 8px; }
    .metric-card:hover { border-color: #00D4AA; }
    .metric-label { color: #8494A7; font-size: 11px; font-weight: 500; }
    .metric-value { color: #E8ECF1; font-size: 22px; font-weight: 700; margin: 4px 0; }
    .metric-sub { color: #566375; font-size: 10px; }
    .section-header { background: linear-gradient(135deg, #00D4AA, #4D9EFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 18px; font-weight: 700; margin: 8px 0 12px 0; }
    .accent { color: #00D4AA; }
    .info-bar { background: #171C24; border-radius: 8px; padding: 8px 16px; border: 1px solid #1E2530; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

DEFAULTS = {"api_key": None, "user": None, "tape_id": None, "tape": None,
            "df": None, "filename": None, "analysis": None, "validation": None, "page": "Pool Overview"}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Drill-down state
for dk in ["_drill_show", "_drill_title", "_drill_bucket", "_drill_mp"]:
    if dk not in st.session_state:
        st.session_state[dk] = None if dk != "_drill_show" else False

def get_client(): return LTAClient(api_key=st.session_state.api_key)


# ═══════════════════════════════════════════════════════════════
# DRILL-DOWN DIALOG
# ═══════════════════════════════════════════════════════════════

@st.dialog("Drill Down", width="large")
def drill_down_dialog():
    """Modal popup showing filtered loans for a bucket — matches React version."""
    from logic import parse_numeric

    title = st.session_state.get("_drill_title", "Drill Down")
    bucket = st.session_state.get("_drill_bucket", {})
    mp = st.session_state.get("_drill_mp", {})
    df_raw = st.session_state.get("df")

    if df_raw is None or not bucket:
        st.warning("No data available.")
        return

    # Filter
    filtered, src_col = _filter_bucket(df_raw, mp, bucket)
    if filtered is None or len(filtered) == 0:
        st.info("No loans match this bucket.")
        return

    # Smart column ordering
    ordered = _smart_col_order(filtered, mp, src_col)
    filtered = filtered[ordered]

    # Header
    bal_col = mp.get("current_balance", "")
    total_bal = 0
    if bal_col and bal_col in filtered.columns:
        total_bal = pd.to_numeric(filtered[bal_col].astype(str).str.replace(r'[$,%\s]', '', regex=True), errors='coerce').sum()

    st.markdown(f"""
    <div style="margin-bottom:12px">
        <div style="font-size:18px;font-weight:700;color:#E8ECF1">{title}</div>
        <div style="font-size:12px;color:#8494A7">{len(filtered):,} loans · {fmt_c(total_bal)} balance</div>
    </div>""", unsafe_allow_html=True)

    # Column selector pills
    col_sel = st.multiselect(
        f"📋 Columns ({len(ordered)})",
        options=ordered,
        default=ordered[:min(15, len(ordered))],
        key="_drill_cols"
    )
    if not col_sel:
        col_sel = ordered[:15]

    # Search filter
    sr1, sr2 = st.columns([3, 1])
    with sr1:
        search = st.text_input("🔍 Filter", placeholder="Search loans...", key="_drill_search", label_visibility="collapsed")
    with sr2:
        csv_data = filtered[col_sel].to_csv(index=False)
        st.download_button("📥 CSV", csv_data, f"drill_{title.replace(' ','_')}.csv", "text/csv", key="_drill_dl")

    display = filtered[col_sel]
    if search:
        mask = display.apply(lambda r: r.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        display = display[mask]

    # Paginated table
    PAGE_SIZE = 30
    total_pages = max(1, (len(display) + PAGE_SIZE - 1) // PAGE_SIZE)
    if "_drill_page" not in st.session_state:
        st.session_state["_drill_page"] = 0
    pg = st.session_state["_drill_page"]
    pg = min(pg, total_pages - 1)

    start = pg * PAGE_SIZE
    page_df = display.iloc[start:start + PAGE_SIZE]

    st.dataframe(page_df, use_container_width=True, height=min(500, len(page_df) * 35 + 45), hide_index=True)

    # Pagination controls
    if total_pages > 1:
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("◀ Prev", disabled=(pg == 0), key="_drill_prev"):
                st.session_state["_drill_page"] = pg - 1
        with p2:
            st.markdown(f'<div style="text-align:center;color:#8494A7;font-size:11px;padding-top:8px">Page {pg+1} / {total_pages}</div>', unsafe_allow_html=True)
        with p3:
            if st.button("Next ▶", disabled=(pg >= total_pages - 1), key="_drill_next"):
                st.session_state["_drill_page"] = pg + 1

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def fmt_c(v):
    if v is None: return "—"
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:,.0f}"

def fmt_p(v): return f"{(v or 0):.1f}%"
def fmt_r(v): return f"{(v or 0):.2f}%"
def fmt_s(v): return str(round(v or 0))

def card(label, value, sub=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

def chart(data, title, color="#00D4AA", key=None):
    """Render a non-clickable bar chart."""
    if not data or all(d.get("count",0)==0 for d in data): return
    df = pd.DataFrame(data)
    fig = px.bar(df, y="name", x="pct", orientation="h", color_discrete_sequence=[color], text="pct")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto')
    fig.update_layout(plot_bgcolor="#12161C", paper_bgcolor="#0B0E11", font_color="#8494A7",
        height=max(200, len(data)*32), title=dict(text=title, font=dict(size=13, color="#E8ECF1")),
        xaxis=dict(title="% Balance", showgrid=True, gridcolor="#1E2530"),
        yaxis=dict(title="", autorange="reversed"), margin=dict(l=10,r=10,t=40,b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=key)

def chart_clickable(data, title, color="#00D4AA", key=None):
    """Just render chart - no click handling."""
    if not data or all(d.get("count",0)==0 for d in data): return
    df = pd.DataFrame(data)
    fig = px.bar(df, y="name", x="pct", orientation="h", color_discrete_sequence=[color], text="pct")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto')
    fig.update_layout(plot_bgcolor="#12161C", paper_bgcolor="#0B0E11", font_color="#8494A7",
        height=max(200, len(data)*32), title=dict(text=title, font=dict(size=13, color="#E8ECF1")),
        xaxis=dict(title="% Balance", showgrid=True, gridcolor="#1E2530"),
        yaxis=dict(title="", autorange="reversed"), margin=dict(l=10,r=10,t=40,b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=key)

def table(data):
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df["balance"] = df["balance"].apply(fmt_c)
    df["pct"] = df["pct"].apply(lambda x: f"{x:.1f}%")
    df["avg_bal"] = df["avg_bal"].apply(fmt_c) if "avg_bal" in df.columns else "—"
    df["min_bal"] = df["min_bal"].apply(fmt_c) if "min_bal" in df.columns else "—"
    df["max_bal"] = df["max_bal"].apply(fmt_c) if "max_bal" in df.columns else "—"
    cols = ["name","count","balance","avg_bal","min_bal","max_bal","pct"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].rename(columns={
        "name":"Bucket","count":"Count","balance":"Total Bal",
        "avg_bal":"Avg Bal","min_bal":"Min Bal","max_bal":"Max Bal","pct":"% Bal"
    })


def _smart_col_order(df_raw, mp, relevant_field=""):
    """Order columns: ID, relevant field, key fields, then rest."""
    cols = list(df_raw.columns)
    priority = []
    id_col = mp.get("loan_id", "")
    if id_col and id_col in cols:
        priority.append(id_col)
    if relevant_field and relevant_field in cols and relevant_field not in priority:
        priority.append(relevant_field)
    for fk in ["current_balance", "original_balance", "interest_rate", "fico_origination",
                "fico_current", "dti", "loan_status", "grade", "state", "loan_purpose",
                "months_on_book", "origination_date"]:
        mc = mp.get(fk, "")
        if mc and mc in cols and mc not in priority:
            priority.append(mc)
    rest = [c for c in cols if c not in priority]
    return priority + rest


def _filter_bucket(df_raw, mp, bucket_info):
    """Filter raw df by a bucket. Returns filtered df or None."""
    from logic import parse_numeric
    name = bucket_info.get("name", "")
    field = bucket_info.get("field", "")
    src_col = mp.get(field, "")

    if "min" in bucket_info and "max" in bucket_info and src_col and src_col in df_raw.columns:
        vals = df_raw[src_col].apply(parse_numeric)
        mask = (vals >= bucket_info["min"]) & (vals <= bucket_info["max"]) & vals.notna()
        return df_raw[mask].copy(), src_col
    elif src_col and src_col in df_raw.columns:
        filtered = df_raw[df_raw[src_col].fillna("").astype(str).str.strip() == name].copy()
        return filtered, src_col
    else:
        return None, ""




def chart_with_drilldown(data, title, color, chart_key, df_raw, mp, grid_prefix):
    """Render chart + enriched summary table + auto-drill selectbox."""
    if not data or all(d.get("count", 0) == 0 for d in data):
        return
    chart_clickable(data, title, color, chart_key)

    # Enrich with avg/min/max balance from raw data
    enriched = []
    bal_col = mp.get("current_balance", "") if mp else ""
    for bucket in data:
        b = dict(bucket)
        if df_raw is not None and bal_col and bal_col in df_raw.columns:
            filtered, _ = _filter_bucket(df_raw, mp, bucket)
            if filtered is not None and len(filtered) > 0:
                from logic import parse_numeric
                bals = filtered[bal_col].apply(parse_numeric).dropna()
                if len(bals) > 0:
                    b["avg_bal"] = float(bals.mean())
                    b["min_bal"] = float(bals.min())
                    b["max_bal"] = float(bals.max())
        if "avg_bal" not in b:
            cnt = b.get("count", 0)
            b["avg_bal"] = b.get("balance", 0) / cnt if cnt > 0 else 0
            b["min_bal"] = 0
            b["max_bal"] = 0
        enriched.append(b)

    # Clean summary table
    st.dataframe(table(enriched), use_container_width=True, hide_index=True)

    # Inline drill selector
    if df_raw is not None:
        bucket_names = [b["name"] for b in data if b.get("count", 0) > 0]
        if bucket_names:
            sel_key = f"dsel_{grid_prefix}_{chart_key}"

            def _on_drill_select(bnames=bucket_names, t=title, d=data, m=mp, k=sel_key):
                val = st.session_state.get(k, "")
                if val:
                    bucket = next((b for b in d if b["name"] == val), None)
                    if bucket:
                        st.session_state["_pending_drill"] = {
                            "title": f"{t}: {bucket['name']}",
                            "bucket": bucket,
                            "mp": m,
                        }
                    st.session_state[k] = ""

            st.selectbox("🔍", [""] + bucket_names, key=sel_key,
                         label_visibility="collapsed",
                         format_func=lambda x: "🔍 Select bucket to drill…" if x == "" else f"🔍 {x}",
                         on_change=_on_drill_select)


# ═══════════════════════════════════════════════════════════════
# TRIGGER DRILL-DOWN DIALOG
# ═══════════════════════════════════════════════════════════════

# Pick up pending drill from chart click (set during previous rerun)
if st.session_state.get("_pending_drill"):
    pending = st.session_state.pop("_pending_drill")
    st.session_state["_drill_title"] = pending["title"]
    st.session_state["_drill_bucket"] = pending["bucket"]
    st.session_state["_drill_mp"] = pending["mp"]
    st.session_state["_drill_page"] = 0
    drill_down_dialog()

# ═══════════════════════════════════════════════════════════════
# LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════════

if not st.session_state.api_key:
    import base64 as _b64l
    _lpl = os.path.join(os.path.dirname(__file__), "lta-logo.svg")
    if os.path.exists(_lpl):
        with open(_lpl, "r", encoding="utf-8") as _fl:
            _lsl = _fl.read()
        _lbl = _b64l.b64encode(_lsl.encode()).decode()
        st.sidebar.markdown(f'<img src="data:image/svg+xml;base64,{_lbl}" style="width:50px;height:50px;border-radius:10px;margin-bottom:4px">', unsafe_allow_html=True)
    st.sidebar.markdown('<span style="color:#E8ECF1;font-size:14px;font-weight:700">Loan Tape Analyzer</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<span style="color:#566375;font-size:10px">ABS TAPE CRACKING PLATFORM</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Check backend
    try:
        LTAClient().health()
    except Exception:
        st.sidebar.error("Backend offline")
        st.error("❌ Cannot reach backend at http://127.0.0.1:8000\n\nStart it: `cd backend && uvicorn app.main:app --reload`")
        st.stop()

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        # Logo on login
        import base64 as _b64
        _lp = os.path.join(os.path.dirname(__file__), "lta-logo.svg")
        if os.path.exists(_lp):
            with open(_lp, "r", encoding="utf-8") as _f:
                _ls = _f.read()
            _lb = _b64.b64encode(_ls.encode()).decode()
            st.markdown(f'<div style="text-align:center"><img src="data:image/svg+xml;base64,{_lb}" style="width:100px;height:100px;border-radius:20px;margin-bottom:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center"><span class="section-header" style="font-size:28px">Loan Tape Analyzer</span></div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#566375;font-size:12px;letter-spacing:2px">ABS TAPE CRACKING PLATFORM</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑 Login", "📝 Register"])
        with tab_login:
            key_in = st.text_input("API Key", type="password", placeholder="Paste your API key")
            if st.button("Login", use_container_width=True) and key_in:
                try:
                    LTAClient(api_key=key_in).list_tapes()
                    st.session_state.api_key = key_in
                except Exception as e:
                    st.error("Invalid API key")
                    st.stop()
                st.rerun()
        with tab_reg:
            rn = st.text_input("Name", placeholder="Your name")
            re_ = st.text_input("Email", placeholder="you@example.com")
            if st.button("Register", use_container_width=True) and rn and re_:
                try:
                    u = LTAClient().create_user(rn, re_)
                    st.session_state["_new_api_key"] = u["api_key"]
                    st.rerun()
                except Exception as e: st.error(f"Failed: {e}")

            # Show key after registration
            if st.session_state.get("_new_api_key"):
                new_key = st.session_state["_new_api_key"]
                st.success("✅ Registration successful!")
                st.code(new_key, language=None)
                st.warning("⚠️ Copy and save this API key! It won't be shown again.")
                if st.button("✅ I've saved my key — Continue", use_container_width=True):
                    st.session_state.api_key = new_key
                    del st.session_state["_new_api_key"]
                    st.rerun()
    st.stop()

client = get_client()

# ═══════════════════════════════════════════════════════════════
# UPLOAD SCREEN (no tape loaded)
# ═══════════════════════════════════════════════════════════════

if not st.session_state.tape_id:
    import base64 as _b64u
    _lpu = os.path.join(os.path.dirname(__file__), "lta-logo.svg")
    if os.path.exists(_lpu):
        with open(_lpu, "r", encoding="utf-8") as _fu:
            _lsu = _fu.read()
        _lbu = _b64u.b64encode(_lsu.encode()).decode()
        st.sidebar.markdown(f'<img src="data:image/svg+xml;base64,{_lbu}" style="width:50px;height:50px;border-radius:10px;margin-bottom:4px">', unsafe_allow_html=True)
    st.sidebar.markdown('<span style="color:#E8ECF1;font-size:14px;font-weight:700">Loan Tape Analyzer</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**No tape loaded**")
    st.sidebar.markdown("Upload a CSV to begin.")
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k in DEFAULTS: st.session_state[k] = DEFAULTS[k]
        st.rerun()

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        _lpu2 = os.path.join(os.path.dirname(__file__), "lta-logo.svg")
        if os.path.exists(_lpu2):
            with open(_lpu2, "r", encoding="utf-8") as _fu2:
                _lsu2 = _fu2.read()
            _lbu2 = _b64u.b64encode(_lsu2.encode()).decode()
            st.markdown(f'<div style="text-align:center"><img src="data:image/svg+xml;base64,{_lbu2}" style="width:80px;height:80px;border-radius:16px;margin-bottom:8px"></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center"><span class="section-header" style="font-size:24px">Loan Tape Analyzer</span></div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#8494A7;font-size:13px">Upload a CSV loan tape to begin analysis</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv","tsv"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Uploading & analyzing..."):
                csv_bytes = uploaded.read()
                tape = client.upload_tape(uploaded.name, csv_bytes)
                st.session_state.tape_id = tape["id"]
                st.session_state.tape = tape
                st.session_state.filename = tape["filename"]
                uploaded.seek(0)
                st.session_state.df = pd.read_csv(uploaded)
                # Auto-run rule-based match
                try:
                    updated = client.auto_match(tape["id"], mode="rule")
                    st.session_state.tape = updated
                except: pass
                st.session_state.analysis = client.get_analysis(tape["id"])
                st.session_state.validation = client.get_validation(tape["id"])
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#566375;font-size:11px">Or load sample data:</p>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        for col_btn, name, label in [(sc1, "consumer_unsecured_loan_tape.csv", "🏦 Consumer Unsecured"),
                                      (sc2, "quickdrive_auto_loan_tape.csv", "🚗 QuickDrive Auto")]:
            with col_btn:
                if st.button(label, use_container_width=True):
                    path = os.path.join(os.path.dirname(__file__), "sample_data", name)
                    if os.path.exists(path):
                        with open(path, "rb") as f: csv_bytes = f.read()
                        with st.spinner("Uploading & analyzing..."):
                            tape = client.upload_tape(name, csv_bytes)
                            st.session_state.tape_id = tape["id"]
                            st.session_state.tape = tape
                            st.session_state.filename = tape["filename"]
                            st.session_state.df = pd.read_csv(path)
                            # Auto-run rule-based match
                            try:
                                updated = client.auto_match(tape["id"], mode="rule")
                                st.session_state.tape = updated
                            except: pass
                            st.session_state.analysis = client.get_analysis(tape["id"])
                            st.session_state.validation = client.get_validation(tape["id"])
                        st.rerun()

        # Previous tapes
        st.markdown("---")
        st.markdown("**Previous Tapes:**")
        try:
            tapes = client.list_tapes()
            for t in tapes:
                c1, c2, c3 = st.columns([3,2,1])
                with c1: st.markdown(f'<span style="color:#E8ECF1;font-weight:600">{t["filename"]}</span>', unsafe_allow_html=True)
                with c2: st.markdown(f'<span style="color:#8494A7;font-size:11px">{t["row_count"]:,} rows</span>', unsafe_allow_html=True)
                with c3:
                    if st.button("→", key=f"o_{t['id']}", use_container_width=True):
                        st.session_state.tape_id = t["id"]
                        st.session_state.tape = t
                        st.session_state.filename = t["filename"]
                        st.session_state.analysis = client.get_analysis(t["id"])
                        st.session_state.validation = client.get_validation(t["id"])
                        st.rerun()
            if not tapes:
                st.markdown('<span style="color:#566375;font-size:11px">No tapes yet.</span>', unsafe_allow_html=True)
        except: pass
    st.stop()


# ═══════════════════════════════════════════════════════════════
# MAIN APP — SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════

tape = st.session_state.tape
an = st.session_state.analysis
vl = st.session_state.validation
mp = tape.get("mapping", {}) if tape else {}
hdrs = tape.get("headers", []) if tape else []

# ── Sidebar with logo ──
import base64, os
_logo_path = os.path.join(os.path.dirname(__file__), "lta-logo.svg")
if os.path.exists(_logo_path):
    with open(_logo_path, "r", encoding="utf-8") as f:
        _logo_svg = f.read()
    _logo_b64 = base64.b64encode(_logo_svg.encode()).decode()
    st.sidebar.markdown(f'<img src="data:image/svg+xml;base64,{_logo_b64}" style="width:60px;height:60px;border-radius:12px;margin-bottom:4px">', unsafe_allow_html=True)
else:
    st.sidebar.markdown("### 📊 LTA")
st.sidebar.markdown(f'<span style="color:#E8ECF1;font-size:14px;font-weight:700">Loan Tape Analyzer</span>', unsafe_allow_html=True)
st.sidebar.markdown(f'<span style="color:#566375;font-size:10px">{st.session_state.filename or ""}</span>', unsafe_allow_html=True)

if an:
    st.sidebar.markdown(f"""
    <div class="info-bar">
        <span style="color:#8494A7;font-size:10px">LOANS</span><br>
        <span style="color:#E8ECF1;font-size:16px;font-weight:700">{an['N']:,}</span>
        <span style="color:#566375;font-size:10px;margin-left:12px">BAL</span>
        <span style="color:#00D4AA;font-size:12px;font-weight:600">{fmt_c(an['tb'])}</span>
    </div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")

PAGES = [
    "📋 Column Mapping",
    "📊 Pool Overview",
    "🔍 Tape Cracking",
    "📈 Stratifications",
    "🔧 Custom Strats",
    "📉 Charts & Regression",
    "✅ Data Quality",
    "⚡ Concentration",
    "🛡️ Risk Summary",
    "🗄️ Raw Data",
    "⚙️ Admin / Templates",
]

page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
sc1, sc2 = st.sidebar.columns(2)
with sc1:
    if st.button("🔄 New Tape", use_container_width=True):
        for k in ["tape_id","tape","df","filename","analysis","validation"]:
            st.session_state[k] = None
        st.rerun()
with sc2:
    if st.button("🚪 Logout", use_container_width=True):
        for k in DEFAULTS: st.session_state[k] = DEFAULTS[k]
        st.rerun()

st.sidebar.markdown(f'<span style="color:#566375;font-size:9px">Mapped: {len(mp)} fields</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: POOL OVERVIEW
# ═══════════════════════════════════════════════════════════════

if page == "📊 Pool Overview":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Pool Overview</div>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        with m1: card("Total Loans", f"{an['N']:,}")
        with m2: card("Total Balance", fmt_c(an["tb"]))
        with m3: card("Orig Balance", fmt_c(an["tob"]))
        with m4: card("Avg Balance", fmt_c(an["avg"]))
        with m5: card("Pool Factor", fmt_p(an["pool_factor"]))
        with m6: card("WA Seasoning", f"{an['wa_mob']:.0f} mo")

        st.markdown("<br>", unsafe_allow_html=True)
        m7,m8,m9,m10,m11,m12 = st.columns(6)
        with m7: card("WA Rate", fmt_r(an["wa_rate"]))
        with m8: card("Rate Range", f"{fmt_r(an.get('min_rate',0))} – {fmt_r(an.get('max_rate',0))}")
        with m9: card("WA FICO Orig", fmt_s(an["wa_fico_orig"]))
        with m10: card("WA FICO Curr", fmt_s(an["wa_fico_curr"]))
        with m11: card("WA DTI", fmt_p(an["wa_dti"]))
        with m12: card("HHI", f"{an['hhi']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: chart_with_drilldown(an.get("fico_dist",[]), "FICO Distribution", "#4D9EFF", "po_fico", st.session_state.df, mp, "po")
        with c2: chart_with_drilldown(an.get("rate_dist",[]), "Rate Distribution", "#00D4AA", "po_rate", st.session_state.df, mp, "po")
        c3, c4 = st.columns(2)
        with c3: chart_with_drilldown(an.get("stat",[])[:10], "Loan Status", "#FFB224", "po_stat", st.session_state.df, mp, "po")
        with c4: chart_with_drilldown(an.get("purp",[])[:10], "Loan Purpose", "#A78BFA", "po_purp", st.session_state.df, mp, "po")


# ═══════════════════════════════════════════════════════════════
# PAGE: TAPE CRACKING
# ═══════════════════════════════════════════════════════════════

elif page == "🔍 Tape Cracking":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Tape Cracking</div>', unsafe_allow_html=True)
        d1,d2,d3,d4,d5,d6 = st.columns(6)
        with d1: card("30+ DPD", fmt_p(an["dq30"]), "of balance")
        with d2: card("60+ DPD", fmt_p(an["dq60"]), "of balance")
        with d3: card("90+ DPD", fmt_p(an["dq90"]), "of balance")
        with d4: card("Gross Loss Rate", fmt_p(an["gross_loss_rate"]), "of orig bal")
        with d5: card("Net Loss", fmt_c(an["net_loss"]))
        with d6: card("Recoveries", fmt_c(an["recoveries"]))

        st.markdown("<br>", unsafe_allow_html=True)
        if an.get("stat"):
            st.markdown("**Loan Status Breakdown**")
            chart_with_drilldown(an["stat"], "Loan Status", "#FFB224", "tc_stat", st.session_state.df, mp, "tc")
        if an.get("vint"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Vintage Distribution**")
            chart_with_drilldown(an["vint"], "Vintage", "#A78BFA", "tc_vint", st.session_state.df, mp, "tc")


# ═══════════════════════════════════════════════════════════════
# PAGE: STRATIFICATIONS
# ═══════════════════════════════════════════════════════════════

elif page == "📈 Stratifications":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Stratifications</div>', unsafe_allow_html=True)

        strats = [("FICO Distribution", an.get("fico_dist",[]), "#4D9EFF"),
                  ("Rate Distribution", an.get("rate_dist",[]), "#00D4AA"),
                  ("DTI Distribution", an.get("dti_dist",[]), "#FFB224"),
                  ("Term Distribution", an.get("term_dist",[]), "#FF4D6A")]
        for i in range(0, len(strats), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i+j < len(strats):
                    name, data, color = strats[i+j]
                    with col:
                        chart_with_drilldown(data, name, color, f"st_{i+j}", st.session_state.df, mp, "st")

        st.markdown("---")
        cats = [("Purpose", an.get("purp",[]), "#A78BFA"), ("Grade", an.get("grad",[]), "#00D4AA"),
                ("Channel", an.get("chan",[]), "#4D9EFF"), ("Verification", an.get("veri",[]), "#FFB224"),
                ("Geography (Top 20)", an.get("geo",[]), "#4D9EFF")]
        for i in range(0, len(cats), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i+j < len(cats):
                    name, data, color = cats[i+j]
                    with col:
                        if data:
                            chart_with_drilldown(data, name, color, f"ct_{i+j}", st.session_state.df, mp, "ct")




# ═══════════════════════════════════════════════════════════════
# PAGE: CUSTOM STRATS
# ═══════════════════════════════════════════════════════════════

elif page == "🔧 Custom Strats":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Custom Stratifications</div>', unsafe_allow_html=True)
        st.markdown('<span style="color:#8494A7;font-size:12px">Build your own strats on any column — categorical or numeric with custom buckets</span>', unsafe_allow_html=True)

        if "_custom_strats" not in st.session_state:
            st.session_state["_custom_strats"] = []

        from logic import parse_numeric
        df_cs = st.session_state.df

        if df_cs is not None:
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Builder form ──
            st.markdown("**➕ New Custom Strat**")
            cs1, cs2 = st.columns(2)
            with cs1:
                cs_col = st.selectbox("Column", hdrs, key="_cs_col")
            with cs2:
                cs_type = st.radio("Type", ["Categorical", "Numeric Buckets"], horizontal=True, key="_cs_type")

            if cs_type == "Categorical":
                cs_top = st.slider("Top N values", 5, 50, 15, key="_cs_top")
                if st.button("▶ Build", key="_cs_build_cat", use_container_width=False):
                    bal_col = mp.get("current_balance", "")
                    groups = {}
                    for _, row in df_cs.iterrows():
                        k = str(row.get(cs_col, "") or "").strip() or "Unknown"
                        b = parse_numeric(row.get(bal_col, 0)) if bal_col else 0
                        if k not in groups:
                            groups[k] = {"count": 0, "balance": 0}
                        groups[k]["count"] += 1
                        groups[k]["balance"] += (b or 0)
                    tb = sum(g["balance"] for g in groups.values())
                    buckets = sorted(groups.items(), key=lambda x: x[1]["balance"], reverse=True)[:cs_top]
                    data = [{"name": k, "count": v["count"], "balance": v["balance"],
                             "pct": (v["balance"]/tb*100) if tb > 0 else 0,
                             "field": ""} for k, v in buckets]
                    st.session_state["_custom_strats"].append({
                        "title": f"{cs_col} (Categorical)",
                        "data": data, "col": cs_col, "type": "cat"
                    })
                    st.rerun()

            else:  # Numeric Buckets
                vals = df_cs[cs_col].apply(parse_numeric).dropna()
                if len(vals) > 0:
                    v_min, v_max = float(vals.min()), float(vals.max())
                    st.markdown(f'<span style="color:#8494A7;font-size:11px">Range: {v_min:,.2f} – {v_max:,.2f} · {len(vals):,} values</span>', unsafe_allow_html=True)

                    cs_method = st.radio("Method", ["Equal Width", "Percentile", "Custom Breaks"], horizontal=True, key="_cs_method")

                    if cs_method == "Equal Width":
                        n_buckets = st.slider("Buckets", 3, 20, 6, key="_cs_nbuckets")
                        if st.button("▶ Build", key="_cs_build_num"):
                            step = (v_max - v_min) / n_buckets
                            breaks = [v_min + step * i for i in range(n_buckets + 1)]
                            bal_col = mp.get("current_balance", "")
                            data = []
                            for bi in range(len(breaks) - 1):
                                lo, hi = breaks[bi], breaks[bi + 1]
                                label = f"{lo:,.0f}–{hi:,.0f}"
                                m = (vals >= lo) & (vals < hi if bi < len(breaks) - 2 else vals <= hi)
                                idxs = vals[m].index
                                bal = df_cs.loc[idxs, bal_col].apply(parse_numeric).sum() if bal_col and bal_col in df_cs.columns else 0
                                tb_all = df_cs[bal_col].apply(parse_numeric).sum() if bal_col and bal_col in df_cs.columns else 1
                                data.append({"name": label, "count": len(idxs), "balance": bal,
                                             "pct": (bal/tb_all*100) if tb_all > 0 else 0,
                                             "field": "", "min": lo, "max": hi})
                            st.session_state["_custom_strats"].append({
                                "title": f"{cs_col} ({n_buckets} buckets)",
                                "data": data, "col": cs_col, "type": "num"
                            })
                            st.rerun()

                    elif cs_method == "Percentile":
                        import numpy as np
                        n_pct = st.slider("Percentile groups", 3, 10, 5, key="_cs_npct")
                        if st.button("▶ Build", key="_cs_build_pct"):
                            pcts = np.linspace(0, 100, n_pct + 1)
                            breaks = [float(np.percentile(vals, p)) for p in pcts]
                            # Deduplicate
                            breaks = sorted(set(breaks))
                            bal_col = mp.get("current_balance", "")
                            data = []
                            for bi in range(len(breaks) - 1):
                                lo, hi = breaks[bi], breaks[bi + 1]
                                label = f"{lo:,.0f}–{hi:,.0f}"
                                m = (vals >= lo) & (vals < hi if bi < len(breaks) - 2 else vals <= hi)
                                idxs = vals[m].index
                                bal = df_cs.loc[idxs, bal_col].apply(parse_numeric).sum() if bal_col and bal_col in df_cs.columns else 0
                                tb_all = df_cs[bal_col].apply(parse_numeric).sum() if bal_col and bal_col in df_cs.columns else 1
                                data.append({"name": label, "count": len(idxs), "balance": bal,
                                             "pct": (bal/tb_all*100) if tb_all > 0 else 0,
                                             "field": "", "min": lo, "max": hi})
                            st.session_state["_custom_strats"].append({
                                "title": f"{cs_col} (P{n_pct})",
                                "data": data, "col": cs_col, "type": "num"
                            })
                            st.rerun()

                    else:  # Custom Breaks
                        breaks_str = st.text_input("Break points (comma-separated)",
                            value=f"{v_min:.0f}, {(v_min+v_max)/2:.0f}, {v_max:.0f}", key="_cs_breaks")
                        if st.button("▶ Build", key="_cs_build_custom"):
                            try:
                                breaks = sorted([float(x.strip()) for x in breaks_str.split(",")])
                                if breaks[0] > v_min: breaks.insert(0, v_min)
                                if breaks[-1] < v_max: breaks.append(v_max)
                                bal_col = mp.get("current_balance", "")
                                data = []
                                for bi in range(len(breaks) - 1):
                                    lo, hi = breaks[bi], breaks[bi + 1]
                                    label = f"{lo:,.0f}–{hi:,.0f}"
                                    m = (vals >= lo) & (vals < hi if bi < len(breaks) - 2 else vals <= hi)
                                    idxs = vals[m].index
                                    bal = df_cs.loc[idxs, bal_col].apply(parse_numeric).sum() if bal_col and bal_col in df_cs.columns else 0
                                    tb_all = df_cs[bal_col].apply(parse_numeric).sum() if bal_col and bal_col in df_cs.columns else 1
                                    data.append({"name": label, "count": len(idxs), "balance": bal,
                                                 "pct": (bal/tb_all*100) if tb_all > 0 else 0,
                                                 "field": "", "min": lo, "max": hi})
                                st.session_state["_custom_strats"].append({
                                    "title": f"{cs_col} (custom)",
                                    "data": data, "col": cs_col, "type": "num"
                                })
                                st.rerun()
                            except Exception as e:
                                st.error(f"Invalid breaks: {e}")
                else:
                    st.warning(f"No numeric values in {cs_col}")

        # ── Render existing custom strats ──
        custom_strats = st.session_state.get("_custom_strats", [])
        if custom_strats:
            st.markdown("---")
            CPAL = ["#36D7B7", "#F472B6", "#818CF8", "#FBBF24", "#34D399", "#FF4D6A", "#4D9EFF", "#A78BFA"]
            for ci in range(0, len(custom_strats), 2):
                c1, c2 = st.columns(2)
                for j, col in enumerate([c1, c2]):
                    idx = ci + j
                    if idx < len(custom_strats):
                        cs = custom_strats[idx]
                        with col:
                            cs_mp = dict(mp)
                            cs_field_key = f"_custom_{idx}"
                            cs_mp[cs_field_key] = cs["col"]
                            tagged_data = []
                            for b in cs["data"]:
                                tb2 = dict(b)
                                tb2["field"] = cs_field_key
                                tagged_data.append(tb2)
                            chart_with_drilldown(tagged_data, cs["title"], CPAL[idx % len(CPAL)],
                                                 f"cust_{idx}", st.session_state.df, cs_mp, "cust")
                            if st.button("🗑 Remove", key=f"del_cust_{idx}"):
                                custom_strats.pop(idx)
                                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear All Custom Strats"):
                st.session_state["_custom_strats"] = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: CHARTS & REGRESSION
# ═══════════════════════════════════════════════════════════════

elif page == "📉 Charts & Regression":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Charts & Regression</div>', unsafe_allow_html=True)

        if st.session_state.df is not None:
            from logic import parse_numeric
            import numpy as np
            df = st.session_state.df

            # Identify numeric columns
            num_cols = []
            for h in hdrs:
                vals = df[h].head(50).apply(parse_numeric)
                if vals.notna().sum() >= 25:
                    num_cols.append(h)
            if not num_cols:
                num_cols = hdrs

            def _pick_default(candidates, fallback_list):
                for c in candidates:
                    col = mp.get(c, "")
                    if col and col in fallback_list:
                        return fallback_list.index(col)
                return 0

            # ── Chart slots stored in session state ──
            if "_chart_slots" not in st.session_state:
                x_def = _pick_default(["fico_origination", "interest_rate", "dti"], num_cols)
                y_def = _pick_default(["current_balance", "original_balance", "monthly_payment"], num_cols)
                if y_def == x_def:
                    y_def = min(x_def + 1, len(num_cols) - 1)
                st.session_state["_chart_slots"] = [{"x": num_cols[x_def], "y": num_cols[y_def]}]

            slots = st.session_state["_chart_slots"]

            for si, slot in enumerate(slots):
                if si > 0:
                    st.markdown("---")

                sc1, sc2, sc3, sc4 = st.columns([3, 3, 2, 1.5])
                with sc1:
                    x_col = st.selectbox("X axis", num_cols,
                        index=num_cols.index(slot["x"]) if slot["x"] in num_cols else 0,
                        key=f"cx_{si}")
                    slot["x"] = x_col
                with sc2:
                    y_col = st.selectbox("Y axis", num_cols,
                        index=num_cols.index(slot["y"]) if slot["y"] in num_cols else min(1, len(num_cols)-1),
                        key=f"cy_{si}")
                    slot["y"] = y_col
                with sc3:
                    color_col = st.selectbox("Color by", ["None"] + hdrs, key=f"cc_{si}")
                with sc4:
                    show_trend = st.checkbox("Trend", value=True, key=f"ct_{si}")

                xv = df[x_col].apply(parse_numeric)
                yv = df[y_col].apply(parse_numeric)
                mask = xv.notna() & yv.notna()

                if mask.sum() >= 3:
                    plot_df = pd.DataFrame({"x": xv[mask], "y": yv[mask]})
                    if color_col != "None":
                        plot_df["color"] = df.loc[mask, color_col].fillna("Unknown").astype(str)
                    if len(plot_df) > 2000:
                        plot_df = plot_df.sample(2000, random_state=42)

                    if color_col != "None":
                        fig = px.scatter(plot_df, x="x", y="y", color="color",
                            labels={"x": x_col, "y": y_col}, opacity=0.6)
                    else:
                        fig = px.scatter(plot_df, x="x", y="y",
                            labels={"x": x_col, "y": y_col},
                            color_discrete_sequence=["#00D4AA"], opacity=0.6)

                    # Manual trendline with numpy
                    if show_trend:
                        try:
                            x_arr = plot_df["x"].values
                            y_arr = plot_df["y"].values
                            slope, intercept = np.polyfit(x_arr, y_arr, 1)
                            x_line = np.array([x_arr.min(), x_arr.max()])
                            y_line = slope * x_line + intercept
                            fig.add_scatter(x=x_line, y=y_line, mode="lines",
                                line=dict(color="#FF4D6A", width=2, dash="dash"),
                                name="Trendline", showlegend=True)
                        except Exception:
                            pass

                    fig.update_layout(plot_bgcolor="#12161C", paper_bgcolor="#0B0E11", font_color="#8494A7", height=420,
                        title=dict(text=f"{y_col} vs {x_col}", font=dict(size=13, color="#E8ECF1")),
                        xaxis=dict(showgrid=True, gridcolor="#1E2530"),
                        yaxis=dict(showgrid=True, gridcolor="#1E2530"),
                        margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{si}")

                    # Inline regression stats
                    x_arr = plot_df["x"].values
                    y_arr = plot_df["y"].values
                    try:
                        slope, intercept = np.polyfit(x_arr, y_arr, 1)
                        y_pred = slope * x_arr + intercept
                        ss_res = np.sum((y_arr - y_pred) ** 2)
                        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        r = np.sqrt(abs(r2)) * (1 if slope >= 0 else -1)
                        n = len(x_arr)
                        strength = "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak"
                        direction = "positive" if r > 0 else "negative"

                        # Equation on its own line
                        sign = "+" if intercept >= 0 else "−"
                        eq = f"ŷ = {slope:.4f} · x {sign} {abs(intercept):.2f}"
                        st.markdown(f'<div style="background:#171C24;border:1px solid #1E2530;border-radius:8px;padding:10px 16px;margin:8px 0;font-family:monospace;font-size:14px;color:#00D4AA">{eq}</div>', unsafe_allow_html=True)

                        re1, re2, re3, re4 = st.columns(4)
                        with re1: card("R²", f"{r2:.4f}")
                        with re2: card("r", f"{r:.4f}")
                        with re3: card("Slope", f"{slope:.4f}")
                        with re4: card("n", f"{n:,}")

                        st.markdown(f'<span style="color:#8494A7;font-size:11px">{strength} {direction} correlation · slope={slope:.4f} · intercept={intercept:.2f}</span>', unsafe_allow_html=True)
                    except Exception:
                        pass
                else:
                    st.warning(f"Not enough numeric data for {x_col} vs {y_col} ({mask.sum()} valid pairs, need ≥3)")

            # Add / remove chart buttons
            st.markdown("<br>", unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns([1, 1, 4])
            with ac1:
                if st.button("➕ Add Chart", use_container_width=True):
                    # Pick next reasonable defaults
                    used_pairs = [(s["x"], s["y"]) for s in slots]
                    new_x = num_cols[min(len(slots), len(num_cols)-1)]
                    new_y = num_cols[min(len(slots)+1, len(num_cols)-1)]
                    slots.append({"x": new_x, "y": new_y})
                    st.rerun()
            with ac2:
                if len(slots) > 1 and st.button("🗑 Remove Last", use_container_width=True):
                    slots.pop()
                    st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: DATA QUALITY
# ═══════════════════════════════════════════════════════════════

elif page == "✅ Data Quality":
    if not vl:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Data Quality</div>', unsafe_allow_html=True)
        comp = vl.get("completeness", 0)
        grade = "A" if comp>=95 else "B" if comp>=85 else "C" if comp>=70 else "D"
        q1,q2,q3,q4 = st.columns(4)
        with q1: card("Quality Grade", grade, f"Completeness: {comp:.1f}%")
        with q2: card("Missing Values", f"{vl.get('missing_count',0):,}")
        with q3: card("Out of Range", f"{vl.get('oor_count',0):,}")
        with q4: card("Fields Mapped", f"{len(mp)}")

        if vl.get("issues"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Issues Found:**")
            for iss in vl["issues"]:
                st.markdown(f'<span style="color:#FF4D6A;font-size:11px">⚠ {iss}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Field Mapping Audit:**")
        audit = [{"Field": k, "Mapped To": v, "Status": "✅"} for k, v in sorted(mp.items())]
        st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: CONCENTRATION
# ═══════════════════════════════════════════════════════════════

elif page == "⚡ Concentration":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Concentration Analysis</div>', unsafe_allow_html=True)
        cn1,cn2,cn3 = st.columns(3)
        hhi_class = "Low" if an["hhi"]<0.15 else "Moderate" if an["hhi"]<0.25 else "High"
        with cn1: card("Geographic HHI", f"{an['hhi']:.4f}", hhi_class)
        with cn2: card("Top State", an['geo'][0]['name'] if an.get('geo') else '—', f"{an.get('top_state_conc',0):.1f}% of balance")
        with cn3: card("States Represented", f"{len(an.get('geo',[]))}")
        st.markdown("<br>", unsafe_allow_html=True)
        chart_with_drilldown(an.get("geo",[]), "Geographic Distribution", "#4D9EFF", "conc_geo", st.session_state.df, mp, "conc")


# ═══════════════════════════════════════════════════════════════
# PAGE: RISK SUMMARY
# ═══════════════════════════════════════════════════════════════

elif page == "🛡️ Risk Summary":
    if not an:
        st.info("Upload a tape first.")
    else:
        st.markdown('<div class="section-header">Risk Dashboard</div>', unsafe_allow_html=True)
        def rl(val, g, y): return "🟢" if val<=g else "🟡" if val<=y else "🔴"

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Credit Risk**")
            st.markdown(f'{rl(an["wa_fico_orig"],999,660)} WA FICO Orig: **{fmt_s(an["wa_fico_orig"])}**')
            st.markdown(f'{rl(an["wa_dti"],35,45)} WA DTI: **{fmt_p(an["wa_dti"])}**')
            st.markdown(f'{rl(an["wa_rate"],15,20)} WA Rate: **{fmt_r(an["wa_rate"])}**')
        with r2:
            st.markdown("**Performance Risk**")
            st.markdown(f'{rl(an["dq30"],3,8)} 30+ DPD: **{fmt_p(an["dq30"])}**')
            st.markdown(f'{rl(an["dq90"],1,5)} 90+ DPD: **{fmt_p(an["dq90"])}**')
            st.markdown(f'{rl(an["net_loss_rate"],2,5)} Net Loss Rate: **{fmt_p(an["net_loss_rate"])}**')

        st.markdown("<br>", unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            st.markdown("**Concentration Risk**")
            st.markdown(f'{rl(an["hhi"],0.15,0.25)} HHI: **{an["hhi"]:.4f}**')
            st.markdown(f'{rl(an["top_state_conc"],20,35)} Top State: **{fmt_p(an["top_state_conc"])}**')
        with r4:
            st.markdown("**Data Quality**")
            c = vl.get("completeness",0) if vl else 0
            o = vl.get("oor_count",0) if vl else 0
            st.markdown(f'{rl(100-c,5,15)} Completeness: **{fmt_p(c)}**')
            st.markdown(f'{rl(o,10,50)} Out of Range: **{o}**')


# ═══════════════════════════════════════════════════════════════
# PAGE: COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════

elif page == "📋 Column Mapping":
    st.markdown('<div class="section-header">Column Mapping</div>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#566375;font-size:11px">{len(mp)} fields mapped · {len(hdrs)} source columns</span>', unsafe_allow_html=True)

    # ── Templates + Actions in one row ──
    try:
        templates = client.list_templates()
    except:
        templates = []

    if templates:
        tpl_options = ["— Select template —"] + [f"{t['name']} ({t['originator']} · {len(t['mapping'])} fields)" for t in templates]
        tc1, tc2, tc3 = st.columns([4, 1.5, 1.5])
        with tc1:
            tpl_sel = st.selectbox("Template", tpl_options, key="tpl_load", label_visibility="collapsed")
        with tc2:
            if tpl_sel != "— Select template —":
                idx = tpl_options.index(tpl_sel) - 1
                tpl = templates[idx]
                applicable = {k: v for k, v in tpl["mapping"].items() if v in hdrs}
                if st.button("📥 Apply", use_container_width=True):
                    with st.spinner("Applying..."):
                        merged = dict(mp)
                        merged.update(applicable)
                        updated = client.update_mapping(st.session_state.tape_id, merged)
                        st.session_state.tape = updated
                        st.session_state.analysis = client.get_analysis(st.session_state.tape_id)
                        st.session_state.validation = client.get_validation(st.session_state.tape_id)
                    st.rerun()
        with tc3:
            if st.button("🧠 AI Match", use_container_width=True):
                with st.spinner("AI matching..."):
                    try:
                        updated = client.auto_match(st.session_state.tape_id, mode="ai")
                        st.session_state.tape = updated
                        st.session_state.analysis = client.get_analysis(st.session_state.tape_id)
                        st.session_state.validation = client.get_validation(st.session_state.tape_id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI match failed: {e}")
    else:
        if st.button("🧠 AI Match", use_container_width=True):
            with st.spinner("AI matching..."):
                try:
                    updated = client.auto_match(st.session_state.tape_id, mode="ai")
                    st.session_state.tape = updated
                    st.session_state.analysis = client.get_analysis(st.session_state.tape_id)
                    st.session_state.validation = client.get_validation(st.session_state.tape_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"AI match failed: {e}")

    st.markdown("---")

    # Get all field definitions (standard + custom)
    try:
        std_fields = client.list_standard_fields()
    except:
        std_fields = {}
    try:
        custom_fields = client.list_fields()
    except:
        custom_fields = []

    # Build full standard field list: key -> label
    all_flds = {}
    for k, v in std_fields.items():
        all_flds[k] = v.get("label", k)
    for cf in custom_fields:
        all_flds[cf["key"]] = f"★ {cf['label']}"

    # Determine reference field set based on template selection
    selected_tpl = None
    if templates:
        tpl_sel_val = st.session_state.get("tpl_load", "— Select template —")
        if tpl_sel_val != "— Select template —":
            tpl_opts = ["— Select template —"] + [f"{t['name']} ({t['originator']} · {len(t['mapping'])} fields)" for t in templates]
            if tpl_sel_val in tpl_opts:
                tidx = tpl_opts.index(tpl_sel_val) - 1
                selected_tpl = templates[tidx]

    if selected_tpl:
        # Template mode: reference fields = template's fields
        ref_fields = {}
        for k, v in selected_tpl["mapping"].items():
            ref_fields[k] = all_flds.get(k, k)
        ref_label = f"Template: {selected_tpl['name']}"
        total_ref = len(ref_fields)
    else:
        # Standard mode: reference fields = all standard + custom fields
        ref_fields = dict(all_flds)
        ref_label = "Standard Fields"
        total_ref = len(ref_fields)

    # Split into mapped and unmapped within the reference set
    mapped_fields = {}
    for k in ref_fields:
        if k in mp and mp[k] in hdrs:
            mapped_fields[k] = ref_fields[k]
    unmapped_ref = {k: v for k, v in ref_fields.items() if k not in mapped_fields}

    # Also track extra mapped fields not in reference (from rule-match or other sources)
    extra_mapped = {}
    for k in mp:
        if k not in ref_fields:
            extra_mapped[k] = all_flds.get(k, k)

    options = ["— (unmapped)"] + sorted(hdrs)
    new_mp = dict(mp)
    changed = False

    st.markdown(f'<span style="color:#8494A7;font-size:11px">{ref_label} · {len(mapped_fields)}/{total_ref} mapped</span>', unsafe_allow_html=True)

    # ── Mapped Fields ──
    if mapped_fields:
        st.markdown(f'<span style="color:#00D4AA;font-size:13px;font-weight:600">✓ Mapped ({len(mapped_fields)})</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        mapped_keys = sorted(mapped_fields.keys())
        mid = (len(mapped_keys) + 1) // 2

        for i, fk in enumerate(mapped_keys):
            container = col_l if i < mid else col_r
            with container:
                label = mapped_fields[fk]
                current = mp.get(fk, "")
                default_idx = options.index(current) if current in options else 0
                sel = st.selectbox(f"🟢 {label}", options, index=default_idx, key=f"m_{fk}")
                if sel == "— (unmapped)":
                    if fk in new_mp:
                        del new_mp[fk]
                        changed = True
                else:
                    if sel != current:
                        new_mp[fk] = sel
                        changed = True
    else:
        st.info("No fields mapped yet. Apply a template or use AI Match above.")

    # ── Unmapped Reference Fields (collapsed) ──
    if unmapped_ref:
        with st.expander(f"⚪ Unmapped ({len(unmapped_ref)})", expanded=False):
            col_l2, col_r2 = st.columns(2)
            unmapped_keys = sorted(unmapped_ref.keys())
            mid2 = (len(unmapped_keys) + 1) // 2

            for i, fk in enumerate(unmapped_keys):
                container = col_l2 if i < mid2 else col_r2
                with container:
                    label = unmapped_ref[fk]
                    sel = st.selectbox(f"⚪ {label}", options, index=0, key=f"m_{fk}")
                    if sel != "— (unmapped)":
                        new_mp[fk] = sel
                        changed = True

    # ── Extra mapped fields (from rule-match, not in selected template/standard set) ──
    if extra_mapped:
        with st.expander(f"➕ Additional Mapped Fields ({len(extra_mapped)})", expanded=False):
            col_l3, col_r3 = st.columns(2)
            extra_keys = sorted(extra_mapped.keys())
            mid3 = (len(extra_keys) + 1) // 2

            for i, fk in enumerate(extra_keys):
                container = col_l3 if i < mid3 else col_r3
                with container:
                    label = extra_mapped[fk]
                    current = mp.get(fk, "")
                    default_idx = options.index(current) if current in options else 0
                    sel = st.selectbox(f"🔵 {label}", options, index=default_idx, key=f"m_{fk}")
                    if sel == "— (unmapped)":
                        if fk in new_mp:
                            del new_mp[fk]
                            changed = True
                    else:
                        if sel != current:
                            new_mp[fk] = sel
                            changed = True

    if changed:
        if st.button("💾 Save Mapping Changes", type="primary", use_container_width=True):
            with st.spinner("Updating..."):
                updated = client.update_mapping(st.session_state.tape_id, new_mp)
                st.session_state.tape = updated
                st.session_state.analysis = client.get_analysis(st.session_state.tape_id)
                st.session_state.validation = client.get_validation(st.session_state.tape_id)
            st.rerun()

    # ── Unmapped Source Columns ──
    st.markdown("---")
    mapped_cols = set(mp.values())
    unmapped_cols = [h for h in hdrs if h not in mapped_cols]

    st.markdown(f'<div class="section-header" style="font-size:15px">Unmapped Source Columns ({len(unmapped_cols)})</div>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#566375;font-size:11px">Add as custom field with optional synonyms for future auto-matching.</span>', unsafe_allow_html=True)

    if unmapped_cols:
        # Header row
        uh1, uh2, uh3, uh4 = st.columns([2, 2, 2, 1])
        with uh1: st.markdown('<span style="color:#8494A7;font-size:10px">COLUMN</span>', unsafe_allow_html=True)
        with uh2: st.markdown('<span style="color:#8494A7;font-size:10px">SYNONYMS (optional)</span>', unsafe_allow_html=True)
        with uh3: st.markdown('<span style="color:#8494A7;font-size:10px">DISPLAY LABEL</span>', unsafe_allow_html=True)
        with uh4: st.markdown('<span style="color:#8494A7;font-size:10px"></span>', unsafe_allow_html=True)

        for col_name in unmapped_cols:
            if st.session_state.df is not None:
                samples = st.session_state.df[col_name].dropna().head(3).tolist()
                sample_str = ", ".join(str(s)[:30] for s in samples)
            else:
                sample_str = "—"

            with st.container():
                uc1, uc2, uc3, uc4 = st.columns([2, 2, 2, 1])
                with uc1:
                    st.markdown(f'<span style="color:#E8ECF1;font-size:12px;font-weight:600">{col_name}</span><br><span style="color:#566375;font-size:10px">{sample_str}</span>', unsafe_allow_html=True)
                with uc2:
                    syn_key = f"syn_{col_name}"
                    synonyms = st.text_input("Synonyms", placeholder="alt names, comma separated", key=syn_key, label_visibility="collapsed")
                with uc3:
                    label_key = f"lbl_{col_name}"
                    custom_label = st.text_input("Label", placeholder=col_name, key=label_key, label_visibility="collapsed")
                with uc4:
                    if st.button("➕", key=f"qa_{col_name}", use_container_width=True):
                        key = col_name.lower().replace(" ", "_").replace("-", "_")
                        if key in all_flds:
                            key = key + "_custom"
                        lbl = custom_label if custom_label else col_name
                        pats = [col_name.lower()]
                        if synonyms:
                            pats += [p.strip().lower() for p in synonyms.split(",") if p.strip()]
                        try:
                            client.create_field(key, lbl, pats)
                            new_mp2 = dict(mp)
                            new_mp2[key] = col_name
                            updated = client.update_mapping(st.session_state.tape_id, new_mp2)
                            st.session_state.tape = updated
                            st.session_state.analysis = client.get_analysis(st.session_state.tape_id)
                            st.session_state.validation = client.get_validation(st.session_state.tape_id)
                            st.toast(f"✅ Added & mapped: {lbl} (synonyms: {pats})")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
    else:
        st.markdown('<span style="color:#00D4AA;font-size:12px">✓ All source columns are mapped.</span>', unsafe_allow_html=True)

    # ── Save Template ──
    st.markdown("---")
    st.markdown("**Save Current Mapping as Template:**")
    tc1, tc2, tc3 = st.columns([2, 2, 1])
    with tc1: tn = st.text_input("Template name", placeholder="e.g. LendingClub v1", key="tn")
    with tc2: to = st.text_input("Originator", placeholder="e.g. LendingClub", key="to")
    with tc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Template", use_container_width=True) and tn and to:
            client.create_template(tn, to, mp)
            st.toast(f"✅ Saved: {tn}")
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: RAW DATA
# ═══════════════════════════════════════════════════════════════

elif page == "🗄️ Raw Data":
    st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:#566375;font-size:11px">{tape.get("row_count",0):,} rows · {tape.get("col_count",0)} cols</span>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        search = st.text_input("🔍 Search", key="rs")
        display = st.session_state.df
        if search:
            mask = display.apply(lambda r: r.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
            display = display[mask]
        st.markdown(f'<span style="color:#566375;font-size:11px">Showing {len(display):,} rows</span>', unsafe_allow_html=True)
        st.dataframe(display, use_container_width=True, height=500, hide_index=True)

        csv = display.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "export.csv", "text/csv")
    else:
        st.info("Raw data not available for previously loaded tapes. Re-upload to view.")
        if st.button("📥 Export via API"):
            csv_data = client.export_csv(st.session_state.tape_id)
            st.download_button("Download", csv_data, "export.csv", "text/csv", key="dl2")


# ═══════════════════════════════════════════════════════════════
# PAGE: ADMIN / TEMPLATES
# ═══════════════════════════════════════════════════════════════

elif page == "⚙️ Admin / Templates":
    st.markdown('<div class="section-header">Admin & Templates</div>', unsafe_allow_html=True)

    admin_section = st.radio("Section", ["Templates", "Custom Fields", "Account"], horizontal=True)

    if admin_section == "Templates":
        st.markdown("**Saved Mapping Templates**")
        st.markdown('<span style="color:#8494A7;font-size:11px">Templates save column mappings for specific originators. Load them on the Column Mapping page.</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            templates = client.list_templates()
            if templates:
                for t in templates:
                    c1,c2,c3 = st.columns([3,2,1])
                    with c1: st.markdown(f'<span style="color:#E8ECF1;font-weight:600">{t["name"]}</span>', unsafe_allow_html=True)
                    with c2: st.markdown(f'<span style="color:#8494A7">{t.get("originator","")} · {len(t.get("mapping",{}))} fields</span>', unsafe_allow_html=True)
                    with c3:
                        if st.button("🗑️", key=f"dt_{t['id']}"):
                            client.delete_template(t["id"])
                            st.rerun()
                    # Show mapping details in expander
                    with st.expander(f"View mapping: {t['name']}", expanded=False):
                        mapping = t.get("mapping", {})
                        if mapping:
                            for std_field, src_col in sorted(mapping.items()):
                                st.markdown(f'<span style="color:#00D4AA;font-size:11px">{std_field}</span> → <span style="color:#E8ECF1;font-size:11px">{src_col}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color:#566375">Empty mapping</span>', unsafe_allow_html=True)
            else:
                st.info("No templates saved yet. Go to Column Mapping page and use 'Save Template' to create one.")
        except Exception as e:
            st.error(f"Failed to load templates: {e}")

    elif admin_section == "Custom Fields":
        st.markdown("**Custom Standard Fields**")
        st.markdown('<span style="color:#8494A7;font-size:11px">Add custom fields beyond the 47 built-in ABS standard fields. These are used during auto-matching.</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            fields = client.list_fields()
            if fields:
                for f in fields:
                    c1, c2 = st.columns([4,1])
                    with c1: st.markdown(f'★ **{f["label"]}** (`{f["key"]}`) — patterns: {f.get("patterns",[])}')
                    with c2:
                        if st.button("🗑️", key=f"df_{f['id']}"):
                            client.delete_field(f["id"])
                            st.rerun()
            else:
                st.info("No custom fields added yet.")
        except Exception as e:
            st.error(f"Failed to load fields: {e}")

        st.markdown("---")
        st.markdown("**Add Custom Field:**")
        af1,af2,af3 = st.columns(3)
        with af1: nk = st.text_input("Field key", placeholder="vehicle_make", key="cfk")
        with af2: nl = st.text_input("Label", placeholder="Vehicle Make", key="cfl")
        with af3: np_ = st.text_input("Patterns", placeholder="make, manufacturer", key="cfp")
        if st.button("➕ Add Field") and nk and nl:
            pats = [p.strip() for p in np_.split(",") if p.strip()] if np_ else []
            client.create_field(nk, nl, pats)
            st.toast(f"✅ Added: {nl}")
            st.rerun()

    elif admin_section == "Account":
        st.markdown(f'**API Key:** `{st.session_state.api_key}`')
        st.markdown(f'<span style="color:#566375;font-size:11px">Use this key for programmatic API access.</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**API Endpoints:**")
        st.code(f"""
# Upload a tape
curl -X POST http://127.0.0.1:8000/api/tapes \\
  -H "X-API-Key: {st.session_state.api_key}" \\
  -F "file=@your_tape.csv"

# Get analysis
curl http://127.0.0.1:8000/api/tapes/{{tape_id}}/analysis \\
  -H "X-API-Key: {st.session_state.api_key}"

# Full API docs: http://127.0.0.1:8000/docs
""", language="bash")
