"""
Loan Tape Analyzer — Streamlit App

Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from logic import (
    STD_FIELDS, parse_numeric, format_currency, format_pct, format_rate, format_score,
    rule_match, score_template, analyze, validate, get_numeric, get_string,
    calc_regression, calc_multi_regression, bucket, group_by
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Loan Tape Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0B0E11; }
    .metric-card {
        background: #171C24; border-radius: 10px; padding: 14px 18px;
        border: 1px solid #1E2530; cursor: pointer; transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #00D4AA; }
    .metric-label { color: #8494A7; font-size: 11px; font-weight: 500; }
    .metric-value { color: #E8ECF1; font-size: 22px; font-weight: 700; margin: 4px 0; }
    .metric-sub { color: #566375; font-size: 10px; }
    .accent { color: #00D4AA; }
    .warn { color: #FFB224; }
    .danger { color: #FF4D6A; }
    .blue { color: #4D9EFF; }
    .section-header {
        background: linear-gradient(135deg, #00D4AA, #4D9EFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 18px; font-weight: 700; margin: 16px 0 8px 0;
    }
    .strat-row { background: #12161C; border-radius: 6px; padding: 6px 10px; margin: 2px 0; }
    div[data-testid="stDataFrame"] { border: 1px solid #1E2530; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "df": None, "filename": None, "mapping": {},
        "custom_fields": [], "templates": [],
        "analysis": None, "validation": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def all_fields():
    """Merge STD_FIELDS + custom fields."""
    merged = dict(STD_FIELDS)
    for cf in st.session_state.custom_fields:
        pats = []
        for s in cf.get("patterns", []):
            if any(c in s for c in r".?*+[]()\\^$|{}"):
                pats.append(s)
            else:
                pats.append(s.strip().replace(" ", ".?").replace("_", ".?"))
        merged[cf["key"]] = {"label": cf["label"], "patterns": pats}
    return merged


def run_analysis():
    """Run analysis + validation with current mapping."""
    df = st.session_state.df
    mp = st.session_state.mapping
    if df is not None and mp:
        st.session_state.analysis = analyze(df, mp)
        st.session_state.validation = validate(df, mp)


def load_data(df, filename):
    """Load a DataFrame, auto-match columns, run analysis."""
    st.session_state.df = df
    st.session_state.filename = filename

    # Try template auto-detect
    hdrs = list(df.columns)
    best_tpl, best_score = None, 0
    for tpl in st.session_state.templates:
        sc = score_template(tpl["mapping"], hdrs)
        if sc > best_score:
            best_score = sc
            best_tpl = tpl

    if best_tpl and best_score >= 0.8:
        st.session_state.mapping = {k: v for k, v in best_tpl["mapping"].items() if v in hdrs}
        st.toast(f"✅ Auto-applied template: {best_tpl['name']} ({best_score:.0%} match)")
    else:
        st.session_state.mapping = rule_match(df, all_fields())
        if best_tpl and best_score >= 0.4:
            st.toast(f"💡 Possible template match: {best_tpl['name']} ({best_score:.0%})")

    run_analysis()


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def strat_chart(data, title, color="#00D4AA"):
    """Create a horizontal bar chart for a stratification."""
    if not data or all(d["count"] == 0 for d in data):
        return None
    df_chart = pd.DataFrame(data)
    fig = px.bar(df_chart, y="name", x="pct", orientation="h",
                 color_discrete_sequence=[color], text="pct")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='auto')
    fig.update_layout(
        plot_bgcolor="#12161C", paper_bgcolor="#0B0E11",
        font_color="#8494A7", height=max(200, len(data) * 32),
        title=dict(text=title, font=dict(size=13, color="#E8ECF1")),
        xaxis=dict(title="% Balance", showgrid=True, gridcolor="#1E2530"),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=20), showlegend=False,
    )
    return fig


def strat_table(data):
    """Create a DataFrame from strat data for display."""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["balance"] = df["balance"].apply(format_currency)
    df["pct"] = df["pct"].apply(lambda x: f"{x:.1f}%")
    return df[["name", "count", "balance", "pct"]].rename(
        columns={"name": "Bucket", "count": "Count", "balance": "Balance", "pct": "% Bal"}
    )


# ═══════════════════════════════════════════════════════════════
# UPLOAD SCREEN
# ═══════════════════════════════════════════════════════════════

if st.session_state.df is None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align:center"><span class="section-header" style="font-size:28px">📊 Loan Tape Analyzer</span></div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#566375;font-size:12px;letter-spacing:2px">ABS TAPE CRACKING PLATFORM</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#8494A7;font-size:13px">AI column mapping, stratification, tape cracking with click-to-drill analytics.</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a CSV loan tape", type=["csv", "tsv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            load_data(df, uploaded.name)
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#566375;font-size:11px">Or try with sample data:</p>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("🏦 Consumer Unsecured\n\n1,000 loans · 88 cols · Clean", use_container_width=True):
                path = os.path.join(os.path.dirname(__file__), "sample_data", "consumer_unsecured_loan_tape.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    load_data(df, "consumer_unsecured_loan_tape.csv")
                    st.rerun()
        with sc2:
            if st.button("🚗 QuickDrive Auto Loans\n\n500 loans · 73 cols · Messy", use_container_width=True):
                path = os.path.join(os.path.dirname(__file__), "sample_data", "quickdrive_auto_loan_tape.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    load_data(df, "quickdrive_auto_loan_tape.csv")
                    st.rerun()

    st.stop()


# ═══════════════════════════════════════════════════════════════
# MAIN APP (data loaded)
# ═══════════════════════════════════════════════════════════════

df = st.session_state.df
mp = st.session_state.mapping
an = st.session_state.analysis
vl = st.session_state.validation
hdrs = list(df.columns)

# Header bar
mapped_count = sum(1 for v in mp.values() if v and v in hdrs)
total_fields = len(all_fields())

hc1, hc2, hc3, hc4 = st.columns([3, 1, 1, 1])
with hc1:
    st.markdown(f'<span class="section-header" style="font-size:16px">📊 Loan Tape Analyzer</span> <span style="color:#566375;font-size:11px">{st.session_state.filename}</span>', unsafe_allow_html=True)
with hc2:
    st.markdown(f'<span style="color:#8494A7;font-size:11px">Loans:</span> <span style="color:#E8ECF1;font-size:13px;font-weight:700">{len(df):,}</span>', unsafe_allow_html=True)
with hc3:
    st.markdown(f'<span style="color:#8494A7;font-size:11px">Mapped:</span> <span class="accent" style="font-size:13px;font-weight:700">{mapped_count}/{total_fields}</span>', unsafe_allow_html=True)
with hc4:
    if st.button("🔄 Reset", use_container_width=True):
        for k in ["df", "filename", "mapping", "analysis", "validation"]:
            st.session_state[k] = None if k != "mapping" else {}
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tabs = st.tabs(["📋 Column Mapping", "📊 Pool Overview", "🔍 Tape Cracking",
                "📈 Stratifications", "🎯 Custom Strats", "📉 Charts & Regression",
                "✅ Data Quality", "⚡ Concentration", "🛡️ Risk Summary",
                "🗄️ Raw Data", "⚙️ Admin / Templates"])


# ── TAB 0: COLUMN MAPPING ──
with tabs[0]:
    st.markdown('<div class="section-header">Column Mapping</div>', unsafe_allow_html=True)

    # Template controls
    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc1:
        tpl_names = ["(none)"] + [f"{t['name']} ({t['originator']})" for t in st.session_state.templates]
        tpl_sel = st.selectbox("Load Template", tpl_names, label_visibility="collapsed")
        if tpl_sel != "(none)":
            idx = tpl_names.index(tpl_sel) - 1
            tpl = st.session_state.templates[idx]
            st.session_state.mapping = {k: v for k, v in tpl["mapping"].items() if v in hdrs}
            run_analysis()
            st.rerun()

    with tc2:
        save_name = st.text_input("Template name", placeholder="e.g. LendingClub v1", label_visibility="collapsed")
    with tc3:
        save_orig = st.text_input("Originator", placeholder="e.g. LendingClub", label_visibility="collapsed")

    if save_name and save_orig:
        if st.button("💾 Save as Template"):
            clean_mp = {k: v for k, v in mp.items() if v and v in hdrs}
            st.session_state.templates.append({
                "name": save_name, "originator": save_orig, "mapping": clean_mp,
            })
            st.toast(f"✅ Template saved: {save_name}")

    st.markdown("---")

    # Mapping grid
    fields = all_fields()
    options = ["— (unmapped)"] + sorted(hdrs)
    new_mp = dict(mp)

    # Show in 2-column layout
    col_l, col_r = st.columns(2)
    field_keys = list(fields.keys())
    mid = (len(field_keys) + 1) // 2

    for i, fk in enumerate(field_keys):
        container = col_l if i < mid else col_r
        with container:
            fdef = fields[fk]
            is_custom = fk not in STD_FIELDS
            label_txt = f"{'★ ' if is_custom else ''}{fdef['label']} ({fk})"
            current = mp.get(fk, "")
            default_idx = options.index(current) if current in options else 0
            sel = st.selectbox(label_txt, options, index=default_idx, key=f"map_{fk}")
            if sel == "— (unmapped)":
                new_mp.pop(fk, None)
            else:
                new_mp[fk] = sel

    if new_mp != mp:
        st.session_state.mapping = new_mp
        run_analysis()
        st.rerun()

    # Unmapped source columns
    mapped_cols = set(mp.values())
    unmapped = [h for h in hdrs if h not in mapped_cols]
    if unmapped:
        st.markdown("---")
        st.markdown('<div class="section-header">Unmapped Source Columns</div>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#566375;font-size:11px">{len(unmapped)} columns not mapped to any field</span>', unsafe_allow_html=True)

        for col in unmapped[:20]:
            samples = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(s) for s in samples)
            uc1, uc2, uc3 = st.columns([2, 3, 1])
            with uc1:
                st.markdown(f'<span style="color:#E8ECF1;font-size:12px;font-weight:600">{col}</span>', unsafe_allow_html=True)
            with uc2:
                st.markdown(f'<span style="color:#566375;font-size:11px">{sample_str}</span>', unsafe_allow_html=True)
            with uc3:
                if st.button("+ Add as Field", key=f"qaf_{col}"):
                    key = col.lower().replace(" ", "_").replace("-", "_")
                    if key in fields:
                        key = key + "_custom"
                    st.session_state.custom_fields.append({
                        "key": key, "label": col, "patterns": [col.lower()],
                    })
                    st.session_state.mapping[key] = col
                    run_analysis()
                    st.rerun()


# ── TAB 1: POOL OVERVIEW ──
with tabs[1]:
    if an is None:
        st.info("Map columns first to see analytics.")
    else:
        st.markdown('<div class="section-header">Pool Overview</div>', unsafe_allow_html=True)

        # Metric cards row 1
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: metric_card("Total Loans", f"{an['N']:,}")
        with m2: metric_card("Total Balance", format_currency(an["tb"]))
        with m3: metric_card("Orig Balance", format_currency(an["tob"]))
        with m4: metric_card("Avg Balance", format_currency(an["avg"]))
        with m5: metric_card("Pool Factor", format_pct(an["pool_factor"]))
        with m6: metric_card("WA Seasoning", f"{an['wa_mob']:.0f} mo")

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric cards row 2
        m7, m8, m9, m10, m11, m12 = st.columns(6)
        with m7: metric_card("WA Rate", format_rate(an["wa_rate"]))
        with m8: metric_card("Rate Range", f"{format_rate(an['min_rate'])} – {format_rate(an['max_rate'])}")
        with m9: metric_card("WA FICO Orig", format_score(an["wa_fico_orig"]))
        with m10: metric_card("WA FICO Curr", format_score(an["wa_fico_curr"]))
        with m11: metric_card("WA DTI", format_pct(an["wa_dti"]))
        with m12: metric_card("HHI", f"{an['hhi']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # FICO & Rate distribution charts
        ch1, ch2 = st.columns(2)
        with ch1:
            fig = strat_chart(an["fico_dist"], "FICO Distribution", "#4D9EFF")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="chart_1")
        with ch2:
            fig = strat_chart(an["rate_dist"], "Rate Distribution", "#00D4AA")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="chart_2")

        # Status & Purpose
        ch3, ch4 = st.columns(2)
        with ch3:
            fig = strat_chart(an["stat"][:10], "Loan Status", "#FFB224")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="chart_3")
        with ch4:
            fig = strat_chart(an["purp"][:10], "Loan Purpose", "#A78BFA")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="chart_4")


# ── TAB 2: TAPE CRACKING ──
with tabs[2]:
    if an is None:
        st.info("Map columns first to see analytics.")
    else:
        st.markdown('<div class="section-header">Tape Cracking</div>', unsafe_allow_html=True)

        # DPD metrics
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        with d1: metric_card("30+ DPD", format_pct(an["dq30"]), "of balance")
        with d2: metric_card("60+ DPD", format_pct(an["dq60"]), "of balance")
        with d3: metric_card("90+ DPD", format_pct(an["dq90"]), "of balance")
        with d4: metric_card("Gross Loss Rate", format_pct(an["gross_loss_rate"]), "of orig bal")
        with d5: metric_card("Net Loss", format_currency(an["net_loss"]))
        with d6: metric_card("Recoveries", format_currency(an["recoveries"]))

        st.markdown("<br>", unsafe_allow_html=True)

        # Status table with drill-down
        st.markdown("**Loan Status Breakdown**")
        if an["stat"]:
            stat_df = strat_table(an["stat"])
            st.dataframe(stat_df, use_container_width=True, hide_index=True)

        # Vintage table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Vintage Distribution**")
        if an["vint"]:
            vint_df = strat_table(an["vint"])
            st.dataframe(vint_df, use_container_width=True, hide_index=True)


# ── TAB 3: STRATIFICATIONS ──
with tabs[3]:
    if an is None:
        st.info("Map columns first to see analytics.")
    else:
        st.markdown('<div class="section-header">Stratifications</div>', unsafe_allow_html=True)
        st.markdown('<span style="color:#566375;font-size:11px">Click any chart bar or table row to explore</span>', unsafe_allow_html=True)

        strats = [
            ("FICO Distribution", an["fico_dist"], "#4D9EFF"),
            ("Rate Distribution", an["rate_dist"], "#00D4AA"),
            ("DTI Distribution", an["dti_dist"], "#FFB224"),
            ("Term Distribution", an["term_dist"], "#FF4D6A"),
        ]

        for i in range(0, len(strats), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i + j < len(strats):
                    name, data, color = strats[i + j]
                    with col:
                        fig = strat_chart(data, name, color)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"strat_{i+j}")
                        st.dataframe(strat_table(data), use_container_width=True, hide_index=True)

        # Categorical strats
        cat_strats = [
            ("Purpose", an["purp"], "#A78BFA"),
            ("Grade", an["grad"], "#00D4AA"),
            ("Channel", an["chan"], "#4D9EFF"),
            ("Verification", an["veri"], "#FFB224"),
            ("Housing", an["hous"], "#FF4D6A"),
            ("Geography (Top 20)", an["geo"], "#4D9EFF"),
        ]

        st.markdown("<br>", unsafe_allow_html=True)
        for i in range(0, len(cat_strats), 2):
            c1, c2 = st.columns(2)
            for j, col in enumerate([c1, c2]):
                if i + j < len(cat_strats):
                    name, data, color = cat_strats[i + j]
                    with col:
                        if data:
                            fig = strat_chart(data, name, color)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"cat_{i+j}")
                            st.dataframe(strat_table(data), use_container_width=True, hide_index=True)


# ── TAB 4: CUSTOM STRATS ──
with tabs[4]:
    if an is None:
        st.info("Map columns first.")
    else:
        st.markdown('<div class="section-header">Custom Stratification Builder</div>', unsafe_allow_html=True)

        cs1, cs2, cs3 = st.columns([2, 1, 1])
        with cs1:
            strat_col = st.selectbox("Select column", hdrs, key="custom_strat_col")
        with cs2:
            strat_type = st.radio("Type", ["Categorical", "Numeric Buckets"], horizontal=True, key="custom_strat_type")
        with cs3:
            if strat_col:
                samples = df[strat_col].dropna().head(5).tolist()
                st.markdown(f'<span style="color:#566375;font-size:10px">Samples: {", ".join(str(s) for s in samples)}</span>', unsafe_allow_html=True)

        if strat_col:
            if strat_type == "Categorical":
                vals = df[strat_col].fillna("Unknown").astype(str).str.strip()
                bals = get_numeric(df, mp, "current_balance").fillna(0)
                grouped = pd.DataFrame({"val": vals, "bal": bals})
                agg = grouped.groupby("val").agg(count=("bal", "size"), balance=("bal", "sum")).reset_index()
                tb = bals.sum()
                agg["pct"] = agg["balance"] / tb * 100 if tb > 0 else 0
                agg = agg.sort_values("balance", ascending=False)

                chart_data = [{"name": r["val"], "count": int(r["count"]), "balance": r["balance"], "pct": r["pct"]}
                              for _, r in agg.iterrows()]
                fig = strat_chart(chart_data[:20], f"{strat_col} — Categorical", "#A78BFA")
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="chart_7")
                st.dataframe(strat_table(chart_data), use_container_width=True, hide_index=True)

            else:
                st.markdown("Define numeric buckets:")
                n_buckets = st.number_input("Number of buckets", 2, 10, 4)
                bucket_defs = []
                for i in range(n_buckets):
                    bc1, bc2, bc3 = st.columns([2, 1, 1])
                    with bc1:
                        bname = st.text_input(f"Label {i+1}", f"Bucket {i+1}", key=f"bn_{i}")
                    with bc2:
                        bmin = st.number_input(f"Min {i+1}", value=float(i * 100), key=f"bmin_{i}")
                    with bc3:
                        bmax = st.number_input(f"Max {i+1}", value=float((i + 1) * 100), key=f"bmax_{i}")
                    bucket_defs.append({"label": bname, "min": bmin, "max": bmax})

                if st.button("Build Stratification"):
                    fake_mp = {"current_balance": mp.get("current_balance", ""), "__custom": strat_col}
                    vals = df[strat_col].apply(parse_numeric)
                    bals = get_numeric(df, mp, "current_balance").fillna(0)
                    tb = bals.sum()
                    results = []
                    for b in bucket_defs:
                        mask = (vals >= b["min"]) & (vals <= b["max"]) & vals.notna()
                        bal = bals[mask].sum()
                        results.append({
                            "name": b["label"], "count": int(mask.sum()),
                            "balance": bal, "pct": (bal / tb * 100) if tb > 0 else 0,
                        })
                    fig = strat_chart(results, f"{strat_col} — Custom Buckets", "#FF4D6A")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="chart_8")
                    st.dataframe(strat_table(results), use_container_width=True, hide_index=True)


# ── TAB 5: CHARTS & REGRESSION ──
with tabs[5]:
    if an is None:
        st.info("Map columns first.")
    else:
        st.markdown('<div class="section-header">Charts & Regression</div>', unsafe_allow_html=True)

        numeric_cols = [h for h in hdrs if df[h].apply(parse_numeric).notna().sum() > len(df) * 0.3]

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            x_col = st.selectbox("X axis", numeric_cols, key="reg_x")
        with rc2:
            y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="reg_y")
        with rc3:
            color_col = st.selectbox("Color by (optional)", ["None"] + hdrs, key="reg_color")
        with rc4:
            z_col = st.selectbox("Z (multiple reg)", ["None"] + numeric_cols, key="reg_z")

        show_reg = st.checkbox("Show Regression Line", value=True)

        if x_col and y_col:
            x_vals = df[x_col].apply(parse_numeric)
            y_vals = df[y_col].apply(parse_numeric)
            mask = x_vals.notna() & y_vals.notna()

            if mask.sum() < 3:
                st.warning("Need at least 3 valid data points.")
            else:
                plot_df = pd.DataFrame({"x": x_vals[mask], "y": y_vals[mask]})
                if color_col != "None":
                    plot_df["color"] = df.loc[mask, color_col].fillna("Unknown").astype(str)

                # Sample if too many points
                if len(plot_df) > 2000:
                    plot_df = plot_df.sample(2000, random_state=42)

                if color_col != "None":
                    fig = px.scatter(plot_df, x="x", y="y", color="color",
                                     labels={"x": x_col, "y": y_col, "color": color_col},
                                     opacity=0.6)
                else:
                    fig = px.scatter(plot_df, x="x", y="y",
                                     labels={"x": x_col, "y": y_col},
                                     color_discrete_sequence=["#00D4AA"], opacity=0.6)

                # Regression line
                reg = None
                if show_reg:
                    x_arr = plot_df["x"].values
                    y_arr = plot_df["y"].values
                    reg = calc_regression(x_arr, y_arr)
                    if reg:
                        x_line = np.linspace(reg["x_min"], reg["x_max"], 100)
                        y_line = reg["slope"] * x_line + reg["intercept"]
                        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                                  line=dict(color="#FF4D6A", dash="dash", width=2),
                                                  name="OLS Fit"))

                fig.update_layout(
                    plot_bgcolor="#12161C", paper_bgcolor="#0B0E11",
                    font_color="#8494A7", height=450,
                    xaxis=dict(showgrid=True, gridcolor="#1E2530"),
                    yaxis=dict(showgrid=True, gridcolor="#1E2530"),
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_9")

                # Regression stats
                if reg and show_reg:
                    rs1, rs2, rs3, rs4 = st.columns(4)
                    with rs1: metric_card("R²", f"{reg['r2']:.4f}")
                    with rs2: metric_card("r", f"{reg['r']:.4f}")
                    with rs3: metric_card("Slope", f"{reg['slope']:.4f}")
                    with rs4: metric_card("Intercept", f"{reg['intercept']:.4f}")

                    strength = "Strong" if abs(reg["r"]) > 0.7 else "Moderate" if abs(reg["r"]) > 0.4 else "Weak"
                    direction = "positive" if reg["r"] > 0 else "negative"
                    st.markdown(f'<span style="color:#8494A7;font-size:12px">{strength} {direction} correlation (n={reg["n"]:,}). y = {reg["slope"]:.4f}x + {reg["intercept"]:.4f}</span>', unsafe_allow_html=True)

                # Multiple regression
                if z_col != "None":
                    z_vals = df[z_col].apply(parse_numeric)
                    mask3 = x_vals.notna() & y_vals.notna() & z_vals.notna()
                    if mask3.sum() >= 5:
                        mreg = calc_multi_regression(
                            x_vals[mask3].values, z_vals[mask3].values, y_vals[mask3].values
                        )
                        if mreg:
                            st.markdown("---")
                            st.markdown(f'<div class="section-header" style="font-size:14px">Multiple Regression: {y_col} = β₀ + β₁·{x_col} + β₂·{z_col}</div>', unsafe_allow_html=True)
                            mr1, mr2, mr3, mr4, mr5 = st.columns(5)
                            with mr1: metric_card("β₀", f"{mreg['b0']:.4f}")
                            with mr2: metric_card(f"β₁ ({x_col})", f"{mreg['b1']:.4f}")
                            with mr3: metric_card(f"β₂ ({z_col})", f"{mreg['b2']:.4f}")
                            with mr4: metric_card("R²", f"{mreg['r2']:.4f}")
                            with mr5: metric_card("Adj R²", f"{mreg['adj_r2']:.4f}")


# ── TAB 6: DATA QUALITY ──
with tabs[6]:
    if vl is None:
        st.info("Map columns first.")
    else:
        st.markdown('<div class="section-header">Data Quality</div>', unsafe_allow_html=True)

        comp = vl["completeness"]
        grade = "A" if comp >= 95 else "B" if comp >= 85 else "C" if comp >= 70 else "D"
        grade_color = "#00D4AA" if grade in "AB" else "#FFB224" if grade == "C" else "#FF4D6A"

        q1, q2, q3, q4 = st.columns(4)
        with q1: metric_card("Quality Grade", grade, f"Completeness: {comp:.1f}%")
        with q2: metric_card("Missing Values", f"{vl['missing_count']:,}")
        with q3: metric_card("Out of Range", f"{vl['oor_count']:,}")
        with q4: metric_card("Fields Mapped", f"{mapped_count}/{total_fields}")

        if vl["issues"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Issues Found:**")
            for issue in vl["issues"]:
                st.markdown(f'<span style="color:#FF4D6A;font-size:11px">⚠ {issue}</span>', unsafe_allow_html=True)

        # Field mapping audit
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Field Mapping Audit**")
        audit_data = []
        for fk, fdef in all_fields().items():
            mapped_to = mp.get(fk, "—")
            status = "✅" if fk in mp and mp[fk] in hdrs else "❌"
            audit_data.append({"Field": fk, "Label": fdef["label"], "Mapped To": mapped_to, "Status": status})
        st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)


# ── TAB 7: CONCENTRATION ──
with tabs[7]:
    if an is None:
        st.info("Map columns first.")
    else:
        st.markdown('<div class="section-header">Concentration Analysis</div>', unsafe_allow_html=True)

        cn1, cn2, cn3 = st.columns(3)
        with cn1:
            hhi_class = "Low" if an["hhi"] < 0.15 else "Moderate" if an["hhi"] < 0.25 else "High"
            metric_card("Geographic HHI", f"{an['hhi']:.4f}", hhi_class)
        with cn2:
            metric_card("Top State", f"{an['geo'][0]['name'] if an['geo'] else '—'}", f"{an['top_state_conc']:.1f}% of balance")
        with cn3:
            metric_card("States Represented", f"{len(an['geo'])}")

        st.markdown("<br>", unsafe_allow_html=True)
        if an["geo"]:
            fig = strat_chart(an["geo"], "Geographic Distribution", "#4D9EFF")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="chart_10")
            st.dataframe(strat_table(an["geo"]), use_container_width=True, hide_index=True)


# ── TAB 8: RISK SUMMARY ──
with tabs[8]:
    if an is None:
        st.info("Map columns first.")
    else:
        st.markdown('<div class="section-header">Risk Dashboard</div>', unsafe_allow_html=True)

        def risk_light(val, green_max, yellow_max):
            if val <= green_max: return "🟢"
            if val <= yellow_max: return "🟡"
            return "🔴"

        ri1, ri2 = st.columns(2)
        with ri1:
            st.markdown("**Credit Risk**")
            st.markdown(f'{risk_light(an["wa_fico_orig"], 999, 660)} WA FICO Orig: **{format_score(an["wa_fico_orig"])}**')
            st.markdown(f'{risk_light(an["wa_dti"], 35, 45)} WA DTI: **{format_pct(an["wa_dti"])}**')
            st.markdown(f'{risk_light(an["wa_rate"], 15, 20)} WA Rate: **{format_rate(an["wa_rate"])}**')

        with ri2:
            st.markdown("**Performance Risk**")
            st.markdown(f'{risk_light(an["dq30"], 3, 8)} 30+ DPD: **{format_pct(an["dq30"])}**')
            st.markdown(f'{risk_light(an["dq90"], 1, 5)} 90+ DPD: **{format_pct(an["dq90"])}**')
            st.markdown(f'{risk_light(an["net_loss_rate"], 2, 5)} Net Loss Rate: **{format_pct(an["net_loss_rate"])}**')

        st.markdown("<br>", unsafe_allow_html=True)
        ri3, ri4 = st.columns(2)
        with ri3:
            st.markdown("**Concentration Risk**")
            st.markdown(f'{risk_light(an["hhi"], 0.15, 0.25)} HHI: **{an["hhi"]:.4f}**')
            st.markdown(f'{risk_light(an["top_state_conc"], 20, 35)} Top State: **{format_pct(an["top_state_conc"])}**')

        with ri4:
            st.markdown("**Data Quality**")
            comp = vl["completeness"] if vl else 0
            oor = vl["oor_count"] if vl else 0
            st.markdown(f'{risk_light(100-comp, 5, 15)} Completeness: **{format_pct(comp)}**')
            st.markdown(f'{risk_light(oor, 10, 50)} Out of Range: **{oor}**')


# ── TAB 9: RAW DATA ──
with tabs[9]:
    st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)

    search = st.text_input("🔍 Search", placeholder="Filter across all columns...", key="raw_search")
    display_df = df
    if search:
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        display_df = df[mask]

    st.markdown(f'<span style="color:#566375;font-size:11px">Showing {len(display_df):,} of {len(df):,} rows · {len(hdrs)} columns</span>', unsafe_allow_html=True)
    st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)

    csv = display_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"loan_tape_export.csv", "text/csv")


# ── TAB 10: ADMIN / TEMPLATES ──
with tabs[10]:
    admin_tab = st.radio("Section", ["Standard Fields", "Templates"], horizontal=True)

    if admin_tab == "Standard Fields":
        st.markdown('<div class="section-header">Standard Fields Manager</div>', unsafe_allow_html=True)

        # Show all fields
        fields_data = []
        for fk, fdef in STD_FIELDS.items():
            fields_data.append({"Key": fk, "Label": fdef["label"], "Patterns": len(fdef["patterns"]), "Type": "Standard"})
        for cf in st.session_state.custom_fields:
            fields_data.append({"Key": cf["key"], "Label": cf["label"], "Patterns": len(cf.get("patterns", [])), "Type": "★ Custom"})

        st.dataframe(pd.DataFrame(fields_data), use_container_width=True, hide_index=True)

        # Add custom field
        st.markdown("---")
        st.markdown("**Add Custom Field**")
        af1, af2, af3 = st.columns(3)
        with af1:
            new_key = st.text_input("Field key", placeholder="e.g. vehicle_make")
        with af2:
            new_label = st.text_input("Display label", placeholder="e.g. Vehicle Make")
        with af3:
            new_patterns = st.text_input("Alternate names (comma-separated)", placeholder="e.g. make, manufacturer")

        if st.button("➕ Add Custom Field") and new_key and new_label:
            pats = [p.strip() for p in new_patterns.split(",") if p.strip()] if new_patterns else []
            existing_keys = set(STD_FIELDS.keys()) | {cf["key"] for cf in st.session_state.custom_fields}
            if new_key in existing_keys:
                st.error(f"Key '{new_key}' already exists.")
            else:
                st.session_state.custom_fields.append({"key": new_key, "label": new_label, "patterns": pats})
                st.toast(f"✅ Added custom field: {new_label}")
                st.rerun()

        # Delete custom fields
        if st.session_state.custom_fields:
            st.markdown("---")
            st.markdown("**Delete Custom Field**")
            del_key = st.selectbox("Select field to delete",
                                    [cf["key"] for cf in st.session_state.custom_fields])
            if st.button("🗑️ Delete") and del_key:
                st.session_state.custom_fields = [cf for cf in st.session_state.custom_fields if cf["key"] != del_key]
                st.session_state.mapping.pop(del_key, None)
                run_analysis()
                st.toast(f"Deleted field: {del_key}")
                st.rerun()

    else:
        st.markdown('<div class="section-header">Originator Templates</div>', unsafe_allow_html=True)

        if not st.session_state.templates:
            st.info("No templates saved yet. Save one from the Column Mapping tab.")
        else:
            for i, tpl in enumerate(st.session_state.templates):
                tc1, tc2, tc3, tc4 = st.columns([3, 2, 1, 1])
                with tc1:
                    st.markdown(f'<span style="color:#E8ECF1;font-weight:600">{tpl["name"]}</span>', unsafe_allow_html=True)
                with tc2:
                    st.markdown(f'<span style="color:#8494A7">{tpl["originator"]} · {len(tpl["mapping"])} fields</span>', unsafe_allow_html=True)
                with tc3:
                    if st.button("📥 Export", key=f"exp_{i}"):
                        json_str = json.dumps(tpl, indent=2)
                        st.download_button("Download JSON", json_str, f"{tpl['name']}.json", "application/json", key=f"dl_{i}")
                with tc4:
                    if st.button("🗑️ Delete", key=f"del_tpl_{i}"):
                        st.session_state.templates.pop(i)
                        st.toast(f"Deleted template: {tpl['name']}")
                        st.rerun()
