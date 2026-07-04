"""
Indigenous Australian Preventable Hospitalisations Dashboard
Streamlit app — full interactive analysis & LOS predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indigenous PPH · Australia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
REMOTENESS_ORDER = ["Major cities", "Inner Regional", "Outer Regional", "Remote", "Very Remote"]
SEIFA_ORDER = [
    "SEIFA 1 (Most disadvantaged)", "SEIFA 2", "SEIFA 3",
    "SEIFA 4", "SEIFA 5 (Least disadvantaged)",
]
YEAR_ORDER = ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18"]
CORE_CATS  = ["Acute PPH", "Chronic PPH", "Vaccine preventable PPH"]

REMOTENESS_COLORS = {
    "Major cities":   "#3B82F6",
    "Inner Regional": "#22C55E",
    "Outer Regional": "#F59E0B",
    "Remote":         "#F97316",
    "Very Remote":    "#EF4444",
}
CATEGORY_COLORS = {
    "Chronic PPH":             "#0D9488",
    "Acute PPH":               "#F87171",
    "Vaccine preventable PPH": "#F59E0B",
}
SEIFA_COLORS = {
    "SEIFA 1 (Most disadvantaged)": "#EF4444",
    "SEIFA 2":                       "#F97316",
    "SEIFA 3":                       "#EAB308",
    "SEIFA 4":                       "#22C55E",
    "SEIFA 5 (Least disadvantaged)": "#3B82F6",
}
PT = "plotly_white"   # plotly template

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

/* ── metric cards ── */
.kpi-row { display:flex; gap:14px; margin-bottom:6px; }
.kpi {
    flex:1; background:#fff; border:1px solid #E2E8F0;
    border-radius:9px; padding:18px 20px;
}
.kpi-label {
    font-size:11px; text-transform:uppercase; letter-spacing:.08em;
    color:#94A3B8; font-weight:700; margin-bottom:6px;
}
.kpi-value {
    font-size:30px; font-weight:900; color:#0F172A;
    font-variant-numeric:tabular-nums; line-height:1;
}
.kpi-value.blue  { color:#1D4ED8; }
.kpi-value.green { color:#15803D; }
.kpi-value.amber { color:#B45309; }
.kpi-value.red   { color:#DC2626; }
.kpi-sub { font-size:11.5px; color:#94A3B8; margin-top:4px; }

/* ── section eyebrow ── */
.eyebrow {
    font-size:11px; text-transform:uppercase; letter-spacing:.1em;
    color:#94A3B8; font-weight:700; margin-bottom:2px;
}

/* ── callout box ── */
.callout {
    background:#EFF6FF; border-left:3px solid #1D4ED8;
    border-radius:0 7px 7px 0; padding:13px 17px;
    font-size:14px; line-height:1.6; margin:8px 0 16px;
}

/* ── prediction hero ── */
.pred-card {
    background:linear-gradient(135deg,#0F172A 0%,#1E40AF 100%);
    border-radius:12px; padding:36px 28px; text-align:center; color:#fff;
}
.pred-days {
    font-size:72px; font-weight:900; line-height:1;
    font-variant-numeric:tabular-nums; letter-spacing:-.03em;
}
.pred-unit { font-size:18px; opacity:.65; margin-top:4px; }
.pred-context { font-size:13px; opacity:.55; margin-top:8px; }

/* ── table row highlight ── */
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached so they run once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading PHN dataset…")
def load_master():
    df = pd.read_csv("processed-datasets/master_data_frame.csv")
    df["remoteness_area"] = pd.Categorical(
        df["remoteness_area"], categories=REMOTENESS_ORDER, ordered=True
    )
    df["seifa_quintile"] = pd.Categorical(
        df["seifa_quintile"], categories=SEIFA_ORDER, ordered=True
    )
    rem_map = {r: i+1 for i, r in enumerate(REMOTENESS_ORDER)}
    df["remoteness_rank"] = df["remoteness_area"].astype(str).map(rem_map).astype(float)
    df["seifa_rank"] = df["seifa_quintile"].astype(str).str.extract(r"SEIFA\s+(\d+)")[0].astype(float)
    # Composite risk: normalise each driver 0–1 then weight
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0
    df["risk_score"] = (
        _norm(df["pph_per_100_residents"])    * 0.50 +
        _norm(df["remoteness_rank"])           * 0.35 +
        _norm(6 - df["seifa_rank"])            * 0.15
    ) * 100
    return df


@st.cache_data(show_spinner="Loading Indigenous dataset…")
def load_indigenous():
    df = pd.read_csv("processed-datasets/indigenous_pre_data_frame.csv", encoding="cp1252")
    df["year_num"]   = df["reporting_year"].astype(str).str[:4].astype(int)
    df["year_label"] = df["year_num"].apply(lambda y: f"{y}-{str(y+1)[2:]}")
    df["year_label"] = pd.Categorical(df["year_label"], categories=YEAR_ORDER, ordered=True)
    df = df[~df["pph_condition"].str.startswith("Total")].copy()
    df = df[df["pph_category"].isin(CORE_CATS)].copy()
    return df


@st.cache_resource(show_spinner="Training Random Forest model…")
def get_model():
    """Train the LOS Random Forest and return model + encoders + metrics."""
    df = pd.read_csv("processed-datasets/indigenous_pre_data_frame.csv", encoding="cp1252")
    df["year_num"] = df["reporting_year"].astype(str).str[:4].astype(int)
    df = df.dropna(subset=["avg_length_of_stay"]).copy()
    df = df[df["sex"] != "Persons"].copy()
    df = df[~df["pph_condition"].str.startswith("Total")].copy()

    le_sex  = LabelEncoder()
    le_cat  = LabelEncoder()
    le_cond = LabelEncoder()
    df["sex_enc"]  = le_sex.fit_transform(df["sex"])
    df["cat_enc"]  = le_cat.fit_transform(df["pph_category"])
    df["cond_enc"] = le_cond.fit_transform(df["pph_condition"])

    X = df[["year_num", "sex_enc", "cat_enc", "cond_enc"]]
    y = df["avg_length_of_stay"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    metrics = {
        "test_r2":  r2_score(y_te, y_pred),
        "test_mae": mean_absolute_error(y_te, y_pred),
    }
    return rf, le_sex, le_cat, le_cond, df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, value, sub="", colour=""):
    return f"""
<div class="kpi">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value {colour}">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>"""


def callout(text):
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(master, indig):
    st.markdown("## Overview")
    st.markdown(
        "Statistical analysis of Potentially Preventable Hospitalisations (PPH) "
        "among Indigenous Australians across **31 Primary Health Networks** and "
        "**32 clinical conditions**, 2012–2018."
    )
    st.markdown("---")

    # ── KPI row ──
    st.markdown(
        '<div class="kpi-row">'
        + kpi("PHNs analysed", "31", "Primary Health Networks")
        + kpi("Conditions", "32", "Across 3 categories")
        + kpi("OLS Model R²", "0.739", "PHN-level variance explained", "blue")
        + kpi("RF Model R²", "0.952", "LOS prediction · test set", "green")
        + kpi("Remoteness effect", "+2.82", "extra PPH per remoteness step", "amber")
        + kpi("NT PPH rate", "29.4", "highest in dataset · per 100", "red")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Average PPH Rate by Remoteness")
        rem_agg = (
            master.groupby("remoteness_area", observed=True)["pph_per_100_residents"]
            .mean().reset_index().rename(columns={"pph_per_100_residents": "mean_pph"})
        )
        fig = px.bar(
            rem_agg, x="remoteness_area", y="mean_pph",
            color="remoteness_area",
            color_discrete_map=REMOTENESS_COLORS,
            labels={"remoteness_area": "", "mean_pph": "Mean PPH per 100 Residents"},
            template=PT,
            category_orders={"remoteness_area": REMOTENESS_ORDER},
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f}<extra></extra>"
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Top 10 PHNs by Composite Risk Score")
        top10 = master.nlargest(10, "risk_score")
        fig2 = px.bar(
            top10.sort_values("risk_score"),
            x="risk_score", y="phn_name",
            color="remoteness_area",
            color_discrete_map=REMOTENESS_COLORS,
            orientation="h",
            labels={"risk_score": "Risk Score", "phn_name": "", "remoteness_area": "Remoteness"},
            template=PT,
        )
        fig2.update_layout(
            height=320, margin=dict(t=10, b=0),
            legend=dict(title="", font=dict(size=10), orientation="h", y=-0.2),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Key Findings")

    findings = [
        ("Geography dominates",
         "Remoteness explains more PHN-level variance than socioeconomic disadvantage. "
         "Each step further from a city → <strong>+2.82 PPH per 100 residents</strong> (p=0.011)."),
        ("Chronic burden is disproportionate",
         "Chronic conditions account for <strong>42% of admissions</strong> but <strong>56% of bed days</strong> — "
         "late-stage presentations demand far more resources per episode."),
        ("Severity ≠ Volume",
         "Gangrene (~15 days LOS) and nutritional deficiencies (~14 days) are the most resource-intensive "
         "despite not ranking in the top 10 by admission count."),
        ("No improvement 2012–2018",
         "Total PPH bed day burden remained <strong>broadly stable</strong> across the study period. "
         "No condition showed a sustained downward trend."),
    ]

    c1, c2 = st.columns(2)
    for i, (title, text) in enumerate(findings):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(
                f'<div class="callout"><strong>{title}:</strong> {text}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Bed days summary ──
    indig_p = indig[indig["sex"] == "Persons"]
    st.markdown("#### Condition Burden Summary")
    burden = (
        indig_p
        .groupby(["pph_condition", "pph_category"])[["number_of_pph", "total_pph_bed_days", "avg_length_of_stay"]]
        .agg({"number_of_pph": "sum", "total_pph_bed_days": "sum", "avg_length_of_stay": "mean"})
        .reset_index()
        .sort_values("number_of_pph", ascending=False)
    )
    burden.columns = ["Condition", "Category", "Total Admissions", "Total Bed Days", "Avg LOS (days)"]
    burden["Total Admissions"] = burden["Total Admissions"].map("{:,.0f}".format)
    burden["Total Bed Days"]   = burden["Total Bed Days"].map("{:,.0f}".format)
    burden["Avg LOS (days)"]   = burden["Avg LOS (days)"].map("{:.1f}".format)
    st.dataframe(burden, use_container_width=True, height=350, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — GEOGRAPHIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_geographic(master):
    st.markdown("## Geographic Analysis")
    st.markdown(
        "Explore how geographic remoteness and socioeconomic disadvantage shape PPH rates "
        "across Australia's 31 Primary Health Networks."
    )

    # ── Sidebar filters ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Filters**")
        rem_sel  = st.multiselect("Remoteness", REMOTENESS_ORDER, default=REMOTENESS_ORDER)
        seifa_sel = st.multiselect("SEIFA Quintile", SEIFA_ORDER, default=SEIFA_ORDER)

    filtered = master[
        master["remoteness_area"].isin(rem_sel) &
        master["seifa_quintile"].isin(seifa_sel)
    ]
    st.caption(f"Showing **{len(filtered)}** of {len(master)} PHNs")
    st.markdown("---")

    # ── OLS model (recompute on filtered set) ──
    if len(filtered) >= 5:
        X = filtered[["health_checks_per_1000_indigenous", "remoteness_rank", "seifa_rank"]].dropna()
        y = filtered.loc[X.index, "pph_per_100_residents"]
        if len(X) >= 5:
            ols = sm.OLS(y, sm.add_constant(X)).fit()
            r2  = ols.rsquared
            rem_coef = ols.params.get("remoteness_rank", None)
            vif_df = pd.DataFrame({
                "Feature": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            })
        else:
            r2, rem_coef, vif_df = None, None, None
    else:
        r2, rem_coef, vif_df = None, None, None

    if r2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OLS R²", f"{r2:.3f}", "variance explained")
        c2.metric("Remoteness β", f"+{rem_coef:.2f}" if rem_coef else "—", "PPH per step")
        c3.metric("Sample size", len(filtered), "PHNs")
        c4.metric("Degrees of freedom", max(len(filtered) - 4, 0), "residual df")
        st.markdown("")

    # ── Scatter + remoteness boxplot ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Health Checks vs PPH Rate")
        fig = px.scatter(
            filtered, x="health_checks_per_1000_indigenous", y="pph_per_100_residents",
            color="remoteness_area", hover_name="phn_name",
            color_discrete_map=REMOTENESS_COLORS,
            labels={
                "health_checks_per_1000_indigenous": "Health Checks per 1,000 Indigenous Residents",
                "pph_per_100_residents": "PPH Rate per 100 Residents",
                "remoteness_area": "Remoteness",
            },
            template=PT,
            category_orders={"remoteness_area": REMOTENESS_ORDER},
        )
        fig.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="white")))
        fig.update_layout(height=370, margin=dict(t=10),
                          legend=dict(orientation="h", y=-0.22, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive correlation reflects confounding — health checks target the same remote, high-need regions.")

    with col2:
        st.markdown("#### PPH Rate by Remoteness Category")
        fig2 = px.box(
            filtered, x="remoteness_area", y="pph_per_100_residents",
            color="remoteness_area", hover_name="phn_name",
            color_discrete_map=REMOTENESS_COLORS, points="all",
            labels={"remoteness_area": "", "pph_per_100_residents": "PPH per 100 Residents"},
            template=PT,
            category_orders={"remoteness_area": REMOTENESS_ORDER},
        )
        fig2.update_traces(marker=dict(size=8, opacity=0.75), jitter=0.25)
        fig2.update_layout(showlegend=False, height=370, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── SEIFA + correlation ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### PPH Rate by SEIFA Quintile")
        fig3 = px.box(
            filtered, x="seifa_quintile", y="pph_per_100_residents",
            color="seifa_quintile", hover_name="phn_name",
            color_discrete_map=SEIFA_COLORS, points="all",
            labels={"seifa_quintile": "", "pph_per_100_residents": "PPH per 100 Residents"},
            template=PT,
            category_orders={"seifa_quintile": SEIFA_ORDER},
        )
        fig3.update_traces(marker=dict(size=8, opacity=0.75), jitter=0.25)
        fig3.update_layout(showlegend=False, height=370, margin=dict(t=10))
        fig3.update_xaxes(tickangle=20, tickfont=dict(size=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Correlation Matrix")
        corr = filtered[[
            "pph_per_100_residents", "health_checks_per_1000_indigenous",
            "remoteness_pph_avg", "seifa_pph_avg",
        ]].rename(columns={
            "pph_per_100_residents": "PPH Rate",
            "health_checks_per_1000_indigenous": "Health Checks",
            "remoteness_pph_avg": "Remoteness PPH",
            "seifa_pph_avg": "SEIFA PPH",
        }).corr()
        fig4 = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", template=PT,
        )
        fig4.update_traces(textfont=dict(size=12))
        fig4.update_layout(height=370, margin=dict(t=10),
                            coloraxis_colorbar=dict(title="r", len=0.7))
        st.plotly_chart(fig4, use_container_width=True)

    # ── VIF table ──
    if vif_df is not None:
        st.markdown("---")
        st.markdown("#### Multicollinearity Check (VIF)")
        callout(
            "VIF > 10 indicates multicollinearity. Health checks and remoteness are co-located by "
            "design — MBS Item 715 check-ups are concentrated in the same remote areas with highest "
            "PPH rates. Both predictors are retained to expose this structural policy tension."
        )
        st.dataframe(
            vif_df.style.format({"VIF": "{:.1f}"})
            .highlight_between(subset=["VIF"], left=10, color="#FEE2E2"),
            use_container_width=False, hide_index=True,
        )

    # ── PHN data table ──
    st.markdown("---")
    st.markdown("#### PHN-Level Data")
    display = filtered[[
        "phn_name", "pph_per_100_residents", "health_checks_per_1000_indigenous",
        "remoteness_area", "seifa_quintile", "risk_score",
    ]].copy().sort_values("pph_per_100_residents", ascending=False)
    display.columns = ["PHN", "PPH / 100", "Health Checks / 1k", "Remoteness", "SEIFA", "Risk Score"]
    st.dataframe(
        display.style
        .background_gradient(subset=["PPH / 100", "Risk Score"], cmap="YlOrRd")
        .format({"PPH / 100": "{:.1f}", "Health Checks / 1k": "{:.1f}", "Risk Score": "{:.0f}"}),
        use_container_width=True, height=380, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CONDITION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_conditions(indig):
    st.markdown("## Condition-Level Analysis")
    st.markdown(
        "Explore which conditions drive the greatest hospitalisation burden for Indigenous Australians "
        "across 2012–2018 — by volume, bed days, severity, and sex."
    )

    indig_p = indig[indig["sex"] == "Persons"].copy()
    indig_s = indig[indig["sex"].isin(["Females", "Males"])].copy()

    # ── Sidebar filters ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Filters**")
        cat_sel  = st.multiselect("PPH Category", CORE_CATS, default=CORE_CATS)
        year_sel = st.multiselect("Year", YEAR_ORDER, default=YEAR_ORDER)
        top_n    = st.slider("Top N conditions", 5, 20, 10)

    fp = indig_p[
        indig_p["pph_category"].isin(cat_sel) &
        indig_p["year_label"].isin(year_sel)
    ]
    fs = indig_s[
        indig_s["pph_category"].isin(cat_sel) &
        indig_s["year_label"].isin(year_sel)
    ]
    st.caption(f"Filtered to **{len(fp)}** person-rows across selected categories and years.")
    st.markdown("---")

    # ── Top N conditions + LOS ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Top {top_n} Conditions — Total Admissions")
        top_vol = (
            fp.groupby(["pph_condition", "pph_category"])["number_of_pph"]
            .sum().reset_index()
            .sort_values("number_of_pph", ascending=False)
            .head(top_n)
        )
        fig = px.bar(
            top_vol.sort_values("number_of_pph"),
            x="number_of_pph", y="pph_condition",
            color="pph_category", color_discrete_map=CATEGORY_COLORS,
            orientation="h",
            labels={"number_of_pph": "Total Admissions", "pph_condition": "", "pph_category": "Category"},
            template=PT,
        )
        fig.update_layout(height=420, margin=dict(t=10),
                          legend=dict(orientation="h", y=-0.18, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### Top {top_n} Conditions — Average LOS")
        alos = (
            fp.groupby(["pph_condition", "pph_category"])["avg_length_of_stay"]
            .mean().reset_index().dropna()
            .sort_values("avg_length_of_stay", ascending=False)
            .head(top_n)
        )
        mean_los = alos["avg_length_of_stay"].mean()
        fig2 = px.bar(
            alos.sort_values("avg_length_of_stay"),
            x="avg_length_of_stay", y="pph_condition",
            color="pph_category", color_discrete_map=CATEGORY_COLORS,
            orientation="h",
            labels={"avg_length_of_stay": "Avg LOS (days)", "pph_condition": "", "pph_category": "Category"},
            template=PT,
        )
        fig2.add_vline(x=mean_los, line_dash="dash", line_color="#94A3B8",
                       annotation_text=f"Mean {mean_los:.1f}d", annotation_font_size=11)
        fig2.update_layout(height=420, margin=dict(t=10),
                           legend=dict(orientation="h", y=-0.18, font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Bed days trend + category breakdown ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Bed Day Burden Over Time (Top 5 Conditions)")
        top5_conds = (
            fp.groupby("pph_condition")["total_pph_bed_days"]
            .sum().nlargest(5).index.tolist()
        )
        trend = (
            fp[fp["pph_condition"].isin(top5_conds)]
            .groupby(["year_label", "pph_condition"], observed=True)["total_pph_bed_days"]
            .sum().reset_index()
        )
        fig3 = px.line(
            trend.sort_values("year_label"),
            x="year_label", y="total_pph_bed_days",
            color="pph_condition", markers=True,
            labels={"year_label": "Year", "total_pph_bed_days": "Total Bed Days", "pph_condition": "Condition"},
            template=PT,
        )
        fig3.update_layout(height=360, margin=dict(t=10),
                           legend=dict(font=dict(size=9), orientation="h", y=-0.3))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Admissions vs Bed Days by Category")
        cat_agg = (
            fp.groupby("pph_category")[["number_of_pph", "total_pph_bed_days"]]
            .sum().reset_index()
        )
        cat_agg["adm_pct"] = cat_agg["number_of_pph"]       / cat_agg["number_of_pph"].sum() * 100
        cat_agg["bed_pct"] = cat_agg["total_pph_bed_days"]  / cat_agg["total_pph_bed_days"].sum() * 100

        melted = pd.melt(
            cat_agg, id_vars="pph_category",
            value_vars=["adm_pct", "bed_pct"],
            var_name="Metric", value_name="Pct",
        )
        melted["Metric"] = melted["Metric"].map({"adm_pct": "Admissions", "bed_pct": "Bed Days"})

        fig4 = px.bar(
            melted, x="Metric", y="Pct", color="pph_category",
            color_discrete_map=CATEGORY_COLORS,
            text="Pct",
            labels={"pph_category": "Category", "Pct": "%", "Metric": ""},
            template=PT,
        )
        fig4.update_traces(texttemplate="%{text:.0f}%", textposition="inside",
                           insidetextanchor="middle")
        fig4.update_layout(height=360, margin=dict(t=10), yaxis=dict(range=[0, 110]),
                           legend=dict(font=dict(size=10)))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Sex comparison ──
    st.markdown(f"#### Sex Comparison — Age-Standardised PPH Rates (Top 8 Conditions)")
    top8 = (
        indig_p.groupby("pph_condition")["number_of_pph"]
        .sum().nlargest(8).index.tolist()
    )
    sex_data = (
        fs[fs["pph_condition"].isin(top8)]
        .groupby(["pph_condition", "sex"])["pph_age_standardised"]
        .mean().reset_index()
    )
    fig5 = px.bar(
        sex_data, x="pph_condition", y="pph_age_standardised",
        color="sex", barmode="group",
        color_discrete_map={"Females": "#F87171", "Males": "#0D9488"},
        labels={"pph_condition": "Condition",
                "pph_age_standardised": "Age-Standardised Rate (per 100,000)",
                "sex": "Sex"},
        template=PT,
    )
    fig5.update_layout(height=360, margin=dict(t=10), xaxis_tickangle=25)
    st.plotly_chart(fig5, use_container_width=True)

    callout(
        "<strong>Sex finding:</strong> Males have higher age-standardised rates for COPD (consistent with "
        "higher smoking prevalence). Diabetes complications show balanced rates — chronic disease programs "
        "for diabetes can be designed with a sex-neutral approach."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — LOS PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
def page_predictor():
    st.markdown("## Length of Stay Predictor")
    st.markdown(
        "Use the trained **Random Forest model** (Test R²=**0.952**, MAE=**0.34 days**) "
        "to predict average length of hospital stay for any PPH admission."
    )
    st.markdown("---")

    rf, le_sex, le_cat, le_cond, df_model, metrics = get_model()

    conditions  = sorted(df_model["pph_condition"].unique())
    years       = sorted(df_model["year_num"].unique())
    year_labels = {y: f"{y}–{str(y+1)[2:]}" for y in years}
    overall_avg = df_model["avg_length_of_stay"].mean()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Input Parameters")

        sel_cond = st.selectbox(
            "PPH Condition",
            conditions,
            index=conditions.index("Cellulitis") if "Cellulitis" in conditions else 0,
            help="Select the clinical condition for the PPH admission",
        )

        # Auto-filter categories valid for this condition
        valid_cats = df_model[df_model["pph_condition"] == sel_cond]["pph_category"].unique().tolist()
        sel_cat = st.selectbox("PPH Category", valid_cats)

        sel_sex = st.radio("Sex", ["Females", "Males"], horizontal=True)

        sel_year = st.select_slider(
            "Reporting Year",
            options=years,
            value=years[-1],
            format_func=lambda y: year_labels[y],
        )

        st.markdown("")
        predict = st.button("Predict Length of Stay", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("**Model performance**")
        mc1, mc2 = st.columns(2)
        mc1.metric("Test R²", f"{metrics['test_r2']:.3f}")
        mc2.metric("Test MAE", f"{metrics['test_mae']:.2f} days")

        st.markdown("**Feature importance**")
        feat_labels = ["Reporting Year", "Sex", "PPH Category", "PPH Condition"]
        importances = rf.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_labels, "Importance": importances}).sort_values("Importance")
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#BFDBFE", "#1D4ED8"],
            text=fi_df["Importance"].map(lambda v: f"{v*100:.1f}%"),
            template=PT,
        )
        fig_fi.update_coloraxes(showscale=False)
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            height=220, margin=dict(t=5, b=0, l=0, r=40),
            xaxis=dict(range=[0, importances.max() * 1.25]),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_result:
        st.markdown("### Prediction")

        if predict:
            try:
                sex_enc  = le_sex.transform([sel_sex])[0]
                cat_enc  = le_cat.transform([sel_cat])[0]
                cond_enc = le_cond.transform([sel_cond])[0]
            except ValueError as e:
                st.error(f"Encoding error: {e}")
                return

            pred = rf.predict([[sel_year, sex_enc, cat_enc, cond_enc]])[0]

            cond_avg = df_model[
                df_model["pph_condition"] == sel_cond
            ]["avg_length_of_stay"].mean()
            cond_los_vals = df_model[
                df_model["pph_condition"] == sel_cond
            ]["avg_length_of_stay"].dropna()

            # Prediction hero card
            st.markdown(f"""
<div class="pred-card">
  <div style="font-size:12px;opacity:.55;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">
    Predicted Average Length of Stay
  </div>
  <div class="pred-days">{pred:.1f}</div>
  <div class="pred-unit">days</div>
  <div class="pred-context">
    {sel_cond} &nbsp;·&nbsp; {sel_cat} &nbsp;·&nbsp; {sel_sex} &nbsp;·&nbsp; {year_labels[sel_year]}
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("")

            # Context metrics
            diff_cond  = pred - cond_avg
            diff_ovr   = pred - overall_avg
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "vs Condition Average",
                f"{cond_avg:.1f} days",
                f"{'+'if diff_cond>0 else ''}{diff_cond:.1f} days",
                delta_color="inverse" if diff_cond > 0 else "normal",
            )
            mc2.metric(
                "vs Overall Average",
                f"{overall_avg:.1f} days",
                f"{'+'if diff_ovr>0 else ''}{diff_ovr:.1f} days",
                delta_color="inverse" if diff_ovr > 0 else "normal",
            )
            mc3.metric("Model Error (±MAE)", f"{metrics['test_mae']:.2f} days")

            # Distribution context
            if len(cond_los_vals) > 0:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=cond_los_vals, nbinsx=15,
                    marker_color="#BFDBFE", name="Observed LOS",
                    hovertemplate="LOS: %{x:.1f}d<br>Count: %{y}<extra></extra>",
                ))
                fig_dist.add_vline(
                    x=pred, line_color="#1D4ED8", line_width=2.5, line_dash="solid",
                    annotation_text=f"  Prediction: {pred:.1f}d",
                    annotation_font=dict(color="#1D4ED8", size=11),
                    annotation_position="top right",
                )
                fig_dist.add_vline(
                    x=cond_avg, line_color="#94A3B8", line_width=1.5, line_dash="dash",
                    annotation_text=f"  Avg: {cond_avg:.1f}d",
                    annotation_font=dict(color="#94A3B8", size=10),
                    annotation_position="top left",
                )
                fig_dist.update_layout(
                    title=f"LOS Distribution — {sel_cond}",
                    xaxis_title="Average Length of Stay (days)",
                    yaxis_title="Count",
                    template=PT,
                    height=280,
                    margin=dict(t=40, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        else:
            st.info("Set the parameters on the left and click **Predict Length of Stay**.")

            # Show model comparison while waiting
            st.markdown("#### Four-Model Benchmark")
            perf = pd.DataFrame({
                "Model": ["Linear Regression", "Ridge Regression", "Random Forest ✓", "Gradient Boosting"],
                "Test R²": [0.108, 0.108, 0.952, 0.940],
                "Test MAE": [1.64, 1.65, 0.34, 0.39],
                "Test RMSE": [2.21, 2.21, 0.51, 0.57],
            })
            st.dataframe(
                perf.style
                .highlight_max(subset=["Test R²"], color="#DCFCE7")
                .highlight_min(subset=["Test MAE", "Test RMSE"], color="#DCFCE7")
                .format({"Test R²": "{:.3f}", "Test MAE": "{:.2f}", "Test RMSE": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )
            callout(
                "Tree models (R²≈0.95) vastly outperform linear regression (R²≈0.11). "
                "The LOS–condition relationship is <strong>strongly non-linear</strong> — "
                "the specific diagnosis is by far the dominant driver of hospital resource consumption."
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # Load data
    master = load_master()
    indig  = load_indigenous()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🏥 Indigenous PPH")
        st.markdown("Australia · 2012–2018")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Overview", "Geographic Analysis", "Condition Analysis", "LOS Predictor"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("**Source**\nAIHW HPF 50–51")
        st.caption("**Methods**\nOLS · Random Forest · 5-Fold CV")
        st.caption("**Coverage**\n31 PHNs · 32 conditions · 6 years")

    # Route
    if page == "Overview":
        page_overview(master, indig)
    elif page == "Geographic Analysis":
        page_geographic(master)
    elif page == "Condition Analysis":
        page_conditions(indig)
    elif page == "LOS Predictor":
        page_predictor()


if __name__ == "__main__":
    main()
