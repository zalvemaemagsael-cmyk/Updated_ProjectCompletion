"""
ASENXO MSME Project Completion Risk Management Dashboard
----------------------------------------------------------
Monitor -> Predict -> Explain Risk -> Prioritize -> Intervene -> Monitor

Loads a pre-trained Logistic Regression pipeline (StandardScaler + OneHotEncoder
-> Logistic Regression) and a one-row-per-project MSME dataset, scores every
project on Completion_Probability, and surfaces which projects are at risk of
not completing, why, and which need intervention.

NOTE ON THIS VERSION: the dataset backing this dashboard is now one row per
project (no monthly panel), and the model uses 11 predictors (7 numeric +
4 categorical: Province, Sector, type_of_ownership, size_of_enterprise).
There is no time dimension per project, so month-over-month trend charts and
"declining probability" / "persistent risk" flags from a panel-data design
are not possible here -- see the Early Warning and Project Risk Profile
sections for how those were adapted to this dataset instead of faked.
"""

from pathlib import Path
import pickle
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# Paths & page config
# ==========================================================================
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "MSME_filled.xlsx"
MODEL_PATH = APP_DIR / "logistic_regression_model.pkl"

st.set_page_config(
    page_title="ASENXO | MSME Project Completion Risk Management Dashboard",
    page_icon="\U0001F3DB\uFE0F",
    layout="wide",
)

# --------------------------------------------------------------------------
# Palette (DOST-style: clean light interface, navy/blue accents)
# --------------------------------------------------------------------------
NAVY = "#0B2A4A"
BLUE = "#1F6FB2"
INK = "#1B2A38"
MUTED = "#5B6B79"

TIER_ORDER = ["Critical", "High", "Medium", "Low"]
TIER_COLORS = {
    "Critical": "#B22222",
    "High": "#D97706",
    "Medium": "#C99A00",
    "Low": "#1E7A46",
}
TIER_BG = {
    "Critical": "#FBEAEA",
    "High": "#FDF1E1",
    "Medium": "#FBF6DD",
    "Low": "#E9F6EE",
}
SUGGESTED_ACTION = {
    "Critical": "Immediate case review \u2014 prioritize for intervention",
    "High": "Enhanced monitoring \u2014 schedule a site visit / check-in",
    "Medium": "Periodic monitoring \u2014 watch for further decline",
    "Low": "Routine monitoring \u2014 no immediate action needed",
}

# Approximate provincial centroids, for visualization only (not official
# boundary data). "Negros" matches this dataset's category label.
PROVINCE_COORDS = {
    "Aklan": (11.8166, 122.0942),
    "Antique": (10.7847, 122.0932),
    "Capiz": (11.5588, 122.7373),
    "Iloilo": (10.7202, 122.5621),
    "Guimaras": (10.5931, 122.6316),
    "Negros": (10.6713, 123.0011),
}

NUMERIC_LABELS = {
    "Debt-service coverage": "Debt-Service Coverage (\u00d7)",
    "Penetration Rate": "Penetration Rate (%)",
    "net_profit_margin": "Net Profit Margin (%)",
    "Current capacity utilization rate": "Current Capacity Utilization (%)",
    "Total Employment": "Total Employment",
    "Total Assets": "Total Assets (\u20b1)",
    "Years in operation": "Years in Operation",
}
CATEGORICAL_COLS = ["Province", "Sector", "type_of_ownership", "size_of_enterprise"]

# --------------------------------------------------------------------------
# Light CSS polish
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 1.2rem; max-width: 1300px; }}
        div[data-testid="stMetric"] {{
            background: white; border: 1px solid #E3E8EE; border-radius: 10px;
            padding: 14px 16px 10px 16px;
        }}
        div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
        .asenxo-header {{
            background: linear-gradient(90deg, {NAVY} 0%, {BLUE} 100%);
            color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 6px;
        }}
        .asenxo-header h1 {{ margin: 0; font-size: 1.5rem; font-weight: 700; }}
        .asenxo-header p {{ margin: 4px 0 0 0; font-size: 0.92rem; opacity: 0.92; }}
        .workflow-strip {{
            color: {NAVY}; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.3px;
            margin: 10px 0 4px 0;
        }}
        .demo-banner {{
            background: #FDF1E1; border: 1px solid #E9C46A; color: #7A4B00;
            padding: 10px 16px; border-radius: 8px; font-size: 0.86rem; margin-bottom: 14px;
        }}
        .tier-pill {{
            display:inline-block; padding: 2px 10px; border-radius: 12px; color: white;
            font-size: 0.72rem; font-weight: 700;
        }}
        section[data-testid="stSidebar"] {{ background: #F7F9FB; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Loading (cached)
# ==========================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # No native unique ID in this dataset (Beneficiary_Name repeats for
    # beneficiaries with more than one project) -- Beneficiary_Name + Year
    # is unique in this file, so build a surrogate Project_ID from it.
    df["Project_ID"] = df["Beneficiary_Name"].astype(str) + "__" + df["Year"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def score_projects(df: pd.DataFrame, _model) -> tuple[pd.DataFrame, list, list, np.ndarray]:
    """Score every project and reconstruct the linear model's own contribution
    per transformed feature (numeric z-scores + active one-hot categories),
    so 'why flagged' is read directly off the fitted coefficients -- no SHAP
    needed for a linear model."""
    feat_cols = list(_model.feature_names_in_)
    numeric_cols = [c for c in feat_cols if c not in CATEGORICAL_COLS]

    X = df[feat_cols]
    proba = _model.predict_proba(X)[:, 1]  # P(Completed)

    pre = _model.named_steps["preprocessor"]
    logreg = _model.named_steps["logreg"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    out_names = pre.get_feature_names_out()
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]
    contrib = Xt * coefs  # shape (n_rows, n_transformed_features)

    out = df.copy()
    out["Completion_Probability"] = proba
    out["Noncompletion_Probability"] = 1 - proba
    out["Logit"] = intercept + contrib.sum(axis=1)

    # Flag rows where a categorical value wasn't seen during model training
    # (OneHotEncoder(handle_unknown="ignore") silently zeroes it out).
    unmapped = []
    for col in CATEGORICAL_COLS:
        prefix = f"cat__{col}_"
        idx = [i for i, n in enumerate(out_names) if n.startswith(prefix)]
        if idx:
            row_sum = Xt[:, idx].sum(axis=1)
            bad_rows = np.where(row_sum == 0)[0]
            for r in bad_rows:
                unmapped.append((df.iloc[r]["Project_ID"], col, df.iloc[r][col]))

    return out, feat_cols, numeric_cols, contrib, out_names, unmapped


def classify_tier(prob_pct: float, t_crit: float, t_high: float, t_med: float) -> str:
    if prob_pct < t_crit:
        return "Critical"
    if prob_pct < t_high:
        return "High"
    if prob_pct < t_med:
        return "Medium"
    return "Low"


def _numeric_phrase(col: str, value: float, mean: float) -> str:
    label = NUMERIC_LABELS.get(col, col)
    if col == "Debt-service coverage":
        if value < 1.0:
            return (
                f"Debt-service coverage is critically low at {value:.2f}\u00d7 \u2014 below the "
                f"1.0 break-even point needed to cover debt obligations."
            )
        return f"Debt-service coverage is soft at {value:.2f}\u00d7, below the typical {mean:.2f}\u00d7."
    if col == "Current capacity utilization rate":
        return f"Current capacity utilization is low at {value:.1f}%, versus a typical {mean:.1f}%."
    if col == "net_profit_margin":
        return f"Net profit margin is thin at {value:.1f}%, versus a typical {mean:.1f}%."
    if col == "Penetration Rate":
        return f"Market penetration rate is low at {value:.1f}%, versus a typical {mean:.1f}%."
    if col == "Years in operation":
        return f"The firm is relatively young at {value:.0f} year(s) in operation, versus a typical {mean:.1f}."
    if col == "Total Employment":
        return f"Total employment is low at {value:.0f}, versus a typical {mean:.0f}."
    if col == "Total Assets":
        return f"Total assets are low at \u20b1{value:,.0f}, versus a typical \u20b1{mean:,.0f}."
    return f"{label} is below the typical range ({value})."


def _categorical_phrase(col: str, value: str) -> str:
    label = {
        "Province": "province",
        "Sector": "sector",
        "type_of_ownership": "ownership type",
        "size_of_enterprise": "enterprise size",
    }.get(col, col)
    return f"Belongs to the '{value}' {label}, which the model associates with lower completion rates historically."


def build_reason(row_idx: int, out_names, contrib_row, feat_cols, numeric_cols, means_map, raw_row) -> str:
    if raw_row["Completion_Probability"] >= 0.5:
        return "No major red flags \u2014 fundamentals are in a healthy range."

    drivers = []
    for i, name in enumerate(out_names):
        c = contrib_row[i]
        if c >= 0:
            continue
        if name.startswith("num__"):
            col = name[len("num__"):]
            drivers.append((c, _numeric_phrase(col, raw_row[col], means_map.get(col, float("nan")))))
        elif name.startswith("cat__"):
            rest = name[len("cat__"):]
            # column names may themselves contain underscores, so match against known columns
            for col in CATEGORICAL_COLS:
                if rest.startswith(col + "_"):
                    val = rest[len(col) + 1:]
                    drivers.append((c, _categorical_phrase(col, val)))
                    break

    if not drivers:
        return "Flagged by the model, though no single factor dominates \u2014 worth a closer look."

    drivers.sort(key=lambda d: d[0])
    phrases = [drivers[0][1]]
    if len(drivers) > 1 and abs(drivers[1][0]) >= 0.35 * abs(drivers[0][0]):
        phrases.append(drivers[1][1])
    return " Also, ".join(phrases)


def project_label(row: pd.Series) -> str:
    return f"{row['Beneficiary_Name']} ({row['Year']})"


# ==========================================================================
# Load & score
# ==========================================================================
model = load_model(MODEL_PATH)
raw_df = load_data(DATA_PATH)
scored, feat_cols, numeric_cols, contrib_matrix, out_names, unmapped_categories = score_projects(raw_df, model)

# Means of numeric columns, used only to phrase "typical" comparisons in reasons.
_numeric_means = {c: scored[c].mean() for c in numeric_cols}

# ==========================================================================
# Header
# ==========================================================================
st.markdown(
    """
    <div class="asenxo-header">
        <h1>ASENXO \u2014 MSME Project Completion Risk Management Dashboard</h1>
        <p>Which MSME projects are at risk of not completing, why, and which require intervention.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="workflow-strip">Monitor \u2192 Predict \u2192 Explain Risk \u2192 Prioritize \u2192 Intervene \u2192 Monitor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="demo-banner">
        <b>DEMO MODE \u2014 Synthetic Data:</b> This dashboard uses synthetic data for demonstration.
        Predictions and province/sector patterns do not represent actual DOST statistics.
    </div>
    """,
    unsafe_allow_html=True,
)

if unmapped_categories:
    with st.expander(f"\u26A0\uFE0F Data quality note: {len(unmapped_categories)} record(s) contain category values not seen during model training", expanded=False):
        st.caption(
            "These fields were treated by the model as unmapped for that project (no push toward "
            "either outcome from that field), rather than dropped or guessed. Consider checking these "
            "for data-entry consistency (e.g. spelling variants of the same sector)."
        )
        st.dataframe(
            pd.DataFrame(unmapped_categories, columns=["Project ID", "Field", "Value in data"]),
            hide_index=True, width="stretch",
        )

# ==========================================================================
# Sidebar: filters + configurable thresholds
# ==========================================================================
st.sidebar.header("Filters")
provinces = st.sidebar.multiselect("Province", sorted(scored["Province"].unique()))
sectors = st.sidebar.multiselect("Sector", sorted(scored["Sector"].unique()))
ownerships = st.sidebar.multiselect("Ownership Type", sorted(scored["type_of_ownership"].unique()))
sizes = st.sidebar.multiselect("Enterprise Size", sorted(scored["size_of_enterprise"].str.title().unique()))
yr_min, yr_max = int(scored["Year"].min()), int(scored["Year"].max())
year_range = st.sidebar.slider("Year", yr_min, yr_max, (yr_min, yr_max))

st.sidebar.divider()
st.sidebar.header("Risk Classification Thresholds")
st.sidebar.caption("Proposed / System Risk Classification \u2014 not official DOST standards.")
t_crit = st.sidebar.slider("Critical / High boundary (%)", 0, 100, 25)
t_high = st.sidebar.slider("High / Medium boundary (%)", 0, 100, 50)
t_med = st.sidebar.slider("Medium / Low boundary (%)", 0, 100, 75)
t_crit, t_high, t_med = sorted([t_crit, t_high, t_med])
st.sidebar.caption(
    f"Critical: 0\u2013{t_crit-1}% \u00b7 High: {t_crit}\u2013{t_high-1}% \u00b7 "
    f"Medium: {t_high}\u2013{t_med-1}% \u00b7 Low: {t_med}\u2013100%"
)

scored["Completion_Pct"] = scored["Completion_Probability"] * 100
scored["Risk_Level"] = scored["Completion_Pct"].apply(lambda p: classify_tier(p, t_crit, t_high, t_med))
scored["Suggested_Action"] = scored["Risk_Level"].map(SUGGESTED_ACTION)
scored["Reason"] = [
    build_reason(i, out_names, contrib_matrix[i], feat_cols, numeric_cols, _numeric_means, scored.iloc[i])
    for i in range(len(scored))
]

# Historical track record per beneficiary (used only as a real, disclosed
# signal in Early Warning -- not fed to the model).
status_by_beneficiary = scored.groupby("Beneficiary_Name")["Completion_Status"].apply(list)

# Apply sidebar filters -> filtered view used across tabs
filt = scored.copy()
if provinces:
    filt = filt[filt["Province"].isin(provinces)]
if sectors:
    filt = filt[filt["Sector"].isin(sectors)]
if ownerships:
    filt = filt[filt["type_of_ownership"].isin(ownerships)]
if sizes:
    filt = filt[filt["size_of_enterprise"].str.title().isin(sizes)]
filt = filt[(filt["Year"] >= year_range[0]) & (filt["Year"] <= year_range[1])]

filt = filt.sort_values("Completion_Probability", ascending=True).reset_index(drop=True)
n_total_all = scored["Project_ID"].nunique()
n_total_filt = filt["Project_ID"].nunique()

if n_total_filt < n_total_all:
    st.caption(f"Showing {n_total_filt} of {n_total_all} total projects, based on active filters.")

# ==========================================================================
# Single-page layout (sections stacked top to bottom, no tabs)
# ==========================================================================
st.markdown(
    """
    <div style="font-size:0.85rem; margin: 4px 0 18px 0;">
        <a href="#risk-overview" style="margin-right:14px; text-decoration:none;">1. Risk Overview</a>
        <a href="#at-risk-projects" style="margin-right:14px; text-decoration:none;">2. At-Risk Projects</a>
        <a href="#project-risk-profile" style="margin-right:14px; text-decoration:none;">3. Project Risk Profile</a>
        <a href="#risk-analytics" style="margin-right:14px; text-decoration:none;">4. Risk Analytics</a>
        <a href="#early-warning" style="margin-right:14px; text-decoration:none;">5. Early Warning</a>
        <a href="#intervention-monitoring" style="text-decoration:none;">6. Intervention Monitoring</a>
    </div>
    """,
    unsafe_allow_html=True,
)


def tier_style(val):
    return f"color: {TIER_COLORS.get(val, INK)}; font-weight: 700;"


# --------------------------------------------------------------------------
# 1. RISK OVERVIEW
# --------------------------------------------------------------------------
st.header("1. Risk Overview", anchor="risk-overview")
with st.container():
    n_atrisk = int((filt["Risk_Level"] != "Low").sum())
    n_critical = int((filt["Risk_Level"] == "Critical").sum())
    avg_prob = filt["Completion_Probability"].mean() * 100 if len(filt) else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Unique Projects", f"{n_total_filt}")
    c2.metric("Avg. Completion Probability", f"{avg_prob:.1f}%")
    c3.metric("At-Risk Projects", f"{n_atrisk}", f"{(n_atrisk/n_total_filt*100 if n_total_filt else 0):.0f}% of total")
    c4.metric("Critical Projects", f"{n_critical}")

    st.markdown("##### Risk Distribution")
    dist = filt["Risk_Level"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int).reset_index()
    dist.columns = ["Risk_Level", "Projects"]
    fig_dist = px.bar(
        dist, x="Risk_Level", y="Projects", color="Risk_Level",
        color_discrete_map=TIER_COLORS, category_orders={"Risk_Level": TIER_ORDER}, text="Projects",
    )
    fig_dist.update_layout(
        showlegend=False, height=320, xaxis_title="", yaxis_title="Number of Projects",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig_dist.update_traces(textposition="outside")
    st.plotly_chart(fig_dist, width="stretch")

# --------------------------------------------------------------------------
# 2. AT-RISK PROJECTS
# --------------------------------------------------------------------------
st.divider()
st.header("2. At-Risk Projects", anchor="at-risk-projects")
with st.container():
    st.markdown("##### Searchable list of projects, sorted highest-risk first")
    colf1, colf2 = st.columns([3, 1])
    search = colf1.text_input("Search by beneficiary or Project ID", "")
    only_atrisk = colf2.checkbox("At-risk only (exclude Low)", value=True)

    table = filt.copy()
    if only_atrisk:
        table = table[table["Risk_Level"] != "Low"]
    if search:
        s = search.lower()
        table = table[
            table["Beneficiary_Name"].str.lower().str.contains(s)
            | table["Project_ID"].str.lower().str.contains(s)
        ]

    view = pd.DataFrame(
        {
            "Beneficiary": table["Beneficiary_Name"],
            "Project ID": table["Project_ID"],
            "Province": table["Province"],
            "Sector": table["Sector"],
            "Enterprise Size": table["size_of_enterprise"].str.title(),
            "Total Assets": table["Total Assets"].map(lambda v: f"\u20b1{v:,.0f}"),
            "Completion Probability": (table["Completion_Probability"] * 100).round(1),
            "Risk Level": table["Risk_Level"],
            "Suggested Action": table["Suggested_Action"],
        }
    )

    st.dataframe(
        view.style.map(tier_style, subset=["Risk Level"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Completion Probability": st.column_config.ProgressColumn(
                "Completion Probability", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    st.caption(
        f"{len(view)} project(s) shown. \"Total Assets\" is shown in place of Project Cost, which "
        f"is not present in this dataset. Suggested actions are system-generated prioritization aids, "
        f"not official DOST decisions."
    )

# --------------------------------------------------------------------------
# 3. PROJECT RISK PROFILE
# --------------------------------------------------------------------------
st.divider()
st.header("3. Project Risk Profile", anchor="project-risk-profile")
with st.container():
    if filt.empty:
        st.info("No projects match the current filters.")
    else:
        options = filt["Project_ID"].tolist()
        labels = {pid: project_label(filt[filt["Project_ID"] == pid].iloc[0]) for pid in options}
        selected_pid = st.selectbox("Select a project", options, format_func=lambda pid: labels[pid])
        prow = filt[filt["Project_ID"] == selected_pid].iloc[0]
        row_idx = scored.index[scored["Project_ID"] == selected_pid][0]
        tier = prow["Risk_Level"]

        info_col, prob_col = st.columns([1.3, 1])
        with info_col:
            st.markdown("###### Project Information")
            st.write(f"**Beneficiary:** {prow['Beneficiary_Name']}")
            st.write(f"**Project ID:** {prow['Project_ID']}")
            st.write(f"**Province:** {prow['Province']}  \u00b7  **Sector:** {prow['Sector']}")
            st.write(f"**Ownership Type:** {prow['type_of_ownership']}  \u00b7  **Enterprise Size:** {prow['size_of_enterprise'].title()}")
            st.write(f"**Years in Operation:** {prow['Years in operation']:.0f}")

        with prob_col:
            st.markdown("###### Model Output")
            st.markdown(
                f"""
                <div style="border-left:6px solid {TIER_COLORS[tier]}; background:{TIER_BG[tier]};
                            border-radius:8px; padding:12px 16px;">
                    <div style="font-size:0.8rem; color:{MUTED};">COMPLETION PROBABILITY</div>
                    <div style="font-size:1.9rem; font-weight:800; color:{NAVY};">{prow['Completion_Probability']*100:.1f}%</div>
                    <div style="font-size:0.85rem; color:{INK};">Non-completion probability: {prow['Noncompletion_Probability']*100:.1f}%</div>
                    <span class="tier-pill" style="background:{TIER_COLORS[tier]};">{tier.upper()} RISK</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("###### The Model's Numeric Predictors")
        ncols = st.columns(4)
        for i, col in enumerate(numeric_cols):
            val = prow[col]
            disp = f"\u20b1{val:,.0f}" if col == "Total Assets" else (f"{val:,.0f}" if col == "Total Employment" else f"{val:.2f}")
            ncols[i % 4].metric(NUMERIC_LABELS.get(col, col), disp)

        st.caption(f"**Why flagged:** {prow['Reason']}")

        st.markdown("###### Feature Contribution to This Project's Score")
        st.caption(
            "Each bar is that feature's standardized value (or active category) \u00d7 its fitted "
            "coefficient. Bars pointing left push toward 'Not Completed'; bars pointing right push "
            "toward 'Completed'."
        )
        row_contrib = contrib_matrix[row_idx]
        contrib_df = pd.DataFrame({"Feature": out_names, "Contribution": row_contrib})
        contrib_df = contrib_df[contrib_df["Contribution"] != 0].copy()

        def clean_name(n):
            if n.startswith("num__"):
                col = n[len("num__"):]
                return NUMERIC_LABELS.get(col, col)
            if n.startswith("cat__"):
                rest = n[len("cat__"):]
                for c in CATEGORICAL_COLS:
                    if rest.startswith(c + "_"):
                        return f"{c}: {rest[len(c)+1:]}"
                return rest
            return n

        contrib_df["Feature"] = contrib_df["Feature"].map(clean_name)
        contrib_df = contrib_df.sort_values("Contribution")
        contrib_df["Direction"] = np.where(contrib_df["Contribution"] < 0, "Toward Not Completed", "Toward Completed")
        fig_contrib = px.bar(
            contrib_df, x="Contribution", y="Feature", orientation="h", color="Direction",
            color_discrete_map={"Toward Not Completed": "#B22222", "Toward Completed": "#1E7A46"},
        )
        fig_contrib.update_layout(
            height=max(280, 28 * len(contrib_df)), xaxis_title="Contribution to log-odds",
            yaxis_title="", margin=dict(t=10, b=10, l=10, r=10), legend_title_text="",
        )
        st.plotly_chart(fig_contrib, width="stretch")
        st.caption(
            "This dataset has one record per project (no monthly history), so month-over-month "
            "trend charts are not available here \u2014 this contribution chart is the visual "
            "evidence behind the flag instead."
        )

# --------------------------------------------------------------------------
# 4. RISK ANALYTICS
# --------------------------------------------------------------------------
st.divider()
st.header("4. Risk Analytics", anchor="risk-analytics")
with st.container():
    st.markdown("##### Risk Patterns")
    dim = st.radio(
        "Break down by", ["Province", "Sector", "Enterprise Size", "Ownership Type", "Year"],
        horizontal=True, label_visibility="collapsed",
    )
    dim_col = {
        "Province": "Province", "Sector": "Sector", "Enterprise Size": "size_of_enterprise",
        "Ownership Type": "type_of_ownership", "Year": "Year",
    }[dim]

    base = filt.copy()
    if dim_col == "size_of_enterprise":
        base[dim_col] = base[dim_col].str.title()

    seg = (
        base.groupby(dim_col)
        .agg(Avg_Completion_Pct=("Completion_Probability", lambda s: s.mean() * 100),
             Projects=("Project_ID", "count"),
             At_Risk=("Risk_Level", lambda s: (s != "Low").sum()))
        .reset_index()
        .sort_values("Avg_Completion_Pct")
    )
    fig_seg = px.bar(
        seg, x="Avg_Completion_Pct", y=dim_col, orientation="h",
        text=seg["Avg_Completion_Pct"].round(1).astype(str) + "%",
        hover_data={"Projects": True, "At_Risk": True, "Avg_Completion_Pct": ":.1f"},
        color="Avg_Completion_Pct", color_continuous_scale=["#B22222", "#D97706", "#1E7A46"],
    )
    fig_seg.update_layout(
        height=max(320, 28 * len(seg)), xaxis_title="Average Completion Probability (%)", yaxis_title="",
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
    )
    fig_seg.update_traces(textposition="outside")
    st.plotly_chart(fig_seg, width="stretch")

    st.markdown("##### Geographic Risk Map (approximate province centroids)")
    geo = (
        filt.groupby("Province")
        .agg(Avg_Completion_Pct=("Completion_Probability", lambda s: s.mean() * 100),
             Projects=("Project_ID", "count"),
             At_Risk=("Risk_Level", lambda s: (s != "Low").sum()))
        .reset_index()
    )
    geo["lat"] = geo["Province"].map(lambda p: PROVINCE_COORDS.get(p, (None, None))[0])
    geo["lon"] = geo["Province"].map(lambda p: PROVINCE_COORDS.get(p, (None, None))[1])
    geo = geo.dropna(subset=["lat", "lon"])

    if geo.empty:
        st.info("No mapped provinces match the current filters.")
    else:
        map_kwargs = dict(
            data_frame=geo, lat="lat", lon="lon", size="Projects", color="Avg_Completion_Pct",
            color_continuous_scale=["#B22222", "#D97706", "#1E7A46"],
            hover_name="Province",
            hover_data={"Projects": True, "At_Risk": True, "Avg_Completion_Pct": ":.1f", "lat": False, "lon": False},
            zoom=6.4, size_max=32,
        )
        # plotly >= 5.24 renamed scatter_mapbox -> scatter_map; support either.
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(map_style="open-street-map", **map_kwargs)
        else:
            fig_map = px.scatter_mapbox(mapbox_style="open-street-map", **map_kwargs)
        fig_map.update_layout(
            height=420, margin=dict(t=10, b=10, l=10, r=10), coloraxis_colorbar_title="Avg. Completion %",
        )
        st.plotly_chart(fig_map, width="stretch")
        st.caption(
            "Bubble position uses approximate provincial centroids for visualization only, "
            "not official geographic boundary data."
        )

# --------------------------------------------------------------------------
# 5. EARLY WARNING
# --------------------------------------------------------------------------
st.divider()
st.header("5. Early Warning", anchor="early-warning")
with st.container():
    st.markdown("##### Automatically Flagged Projects")
    st.caption(
        "This dataset has one record per project (no monthly history), so the panel-data notions of "
        "'declining probability' and 'persistent risk over time' don't apply here. The flags below are "
        "adapted to what this dataset actually supports: current risk level, standing well below sector "
        "peers, and a beneficiary's own prior track record on other projects (a real, disclosed signal \u2014 "
        "not a model input)."
    )

    peer_gap_pts = st.slider("Below-peer-average flag threshold (percentage points)", 5, 40, 15)

    sector_avg = filt.groupby("Sector")["Completion_Pct"].transform("mean")
    filt = filt.assign(_sector_avg=sector_avg)

    flags_rows = []
    for _, r in filt.iterrows():
        flags = []
        if r["Risk_Level"] in ("High", "Critical"):
            flags.append(f"{r['Risk_Level']} risk currently")

        gap = r["_sector_avg"] - r["Completion_Pct"]
        if gap >= peer_gap_pts:
            flags.append(f"{gap:.0f} pts below its sector's average completion probability")

        others = [
            s for name2, s in zip(
                scored[scored["Beneficiary_Name"] == r["Beneficiary_Name"]]["Project_ID"],
                scored[scored["Beneficiary_Name"] == r["Beneficiary_Name"]]["Completion_Status"],
            )
            if name2 != r["Project_ID"]
        ]
        if any(s == "Not Completed" for s in others):
            flags.append("Beneficiary has another project on record marked 'Not Completed'")

        if flags:
            flags_rows.append(
                {
                    "Beneficiary": r["Beneficiary_Name"],
                    "Project ID": r["Project_ID"],
                    "Risk Level": r["Risk_Level"],
                    "Completion Probability": round(r["Completion_Pct"], 1),
                    "Flags": " | ".join(flags),
                }
            )

    if not flags_rows:
        st.success("No projects meet the early-warning criteria for the current filters and settings.")
    else:
        fdf = pd.DataFrame(flags_rows).sort_values("Completion Probability")
        st.dataframe(
            fdf.style.map(tier_style, subset=["Risk Level"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Completion Probability": st.column_config.ProgressColumn(
                    "Completion Probability", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
        st.caption(f"{len(fdf)} project(s) flagged.")

# --------------------------------------------------------------------------
# 6. INTERVENTION MONITORING
# --------------------------------------------------------------------------
st.divider()
st.header("6. Intervention Monitoring", anchor="intervention-monitoring")
with st.container():
    st.markdown("##### Intervention Monitoring Log")
    st.caption(
        "System suggestions for prioritization \u2014 not official DOST decisions. "
        "Status, personnel, follow-up dates, and remarks are editable and kept for this session."
    )

    at_risk_now = filt[filt["Risk_Level"] != "Low"].copy()

    if "intervention_log" not in st.session_state:
        st.session_state.intervention_log = pd.DataFrame(
            {
                "Project ID": at_risk_now["Project_ID"],
                "Beneficiary": at_risk_now["Beneficiary_Name"],
                "Risk Level": at_risk_now["Risk_Level"],
                "Suggested Action": at_risk_now["Suggested_Action"],
                "Intervention Status": "Not Started",
                "Assigned Personnel": "",
                "Follow-up Date": date.today(),
                "Remarks": "",
            }
        )

    if st.button("Sync newly at-risk projects into the log"):
        existing_ids = set(st.session_state.intervention_log["Project ID"])
        new_ones = at_risk_now[~at_risk_now["Project_ID"].isin(existing_ids)]
        if len(new_ones):
            addition = pd.DataFrame(
                {
                    "Project ID": new_ones["Project_ID"],
                    "Beneficiary": new_ones["Beneficiary_Name"],
                    "Risk Level": new_ones["Risk_Level"],
                    "Suggested Action": new_ones["Suggested_Action"],
                    "Intervention Status": "Not Started",
                    "Assigned Personnel": "",
                    "Follow-up Date": date.today(),
                    "Remarks": "",
                }
            )
            st.session_state.intervention_log = pd.concat(
                [st.session_state.intervention_log, addition], ignore_index=True
            )
            st.success(f"Added {len(new_ones)} newly at-risk project(s) to the log.")
        else:
            st.info("No new at-risk projects to add.")

    edited = st.data_editor(
        st.session_state.intervention_log,
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "Intervention Status": st.column_config.SelectboxColumn(
                "Intervention Status",
                options=["Not Started", "In Progress", "Completed", "Not Required"],
            ),
            "Follow-up Date": st.column_config.DateColumn("Follow-up Date"),
            "Risk Level": st.column_config.TextColumn("Risk Level", disabled=True),
            "Suggested Action": st.column_config.TextColumn("Suggested Action", disabled=True),
        },
        key="intervention_editor",
    )
    st.session_state.intervention_log = edited

    st.download_button(
        "Download intervention log (CSV)",
        edited.to_csv(index=False).encode("utf-8"),
        file_name="intervention_log.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Model: Logistic Regression (StandardScaler + OneHotEncoder, L2 penalty) on Debt-Service Coverage, "
    "Penetration Rate, Net Profit Margin, Current Capacity Utilization, Total Employment, Total Assets, "
    "Years in Operation, Province, Sector, Ownership Type, and Enterprise Size. "
    "Risk tiers shown are a Proposed / System Risk Classification, not official DOST standards."
)
