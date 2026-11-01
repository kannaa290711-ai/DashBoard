"""
AirAware: An Interactive Air Quality and Pollution Analytics Dashboard
======================================================================
B.Tech Data Science — College Demo Project

Tech stack : Python, Streamlit, Pandas, NumPy, Plotly
Data        : 100 % SYNTHETIC (generated inside this file with a fixed seed)

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AirAware — Air Quality Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"

# ──────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────
BG_MAIN   = "#14181F"
BG_SIDE   = "#11151B"
BG_CARD   = "#1C222B"
BORDER    = "#2A313C"
TEXT_MAIN = "#E6E8EB"
TEXT_MUT  = "#9AA3AF"
ACCENT    = "#4F9D9D"

# Per-city colours (stable across every chart)
_PALETTE = ["#5B8DB8", "#C9A24D", "#B5654A", "#6E9E78", "#8C7AA5",
            "#7FAEC4", "#D4B96A", "#C88068", "#89B592", "#A495B8"]

AQI_CLR = {
    "Good": "#5E9E6E", "Satisfactory": "#9DB35B", "Moderate": "#D1B04C",
    "Poor": "#D18B47", "Very Poor": "#B85C4A", "Severe": "#8C3F3F",
}

HEAT_SCALE = [[0, "#1C222B"], [0.25, "#2E3A40"], [0.5, "#5C6B55"],
              [0.75, "#B5974B"], [1, "#D4B96A"]]
MAP_SCALE  = [[0, "#2E3A40"], [0.35, "#5C6B55"], [0.65, "#B5974B"], [1, "#C9824A"]]

# ──────────────────────────────────────────────
# City metadata
# ──────────────────────────────────────────────
CITY_META = {
    "Delhi":         {"lat": 28.6139, "lon": 77.2090, "base": 1.45},
    "Mumbai":        {"lat": 19.0760, "lon": 72.8777, "base": 0.90},
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "base": 0.85},
    "Chennai":       {"lat": 13.0827, "lon": 80.2707, "base": 0.75},
    "Kolkata":       {"lat": 22.5726, "lon": 88.3639, "base": 1.20},
    "Bengaluru":     {"lat": 12.9716, "lon": 77.5946, "base": 0.70},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "base": 0.65},
    "Ahmedabad":     {"lat": 23.0225, "lon": 72.5714, "base": 1.00},
    "Pune":          {"lat": 18.5204, "lon": 73.8567, "base": 0.78},
    "Lucknow":       {"lat": 26.8467, "lon": 80.9462, "base": 1.30},
}

POLL_UNIT = {
    "pm2_5": "ug/m3", "pm10": "ug/m3", "nitrogen_dioxide": "ug/m3",
    "sulphur_dioxide": "ug/m3", "ozone": "ug/m3", "carbon_monoxide": "mg/m3",
}
POLL_LBL = {
    "pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2", "ozone": "O3", "carbon_monoxide": "CO",
}

# Navigation items
NAV_ITEMS = [
    "Overview", "Trends", "City Comparison", "Pollutants",
    "Heatmap", "Map", "Insights", "Data & Export", "About",
]
NAV_ICONS = {
    "Overview": "◈", "Trends": "◆", "City Comparison": "▬",
    "Pollutants": "◇", "Heatmap": "▦", "Map": "◎",
    "Insights": "▸", "Data & Export": "▤", "About": "○",
}

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}

/* Main background */
.stApp {{ background-color: {BG_MAIN}; color: {TEXT_MAIN}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {BG_SIDE};
    border-right: 1px solid {BORDER};
    padding-top: 0.5rem;
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {{ color: {TEXT_MUT}; font-size: 0.84rem; }}

/* ── Radio-as-nav: hide circles, style as menu ── */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
    display: none !important;           /* hide the radio group label */
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
    gap: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {{
    display: flex !important;
    align-items: center;
    padding: 9px 16px !important;
    margin: 0 !important;
    border-radius: 0 !important;
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    font-size: 0.88rem !important;
    color: {TEXT_MUT} !important;
    font-weight: 500 !important;
    background: transparent !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
    background: rgba(79,157,157,0.08) !important;
    color: {TEXT_MAIN} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
    border-left: 3px solid {ACCENT} !important;
    background: rgba(79,157,157,0.12) !important;
    color: {TEXT_MAIN} !important;
    font-weight: 600 !important;
}}
/* Hide the radio dot */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {{
    pointer-events: none;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] input[type="radio"] {{
    display: none !important;
}}
/* Hide the circle pseudo-element */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {{
    display: none !important;
}}

/* ── Multiselect tag override (remove red) ── */
span[data-baseweb="tag"] {{
    background-color: #2A3744 !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_MAIN} !important;
    border-radius: 4px !important;
}}
span[data-baseweb="tag"] svg {{ fill: {TEXT_MUT} !important; }}

/* ── Slider override ── */
div[data-baseweb="slider"] div[role="progressbar"] {{ background-color: {ACCENT} !important; }}
div[data-baseweb="slider"] div[data-testid="stThumbValue"] {{ color: {TEXT_MAIN} !important; }}

/* ── Buttons ── */
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {{
    background-color: #2A3744 !important; border: 1px solid {BORDER} !important;
    color: {TEXT_MAIN} !important; border-radius: 6px !important;
}}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background-color: {ACCENT} !important; border: none !important;
    color: #fff !important; border-radius: 6px !important;
}}

/* ── KPI cards ── */
div[data-testid="stMetric"] {{
    background-color: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 18px 20px 14px 20px;
}}
div[data-testid="stMetric"] label {{
    color: {TEXT_MUT} !important; font-size: 0.76rem !important;
    font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.04em;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: {TEXT_MAIN} !important; font-weight: 700 !important; font-size: 1.55rem !important;
}}

/* ── Expander ── */
details[data-testid="stExpander"] {{
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px;
}}

/* ── Custom classes ── */
.pg-heading {{ font-size: 22px; font-weight: 600; color: {TEXT_MAIN}; margin-bottom: 2px; }}
.pg-desc {{ font-size: 13px; color: {TEXT_MUT}; margin-top: 0; margin-bottom: 18px; }}
.chart-cap {{ color: {TEXT_MUT}; font-size: 13px; margin-top: 4px; line-height: 1.5; }}
.section-div {{ border: none; border-top: 1px solid {BORDER}; margin: 28px 0 24px 0; }}
.synth-badge {{
    background: #1E2630; border: 1px dashed #6B7280; color: {TEXT_MUT};
    border-radius: 6px; padding: 8px 12px; text-align: center;
    font-size: 0.76rem; margin-top: 10px;
}}
.aqi-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.74rem; font-weight: 600; color: #fff;
}}
.insight-card {{
    background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 8px; font-size: 0.88rem;
    color: {TEXT_MAIN}; line-height: 1.55;
}}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

/* ── Header / footer ── */
header[data-testid="stHeader"] {{ background-color: {BG_MAIN} !important; }}
footer {{ visibility: hidden; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_MAIN}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Plotly template
# ──────────────────────────────────────────────
PLT_TPL = go.layout.Template(layout=go.Layout(
    paper_bgcolor=BG_MAIN, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_MUT, family="Inter, Segoe UI, sans-serif", size=12),
    title_font=dict(color=TEXT_MAIN, size=14, family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title_font_color=TEXT_MUT),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title_font_color=TEXT_MUT),
    colorway=_PALETTE,
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color=TEXT_MUT, font_size=11,
                orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, title_text=""),
    margin=dict(l=48, r=24, t=55, b=60),
))


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def ccol(city: str, idx: int = 0) -> str:
    """Stable colour for a city."""
    all_c = sorted(CITY_META.keys())
    if city in all_c:
        return _PALETTE[all_c.index(city) % len(_PALETTE)]
    return _PALETTE[idx % len(_PALETTE)]

def cmap_for(cities) -> dict:
    """Colour map dict for a list of cities."""
    return {c: ccol(c, i) for i, c in enumerate(sorted(cities))}

def pg_head(title: str, desc: str = ""):
    st.markdown(f'<p class="pg-heading">{title}</p>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<p class="pg-desc">{desc}</p>', unsafe_allow_html=True)

def chart_cap(text: str):
    st.markdown(f'<p class="chart-cap">{text}</p>', unsafe_allow_html=True)

def aqi_badge(cat: str) -> str:
    c = AQI_CLR.get(cat, "#6B7280")
    return f'<span class="aqi-badge" style="background:{c};">{cat}</span>'

def _aqi_cat(v: float) -> str:
    if pd.isna(v): return "Unknown"
    if v <= 50:  return "Good"
    if v <= 100: return "Satisfactory"
    if v <= 200: return "Moderate"
    if v <= 300: return "Poor"
    if v <= 400: return "Very Poor"
    return "Severe"


# ══════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════

@st.cache_data
def generate_data() -> pd.DataFrame:
    """90 days x 24 hours x 10 cities of synthetic air-quality data (seed=42)."""
    rng = np.random.default_rng(seed=42)
    start = datetime(2024, 1, 1)
    hours = 90 * 24
    rows = []
    for city, meta in CITY_META.items():
        b = meta["base"]; lat = meta["lat"]; lon = meta["lon"]
        for h in range(hours):
            ts = start + timedelta(hours=h)
            hr = ts.hour
            rush = 1.25 if 7 <= hr <= 10 else (1.35 if 17 <= hr <= 21 else 0.85)
            spike = rng.choice([1.0]*9 + [2.0])
            f = b * rush * spike
            rows.append({
                "timestamp": ts, "city": city, "latitude": lat, "longitude": lon,
                "pm2_5": round(max(5, rng.normal(60*f, 18)), 1),
                "pm10": round(max(10, rng.normal(110*f, 30)), 1),
                "nitrogen_dioxide": round(max(2, rng.normal(35*f, 10)), 1),
                "sulphur_dioxide": round(max(1, rng.normal(15*f, 5)), 1),
                "carbon_monoxide": round(max(0.2, rng.normal(1.2*f, 0.35)), 2),
                "ozone": round(max(2, rng.normal(40*(0.6+0.4/f), 12)), 1),
                "aqi": round(max(10, rng.normal(55*f+20, 22)), 1),
            })
    df = pd.DataFrame(rows)
    # ~2 % NaN
    tot = len(df) * 6
    pcols = list(POLL_UNIT.keys())
    for _ in range(int(0.02 * tot)):
        df.at[rng.integers(0, len(df)), rng.choice(pcols)] = np.nan
    df["aqi_category"] = df["aqi"].apply(_aqi_cat)
    df["data_source"] = "Synthetic demo data"
    return df


# ══════════════════════════════════════════════
# FILTERING
# ══════════════════════════════════════════════

def filter_data(df, cities, d0, d1):
    mask = df["city"].isin(cities) & (df["timestamp"].dt.date >= d0) & (df["timestamp"].dt.date <= d1)
    return df.loc[mask].copy()


# ══════════════════════════════════════════════
# KPI
# ══════════════════════════════════════════════

def calc_kpis(fdf):
    if fdf.empty:
        return {"latest_aqi": None, "aqi_cat": "—", "latest_pm25": None,
                "latest_pm10": None, "most_polluted": "—"}
    latest = fdf.loc[fdf["timestamp"].idxmax()]
    avg = fdf.groupby("city")["aqi"].mean()
    return {
        "latest_aqi": round(latest["aqi"], 1) if pd.notna(latest["aqi"]) else None,
        "aqi_cat": latest["aqi_category"],
        "latest_pm25": round(latest["pm2_5"], 1) if pd.notna(latest["pm2_5"]) else None,
        "latest_pm10": round(latest["pm10"], 1) if pd.notna(latest["pm10"]) else None,
        "most_polluted": avg.idxmax() if not avg.empty else "—",
    }


# ══════════════════════════════════════════════
# INSIGHTS
# ══════════════════════════════════════════════

def generate_insights(fdf):
    if fdf.empty:
        return ["No data available for the selected filters. Adjust the sidebar and try again."]
    ins = []
    # 1 — highest avg AQI
    avg = fdf.groupby("city")["aqi"].mean()
    top = avg.idxmax(); tv = avg.max()
    ins.append(f"<strong>{top}</strong> has the highest average AQI at <strong>{tv:.0f}</strong> "
               f"({_aqi_cat(tv)}) in the selected period.")
    # 2 — first vs second half
    mid = fdf["timestamp"].min() + (fdf["timestamp"].max() - fdf["timestamp"].min()) / 2
    h1 = fdf[fdf["timestamp"] <= mid]["aqi"].mean()
    h2 = fdf[fdf["timestamp"] > mid]["aqi"].mean()
    if pd.notna(h1) and pd.notna(h2):
        d = h2 - h1; dr = "increased" if d > 0 else "decreased"
        ins.append(f"Overall AQI {dr} by <strong>{abs(d):.1f}</strong> points from the first half "
                   f"to the second half of the selected period.")
    # 3 — peak hour
    tmp = fdf.copy(); tmp["hour"] = tmp["timestamp"].dt.hour
    pk = tmp.groupby("hour")["aqi"].mean().idxmax()
    ins.append(f"Pollution peaks around <strong>{pk}:00–{pk+1}:00</strong> on average.")
    # 4 — largest variation
    sd = fdf.groupby("city")["aqi"].std().dropna()
    if not sd.empty:
        vc = sd.idxmax()
        ins.append(f"<strong>{vc}</strong> shows the largest day-to-day AQI variation "
                   f"(std dev {sd.max():.1f}).")
    # 5 — cleanest city
    if len(avg) > 1:
        bot = avg.idxmin(); bv = avg.min()
        ins.append(f"<strong>{bot}</strong> is the cleanest city in this selection with an "
                   f"average AQI of <strong>{bv:.0f}</strong> ({_aqi_cat(bv)}).")
    # 6 — dominant pollutant
    pm = {p: fdf[p].mean() for p in POLL_UNIT if fdf[p].notna().any()}
    if pm:
        pz = {p: m / (fdf[p].std() or 1) for p, m in pm.items()}
        tp = max(pz, key=pz.get)
        ins.append(f"The most elevated pollutant (relative to variability) is "
                   f"<strong>{POLL_LBL[tp]}</strong> at a mean of {pm[tp]:.1f} {POLL_UNIT[tp]}.")
    return ins


# ══════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════

def _top_cities(fdf, n=5):
    return fdf.groupby("city")["aqi"].mean().sort_values(ascending=False).head(n).index.tolist()


def _daily(fdf, cities, col):
    """Resample to daily mean per city for one column."""
    return (fdf[fdf["city"].isin(cities)].set_index("timestamp")
            .groupby("city").resample("D")[col].mean().reset_index())


# ── Trend: single overlay chart ──
def chart_trend_single(fdf, pol, cities):
    unit = POLL_UNIT.get(pol, ""); lbl = POLL_LBL.get(pol, pol)
    daily = _daily(fdf, cities, pol)
    fig = go.Figure()
    for i, city in enumerate(cities):
        cd = daily[daily["city"] == city].sort_values("timestamp")
        cd["roll"] = cd[pol].rolling(7, min_periods=1).mean()
        c = ccol(city, i)
        # faint raw
        fig.add_trace(go.Scatter(x=cd["timestamp"], y=cd[pol], name=city,
            legendgroup=city, showlegend=False, mode="lines",
            line=dict(color=c, width=0.7), opacity=0.18, hoverinfo="skip"))
        # bold 7-day
        fig.add_trace(go.Scatter(x=cd["timestamp"], y=cd["roll"], name=city,
            legendgroup=city, mode="lines", line=dict(color=c, width=2.2)))
    fig.update_layout(
        template=PLT_TPL,
        title=dict(text=f"{lbl} daily average, 7-day smoothed ({unit})", x=0, y=0.98, xanchor="left", yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, title_text=""),
        margin=dict(l=48, r=24, t=65, b=44),
        xaxis_title="Date", yaxis_title=f"{lbl} ({unit})", hovermode="x unified", height=420
    )
    return fig


# ── Trend: small multiples ──
def chart_trend_sm(fdf, pol, cities):
    n = len(cities); cols = 3; rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cities,
                        shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.10)
    unit = POLL_UNIT.get(pol, ""); lbl = POLL_LBL.get(pol, pol)
    daily = _daily(fdf, cities, pol)
    for i, city in enumerate(cities):
        r, c = i // cols + 1, i % cols + 1
        cd = daily[daily["city"] == city].sort_values("timestamp")
        cd["roll"] = cd[pol].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=cd["timestamp"], y=cd["roll"], mode="lines",
            line=dict(color=ccol(city, i), width=1.8), showlegend=False), row=r, col=c)
    fig.update_layout(
        template=PLT_TPL,
        title=dict(text=f"{lbl} 7-day average by city ({unit})", x=0, y=0.98, xanchor="left", yanchor="top"),
        height=240*rows, showlegend=False,
        margin=dict(l=48, r=24, t=60, b=44)
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=TEXT_MUT)
    return fig


# ── CO separate chart ──
def chart_co(fdf, cities):
    daily = _daily(fdf, cities, "carbon_monoxide")
    fig = go.Figure()
    for i, city in enumerate(cities):
        cd = daily[daily["city"] == city].sort_values("timestamp")
        cd["roll"] = cd["carbon_monoxide"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=cd["timestamp"], y=cd["roll"], name=city,
            mode="lines", line=dict(color=ccol(city, i), width=2)))
    fig.update_layout(
        template=PLT_TPL,
        title=dict(text="CO daily average, 7-day smoothed (mg/m3)", x=0, y=0.98, xanchor="left", yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, title_text=""),
        margin=dict(l=48, r=24, t=65, b=44),
        xaxis_title="Date", yaxis_title="CO (mg/m3)", hovermode="x unified", height=340
    )
    return fig


# ── City bar ──
def chart_city_bar(fdf, metric, n):
    avg = fdf.groupby("city")[metric].mean().dropna().sort_values(ascending=True).tail(n)
    unit = POLL_UNIT.get(metric, ""); lbl = POLL_LBL.get(metric, "AQI")
    colours = [ccol(c, i) for i, c in enumerate(avg.index)]
    fig = go.Figure(go.Bar(y=avg.index, x=avg.values, orientation="h",
        marker_color=colours, text=[f"{v:.1f}" for v in avg.values],
        textposition="outside", textfont=dict(color=TEXT_MUT, size=11)))
    ttl = f"Average {lbl} by city ({unit})" if unit else f"Average {lbl} by city"
    fig.update_layout(
        template=PLT_TPL,
        title=dict(text=ttl, x=0, y=0.98, xanchor="left", yanchor="top"),
        xaxis_title=f"{lbl} ({unit})" if unit else lbl,
        yaxis_title="", yaxis=dict(categoryorder="total ascending"),
        height=max(300, n*42+100), showlegend=False,
        margin=dict(l=48, r=24, t=50, b=44)
    )
    return fig


# ── AQI category distribution (stacked bar) ──
def chart_aqi_dist(fdf):
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    counts = fdf.groupby(["city", "aqi_category"]).size().unstack(fill_value=0)
    # Make sure all categories exist
    for c in cats:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[cats]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct = pct.sort_values("Good", ascending=True)

    fig = go.Figure()
    for cat in cats:
        fig.add_trace(go.Bar(
            y=pct.index, x=pct[cat], name=cat, orientation="h",
            marker_color=AQI_CLR.get(cat, "#6B7280"),
            text=[f"{v:.0f}%" if v > 5 else "" for v in pct[cat]],
            textposition="inside", textfont=dict(size=10, color="#fff"),
        ))
    fig.update_layout(
        template=PLT_TPL, barmode="stack",
        title=dict(text="Share of hours in each AQI category by city", x=0, y=0.98, xanchor="left", yanchor="top"),
        xaxis_title="Percentage of hours", yaxis_title="",
        height=max(300, len(pct)*36+120), showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, title_text=""),
        margin=dict(l=48, r=24, t=50, b=65)
    )
    return fig


# ── Histogram ──
def chart_hist(fdf, metric):
    unit = POLL_UNIT.get(metric, ""); lbl = POLL_LBL.get(metric, metric)
    fig = px.histogram(fdf, x=metric, color="city", barmode="overlay", opacity=0.55,
        color_discrete_map=cmap_for(fdf["city"].unique()), template=PLT_TPL,
        labels={metric: f"{lbl} ({unit})", "city": ""}, height=400)
    fig.update_layout(
        title=dict(text=f"Distribution of {lbl} readings ({unit})", x=0, y=0.98, xanchor="left", yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5, title_text=""),
        margin=dict(l=48, r=24, t=50, b=70),
    )
    return fig


# ── Box plot ──
def chart_box(fdf, metric):
    unit = POLL_UNIT.get(metric, ""); lbl = POLL_LBL.get(metric, metric)
    fig = px.box(fdf, x="city", y=metric, color="city",
        color_discrete_map=cmap_for(fdf["city"].unique()), template=PLT_TPL,
        labels={metric: f"{lbl} ({unit})", "city": ""}, height=400)
    fig.update_layout(
        title=dict(text=f"Spread of {lbl} by city ({unit})", x=0, y=0.98, xanchor="left", yanchor="top"),
        margin=dict(l=48, r=24, t=50, b=70),
        showlegend=False, xaxis_tickangle=-30
    )
    return fig


# ── Correlation heatmap ──
def chart_corr(fdf):
    pcols = [p for p in POLL_UNIT if fdf[p].notna().any()]
    if len(pcols) < 2:
        return None
    corr = fdf[pcols].corr()
    labels = [POLL_LBL.get(p, p) for p in pcols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=HEAT_SCALE, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT_MAIN),
        colorbar_title="r",
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLT_TPL,
        title=dict(text="Pollutant correlation matrix", x=0, y=0.98, xanchor="left", yanchor="top"),
        height=380, xaxis_side="bottom",
        margin=dict(l=48, r=24, t=50, b=44)
    )
    return fig


# ── Heatmap (hour x date) ──
def chart_heatmap(fdf, metric):
    tmp = fdf.copy(); tmp["hour"] = tmp["timestamp"].dt.hour; tmp["date"] = tmp["timestamp"].dt.date
    pivot = tmp.pivot_table(index="hour", columns="date", values=metric, aggfunc="mean")
    unit = POLL_UNIT.get(metric, ""); lbl = POLL_LBL.get(metric, metric)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[str(d) for d in pivot.columns], y=pivot.index,
        colorscale=HEAT_SCALE, colorbar_title=f"{lbl} ({unit})",
        hovertemplate="Date: %{x}<br>Hour: %{y}<br>Value: %{z:.1f}<extra></extra>"))
    fig.update_layout(
        template=PLT_TPL,
        title=dict(text=f"{lbl} by hour of day and date — mean across selected cities ({unit})", x=0, y=0.98, xanchor="left", yanchor="top"),
        xaxis_title="Date", yaxis_title="Hour of Day", height=440, yaxis=dict(dtick=2),
        margin=dict(l=48, r=24, t=50, b=44)
    )
    return fig


# ── Map ──
def chart_map(fdf):
    agg = fdf.groupby("city").agg({"aqi": "mean", "latitude": "first", "longitude": "first"}).reset_index()
    agg["aqi"] = agg["aqi"].round(1)
    agg["category"] = agg["aqi"].apply(_aqi_cat)
    fig = px.scatter_map(agg, lat="latitude", lon="longitude", color="aqi", size="aqi",
        color_continuous_scale=MAP_SCALE, hover_name="city",
        hover_data={"aqi": ":.1f", "category": True, "latitude": False, "longitude": False},
        zoom=4, center={"lat": 20.5, "lon": 79}, height=500)
    fig.update_layout(
        template=PLT_TPL, map_style="carto-darkmatter",
        title=dict(text="Average AQI by city", x=0, y=0.98, xanchor="left", yanchor="top"),
        coloraxis_colorbar_title="AQI",
        margin=dict(l=24, r=24, t=50, b=24)
    )
    return fig


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
df_full = generate_data()
all_cities = sorted(df_full["city"].unique())


# ══════════════════════════════════════════════
# SIDEBAR: logo + navigation + filters
# ══════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown(f'<p style="font-size:22px;font-weight:700;color:{TEXT_MAIN};margin-bottom:0;">'
                f'AirAware</p>'
                f'<p style="font-size:11px;color:{TEXT_MUT};margin-top:2px;margin-bottom:12px;">'
                f'Air Quality Analytics</p>', unsafe_allow_html=True)

    # Navigation radio (styled as menu via CSS)
    nav_labels = [f"{NAV_ICONS.get(n, '·')}  {n}" for n in NAV_ITEMS]
    sel_label = st.radio("nav", nav_labels, index=NAV_ITEMS.index(st.session_state.page),
                         label_visibility="collapsed", key="nav_radio")
    # Extract page name back from label
    st.session_state.page = sel_label.split("  ", 1)[1] if "  " in sel_label else sel_label

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:12px 0 14px 0;">',
                unsafe_allow_html=True)

    # Global filters
    with st.expander("Filters", expanded=True):
        sel_cities = st.multiselect("Cities", options=all_cities, default=all_cities,
                                    help="Choose one or more cities.")
        min_d = df_full["timestamp"].dt.date.min()
        max_d = df_full["timestamp"].dt.date.max()
        date_range = st.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
        sel_pol = st.selectbox("Primary pollutant", options=list(POLL_UNIT.keys()),
                               format_func=lambda p: f"{POLL_LBL[p]} ({POLL_UNIT[p]})")
        if st.button("Reset filters", use_container_width=True):
            st.rerun()

    st.markdown('<div class="synth-badge"><strong>Synthetic demo data</strong><br>'
                'Not real measurements</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# VALIDATE FILTERS
# ══════════════════════════════════════════════
if isinstance(date_range, (list, tuple)):
    d0, d1 = (date_range[0], date_range[1]) if len(date_range) == 2 else (date_range[0], date_range[0])
else:
    d0 = d1 = date_range

if d0 > d1:
    st.warning("Start date is after end date — swapping automatically.")
    d0, d1 = d1, d0

if not sel_cities:
    st.warning("No cities selected. Pick at least one city from the sidebar.")
    st.stop()

fdf = filter_data(df_full, sel_cities, d0, d1)
if fdf.empty:
    st.info("No data matches the current filters. Try widening the date range or adding more cities.")
    st.stop()

page = st.session_state.page


# ══════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════
if page == "Overview":
    # Header
    st.markdown(f'<p style="font-size:28px;font-weight:700;color:{TEXT_MAIN};margin-bottom:2px;">'
                f'AirAware</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:14px;color:{TEXT_MUT};margin-top:0;margin-bottom:18px;">'
                f'Synthetic air-quality data for 10 Indian cities &middot; '
                f'Synthetic AQI (CPCB-style categories) &middot; '
                f'Latest data: {fdf["timestamp"].max().strftime("%d %b %Y, %H:%M")}</p>',
                unsafe_allow_html=True)

    # KPI cards
    kpis = calc_kpis(fdf)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Latest AQI", kpis["latest_aqi"] if kpis["latest_aqi"] is not None else "—")
        st.markdown(f'<span style="font-size:12px;color:{TEXT_MUT};">Synthetic AQI '
                    f'{aqi_badge(kpis["aqi_cat"])}</span>', unsafe_allow_html=True)
    with k2:
        st.metric("Latest PM2.5", kpis["latest_pm25"] if kpis["latest_pm25"] is not None else "—")
        st.markdown(f'<span style="font-size:12px;color:{TEXT_MUT};">ug/m3</span>',
                    unsafe_allow_html=True)
    with k3:
        st.metric("Latest PM10", kpis["latest_pm10"] if kpis["latest_pm10"] is not None else "—")
        st.markdown(f'<span style="font-size:12px;color:{TEXT_MUT};">ug/m3</span>',
                    unsafe_allow_html=True)
    with k4:
        st.metric("Most Polluted City", kpis["most_polluted"])
        if kpis["most_polluted"] != "—":
            mp = fdf[fdf["city"] == kpis["most_polluted"]]["aqi"].mean()
            st.markdown(f'<span style="font-size:12px;color:{TEXT_MUT};">Avg AQI {mp:.0f}</span>',
                        unsafe_allow_html=True)

    # Headline insight
    st.markdown('<hr class="section-div">', unsafe_allow_html=True)
    ins = generate_insights(fdf)
    if ins:
        st.markdown(f'<div class="insight-card">{ins[0]}</div>', unsafe_allow_html=True)

    # City ranking + AQI distribution side by side
    ov1, ov2 = st.columns(2)
    with ov1:
        fig = chart_city_bar(fdf, "aqi", min(len(sel_cities), 10))
        st.plotly_chart(fig, use_container_width=True)
        chart_cap("How to read: cities ranked by mean AQI over the filtered period. "
                  "Longer bars mean worse average air quality.")
    with ov2:
        fig = chart_aqi_dist(fdf)
        st.plotly_chart(fig, use_container_width=True)
        chart_cap("How to read: each bar shows what share of hours fell into each "
                  "AQI category. Greener is healthier.")


# ══════════════════════════════════════════════
# PAGE: Trends
# ══════════════════════════════════════════════
elif page == "Trends":
    pg_head("Pollution Trends",
            "7-day rolling average to reduce daily noise. CO is plotted separately (different unit).")

    mode = st.radio("Show", ["Top 5 most polluted", "All selected cities (max 5)",
                             "Small multiples (one chart per city)"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Top 5 most polluted":
        tc = _top_cities(fdf, 5)
    elif mode.startswith("All selected"):
        tc = sel_cities[:5]
    else:
        tc = sel_cities

    if sel_pol == "carbon_monoxide":
        st.plotly_chart(chart_co(fdf, tc), use_container_width=True)
        chart_cap("Showing CO only (mg/m3). CO uses a different unit from other pollutants.")
    elif mode.startswith("Small"):
        st.plotly_chart(chart_trend_sm(fdf, sel_pol, tc), use_container_width=True)
        chart_cap("Each panel shows one city. Y-axis is shared so values are directly comparable. "
                  "Showing the 7-day average to reduce noise.")
    else:
        st.plotly_chart(chart_trend_single(fdf, sel_pol, tc), use_container_width=True)
        chart_cap("Faint lines show raw daily averages. Bold lines are the 7-day rolling mean. "
                  "Showing the 7-day average to reduce noise.")

    if sel_pol != "carbon_monoxide":
        with st.expander("Show CO trend (separate axis)"):
            st.plotly_chart(chart_co(fdf, tc), use_container_width=True)
            chart_cap("CO is in mg/m3 — plotted separately from the ug/m3 pollutants.")


# ══════════════════════════════════════════════
# PAGE: City Comparison
# ══════════════════════════════════════════════
elif page == "City Comparison":
    pg_head("City Comparison", "Cities ranked by the average value of the selected metric.")
    c1, c2 = st.columns([2, 1])
    with c1:
        cm = st.selectbox("Metric", options=list(POLL_UNIT.keys()) + ["aqi"],
                          format_func=lambda p: POLL_LBL.get(p, "AQI"), key="cmp_m")
    with c2:
        nc = st.slider("Cities to show", 2, len(sel_cities), min(len(sel_cities), 10), key="cmp_n")
    st.plotly_chart(chart_city_bar(fdf, cm, nc), use_container_width=True)
    chart_cap("How to read: longer bars mean higher average concentration or AQI. "
              "Values are averaged over the entire filtered date range.")


# ══════════════════════════════════════════════
# PAGE: Pollutants
# ══════════════════════════════════════════════
elif page == "Pollutants":
    pg_head("Pollutant Distribution",
            "Histogram, box plot, and correlation matrix for the selected pollutant.")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.plotly_chart(chart_hist(fdf, sel_pol), use_container_width=True)
        chart_cap("How to read: overlapping bars show how often each concentration range was "
                  "observed per city.")
    with pc2:
        st.plotly_chart(chart_box(fdf, sel_pol), use_container_width=True)
        chart_cap("How to read: the box is the middle 50 % of readings. "
                  "The line inside is the median. Dots are outliers.")

    # Correlation heatmap
    st.markdown('<hr class="section-div">', unsafe_allow_html=True)
    fig_corr = chart_corr(fdf)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        chart_cap("How to read: values range from -1 (inverse relationship) to +1 (move together). "
                  "Computed from all selected cities and dates.")
    else:
        st.info("Not enough non-null pollutant data to compute correlations.")


# ══════════════════════════════════════════════
# PAGE: Heatmap
# ══════════════════════════════════════════════
elif page == "Heatmap":
    lbl = POLL_LBL.get(sel_pol, sel_pol)
    pg_head("Heatmap",
            f"Mean {lbl} by hour of day and date, averaged across selected cities.")
    st.plotly_chart(chart_heatmap(fdf, sel_pol), use_container_width=True)
    chart_cap(f"How to read: each cell is the mean {lbl} for that hour and date. "
              "Warmer (lighter) cells indicate higher concentrations. "
              "The aggregation averages across all selected cities.")


# ══════════════════════════════════════════════
# PAGE: Map
# ══════════════════════════════════════════════
elif page == "Map":
    pg_head("Map", "Each city sized and coloured by its average AQI for the selected period.")
    st.plotly_chart(chart_map(fdf), use_container_width=True)
    chart_cap("How to read: larger, warmer-toned bubbles indicate worse average air quality. "
              "Hover over a city for details.")


# ══════════════════════════════════════════════
# PAGE: Insights
# ══════════════════════════════════════════════
elif page == "Insights":
    pg_head("Air Quality Insights",
            "All insights are computed from the current filter selections. Nothing is hardcoded.")
    insights = generate_insights(fdf)
    for i in insights:
        st.markdown(f'<div class="insight-card">{i}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: Data & Export
# ══════════════════════════════════════════════
elif page == "Data & Export":
    pg_head("Data & Export",
            f"{len(fdf):,} rows for {len(sel_cities)} cities in the selected date range.")

    disp = fdf.copy()
    disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(disp, use_container_width=True, height=440)

    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv,
                       file_name="airaware_filtered_data.csv", mime="text/csv")


# ══════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════
elif page == "About":
    pg_head("About / Methodology")
    st.markdown(f"""
**Data source.** All data in this dashboard is **100 % synthetic**, generated
with `numpy.random` using a fixed seed (42) for reproducibility.
It does **not** represent real air-quality measurements.

**How the data is generated.**
Each city has a base pollution multiplier. Delhi, Lucknow and Kolkata have higher
baselines; coastal cities like Chennai and Visakhapatnam are lower. Rush-hour
windows (7–10 AM, 5–9 PM) increase values, and random spikes occur about 10 %
of the time. About 2 % of pollutant cells are left as NaN to simulate sensor
gaps — they are never filled with zero.

**AQI categories** follow the Indian CPCB breakpoints:
Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300),
Very Poor (301–400), Severe (401+).
Labelled "Synthetic AQI (CPCB-style categories)".

**Trend smoothing.** The 7-day rolling average reduces hourly and daily noise
so the underlying pattern is easier to see.

**Units.** PM2.5, PM10, NO2, SO2, O3 in ug/m3. CO in mg/m3.
CO is always on a separate chart to avoid scale distortion.

**Insights** are computed live from the filtered data — nothing is hardcoded.

**Correlation matrix.** Pearson correlation coefficients between pollutants,
computed from all filtered readings. Values near +1 mean two pollutants tend
to rise and fall together; near 0 means little linear relationship.
    """)
