"""
BambooSense v4.0 — Streamlit App
Brand palette: Cream #E8DEBB · Khaki #9B8C60 · Sage #A3B86A · Forest #4D7A1E · Burnt-Orange #D45A10
SDG 13 (Climate Action) · SDG 15 (Life on Land)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import json
import datetime
from PIL import Image

# ── ReportLab ────────────────────────────────────────────────
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch

# ── Local modules ────────────────────────────────────────────
from calibration import run_calibration, SCALE_CACHE, DEFAULT_CALIB_CONFIG
from inference   import (analyse_image, chart_biomass, chart_health,
                          chart_height, chart_carbon, chart_health_donut,
                          fig_to_bytes, CARBON_FACTOR, HEALTH_CONFIG,
                          C_CREAM, C_KHAKI, C_SAGE, C_FOREST, C_ORANGE, C_DARK)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BambooSense",
    page_icon="🎋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  — brand palette + SDG-themed design
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #2A3D14;
}
.stApp {
    background: #F5F2E8;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(140deg, #2A3D14 0%, #4D7A1E 55%, #5e8f26 100%);
    border-radius: 20px;
    padding: 48px 48px 38px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(42,61,20,0.28);
}
.hero::before {
    content: '🎋';
    position: absolute; right: 44px; top: 50%;
    transform: translateY(-50%);
    font-size: 140px; opacity: 0.09; line-height: 1;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 75% 40%, rgba(163,184,106,0.14), transparent 60%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px; font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #A3B86A;
    margin: 0 0 10px;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 54px; font-weight: 900;
    color: #E8DEBB; margin: 0 0 4px;
    letter-spacing: -0.03em; line-height: 1.08;
}
.hero h1 span { color: #A3B86A; }
.hero .tagline {
    color: rgba(232,222,187,0.7);
    font-size: 12.5px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0 0 20px;
}
.hero .mission {
    color: rgba(232,222,187,0.82);
    font-size: 13.5px; line-height: 1.75;
    max-width: 540px;
    border-left: 3px solid rgba(163,184,106,0.55);
    padding-left: 14px;
    margin: 0;
}
.hero .sdg-row {
    display: flex; gap: 10px; margin-top: 22px; flex-wrap: wrap;
}
.sdg-badge {
    background: rgba(163,184,106,0.18);
    border: 1.5px solid rgba(163,184,106,0.4);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: #A3B86A;
    letter-spacing: 0.05em;
}

/* ── Stat Pills ── */
.stats-row { display:flex; gap:12px; flex-wrap:wrap; margin: 24px 0 8px; }
.stat-pill {
    flex:1; min-width:125px;
    background: linear-gradient(145deg, #FDFAF2, #F0ECD8);
    border: 1.5px solid rgba(155,140,96,0.22);
    border-radius: 16px; padding: 18px 14px; text-align:center;
    box-shadow: 0 3px 12px rgba(42,61,20,0.07);
    transition: transform 0.18s, box-shadow 0.18s;
}
.stat-pill:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(42,61,20,0.12); }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 32px; font-weight: 900;
    color: #4D7A1E; line-height: 1;
}
.stat-val.orange { color: #D45A10; }
.stat-val.khaki  { color: #9B8C60; }
.stat-val.dark   { color: #2A3D14; }
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 9px; text-transform: uppercase;
    letter-spacing: 0.09em; color: #9B8C60; margin-top: 5px;
}
.stat-icon { font-size: 18px; margin-bottom: 4px; }

/* ── Section Title ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px; font-weight: 700;
    color: #2A3D14; margin: 0 0 14px;
    display: flex; align-items: center; gap: 9px;
}
.section-rule {
    border: none; border-top: 1px solid rgba(155,140,96,0.25);
    margin: 24px 0;
}

/* ── Carbon Banner ── */
.carbon-banner {
    background: linear-gradient(135deg, #2A3D14 0%, #4D7A1E 60%, #5e8f26 100%);
    border-radius: 18px; padding: 24px 32px;
    color: #E8DEBB; margin: 18px 0 26px;
    display: flex; align-items: center; gap: 24px;
    box-shadow: 0 6px 28px rgba(42,61,20,0.28);
    border: 1px solid rgba(163,184,106,0.25);
}
.co2-num {
    font-family: 'Playfair Display', serif;
    font-size: 52px; font-weight: 900; line-height: 1; flex-shrink: 0;
    color: #E8DEBB;
}
.co2-unit { font-size: 20px; font-weight: 400; color: #A3B86A; }
.co2-detail { font-size: 13px; opacity: 0.88; line-height: 1.7; }
.co2-detail strong { color: #A3B86A; }
.co2-equiv {
    background: rgba(163,184,106,0.18);
    border: 1px solid rgba(163,184,106,0.35);
    border-radius: 10px; padding: 10px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px; color: #A3B86A;
    margin-top: 10px; display: inline-block;
}

/* ── Health Legend ── */
.health-legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0 18px; }
.health-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 16px; border-radius: 22px;
    font-size: 13px; font-weight: 600;
    border: 1.5px solid rgba(0,0,0,0.10);
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.health-badge .dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
.hb-green  { background: #eef5e8; color: #2A3D14; }
.hb-yellow { background: #f5f2e0; color: #6B5B10; }
.hb-dry    { background: #faeee6; color: #8B3010; }

/* ── Calibration Card ── */
.calib-card {
    background: #F0ECD8;
    border: 1.5px solid rgba(155,140,96,0.35);
    border-radius: 14px; padding: 18px 22px;
    margin-bottom: 18px; font-size: 13px;
    color: #2A3D14; line-height: 1.75;
}
.calib-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 16px; color: #2A3D14; margin: 0 0 10px;
}
.calib-ok   { color: #4D7A1E; font-weight: 700; }
.calib-warn { color: #D45A10; font-weight: 700; }

/* ── Sidebar Info Cards ── */
.info-card {
    background: #F0ECD8;
    border-left: 4px solid #9B8C60;
    border-radius: 0 10px 10px 0;
    padding: 12px 14px; margin-bottom: 11px;
    font-size: 13px; color: #2A3D14; line-height: 1.6;
}
.info-card b { color: #4D7A1E; }
.info-card code {
    background: rgba(155,140,96,0.18);
    padding: 1px 5px; border-radius: 4px;
    font-family: 'DM Mono', monospace; font-size: 11.5px;
    color: #2A3D14;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4D7A1E, #2A3D14) !important;
    color: #E8DEBB !important; border: none !important;
    border-radius: 10px !important; padding: 12px 32px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; font-size: 14.5px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 18px rgba(77,122,30,0.38) !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 30px rgba(77,122,30,0.52) !important;
    transform: translateY(-1px) !important;
}

/* ── File Uploader ── */
div[data-testid="stFileUploader"] {
    background: #FDFAF2;
    border: 2px dashed rgba(155,140,96,0.4);
    border-radius: 14px; padding: 6px;
}

/* ── Misc ── */
hr { border: none; border-top: 1px solid rgba(155,140,96,0.2) !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stExpander"] {
    background: #F0ECD8;
    border-radius: 12px;
    border: 1px solid rgba(155,140,96,0.25);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Playfair Display',serif;font-size:22px;
                font-weight:900;color:#4D7A1E;margin-bottom:4px;">
        🎋 BambooSense
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;
                color:#9B8C60;letter-spacing:0.08em;text-transform:uppercase;
                margin-bottom:18px;">
        v4.0 · YOLOv8 · Health-Aware
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <b>Mission</b><br>
        Scalable, automated bamboo forest monitoring for biomass estimation,
        health assessment, and climate impact reporting.
    </div>
    <div class="info-card">
        <b>Biomass Formula</b><br>
        <code>B = K × D² × H × health_factor</code><br>
        K = 0.03 · D = diameter (cm) · H = height (cm)<br>
        Green = 1.0 · Yellow = 0.65 · Dry = 0.35
    </div>
    <div class="info-card">
        <b>Carbon Sequestration</b><br>
        Bamboo ≈ <b>0.25 kg CO₂ per kg</b> dry biomass
        (INBAR TR-37 / IPCC Tier 1).<br>
        Dry culms: additional carbon penalty (×0.20) —
        decomposition has released sequestered carbon.<br>
        <code>CO₂ = Biomass × 0.25 × c_factor</code>
    </div>
    <div class="info-card">
        <b>Health Classification — HSV + Saturation</b><br>
        🟢 <b>Green</b>  — healthy, full biomass (100%)<br>
        🟡 <b>Yellow</b> — stressed / transitional (65%)<br>
        🔴 <b>Dry</b>    — dead/dry (35% biomass, 20% carbon)<br><br>
        
    </div>
    <div class="info-card">
        <b>Detection Model</b><br>
        YOLOv8-Small fine-tuned on bamboo field imagery.<br>
        Detects: <b>culm</b> (stalk) · <b>node</b> (joint)
    </div>
    <div class="info-card">
        <b>Why Bamboo?</b><br>
        • Fastest-growing plant on Earth<br>
        • Sequesters 2–3× more CO₂ than trees<br>
        • Supports construction, pulp & bioenergy<br>
        • Contributes to SDG 13 · SDG 15 · SDG 11
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("BambooSense v4.0 · Streamlit")


# ─────────────────────────────────────────────────────────────
# MODEL + CALIBRATION  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    return YOLO("model/yolov8s_bamboo.pt")


@st.cache_data(show_spinner=False)
def get_scale_factors():
    if os.path.exists(SCALE_CACHE):
        with open(SCALE_CACHE) as f:
            d = json.load(f)
        return d["internode_scale"], d["diameter_scale"]
    model = load_model()
    try:
        si, sd = run_calibration(model, DEFAULT_CALIB_CONFIG, SCALE_CACHE)
    except RuntimeError:
        si, sd = 0.08, 0.05
    return si, sd


def audit_calibration(scale_i: float, scale_d: float) -> dict:
    warnings = []
    ok = True
    if not (0.02 <= scale_i <= 0.30):
        warnings.append(f"⚠️  internode_scale {scale_i:.5f} cm/px outside [0.02, 0.30].")
        ok = False
    else:
        warnings.append(f"✅  internode_scale {scale_i:.5f} cm/px — within normal range.")
    if not (0.01 <= scale_d <= 0.20):
        warnings.append(f"⚠️  diameter_scale {scale_d:.5f} cm/px outside [0.01, 0.20].")
        ok = False
    else:
        warnings.append(f"✅  diameter_scale {scale_d:.5f} cm/px — within normal range.")
    ratio = scale_i / scale_d if scale_d > 0 else 0
    if not (0.5 <= ratio <= 8.0):
        warnings.append(f"⚠️  Scale ratio (internode/diameter) = {ratio:.2f} unusual (expected 0.5–8).")
        ok = False
    else:
        warnings.append(f"✅  Scale ratio (internode/diameter) = {ratio:.2f} — consistent.")
    return {"ok": ok, "messages": warnings}


# ─────────────────────────────────────────────────────────────
# RICH PDF BUILDER
# ─────────────────────────────────────────────────────────────
def build_pdf(df: pd.DataFrame, chart_dict: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, leftMargin=52, rightMargin=52,
                              topMargin=52, bottomMargin=52)

    # ── Styles ───────────────────────────────────────────────
    FOREST_RL  = rl_colors.HexColor("#4D7A1E")
    DARK_RL    = rl_colors.HexColor("#2A3D14")
    KHAKI_RL   = rl_colors.HexColor("#9B8C60")
    CREAM_RL   = rl_colors.HexColor("#E8DEBB")
    ORANGE_RL  = rl_colors.HexColor("#D45A10")
    SAGE_RL    = rl_colors.HexColor("#A3B86A")
    BG_RL      = rl_colors.HexColor("#F5F2E8")
    LGTEEN_RL  = rl_colors.HexColor("#eef5e8")

    sTitle = ParagraphStyle('sTitle',
        fontName='Helvetica-Bold', fontSize=24,
        textColor=DARK_RL, spaceAfter=4, leading=28)
    sSub = ParagraphStyle('sSub',
        fontName='Helvetica', fontSize=10,
        textColor=KHAKI_RL, spaceAfter=14, leading=14)
    sBody = ParagraphStyle('sBody',
        fontName='Helvetica', fontSize=11,
        textColor=DARK_RL, spaceAfter=6, leading=16)
    sBodyB = ParagraphStyle('sBodyB',
        fontName='Helvetica-Bold', fontSize=11,
        textColor=DARK_RL, spaceAfter=6, leading=16)
    sCO2 = ParagraphStyle('sCO2',
        fontName='Helvetica-Bold', fontSize=15,
        textColor=FOREST_RL, spaceAfter=6, leading=20)
    sChartLabel = ParagraphStyle('sChartLabel',
        fontName='Helvetica-Bold', fontSize=11,
        textColor=FOREST_RL, spaceAfter=4)
    sFooter = ParagraphStyle('sFooter',
        fontName='Helvetica', fontSize=8,
        textColor=KHAKI_RL, spaceAfter=0)

    total_co2     = df["Carbon_CO2_kg"].sum()
    total_biomass = df["Biomass_kg"].sum()
    health_counts = df["Health"].value_counts().to_dict()
    health_str    = "  ·  ".join(f"{h}: {n}" for h, n in health_counts.items())
    trees_eq      = total_co2 / 21.77
    today         = datetime.date.today().strftime("%d %B %Y")

    items = [
        Paragraph("🎋  BambooSense", sTitle),
        Paragraph("Automated Bamboo Growth Monitoring · Biomass & Carbon Assessment Report", sSub),
        HRFlowable(width="100%", thickness=1.5, color=FOREST_RL, spaceAfter=14),

        # Summary grid via table
        Paragraph("Report Summary", sBodyB),
        Spacer(1, 6),
    ]

    # ── Summary table ────────────────────────────────────────
    summary_data = [
        ["Generated",        today,
         "Model",            "YOLOv8s — bamboo fine-tuned"],
        ["Images Processed", str(df['Image'].nunique()),
         "Total Culms",      str(len(df))],
        ["Health Breakdown", health_str,
         "Avg Height",       f"{df['Height_cm'].mean():.2f} cm"],
        ["Total Biomass",    f"{total_biomass:.3f} kg",
         "Avg Diameter",     f"{df['Diameter_cm'].mean():.2f} cm"],
    ]
    summary_tbl = Table(summary_data, colWidths=[1.2*inch, 2.5*inch, 1.2*inch, 2.5*inch])
    summary_tbl.setStyle(TableStyle([
        ('FONTNAME',    (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',    (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 0), (0, -1), FOREST_RL),
        ('TEXTCOLOR',   (2, 0), (2, -1), FOREST_RL),
        ('TEXTCOLOR',   (1, 0), (1, -1), DARK_RL),
        ('TEXTCOLOR',   (3, 0), (3, -1), DARK_RL),
        ('BACKGROUND',  (0, 0), (-1, -1), BG_RL),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [BG_RL, CREAM_RL]),
        ('GRID',        (0, 0), (-1, -1), 0.4, KHAKI_RL),
        ('TOPPADDING',  (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    items += [summary_tbl, Spacer(1, 16)]

    # ── CO2 highlight ────────────────────────────────────────
    items += [
        HRFlowable(width="100%", thickness=1, color=SAGE_RL, spaceAfter=10),
        Paragraph(
            f"🌍  Total CO₂ Sequestered: <b>{total_co2:.3f} kg</b>  "
            f"({total_co2/1000:.5f} tonnes)  ·  "
            f"≈ <b>{trees_eq:.1f} trees</b> annual absorption equivalent",
            sCO2),
        Paragraph(
            f"Carbon factor: {CARBON_FACTOR} kg CO₂ / kg dry biomass  "
            f"(INBAR TR-37 / IPCC Tier 1)  ·  "
            f"Dry culm carbon penalty applied (×0.20)",
            sSub),
        HRFlowable(width="100%", thickness=1, color=SAGE_RL, spaceAfter=14),
    ]

    # ── SDG alignment ────────────────────────────────────────
    items += [
        Paragraph("<b>SDG Alignment</b>", sBodyB),
        Paragraph(
            "SDG 13 (Climate Action): Bamboo forests are a measurable, "
            "low-cost natural carbon sink.  SDG 15 (Life on Land): "
            "BambooSense supports sustainable forest management and "
            "biodiversity monitoring.  SDG 11 (Sustainable Cities): "
            "Bamboo biomass data informs construction material supply chains.",
            sBody),
        Spacer(1, 12),
    ]

    # ── Charts ───────────────────────────────────────────────
    Paragraph("Analytics Charts", sBodyB)
    for label, img_b in chart_dict.items():
        items += [
            Paragraph(label, sChartLabel),
            RLImage(io.BytesIO(img_b), width=5.2*inch, height=2.6*inch),
            Spacer(1, 12),
        ]

    items += [Spacer(1, 6)]

    # ── Data table ───────────────────────────────────────────
    items += [
        HRFlowable(width="100%", thickness=1, color=FOREST_RL, spaceAfter=8),
        Paragraph("Detailed Culm Measurements", sBodyB),
        Spacer(1, 6),
    ]

    cols   = ["Image", "Culm", "Health", "Height_cm", "Diameter_cm",
              "Biomass_kg", "Carbon_CO2_kg"]
    tdata  = [["Image", "Culm #", "Health", "Height (cm)",
               "Diameter (cm)", "Biomass (kg)", "CO₂ (kg)"]]
    tdata += [[str(row[c]) for c in cols] for _, row in df.iterrows()]

    col_widths = [1.4*inch, 0.55*inch, 0.75*inch, 0.95*inch,
                  1.0*inch, 0.9*inch, 0.85*inch]
    tbl = Table(tdata, repeatRows=1, colWidths=col_widths)

    row_styles = [
        ('BACKGROUND',   (0, 0), (-1, 0),  DARK_RL),
        ('TEXTCOLOR',    (0, 0), (-1, 0),  CREAM_RL),
        ('FONTNAME',     (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 8),
        ('GRID',         (0, 0), (-1, -1), 0.35, rl_colors.HexColor("#C8C0A0")),
        ('ALIGN',        (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1),
         [rl_colors.HexColor("#FDFAF2"), rl_colors.HexColor("#F0ECD8")]),
    ]
    health_row_colors = {
        "Green":  "#eef5e8",
        "Yellow": "#f8f4dc",
        "Dry":    "#faeee6",
    }
    for row_i, (_, row) in enumerate(df.iterrows(), start=1):
        bg = health_row_colors.get(row["Health"], "#FDFAF2")
        row_styles.append(
            ('BACKGROUND', (2, row_i), (2, row_i), rl_colors.HexColor(bg)))
        # Colour the CO2 cell green gradient
        co2_val = float(row["Carbon_CO2_kg"])
        max_co2 = float(df["Carbon_CO2_kg"].max()) + 1e-9
        intensity = int(180 + 60 * (1 - co2_val / max_co2))
        row_styles.append(
            ('TEXTCOLOR', (6, row_i), (6, row_i), FOREST_RL))

    tbl.setStyle(TableStyle(row_styles))
    items += [tbl, Spacer(1, 20)]

    # ── Footer ───────────────────────────────────────────────
    items += [
        HRFlowable(width="100%", thickness=0.5, color=KHAKI_RL, spaceAfter=8),
        Paragraph(
            f"BambooSense v4.0  ·  YOLOv8-Small  ·  "
            f"Generated {today}  ·  "
            f"Supporting SDG 13 · SDG 15 · SDG 11",
            sFooter),
    ]

    doc.build(items)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🌱 Computer Vision for Sustainable Forestry</div>
  <h1>Bamboo<span>Sense</span></h1>
  <p class="tagline">Growth Monitoring · Biomass Estimation · Carbon Assessment</p>
  <p class="mission">
    Bamboo is one of Earth's most powerful natural carbon sinks — growing up to 90 cm/day
    and sequestering 2–3× more CO₂ than equivalent tree plantations.
    BambooSense uses computer vision to provide scalable, field-ready measurement
    for forest management, biomass estimation, and climate impact reporting.
    <br><br>
  
  </p>
  <div class="sdg-row">
    <span class="sdg-badge">🌍 SDG 13 · Climate Action</span>
    <span class="sdg-badge">🌿 SDG 15 · Life on Land</span>
    <span class="sdg-badge">🏙 SDG 11 · Sustainable Cities</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# HEALTH LEGEND
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🌿 Culm Health Classification</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="health-legend">
  <div class="health-badge hb-green">
    <div class="dot" style="background:#4D7A1E"></div>
    🟢 <b>Green</b> — Healthy · Biomass ×1.00 · Carbon ×1.00
  </div>
  <div class="health-badge hb-yellow">
    <div class="dot" style="background:#A3B86A"></div>
    🟡 <b>Yellow</b> — Stressed · Biomass ×0.65 · Carbon ×0.65
  </div>
  <div class="health-badge hb-dry">
    <div class="dot" style="background:#D45A10"></div>
    🔴 <b>Dry / Dead</b> — Biomass ×0.35 · Carbon ×0.20
  </div>
</div>

<hr class="section-rule">
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# UPLOAD
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📸 Upload Field Images</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload bamboo field images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("🔬  Analyse Images",
                    use_container_width=True, disabled=not uploaded)
with col_info:
    if not uploaded:
        st.caption("Upload one or more field images, then click **Analyse Images**.")
    else:
        st.caption(f"**{len(uploaded)} image(s)** ready — click **Analyse Images** to begin.")

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═════════════════════════════════════════════════════════════
if run and uploaded:

    with st.spinner("Loading detection model…"):
        model = load_model()

    scale_i, scale_d = get_scale_factors()

    all_rows, annotated = [], {}
    prog = st.progress(0, text="Running YOLO detection…")

    for idx, f in enumerate(uploaded):
        pil = Image.open(f).convert("RGB")
        rows, ann_img, n_nodes = analyse_image(model, pil, scale_i, scale_d)
        for r in rows:
            r["Image"] = f.name
        all_rows.extend(rows)
        annotated[f.name] = (ann_img, n_nodes)
        prog.progress((idx + 1) / len(uploaded), text=f"Processed: {f.name}")

    prog.empty()

    if not all_rows:
        st.error("No bamboo culms detected. Try images with clearer culm visibility "
                 "or lower the confidence threshold.")
        st.stop()

    df = pd.DataFrame(all_rows)
    total_biomass = df["Biomass_kg"].sum()
    total_carbon  = df["Carbon_CO2_kg"].sum()
    health_counts = df["Health"].value_counts().to_dict()
    n_green  = health_counts.get("Green",  0)
    n_yellow = health_counts.get("Yellow", 0)
    n_dry    = health_counts.get("Dry",    0)

    # ── STAT PILLS ────────────────────────────────────────────
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-pill">
        <div class="stat-icon">🖼️</div>
        <div class="stat-val dark">{df['Image'].nunique()}</div>
        <div class="stat-lbl">Images Processed</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">🎋</div>
        <div class="stat-val dark">{len(df)}</div>
        <div class="stat-lbl">Culms Detected</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(77,122,30,0.35)">
        <div class="stat-icon">🟢</div>
        <div class="stat-val">{n_green}</div>
        <div class="stat-lbl">Green Culms</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(163,184,106,0.45)">
        <div class="stat-icon">🟡</div>
        <div class="stat-val khaki">{n_yellow}</div>
        <div class="stat-lbl">Yellow Culms</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(212,90,16,0.35)">
        <div class="stat-icon">🔴</div>
        <div class="stat-val orange">{n_dry}</div>
        <div class="stat-lbl">Dry Culms</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">📏</div>
        <div class="stat-val dark">{df['Height_cm'].mean():.1f}</div>
        <div class="stat-lbl">Avg Height (cm)</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">⚖️</div>
        <div class="stat-val">{total_biomass:.2f}</div>
        <div class="stat-lbl">Total Biomass (kg)*</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(42,61,20,0.3)">
        <div class="stat-icon">🌍</div>
        <div class="stat-val dark">{total_carbon:.2f}</div>
        <div class="stat-lbl">CO₂ Sequestered (kg)*</div>
      </div>
    </div>
    <p style="font-size:11px;color:#9B8C60;margin:-6px 0 14px">
       dry/stressed culms contribute reduced biomass and carbon.
    </p>
    """, unsafe_allow_html=True)

    # ── CARBON BANNER ─────────────────────────────────────────
    trees_eq = total_carbon / 21.77
    st.markdown(f"""
    <div class="carbon-banner">
      <div style="font-size:46px;flex-shrink:0">🌍</div>
      <div>
        <div class="co2-num">{total_carbon:.2f}
          <span class="co2-unit">kg CO₂</span>
        </div>
        <div class="co2-detail">
          Carbon sequestered by this bamboo stand (health-adjusted)  ·
          Method: INBAR Technical Report 37 · Factor = {CARBON_FACTOR} kg CO₂/kg dry biomass
        </div>
        <div class="co2-equiv">
          🌳  ≈ {trees_eq:.1f} trees — annual CO₂ absorption equivalent
          &nbsp;·&nbsp; 🌱 {total_biomass:.2f} kg total biomass
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── ANNOTATED IMAGES ──────────────────────────────────────
    st.markdown('<div class="section-title">🖼️ Detection Results</div>',
                unsafe_allow_html=True)
    st.caption("Box colours: 🟢 Forest green = healthy · 🟡 Sage green = stressed/yellow · 🔴 Burnt orange = dry/dead · Orange border = node")

    img_cols = st.columns(min(len(annotated), 3))
    for i, (name, (img, n_nodes)) in enumerate(annotated.items()):
        with img_cols[i % 3]:
            st.image(img, caption=f"{name}  ·  {n_nodes} node(s)", use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Analytics</div>',
                unsafe_allow_html=True)

    f_bio    = chart_biomass(df)
    f_health = chart_health(df)
    f_hgt    = chart_height(df)
    f_co2    = chart_carbon(df)
    f_donut  = chart_health_donut(df)

    c1, c2 = st.columns(2)
    with c1: st.pyplot(f_bio,    use_container_width=True)
    with c2: st.pyplot(f_health, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3: st.pyplot(f_hgt,   use_container_width=True)
    with c4: st.pyplot(f_co2,   use_container_width=True)

    # Donut centred
    _, cd, _ = st.columns([1, 2, 1])
    with cd: st.pyplot(f_donut, use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── DATA TABLE ────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Measurement Data</div>',
                unsafe_allow_html=True)

    def _health_row_color(val):
        return {
            "Green":  "background-color:#eef5e8; color:#2A3D14",
            "Yellow": "background-color:#f8f4dc; color:#5B4A10",
            "Dry":    "background-color:#faeee6; color:#7A2E10",
        }.get(val, "")

    styled = (
        df.style
          .applymap(_health_row_color, subset=["Health"])
          .format({
              "Height_cm":     "{:.2f}",
              "Diameter_cm":   "{:.2f}",
              "Biomass_kg":    "{:.3f}",
              "Carbon_CO2_kg": "{:.3f}",
          })
          .background_gradient(subset=["Biomass_kg"],    cmap="YlGn")
          .background_gradient(subset=["Carbon_CO2_kg"], cmap="Greens")
    )
    st.dataframe(styled, use_container_width=True,
                 height=min(440, 55 + len(df) * 35))

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── EXPORTS ───────────────────────────────────────────────
    st.markdown('<div class="section-title">⬇️ Export Results</div>',
                unsafe_allow_html=True)

    st.caption("All exports use the BambooSense brand palette. The PDF report includes "
               "an SDG alignment statement, colour-coded data table, and all four charts.")

    b_bio    = fig_to_bytes(f_bio,    dpi=180)
    b_health = fig_to_bytes(f_health, dpi=180)
    b_hgt    = fig_to_bytes(f_hgt,    dpi=180)
    b_co2    = fig_to_bytes(f_co2,    dpi=180)
    b_donut  = fig_to_bytes(f_donut,  dpi=180)
    b_csv    = df.to_csv(index=False).encode()
    b_pdf    = build_pdf(df, {
        "Biomass per Image": b_bio,
        "Biomass by Health Status":             b_health,
        "Culm Height Distribution":             b_hgt,
        "CO₂ Sequestered":    b_co2,
    })

    d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
    with d1:
        st.download_button("📄 CSV",         b_csv,   "bamboo_data.csv",
                           "text/csv",        use_container_width=True)
    with d2:
        st.download_button("📑 PDF Report",  b_pdf,   "BambooSense_Report.pdf",
                           "application/pdf", use_container_width=True)
    with d3:
        st.download_button("📊 Biomass",     b_bio,   "biomass_chart.png",
                           "image/png",       use_container_width=True)
    with d4:
        st.download_button("🌿 Health",      b_health,"health_chart.png",
                           "image/png",       use_container_width=True)
    with d5:
        st.download_button("📈 Heights",     b_hgt,   "height_dist.png",
                           "image/png",       use_container_width=True)
    with d6:
        st.download_button("🌍 CO₂ Chart",   b_co2,   "carbon_chart.png",
                           "image/png",       use_container_width=True)
    with d7:
        st.download_button("🍩 Donut",       b_donut, "health_donut.png",
                           "image/png",       use_container_width=True)

    plt.close('all')