"""
BambooSense — Streamlit App
All calibration logic lives in  calibration.py
All inference  logic lives in   inference.py
This file is UI + orchestration only.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import json
from PIL import Image

# ── ReportLab ────────────────────────────────────────────────
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ── Local modules ────────────────────────────────────────────
from calibration import run_calibration, SCALE_CACHE, DEFAULT_CALIB_CONFIG
from inference   import (analyse_image, chart_biomass, chart_health,
                          chart_height, chart_carbon, fig_to_bytes,
                          CARBON_FACTOR, HEALTH_CONFIG)

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
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #eef6e4 0%, #f5fdf0 40%, #e8f5dc 100%);
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a2410 0%, #2e5c1e 55%, #4a7a30 100%);
    border-radius: 22px;
    padding: 48px 44px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,60,20,0.22);
}
.hero::before {
    content: '🎋';
    position: absolute; right: 36px; top: 50%;
    transform: translateY(-50%);
    font-size: 130px; opacity: 0.12; line-height: 1;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(138,184,104,0.12), transparent 65%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 52px; font-weight: 900;
    color: #ffffff; margin: 0 0 6px;
    letter-spacing: -0.03em; line-height: 1.1;
}
.hero h1 span { color: #8ab868; }
.hero .tagline {
    color: #c8e0a8;
    font-size: 13px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 18px;
}
.hero .mission {
    color: rgba(200,224,168,0.85);
    font-size: 14px; line-height: 1.7;
    max-width: 520px;
    border-left: 3px solid rgba(138,184,104,0.5);
    padding-left: 14px;
    margin: 0;
}

/* ── Stat Pills ── */
.stats-row { display:flex; gap:14px; flex-wrap:wrap; margin:24px 0; }
.stat-pill {
    flex:1; min-width:130px;
    background: linear-gradient(135deg, #f0f8e8, #e8f5dc);
    border: 1.5px solid rgba(90,138,56,0.25);
    border-radius: 14px; padding:18px 16px; text-align:center;
    box-shadow: 0 2px 10px rgba(58,90,40,0.07);
    transition: transform 0.2s;
}
.stat-pill:hover { transform: translateY(-2px); }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 34px; font-weight:900;
    color: #3a5a28; line-height:1;
}
.stat-val.carbon { color: #1b5e20; }
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 9px; text-transform:uppercase;
    letter-spacing:0.08em; color:#8aaa78; margin-top:5px;
}
.stat-icon { font-size:18px; margin-bottom:4px; }

/* ── Section Title ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 21px; font-weight:700;
    color: #1a2410; margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
}

/* ── Carbon Banner ── */
.carbon-banner {
    background: linear-gradient(135deg, #1b5e20, #2e7d32, #388e3c);
    border-radius: 16px; padding: 22px 28px;
    color: #fff; margin: 16px 0 24px;
    display: flex; align-items: center; gap: 20px;
    box-shadow: 0 4px 20px rgba(27,94,32,0.25);
}
.carbon-banner .co2-num {
    font-family: 'Playfair Display', serif;
    font-size: 48px; font-weight:900; line-height:1;
    flex-shrink:0;
}
.carbon-banner .co2-detail { font-size:13px; opacity:0.88; line-height:1.65; }
.carbon-banner .co2-detail strong { color:#a5d6a7; }

/* ── Health Legend ── */
.health-legend {
    display: flex; gap: 16px; flex-wrap: wrap;
    margin: 12px 0 20px;
}
.health-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 600;
    border: 1.5px solid rgba(0,0,0,0.12);
}
.health-badge .dot {
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
}
.health-green  { background: #e8f5e9; color: #1b5e20; }
.health-yellow { background: #fffde7; color: #f57f17; }
.health-dry    { background: #fbe9e7; color: #bf360c; }

/* ── Calibration Audit Card ── */
.calib-card {
    background: #f0f8e8;
    border: 1.5px solid rgba(90,138,56,0.3);
    border-radius: 14px; padding: 18px 22px;
    margin-bottom: 20px; font-size: 13px;
    color: #2d4a20; line-height: 1.7;
}
.calib-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 16px; color: #1a2410;
    margin: 0 0 10px;
}
.calib-ok   { color: #2e7d32; font-weight: 700; }
.calib-warn { color: #e65100; font-weight: 700; }

/* ── Info Cards (sidebar) ── */
.info-card {
    background: #f0f8e8;
    border-left: 4px solid #5a8a38;
    border-radius: 0 10px 10px 0;
    padding: 12px 14px; margin-bottom:12px;
    font-size:13px; color:#2d4a20; line-height:1.55;
}
.info-card b { color:#1a2410; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #5a8a38, #3a5a28) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; padding:12px 32px !important;
    font-family:'Lato', sans-serif !important; font-weight:700 !important;
    font-size:15px !important; letter-spacing:0.02em !important;
    box-shadow: 0 4px 16px rgba(90,138,56,0.35) !important;
    transition: box-shadow 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 28px rgba(90,138,56,0.48) !important;
}

div[data-testid="stFileUploader"] {
    background:#f8fdf4;
    border: 1.5px dashed rgba(90,138,56,0.35);
    border-radius:12px; padding:4px;
}

hr { border-color:rgba(90,138,56,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR  — about / methodology
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 About BambooSense")
    st.markdown("""
    <div class="info-card">
        <b>Mission</b><br>
        Automated, scalable monitoring of bamboo forests for biomass estimation,
        carbon assessment, and sustainable resource management.
    </div>
    <div class="info-card">
        <b>Biomass Formula</b><br>
        <code>B = K × D² × H × health_factor</code><br>
        K = 0.03 · D = diameter (cm) · H = height (cm)<br>
        Health factor: Green=1.0 · Yellow=0.65 · Dry=0.35
    </div>
    <div class="info-card">
        <b>Carbon Sequestration</b><br>
        Bamboo sequesters ≈ <b>0.25 kg CO₂ per kg</b> of dry biomass
        (INBAR TR-37 / IPCC Tier 1).<br>
        Dry culms receive an additional carbon penalty (factor 0.20)
        as decomposition has already released sequestered carbon.<br>
        <code>CO₂ = Biomass × CARBON_FACTOR × c_factor</code>
    </div>
    <div class="info-card">
        <b>Health Classification (HSV)</b><br>
        🟢 <b>Green</b>  — healthy, full biomass (100%)<br>
        🟡 <b>Yellow</b> — stressed / transitional (65%)<br>
        🔴 <b>Dry</b>    — dead or dry culm (35% biomass, 20% carbon)
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
        • Critical for construction, pulp & bioenergy<br>
        • Supports climate resilience & forest management
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("BambooSense v3.0 · YOLOv8 · Health-Aware Biomass · Streamlit")


# ─────────────────────────────────────────────────────────────
# MODEL + CALIBRATION  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    return YOLO("model/yolov8s_bamboo.pt")


@st.cache_data(show_spinner=False)
def get_scale_factors():
    """Load from cache or run calibration."""
    if os.path.exists(SCALE_CACHE):
        with open(SCALE_CACHE) as f:
            d = json.load(f)
        return d["internode_scale"], d["diameter_scale"]

    model = load_model()
    try:
        si, sd = run_calibration(model, DEFAULT_CALIB_CONFIG, SCALE_CACHE)
    except RuntimeError:
        # Calibration images absent — use sensible defaults
        si, sd = 0.08, 0.05
    return si, sd


def audit_calibration(scale_i: float, scale_d: float) -> dict:
    """
    Sanity-check the calibration scale factors against expected real-world ranges.

    Typical bamboo field photography (1–3 m distance, phone/DSLR):
      internode scale : 0.02 – 0.30 cm/px  (internode 15–45 cm, height ~100–500 px)
      diameter  scale : 0.01 – 0.20 cm/px  (diameter  3–12 cm,  width  ~30–300 px)

    Returns a dict with 'ok' bool and 'messages' list.
    """
    warnings = []
    ok = True

    if not (0.02 <= scale_i <= 0.30):
        warnings.append(
            f"⚠️  internode_scale {scale_i:.5f} cm/px is outside the expected range "
            f"[0.02, 0.30]. Check calibration image distance or real-world measurements."
        )
        ok = False
    else:
        warnings.append(f"✅  internode_scale {scale_i:.5f} cm/px — within normal range.")

    if not (0.01 <= scale_d <= 0.20):
        warnings.append(
            f"⚠️  diameter_scale {scale_d:.5f} cm/px is outside the expected range "
            f"[0.01, 0.20]. Verify diameter measurements or image resolution."
        )
        ok = False
    else:
        warnings.append(f"✅  diameter_scale {scale_d:.5f} cm/px — within normal range.")

    ratio = scale_i / scale_d if scale_d > 0 else 0
    if not (0.5 <= ratio <= 8.0):
        warnings.append(
            f"⚠️  Scale ratio (internode/diameter) = {ratio:.2f} is unusual "
            f"(expected 0.5–8.0). Both scales may be using inconsistent reference images."
        )
        ok = False
    else:
        warnings.append(f"✅  Scale ratio (internode/diameter) = {ratio:.2f} — looks consistent.")

    return {"ok": ok, "messages": warnings}


# ─────────────────────────────────────────────────────────────
# PDF BUILDER
# ─────────────────────────────────────────────────────────────
def build_pdf(df: pd.DataFrame, chart_dict: dict) -> bytes:
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, leftMargin=48, rightMargin=48,
                              topMargin=48, bottomMargin=48)

    T  = ParagraphStyle('T', fontName='Helvetica-Bold', fontSize=20,
                         textColor=colors.HexColor('#1a2410'), spaceAfter=4)
    S  = ParagraphStyle('S', fontName='Helvetica', fontSize=10,
                         textColor=colors.HexColor('#5a8a38'), spaceAfter=16)
    B  = ParagraphStyle('B', fontName='Helvetica', fontSize=11,
                         textColor=colors.HexColor('#243318'), spaceAfter=7)
    BC = ParagraphStyle('BC', fontName='Helvetica-Bold', fontSize=13,
                         textColor=colors.HexColor('#1b5e20'), spaceAfter=7)

    total_co2 = df["Carbon_CO2_kg"].sum()

    # Health breakdown
    health_counts = df["Health"].value_counts().to_dict()
    health_str = " · ".join(f"{h}: {n}" for h, n in health_counts.items())

    items = [
        Paragraph("BambooSense — Growth Monitoring Report", T),
        Paragraph("Automated Culm Detection · Biomass Estimation · Carbon Assessment", S),
        Spacer(1, 6),
        Paragraph(f"Images Processed : <b>{df['Image'].nunique()}</b>", B),
        Paragraph(f"Culms Detected   : <b>{len(df)}</b>", B),
        Paragraph(f"Culm Health      : <b>{health_str}</b>", B),
        Paragraph(f"Total Biomass    : <b>{df['Biomass_kg'].sum():.3f} kg</b> "
                  f"(health-corrected)", B),
        Paragraph(f"Avg Height       : <b>{df['Height_cm'].mean():.2f} cm</b>", B),
        Paragraph(f"Avg Diameter     : <b>{df['Diameter_cm'].mean():.2f} cm</b>", B),
        Spacer(1, 8),
        Paragraph(f"🌍  Total CO₂ Sequestered: <b>{total_co2:.3f} kg</b>  "
                  f"({total_co2/1000:.4f} tonne)", BC),
        Paragraph(f"Carbon factor: {CARBON_FACTOR} kg CO₂ / kg biomass  "
                  f"(INBAR TR-37 / IPCC Tier 1) — adjusted for dry/stressed culms", S),
        Spacer(1, 14),
    ]

    for label, img_b in chart_dict.items():
        items += [
            Paragraph(label, S),
            RLImage(io.BytesIO(img_b), width=4.8*inch, height=2.4*inch),
            Spacer(1, 10),
        ]

    cols  = ["Image", "Culm", "Health", "Height_cm", "Diameter_cm",
             "Biomass_kg", "Carbon_CO2_kg"]
    tdata = [cols] + [[str(row[c]) for c in cols] for _, row in df.iterrows()]
    tbl   = Table(tdata, repeatRows=1)

    # Row colour-code by health
    row_styles = [
        ('BACKGROUND',    (0, 0), (-1,  0), colors.HexColor('#3a5a28')),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 8),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#c8e0a8')),
        ('ALIGN',         (2, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND',    (6, 0), (6, -1), colors.HexColor('#e8f5e9')),  # carbon col
    ]
    health_row_colors = {
        "Green":  '#e8f5e9',
        "Yellow": '#fffde7',
        "Dry":    '#fbe9e7',
    }
    for row_i, (_, row) in enumerate(df.iterrows(), start=1):
        bg = health_row_colors.get(row["Health"], '#ffffff')
        row_styles.append(('BACKGROUND', (2, row_i), (2, row_i),
                            colors.HexColor(bg)))

    tbl.setStyle(TableStyle(row_styles))
    items += [Paragraph("Detailed Measurements", S), tbl]
    doc.build(items)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>Bamboo<span>Sense</span></h1>
  <p class="tagline">Automated Growth Monitoring · Biomass Estimation · Carbon Assessment</p>
  <p class="mission">
    Bamboo is one of Earth's most powerful natural carbon sinks — growing up to 90 cm/day
    and sequestering 2–3× more CO₂ than equivalent tree plantations.
    BambooSense uses computer vision to provide scalable, field-ready measurement
    for forest management, biomass estimation, and climate impact reporting.
    <br><br>
    <b style="color:#8ab868">New:</b> Culms are now classified as
    🟢 Green · 🟡 Yellow · 🔴 Dry — biomass and carbon values are automatically
    corrected for culm health status.
  </p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# CALIBRATION AUDIT
# ═════════════════════════════════════════════════════════════
with st.expander("🔧 Calibration Audit", expanded=False):
    scale_i, scale_d = get_scale_factors()
    audit = audit_calibration(scale_i, scale_d)

    status_icon = "✅ Calibration looks correct" if audit["ok"] else "⚠️ Calibration may have issues"
    status_cls  = "calib-ok" if audit["ok"] else "calib-warn"

    msgs_html = "".join(
        f"<div style='margin:4px 0'>{m}</div>" for m in audit["messages"]
    )

    st.markdown(f"""
    <div class="calib-card">
      <h4>📐 Scale Factor Audit</h4>
      <span class="{status_cls}">{status_icon}</span>
      <br><br>
      {msgs_html}
      <br>
      <b>internode_scale</b> = {scale_i:.6f} cm/px &nbsp;·&nbsp;
      <b>diameter_scale</b>  = {scale_d:.6f} cm/px
      <br><br>
      <small>
        <b>How calibration works:</b> The model detects nodes in calibration images.
        Adjacent node-centre Y-positions give internode pixel height;
        culm bounding-box width gives diameter pixels.
        Scale = real_cm / mean_px. If your calibration images were taken from different
        distances or with different zoom levels than your field images, re-run calibration
        with matching conditions.
      </small>
    </div>
    """, unsafe_allow_html=True)

    if not audit["ok"]:
        st.warning("Re-run calibration with images taken at the same distance and zoom "
                   "as your field images, and double-check your real-world measurements.")


# ═════════════════════════════════════════════════════════════
# HEALTH LEGEND
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🌿 Culm Health Classification</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="health-legend">
  <div class="health-badge health-green">
    <div class="dot" style="background:#34A853"></div>
    🟢 <b>Green</b> — Healthy · Biomass ×1.00 · Carbon ×1.00
  </div>
  <div class="health-badge health-yellow">
    <div class="dot" style="background:#DCBC00"></div>
    🟡 <b>Yellow</b> — Stressed · Biomass ×0.65 · Carbon ×0.65
  </div>
  <div class="health-badge health-dry">
    <div class="dot" style="background:#D04020"></div>
    🔴 <b>Dry/Dead</b> — Biomass ×0.35 · Carbon ×0.20
  </div>
</div>
<p style="font-size:12px;color:#6a8a5a;margin-top:4px">
  Classification uses HSV colour-space analysis of each culm's bounding-box region.
  Dry bamboo has significantly lower moisture content and its sequestered carbon is
  actively being released through decomposition — hence the steeper carbon penalty.
</p>
""", unsafe_allow_html=True)

st.markdown("---")


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

st.markdown("---")


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
        st.error("No bamboo culms detected. "
                 "Try images with clearer culm visibility or lower the confidence threshold.")
        st.stop()

    df = pd.DataFrame(all_rows)

    total_biomass = df["Biomass_kg"].sum()
    total_carbon  = df["Carbon_CO2_kg"].sum()

    # Health counts for summary pills
    health_counts = df["Health"].value_counts().to_dict()
    n_green  = health_counts.get("Green",  0)
    n_yellow = health_counts.get("Yellow", 0)
    n_dry    = health_counts.get("Dry",    0)

    # ── STAT PILLS ────────────────────────────────────────
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-pill">
        <div class="stat-icon">🖼️</div>
        <div class="stat-val">{df['Image'].nunique()}</div>
        <div class="stat-lbl">Images Processed</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">🌿</div>
        <div class="stat-val">{len(df)}</div>
        <div class="stat-lbl">Culms Detected</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(52,168,83,0.4)">
        <div class="stat-icon">🟢</div>
        <div class="stat-val" style="color:#2e7d32">{n_green}</div>
        <div class="stat-lbl">Green Culms</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(220,188,0,0.4)">
        <div class="stat-icon">🟡</div>
        <div class="stat-val" style="color:#f57f17">{n_yellow}</div>
        <div class="stat-lbl">Yellow Culms</div>
      </div>
      <div class="stat-pill" style="border-color:rgba(208,64,32,0.4)">
        <div class="stat-icon">🔴</div>
        <div class="stat-val" style="color:#bf360c">{n_dry}</div>
        <div class="stat-lbl">Dry Culms</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">📏</div>
        <div class="stat-val">{df['Height_cm'].mean():.1f}</div>
        <div class="stat-lbl">Avg Height (cm)</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">⚖️</div>
        <div class="stat-val">{total_biomass:.2f}</div>
        <div class="stat-lbl">Total Biomass (kg)*</div>
      </div>
      <div class="stat-pill">
        <div class="stat-icon">🌍</div>
        <div class="stat-val carbon">{total_carbon:.2f}</div>
        <div class="stat-lbl">CO₂ Sequestered (kg)*</div>
      </div>
    </div>
    <p style="font-size:11px;color:#8aaa78;margin:-10px 0 16px">
      * Values are health-corrected: dry and stressed culms contribute less biomass and carbon.
    </p>
    """, unsafe_allow_html=True)

    # ── CARBON HIGHLIGHT BANNER ───────────────────────────
    trees_equiv = total_carbon / 21.77
    st.markdown(f"""
    <div class="carbon-banner">
      <div style="font-size:42px">🌍</div>
      <div>
        <div class="co2-num">{total_carbon:.2f} <span style="font-size:22px;font-weight:400">kg CO₂</span></div>
        <div class="co2-detail">
          Carbon sequestered by this bamboo stand (health-adjusted) ·
          <strong>≈ {trees_equiv:.1f} trees</strong> worth of annual absorption<br>
          Method: INBAR Technical Report 37 · Base factor = {CARBON_FACTOR} kg CO₂ / kg dry biomass ·
          Dry culm penalty applied (×0.20 carbon factor)
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ANNOTATED IMAGES ──────────────────────────────────
    st.markdown('<div class="section-title">🖼️ Detection Results</div>',
                unsafe_allow_html=True)
    st.caption("Bounding-box colours: 🟢 Green = healthy · 🟡 Yellow = stressed · 🔴 Dry = dead/dry · 🟠 Orange = node")
    img_cols = st.columns(min(len(annotated), 3))
    for i, (name, (img, n_nodes)) in enumerate(annotated.items()):
        with img_cols[i % 3]:
            st.image(img,
                     caption=f"{name}  ·  {n_nodes} node(s) detected",
                     use_container_width=True)

    st.markdown("---")

    # ── CHARTS ────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Analytics</div>',
                unsafe_allow_html=True)

    f_bio    = chart_biomass(df)
    f_health = chart_health(df)
    f_hgt    = chart_height(df)
    f_co2    = chart_carbon(df)

    c1, c2 = st.columns(2)
    with c1: st.pyplot(f_bio,    use_container_width=True)
    with c2: st.pyplot(f_health, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3: st.pyplot(f_hgt, use_container_width=True)
    with c4: st.pyplot(f_co2, use_container_width=True)

    st.markdown("---")

    # ── DATA TABLE ────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Measurement Data</div>',
                unsafe_allow_html=True)

    def _health_row_color(val):
        return {
            "Green":  "background-color: #e8f5e9",
            "Yellow": "background-color: #fffde7",
            "Dry":    "background-color: #fbe9e7",
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
                 height=min(420, 55 + len(df) * 35))

    st.markdown("---")

    # ── DOWNLOADS ─────────────────────────────────────────
    st.markdown('<div class="section-title">⬇️ Export Results</div>',
                unsafe_allow_html=True)

    b_bio    = fig_to_bytes(f_bio)
    b_health = fig_to_bytes(f_health)
    b_hgt    = fig_to_bytes(f_hgt)
    b_co2    = fig_to_bytes(f_co2)
    b_csv    = df.to_csv(index=False).encode()
    b_pdf    = build_pdf(df, {
        "Biomass per Image (health-corrected)": b_bio,
        "Biomass by Health Status":             b_health,
        "Culm Height Distribution":             b_hgt,
        "CO₂ Sequestered (health-adjusted)":    b_co2,
    })

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        st.download_button("📄 CSV Data",      b_csv,    "bamboo_data.csv",
                           "text/csv",         use_container_width=True)
    with d2:
        st.download_button("📑 PDF Report",    b_pdf,    "BambooSense_Report.pdf",
                           "application/pdf",  use_container_width=True)
    with d3:
        st.download_button("📊 Biomass Chart", b_bio,    "biomass_chart.png",
                           "image/png",        use_container_width=True)
    with d4:
        st.download_button("🌿 Health Chart",  b_health, "health_chart.png",
                           "image/png",        use_container_width=True)
    with d5:
        st.download_button("📈 Height Dist.",  b_hgt,    "height_distribution.png",
                           "image/png",        use_container_width=True)
    with d6:
        st.download_button("🌍 CO₂ Chart",     b_co2,    "carbon_chart.png",
                           "image/png",        use_container_width=True)

    plt.close('all')