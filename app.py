import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BambooSense",
    page_icon="🎋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #eef6e4 0%, #f5fdf0 40%, #e8f5dc 100%);
}

.hero {
    background: linear-gradient(135deg, #1a2410 0%, #3a5a28 60%, #5a8a38 100%);
    border-radius: 20px;
    padding: 48px 40px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🎋';
    position: absolute;
    right: 40px; top: 50%;
    transform: translateY(-50%);
    font-size: 120px;
    opacity: 0.15;
    line-height: 1;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 8px;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero h1 span { color: #8ab868; }
.hero p {
    color: #c8e0a8;
    font-size: 15px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
}

.stats-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 24px 0;
}
.stat-pill {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, #f0f8e8, #e8f5dc);
    border: 1px solid rgba(90,138,56,0.2);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 900;
    color: #3a5a28;
    line-height: 1;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8aaa78;
    margin-top: 6px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: #1a2410;
    margin-bottom: 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #5a8a38, #3a5a28) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(90,138,56,0.35) !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 24px rgba(90,138,56,0.45) !important;
}

div[data-testid="stFileUploader"] {
    background: #f8fdf4;
    border: 1.5px solid rgba(90,138,56,0.2);
    border-radius: 12px;
    padding: 4px;
}

hr { border-color: rgba(90,138,56,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("model/yolov8s_bamboo.pt")


# ─────────────────────────────────────────────────────────────
# CALIBRATION — completely hidden from the user
# Runs once, caches result to bamboo_scale.json
# ─────────────────────────────────────────────────────────────
CALIB_CONFIG = [
    {"path": "calibration_images/calib1.jpeg", "internode_cm": 27,   "diameter_cm": 5.73},
    {"path": "calibration_images/calib2.jpg",  "internode_cm": 25,   "diameter_cm": 6.69},
]
SCALE_CACHE = "bamboo_scale.json"


def _scale_from_image(model, path, real_internode, real_diam):
    img = cv2.imread(path)
    if img is None:
        return None, None
    results = model(img, verbose=False)
    culms, nodes = [], []
    for r in results:
        for box in r.boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "culm":   culms.append((x1, y1, x2, y2))
            elif label == "node": nodes.append((x1, y1, x2, y2))

    nc = sorted([(y1 + y2) // 2 for (_, y1, _, y2) in nodes])
    internode_px = [nc[i] - nc[i-1] for i in range(1, len(nc))]
    mean_i = np.mean(internode_px) if internode_px else 1
    mean_d = np.mean([(x2 - x1) for (x1, y1, x2, y2) in culms]) if culms else 1
    return real_internode / mean_i, real_diam / mean_d


@st.cache_data(show_spinner=False)
def get_scale_factors():
    if os.path.exists(SCALE_CACHE):
        with open(SCALE_CACHE) as f:
            d = json.load(f)
        return d["internode_scale"], d["diameter_scale"]

    model = load_model()
    si_list, sd_list = [], []
    for cfg in CALIB_CONFIG:
        if not os.path.exists(cfg["path"]):
            continue
        si, sd = _scale_from_image(model, cfg["path"], cfg["internode_cm"], cfg["diameter_cm"])
        if si and sd:
            si_list.append(si)
            sd_list.append(sd)

    # Fallback if calibration images absent
    final_i = float(np.mean(si_list)) if si_list else 0.08
    final_d = float(np.mean(sd_list)) if sd_list else 0.05

    with open(SCALE_CACHE, "w") as f:
        json.dump({"internode_scale": final_i, "diameter_scale": final_d}, f)
    return final_i, final_d


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def analyse_image(model, pil_img, scale_i, scale_d, conf=0.3):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    vis     = img_bgr.copy()
    results = model(img_bgr, verbose=False)
    culms, nodes = [], []

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < conf:
                continue
            cls   = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "culm":   culms.append((x1, y1, x2, y2))
            elif label == "node": nodes.append((x1, y1, x2, y2))

    K = 0.03
    rows = []
    for i, (x1, y1, x2, y2) in enumerate(culms):
        h_cm = (y2 - y1) * scale_i
        d_cm = (x2 - x1) * scale_d
        bm   = K * (d_cm ** 2) * h_cm
        rows.append({"Culm": i+1, "Height_cm": round(h_cm, 2),
                     "Diameter_cm": round(d_cm, 2), "Biomass_kg": round(bm, 3)})
        cv2.rectangle(vis, (x1, y1), (x2, y2), (52, 168, 83), 2)
        cv2.putText(vis, f"#{i+1}  H:{h_cm:.0f}cm  B:{bm:.2f}kg",
                    (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 168, 83), 2)

    for j, (x1, y1, x2, y2) in enumerate(nodes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(vis, f"N{j+1}", (x1, max(y1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    return rows, Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), len(nodes)


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
GREEN_PALETTE = ['#3a5a28','#5a8a38','#8ab868','#c8e0a8','#1a2410']

def _base_fig(w=7, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#f8fdf4')
    ax.set_facecolor('#f8fdf4')
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#c8e0a8')
    ax.yaxis.grid(True, color='#e8f5dc', linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=9, colors='#3a5a28')
    return fig, ax


def chart_biomass(df):
    fig, ax = _base_fig()
    totals = df.groupby("Image")["Biomass_kg"].sum()
    bars   = ax.bar(totals.index, totals.values,
                    color=GREEN_PALETTE[:len(totals)], edgecolor='none', width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.2f}", ha='center', va='bottom',
                fontsize=9, color='#3a5a28', fontweight='bold')
    ax.set_title("Biomass per Image", fontsize=13, fontweight='bold', color='#1a2410', pad=12)
    ax.set_ylabel("Biomass (kg)", color='#5a8a38', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    return fig


def chart_height(df):
    fig, ax = _base_fig()
    ax.hist(df["Height_cm"], bins=max(5, len(df)//2),
            color='#5a8a38', edgecolor='#eef6e4', linewidth=0.5, alpha=0.9)
    ax.axvline(df["Height_cm"].mean(), color='#d4a843', linestyle='--',
               linewidth=1.5, label=f'Mean: {df["Height_cm"].mean():.1f} cm')
    ax.set_title("Height Distribution", fontsize=13, fontweight='bold', color='#1a2410', pad=12)
    ax.set_xlabel("Height (cm)", color='#5a8a38', fontsize=10)
    ax.set_ylabel("Count",       color='#5a8a38', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig




def fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────
def build_pdf(df, chart_dict: dict) -> bytes:
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, leftMargin=48, rightMargin=48,
                             topMargin=48, bottomMargin=48)
    stls = getSampleStyleSheet()
    T    = ParagraphStyle('T', fontName='Helvetica-Bold', fontSize=20,
                          textColor=colors.HexColor('#1a2410'), spaceAfter=6)
    S    = ParagraphStyle('S', fontName='Helvetica', fontSize=10,
                          textColor=colors.HexColor('#5a8a38'), spaceAfter=18)
    B    = ParagraphStyle('B', fontName='Helvetica', fontSize=11,
                          textColor=colors.HexColor('#243318'), spaceAfter=8)

    items = [
        Paragraph("BambooSense — Growth Monitoring Report", T),
        Paragraph("Automated Culm Detection · Biomass Estimation · Carbon Assessment", S),
        Spacer(1, 8),
        Paragraph(f"Images Processed: <b>{df['Image'].nunique()}</b>", B),
        Paragraph(f"Culms Detected:   <b>{len(df)}</b>", B),
        Paragraph(f"Total Biomass:    <b>{df['Biomass_kg'].sum():.3f} kg</b>", B),
        Paragraph(f"Avg Height:       <b>{df['Height_cm'].mean():.2f} cm</b>", B),
        Paragraph(f"Avg Diameter:     <b>{df['Diameter_cm'].mean():.2f} cm</b>", B),
        Spacer(1, 16),
    ]

    for label, img_b in chart_dict.items():
        items += [
            Paragraph(label, S),
            RLImage(io.BytesIO(img_b), width=4.5*inch, height=2.2*inch),
            Spacer(1, 10),
        ]

    cols  = ["Image","Culm","Height_cm","Diameter_cm","Biomass_kg"]
    tdata = [cols] + [[str(row[c]) for c in cols] for _, row in df.iterrows()]
    tbl   = Table(tdata, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3a5a28')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8fdf4'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#c8e0a8')),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
    ]))
    items += [Paragraph("Detailed Measurements", S), tbl]
    doc.build(items)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>Bamboo<span>Sense</span></h1>
  <p>Automated Growth Monitoring &amp; Biomass Estimation · YOLO-Powered</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ──
st.markdown('<div class="section-title">📸 Upload Field Images</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload bamboo field images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("🔬  Analyse Images", use_container_width=True, disabled=not uploaded)
with col_info:
    if not uploaded:
        st.caption("Upload one or more field images, then click **Analyse Images**.")
    else:
        st.caption(f"**{len(uploaded)} image(s)** ready for analysis.")

st.markdown("---")

# ── Pipeline ──
if run and uploaded:

    with st.spinner("Loading detection model…"):
        model = load_model()

    # Calibration runs silently
    scale_i, scale_d = get_scale_factors()

    all_rows, annotated = [], {}
    prog = st.progress(0, text="Running detection…")

    for idx, f in enumerate(uploaded):
        pil = Image.open(f).convert("RGB")
        rows, ann, n_nodes = analyse_image(model, pil, scale_i, scale_d)
        for r in rows:
            r["Image"] = f.name
        all_rows.extend(rows)
        annotated[f.name] = (ann, n_nodes)
        prog.progress((idx + 1) / len(uploaded), text=f"Processed: {f.name}")

    prog.empty()

    if not all_rows:
        st.error("No bamboo culms detected. Try images with clearer culm visibility.")
        st.stop()

    df = pd.DataFrame(all_rows)

    # ── Stats ──
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-pill">
        <div class="stat-val">{df['Image'].nunique()}</div>
        <div class="stat-lbl">Images</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{len(df)}</div>
        <div class="stat-lbl">Culms Detected</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{df['Biomass_kg'].sum():.2f}</div>
        <div class="stat-lbl">Total Biomass (kg)</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{df['Height_cm'].mean():.1f}</div>
        <div class="stat-lbl">Avg Height (cm)</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{df['Diameter_cm'].mean():.2f}</div>
        <div class="stat-lbl">Avg Diameter (cm)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Annotated images ──
    st.markdown('<div class="section-title">🖼️ Detection Results</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(annotated), 3))
    for i, (name, (img, n_nodes)) in enumerate(annotated.items()):
        with cols[i % 3]:
            st.image(img, caption=f"{name}  ·  {n_nodes} nodes detected",
                     use_container_width=True)

    st.markdown("---")

    # ── Charts ──
    st.markdown('<div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)
    f_bio  = chart_biomass(df)
    f_hgt  = chart_height(df)
    

    c1, c2 = st.columns(2)
    with c1: st.pyplot(f_bio,  use_container_width=True)
    with c2: st.pyplot(f_hgt,  use_container_width=True)
   

    st.markdown("---")

    # ── Data table ──
    st.markdown('<div class="section-title">📋 Measurement Data</div>', unsafe_allow_html=True)
    st.dataframe(
        df.style
          .format({"Height_cm": "{:.2f}", "Diameter_cm": "{:.2f}", "Biomass_kg": "{:.3f}"})
          .background_gradient(subset=["Biomass_kg"], cmap="YlGn"),
        use_container_width=True,
        height=min(420, 55 + len(df) * 35),
    )

    st.markdown("---")

    # ── Downloads ──
    st.markdown('<div class="section-title">⬇️ Export Results</div>', unsafe_allow_html=True)

    b_bio  = fig_bytes(f_bio)
    b_hgt  = fig_bytes(f_hgt)

    b_pdf  = build_pdf(df, {
        "Biomass per Image":   b_bio,
        "Height Distribution": b_hgt,
    })
    b_csv  = df.to_csv(index=False).encode()

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1: st.download_button("📄 CSV",           b_csv,  "bamboo_data.csv",         "text/csv",         use_container_width=True)
    with d2: st.download_button("📑 PDF Report",    b_pdf,  "BambooSense_Report.pdf",  "application/pdf",  use_container_width=True)
    with d3: st.download_button("📊 Biomass Chart", b_bio,  "biomass_chart.png",       "image/png",        use_container_width=True)
    with d4: st.download_button("📈 Height Dist.",  b_hgt,  "height_distribution.png", "image/png",        use_container_width=True)
  

    plt.close('all')