# ================================
# 🌿 INFERENCE
# ================================
# Run standalone (python inference.py) to batch-process test_data/ folder,
# OR import analyse_image / run_batch_inference from app.py.

import cv2
import numpy as np
import pandas as pd
import os
import io
import json
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
BIOMASS_K = 0.03          # allometric constant  (kg / cm³)

# Carbon sequestration factor for bamboo:
#   Bamboo biomass is ~47 % carbon by dry weight.
#   1 kg of carbon = 3.67 kg CO₂ equivalent.
#   → 1 kg biomass ≈ 0.47 × 3.67 ≈ 1.72 kg CO₂ sequestered.
#   We keep a conservative published value of ~0.25 tonne CO₂ per tonne dry biomass
#   (INBAR Technical Report 37 / IPCC guidelines Tier 1).
CARBON_FACTOR = 0.25      # kg CO₂ per kg of dry bamboo biomass


# ─────────────────────────────────────────────────────────
# CORE: analyse a single PIL image
# ─────────────────────────────────────────────────────────
def analyse_image(model, pil_img, scale_internode: float, scale_diameter: float,
                  conf_threshold: float = 0.3):
    """
    Detect culms and nodes in *pil_img*, compute per-culm measurements, and
    return annotated image + per-culm data rows.

    Parameters
    ----------
    model            : loaded YOLO model
    pil_img          : PIL.Image (RGB)
    scale_internode  : cm / px (height axis)
    scale_diameter   : cm / px (width axis)
    conf_threshold   : minimum detection confidence

    Returns
    -------
    rows      : list[dict]  — per-culm measurements
    ann_img   : PIL.Image   — annotated RGB image
    n_nodes   : int         — number of nodes detected
    """
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    vis     = img_bgr.copy()
    results = model(img_bgr, verbose=False)

    culms, nodes = [], []
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < conf_threshold:
                continue
            cls   = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "culm":
                culms.append((x1, y1, x2, y2))
            elif label == "node":
                nodes.append((x1, y1, x2, y2))

    rows = []
    for i, (x1, y1, x2, y2) in enumerate(culms):
        h_cm   = (y2 - y1) * scale_internode
        d_cm   = (x2 - x1) * scale_diameter
        bm_kg  = BIOMASS_K * (d_cm ** 2) * h_cm
        co2_kg = bm_kg * CARBON_FACTOR

        rows.append({
            "Culm":        i + 1,
            "Height_cm":   round(h_cm,   2),
            "Diameter_cm": round(d_cm,   2),
            "Biomass_kg":  round(bm_kg,  3),
            "Carbon_CO2_kg": round(co2_kg, 3),
        })

        # ── Draw culm box ──────────────────────────────
        cv2.rectangle(vis, (x1, y1), (x2, y2), (52, 168, 83), 2)
        cv2.putText(vis,
                    f"#{i+1}  H:{h_cm:.0f}cm  B:{bm_kg:.2f}kg  C:{co2_kg:.2f}kg",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 168, 83), 2)

    # ── Draw node boxes ────────────────────────────────
    for j, (x1, y1, x2, y2) in enumerate(nodes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(vis, f"N{j+1}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    ann_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return rows, ann_img, len(nodes)


# ─────────────────────────────────────────────────────────
# STANDALONE BATCH PIPELINE
# ─────────────────────────────────────────────────────────
def run_batch_inference(model, scale_internode: float, scale_diameter: float,
                        input_folder: str = "test_data",
                        output_folder: str = "output",
                        conf_threshold: float = 0.3):
    """
    Process every image in *input_folder*, save annotated images to
    *output_folder*, and return a consolidated DataFrame.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_data = []

    for image_name in os.listdir(input_folder):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path  = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder,
                                   os.path.splitext(image_name)[0] + "_output.jpg")

        pil_img = Image.open(image_path).convert("RGB")
        rows, ann_img, n_nodes = analyse_image(
            model, pil_img, scale_internode, scale_diameter, conf_threshold)

        for r in rows:
            r["Image"] = image_name
        all_data.extend(rows)

        ann_img.save(output_path)
        print(f"  {image_name}: {len(rows)} culms, {n_nodes} nodes")

    df = pd.DataFrame(all_data)
    return df


# ─────────────────────────────────────────────────────────
# CHART HELPERS  (used by app.py and standalone run)
# ─────────────────────────────────────────────────────────
GREEN_PALETTE = ['#3a5a28', '#5a8a38', '#8ab868', '#c8e0a8', '#1a2410']


def _base_fig(w=7, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#f8fdf4')
    ax.set_facecolor('#f8fdf4')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#c8e0a8')
    ax.yaxis.grid(True, color='#e8f5dc', linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=9, colors='#3a5a28')
    return fig, ax


def chart_biomass(df: pd.DataFrame):
    fig, ax = _base_fig()
    totals = df.groupby("Image")["Biomass_kg"].sum()
    bars   = ax.bar(totals.index, totals.values,
                    color=GREEN_PALETTE[:len(totals)], edgecolor='none', width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.2f}", ha='center', va='bottom',
                fontsize=9, color='#3a5a28', fontweight='bold')
    ax.set_title("Biomass per Image (kg)", fontsize=13,
                 fontweight='bold', color='#1a2410', pad=12)
    ax.set_ylabel("Biomass (kg)", color='#5a8a38', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    return fig


def chart_height(df: pd.DataFrame):
    fig, ax = _base_fig()
    ax.hist(df["Height_cm"], bins=max(5, len(df) // 2),
            color='#5a8a38', edgecolor='#eef6e4', linewidth=0.5, alpha=0.9)
    ax.axvline(df["Height_cm"].mean(), color='#d4a843', linestyle='--',
               linewidth=1.5, label=f'Mean: {df["Height_cm"].mean():.1f} cm')
    ax.set_title("Culm Height Distribution", fontsize=13,
                 fontweight='bold', color='#1a2410', pad=12)
    ax.set_xlabel("Height (cm)", color='#5a8a38', fontsize=10)
    ax.set_ylabel("Count",       color='#5a8a38', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def chart_carbon(df: pd.DataFrame):
    fig, ax = _base_fig()
    totals = df.groupby("Image")["Carbon_CO2_kg"].sum()
    bars   = ax.bar(totals.index, totals.values,
                    color=['#2e7d32', '#388e3c', '#43a047', '#66bb6a', '#1b5e20'][:len(totals)],
                    edgecolor='none', width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                f"{b.get_height():.3f}", ha='center', va='bottom',
                fontsize=9, color='#1b5e20', fontweight='bold')
    ax.set_title("CO₂ Sequestered per Image (kg)", fontsize=13,
                 fontweight='bold', color='#1a2410', pad=12)
    ax.set_ylabel("CO₂ (kg)", color='#5a8a38', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
# STANDALONE ENTRY-POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ultralytics import YOLO

    SCALE_CACHE = "bamboo_scale.json"
    with open(SCALE_CACHE) as f:
        scale = json.load(f)

    _model = YOLO("model/yolov8s_bamboo.pt")
    print("Loaded scale factors:", scale)

    df = run_batch_inference(
        _model,
        scale_internode=scale["internode_scale"],
        scale_diameter=scale["diameter_scale"],
    )

    df.to_csv("output.csv", index=False)
    print("\n✅ CSV saved → output.csv")
    print(df.groupby("Image")[["Biomass_kg", "Carbon_CO2_kg"]].sum())

    chart_biomass(df).savefig("biomass_chart.png")
    chart_height(df).savefig("height_distribution.png")
    chart_carbon(df).savefig("carbon_chart.png")
    print("📊 Charts saved")
    plt.close('all')