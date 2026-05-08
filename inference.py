# ================================
# 🌿 INFERENCE  — BambooSense v4.0
# ================================
# Palette :
#   Cream      #E8DEBB   background / parchment
#   Khaki      #9B8C60   aged / neutral
#   Sage       #A3B86A   yellow-green / transitional culms
#   Forest     #4D7A1E   healthy green culms
#   Burnt-Org  #D45A10   dry / dead culms  (also accent)

import cv2
import numpy as np
import pandas as pd
import os
import io
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import datetime


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
BIOMASS_K     = 0.03      # allometric constant  (kg / cm³)
CARBON_FACTOR = 0.25      # kg CO₂ per kg dry bamboo biomass  (INBAR TR-37)

# Brand palette
C_CREAM  = "#E8DEBB"
C_KHAKI  = "#9B8C60"
C_SAGE   = "#A3B86A"
C_FOREST = "#4D7A1E"
C_ORANGE = "#D45A10"
C_DARK   = "#2A3D14"      # very dark green for text / axes
C_LIGHT  = "#F5F2E8"      # near-white warm background


# ─────────────────────────────────────────────────────────
# 🌿 BAMBOO HEALTH CLASSIFICATION  — HSV + Saturation-aware
# ─────────────────────────────────────────────────────────
#
#   The critical problem:  a *yellow* bamboo culm has a warm hue (≈20-34 in
#   OpenCV HSV 0-180 space) AND moderate-to-good saturation (≥60).
#   A *dry/dead* culm is also brownish/yellowish but with LOW saturation
#   (the colour has "washed out" to tan or grey-brown).
#
#   logic:
#     1. Mask out very dark + very desaturated pixels (background/shadow).
#     2. Compute per-label hue-match fractions AS BEFORE.
#     3. Apply a saturation tiebreaker:
#        • If the dominant hue falls in the Yellow range AND mean_sat ≥ 70
#          → always call it Yellow (stressed), not Dry.
#        • If the dominant hue falls in the Green range but mean_sat < 40
#          → downgrade to Dry (faded / very pale).
#        • Brown/orange hues with low saturation (mean_sat < 55) → Dry.
#        • Brown/orange hues with moderate saturation → Yellow.
#
#   This matches the bamboo palette swatches:
#     C_FOREST (#4D7A1E) → hue ≈ 79   → Green bucket  ✔
#     C_SAGE   (#A3B86A) → hue ≈ 75   → Green bucket, but low sat → Yellow ✔
#     C_KHAKI  (#9B8C60) → hue ≈ 33   → Yellow bucket, low sat → Dry ✔
#     C_ORANGE (#D45A10) → hue ≈ 13   → Dry bucket  ✔

HEALTH_CONFIG = {
    "Healthy":  (1.00, 1.00, (30, 122, 77), C_FOREST),
    "Stressed": (0.65, 0.65, (106, 184, 163), C_SAGE),
    "Degraded": (0.35, 0.20, (16, 90, 212), C_ORANGE),
}

# HSV hue ranges  (0–180 in OpenCV)
_HSV_RANGES = {
    "Green":  [(36,  90)],            # true greens including forest & sage
    "Yellow": [(15,  35), (91, 100)], # warm yellow-green, yellow-brown bridge
    "Dry":    [(0,   14), (101, 180)],# red/brown/grey, wraps at 0
}

# Saturation thresholds for tiebreaking
_SAT_VIVID      = 70   # above → colour is vivid enough to trust hue
_SAT_WASHED_OUT = 50   # below → bamboo has lost colour → Dry


def _classify_culm_health(img_bgr: np.ndarray,
                           x1: int, y1: int,
                           x2: int, y2: int) -> str:
    """
    Returns HEALTH STATUS (not color labels):
    Healthy / Stressed / Degraded
    """

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "Degraded"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].flatten().astype(float)
    sat = hsv[:, :, 1].flatten().astype(float)
    val = hsv[:, :, 2].flatten().astype(float)

    mask = (sat > 25) & (val > 45) & (val < 245)
    if mask.sum() < 40:
        return "Degraded"

    hue_m = hue[mask]
    sat_m = sat[mask]
    mean_sat = float(np.mean(sat_m))

    total = len(hue_m)
    fracs = {}

    for label, ranges in _HSV_RANGES.items():
        n = sum(((hue_m >= lo) & (hue_m <= hi)).sum() for lo, hi in ranges)
        fracs[label] = n / total

    dominant = max(fracs, key=fracs.get)

    # ── Mapping to HEALTH STATUS ─────────────────────

    if dominant == "Green":
        if mean_sat < _SAT_WASHED_OUT - 10:
            return "Degraded"
        if mean_sat < _SAT_VIVID:
            return "Stressed"
        return "Healthy"

    if dominant == "Yellow":
        if mean_sat < _SAT_WASHED_OUT:
            return "Degraded"
        return "Stressed"

    # dominant == "Dry"
    if mean_sat >= _SAT_VIVID:
        return "Stressed"

    return "Degraded"
# ─────────────────────────────────────────────────────────
# CORE: analyse a single PIL image
# ─────────────────────────────────────────────────────────
def analyse_image(model, pil_img, scale_internode: float, scale_diameter: float,
                  conf_threshold: float = 0.3):
    """
    Detect culms and nodes in *pil_img*, compute per-culm measurements, and
    return annotated image + per-culm data rows.
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
            x1_, y1_, x2_, y2_ = map(int, box.xyxy[0])
            if label == "culm":
                culms.append((x1_, y1_, x2_, y2_))
            elif label == "node":
                nodes.append((x1_, y1_, x2_, y2_))

    rows = []
    for i, (x1, y1, x2, y2) in enumerate(culms):
        h_cm = (y2 - y1) * scale_internode
        d_cm = (x2 - x1) * scale_diameter

        health = _classify_culm_health(img_bgr, x1, y1, x2, y2)
        bm_factor, c_factor, box_color_bgr, _ = HEALTH_CONFIG[health]

        bm_kg  = BIOMASS_K * (d_cm ** 2) * h_cm * bm_factor
        co2_kg = bm_kg * CARBON_FACTOR * c_factor

        rows.append({
            "Culm":          i + 1,
            "Health":        health,
            "Height_cm":     round(h_cm,   2),
            "Diameter_cm":   round(d_cm,   2),
            "Biomass_kg":    round(bm_kg,  3),
            "Carbon_CO2_kg": round(co2_kg, 3),
        })

        # ── Draw culm bounding box ──────────────────────────
        overlay = vis.copy()

        # Soft transparent fill
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            box_color_bgr,
            -1
        )

        # Slightly stronger visibility
        alpha = 0.14
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Darker/thicker outer bbox
        dark_box = tuple(max(c - 40, 0) for c in box_color_bgr)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            dark_box,
            3
        )

       

    # ── Node boxes ─────────────────────────────────────────
    node_color = (16, 90, 212)   # burnt orange in BGR
    for j, (x1, y1, x2, y2) in enumerate(nodes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), node_color, 2)
        

    

    ann_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    return rows, ann_img, len(culms)


# ─────────────────────────────────────────────────────────
# BATCH PIPELINE
# ─────────────────────────────────────────────────────────
def run_batch_inference(model, scale_internode: float, scale_diameter: float,
                        input_folder: str = "test_data",
                        output_folder: str = "output",
                        conf_threshold: float = 0.3):
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
        summary = {}
        for r in rows:
            summary[r["Health"]] = summary.get(r["Health"], 0) + 1
        print(f"  {image_name}: {len(rows)} culms {summary}, {n_nodes} nodes")
    return pd.DataFrame(all_data)


# ─────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────
HEALTH_COLORS = {
    "Healthy": C_FOREST,
    "Stressed": C_SAGE,
    "Degraded": C_ORANGE,
}

# Custom green-to-orange gradient colormap for sequential data
_BAMBOO_CMAP = LinearSegmentedColormap.from_list(
    "bamboo", [C_CREAM, C_SAGE, C_FOREST, C_DARK])

_CARBON_CMAP = LinearSegmentedColormap.from_list(
    "carbon", [C_CREAM, C_KHAKI, C_FOREST, C_DARK])


def _watermark(fig, text="BambooSense · SDG 13 · SDG 15"):
    """Add a subtle watermark / brand strip at the bottom of a figure."""
    fig.text(0.99, 0.01, text,
             ha="right", va="bottom", fontsize=7,
             color=C_KHAKI, alpha=0.7, style="italic",
             fontfamily="monospace")


def _base_fig(w=7.5, h=4.0, title=""):
    """Styled base figure using the brand palette."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_CREAM)
    ax.set_facecolor(C_LIGHT)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(C_KHAKI)
    ax.yaxis.grid(True, color=C_CREAM, linestyle='--', linewidth=0.9, alpha=0.8)
    ax.tick_params(labelsize=9, colors=C_DARK)
    ax.xaxis.label.set_color(C_DARK)
    ax.yaxis.label.set_color(C_DARK)
    _watermark(fig)
    return fig, ax


def chart_biomass(df: pd.DataFrame):
    fig, ax = _base_fig(w=7.5, h=4.2)
    totals = df.groupby("Image")["Biomass_kg"].sum()
    n = len(totals)
    palette = [C_FOREST, C_SAGE, C_KHAKI, C_DARK, C_ORANGE]
    bars = ax.bar(range(n), totals.values,
                  color=palette[:n], edgecolor=C_CREAM, linewidth=1.2, width=0.55)
    ax.set_xticks(range(n))
    ax.set_xticklabels(totals.index, rotation=15, ha="right", fontsize=9)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f"{b.get_height():.2f} kg",
                ha='center', va='bottom', fontsize=9, color=C_DARK, fontweight='bold')
    ax.set_title("Biomass per Image",
                 fontsize=13, fontweight='bold', color=C_DARK, pad=14,
                 fontfamily="serif")
    ax.set_ylabel("Biomass (kg)", fontsize=10)
   
    plt.tight_layout()
    return fig


def chart_health(df: pd.DataFrame):
    """Stacked bar: biomass contribution by health category per image."""
    fig, ax = _base_fig(w=7.5, h=4.2)
    images  = df["Image"].unique()
    n       = len(images)
    bottoms = np.zeros(n)
    for health in ["Healthy", "Stressed", "Degraded"]:
        vals = np.array([
            df[(df["Image"] == img) & (df["Health"] == health)]["Biomass_kg"].sum()
            for img in images
        ])
        bars = ax.bar(range(n), vals, bottom=bottoms,
                      color=HEALTH_COLORS[health],
                      label=health, edgecolor=C_CREAM, linewidth=0.8,
                      width=0.55, alpha=0.92)
        # Label non-zero segments
        for rect, v, bot in zip(bars, vals, bottoms):
            if v > 0.005:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        bot + v / 2,
                        f"{v:.2f}", ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')
        bottoms += vals
    ax.set_xticks(range(n))
    ax.set_xticklabels(images, rotation=15, ha="right", fontsize=9)
    ax.set_title("Biomass by Culm Health Status",
                 fontsize=13, fontweight='bold', color=C_DARK, pad=14,
                 fontfamily="serif")
    ax.set_ylabel("Biomass (kg)", fontsize=10)
    leg = ax.legend(title="Health Status", fontsize=9, title_fontsize=9,
                    loc="upper right", framealpha=0.85,
                    facecolor=C_CREAM, edgecolor=C_KHAKI)
    leg.get_title().set_color(C_DARK)
    
    plt.tight_layout()
    return fig


def chart_height(df: pd.DataFrame):
    fig, ax = _base_fig(w=7.5, h=4.2)
    ax.hist(df["Height_cm"],
            bins=max(5, len(df) // 2),
            color=C_FOREST, edgecolor=C_CREAM, linewidth=0.8, alpha=0.88)
    mean_h = df["Height_cm"].mean()
    ax.axvline(mean_h, color=C_ORANGE, linestyle='--', linewidth=2,
               label=f'Mean: {mean_h:.1f} cm')
    # Shade ±1 std
    std_h = df["Height_cm"].std()
    ax.axvspan(mean_h - std_h, mean_h + std_h,
               alpha=0.10, color=C_SAGE, label=f'±1 SD: {std_h:.1f} cm')
    ax.set_title("Culm Height Distribution",
                 fontsize=13, fontweight='bold', color=C_DARK, pad=14,
                 fontfamily="serif")
    ax.set_xlabel("Height (cm)", fontsize=10)
    ax.set_ylabel("Count",       fontsize=10)
    leg = ax.legend(fontsize=9, facecolor=C_CREAM, edgecolor=C_KHAKI)
    
    plt.tight_layout()
    return fig


def chart_carbon(df: pd.DataFrame):
    fig, ax = _base_fig(w=7.5, h=4.2)
    totals = df.groupby("Image")["Carbon_CO2_kg"].sum()
    n = len(totals)
    # Gradient colouring: more carbon = deeper forest green
    norm_vals = totals.values / (totals.values.max() + 1e-9)
    bar_colors = [_CARBON_CMAP(v * 0.8 + 0.2) for v in norm_vals]
    bars = ax.bar(range(n), totals.values,
                  color=bar_colors, edgecolor=C_CREAM, linewidth=1.2, width=0.55)
    ax.set_xticks(range(n))
    ax.set_xticklabels(totals.index, rotation=15, ha="right", fontsize=9)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                f"{b.get_height():.3f} kg",
                ha='center', va='bottom', fontsize=9, color=C_DARK, fontweight='bold')
    ax.set_title("CO₂ Sequestered per Image",
                 fontsize=13, fontweight='bold', color=C_DARK, pad=14,
                 fontfamily="serif")
    ax.set_ylabel("CO₂ (kg)", fontsize=10)
    plt.tight_layout()
    return fig


def chart_health_donut(df: pd.DataFrame):
    """Donut chart: culm count by health category."""
    counts = df["Health"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [HEALTH_COLORS.get(l, C_KHAKI) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(C_CREAM)
    wedges, texts, autotexts = ax.pie(
        values, labels=None,
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=C_CREAM, linewidth=2),
        autopct="%1.0f%%", pctdistance=0.77,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_fontweight("bold")
    # Centre annotation
    total = sum(values)
    ax.text(0, 0, f"{total}\nCulms", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_DARK,
            fontfamily="serif")
    legend_patches = [mpatches.Patch(facecolor=c, label=f"{l}  ({v})")
                      for l, v, c in zip(labels, values, colors)]
    ax.legend(handles=legend_patches, loc="lower center", fontsize=10,
              framealpha=0.85, facecolor=C_CREAM, edgecolor=C_KHAKI,
              ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Culm Health Distribution",
                 fontsize=13, fontweight='bold', color=C_DARK, pad=14,
                 fontfamily="serif")
    _watermark(fig)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig, dpi=180) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
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
    print("\n CSV saved → output.csv")
    print(df.groupby("Image")[["Biomass_kg", "Carbon_CO2_kg"]].sum())
    print("\nHealth breakdown:")
    print(df.groupby(["Image", "Health"])[["Biomass_kg", "Carbon_CO2_kg"]].sum())

    chart_biomass(df).savefig("biomass_chart.png")
    chart_health(df).savefig("health_chart.png")
    chart_height(df).savefig("height_distribution.png")
    chart_carbon(df).savefig("carbon_chart.png")
    chart_health_donut(df).savefig("health_donut.png")
    print("Charts saved")
    plt.close('all')