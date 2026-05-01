# ================================
# 🌿 INFERENCE (FULL SYSTEM VERSION)
# ================================

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

model = YOLO("model/yolo11n.pt")

# -------------------------------
# LOAD CALIBRATION
# -------------------------------
with open("bamboo_scale.json", "r") as f:
    scale = json.load(f)

scale_internode = scale["internode_scale"]
scale_diameter = scale["diameter_scale"]

print("Loaded scale factors")

# -------------------------------
# INPUT FOLDER (ALL IMAGES)
# -------------------------------
input_folder = "test_data"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

all_data = []

# ===============================
# PROCESS ALL IMAGES
# ===============================
for image_name in os.listdir(input_folder):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, image_name)
    base_name = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}_output.jpg")

    image = cv2.imread(image_path)
    img_vis = image.copy()

    results = model(image)

    culms = []
    nodes = []

    # ---------------- detection ----------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if conf < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "culm":
                culms.append((x1, y1, x2, y2))
            elif label == "node":
                nodes.append((x1, y1, x2, y2))

    # ---------------- features ----------------
    heights_px = [(y2 - y1) for (x1, y1, x2, y2) in culms]
    widths_px  = [(x2 - x1) for (x1, y1, x2, y2) in culms]

    heights_cm = [h * scale_internode for h in heights_px]
    diam_cm = [w * scale_diameter for w in widths_px]

    k = 0.03
    biomass = [k * (D**2) * H for D, H in zip(diam_cm, heights_cm)]

    # ---------------- save per image data ----------------
    for i in range(len(culms)):
        all_data.append({
            "Image": image_name,
            "Culm_ID": i+1,
            "Height_cm": heights_cm[i],
            "Diameter_cm": diam_cm[i],
            "Biomass_kg": biomass[i]
        })

    # ---------------- visualization ----------------
    for (x1, y1, x2, y2) in culms:
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, (x1, y1, x2, y2) in enumerate(culms):
        cv2.putText(img_vis,
                    f"H:{heights_cm[i]:.1f} B:{biomass[i]:.1f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 2)
        
    # ---------------- NODE VISUALIZATION ----------------
    for (x1, y1, x2, y2) in nodes:
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for i, (x1, y1, x2, y2) in enumerate(nodes):
        cv2.putText(img_vis,
                    f"Node {i+1}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

    cv2.imwrite(output_path, img_vis)

# ===============================
# SAVE CSV
# ===============================
df = pd.DataFrame(all_data)
df.to_csv("output.csv", index=False)

print("\n✅ CSV saved")

# ===============================
# 📊 AUTO CHARTS (NEW FEATURE)
# ===============================

plt.figure()
df.groupby("Image")["Biomass_kg"].sum().plot(kind="bar")
plt.title("Biomass per Image")
plt.ylabel("Biomass (kg)")
plt.savefig("biomass_chart.png")

plt.figure()
df["Height_cm"].hist()
plt.title("Height Distribution")
plt.savefig("height_distribution.png")

print("📊 Charts saved")

# ===============================
# 📄 PDF REPORT (VERY IMPRESSIVE)
# ===============================

doc = SimpleDocTemplate("Bamboo_Report.pdf")
styles = getSampleStyleSheet()
content = []

content.append(Paragraph("🌿 Bamboo Growth Monitoring Report", styles["Title"]))
content.append(Spacer(1, 12))

content.append(Paragraph(f"Total Images Processed: {len(df['Image'].unique())}", styles["Normal"]))
content.append(Paragraph(f"Total Culms Detected: {len(df)}", styles["Normal"]))
content.append(Paragraph(f"Total Biomass: {df['Biomass_kg'].sum():.2f} kg", styles["Normal"]))
content.append(Spacer(1, 12))

doc.build(content)

print("📄 PDF report generated: Bamboo_Report.pdf")