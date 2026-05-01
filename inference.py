# ================================
# 🌿 INFERENCE (USE FOR ALL IMAGES)
# ================================

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import os

model = YOLO("model/yolo11n.pt")

# -------------------------------
# Load saved calibration
# -------------------------------
with open("bamboo_scale.json", "r") as f:
    scale = json.load(f)

scale_internode = scale["internode_scale"]
scale_diameter = scale["diameter_scale"]

print("Loaded scale factors")

# -------------------------------
# Input image
# -------------------------------
image_path = "test_data/1.jpeg"
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = f"output/{base_name}_output.jpg"

image = cv2.imread(image_path)
img_vis = image.copy()

results = model(image)

culms = []
nodes = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "culm":
            culms.append((x1, y1, x2, y2))
        elif label == "node":
            nodes.append((x1, y1, x2, y2))

# -------------------------------
# PIXEL FEATURES
# -------------------------------
heights_px = [(y2 - y1) for (x1, y1, x2, y2) in culms]
widths_px  = [(x2 - x1) for (x1, y1, x2, y2) in culms]

# -------------------------------
# REAL-WORLD CONVERSION
# -------------------------------
heights_cm = [h * scale_internode for h in heights_px]
diam_cm = [w * scale_diameter for w in widths_px]

# -------------------------------
# PRINT OUTPUT
# -------------------------------
data = []

for i in range(len(culms)):
    print(f"\n🌿 Culm {i+1}")
    print("Height (cm):", heights_cm[i])
    print("Diameter (cm):", diam_cm[i])

    data.append({
        "Culm_ID": i+1,
        "Height_cm": heights_cm[i],
        "Diameter_cm": diam_cm[i]
    })

df = pd.DataFrame(data)

df.to_csv("output.csv", index=False)

# -------------------------------
# DRAW BOUNDING BOXES
# -------------------------------
for (x1, y1, x2, y2) in culms:
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

for (x1, y1, x2, y2) in nodes:
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Optional: add labels
for i, (x1, y1, x2, y2) in enumerate(culms):
    cv2.putText(img_vis, f"Culm {i+1}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    


cv2.imwrite(output_path, img_vis)

print("\n✅ Saved output image + CSV")