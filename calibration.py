# ================================
# 🌿 CALIBRATION (RUN ONCE ONLY)
# ================================

import cv2
import numpy as np
from ultralytics import YOLO
import json

model = YOLO("model/yolov8s_bamboo.pt")

def get_scale(image_path, real_internode_cm, real_diameter_cm):

    image = cv2.imread(image_path)
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

    # ---------------- INTERNODE PIXELS ----------------
    node_centers = [(y1 + y2)//2 for (_, y1, _, y2) in nodes]
    node_centers.sort()

    internode_px = [
        node_centers[i] - node_centers[i-1]
        for i in range(1, len(node_centers))
    ]

    mean_internode_px = np.mean(internode_px)

    # ---------------- DIAMETER PIXELS ----------------
    diam_px = [(x2 - x1) for (x1, y1, x2, y2) in culms]
    mean_diam_px = np.mean(diam_px)

    # ---------------- SCALE FACTORS ----------------
    scale_internode = real_internode_cm / mean_internode_px
    scale_diameter = real_diameter_cm / mean_diam_px

    return scale_internode, scale_diameter


# 🔁 YOUR 2 CALIBRATION IMAGES
scale1 = get_scale("calibration_images/calib1.jpeg", real_internode_cm=27, real_diameter_cm=5.73)
scale2 = get_scale("calibration_images/calib2.jpg", real_internode_cm=25, real_diameter_cm=6.69)

# average both images
final_scale_internode = (scale1[0] + scale2[0]) / 2
final_scale_diameter = (scale1[1] + scale2[1]) / 2

print("Final Scale Internode:", final_scale_internode)
print("Final Scale Diameter:", final_scale_diameter)

# save model
scale_data = {
    "internode_scale": final_scale_internode,
    "diameter_scale": final_scale_diameter
}

with open("bamboo_scale.json", "w") as f:
    json.dump(scale_data, f)

print("✅ Calibration saved!")