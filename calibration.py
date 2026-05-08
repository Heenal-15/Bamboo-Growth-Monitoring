# ================================
# 🌿 CALIBRATION
# ================================
# Run standalone (python calibration.py) to generate bamboo_scale.json,
# OR import get_scale / run_calibration from app.py.

import cv2
import numpy as np
import json
import os


# Default config — override by passing your own list to run_calibration()
DEFAULT_CALIB_CONFIG = [
    {"path": "calibration_images/calib1.jpeg", "internode_cm": 27,   "diameter_cm": 5.73},
    {"path": "calibration_images/calib2.jpg",  "internode_cm": 25,   "diameter_cm": 6.69},
]

SCALE_CACHE = "bamboo_scale.json"


# ─────────────────────────────────────────────────────────
# CORE: derive scale factors from a single image
# ─────────────────────────────────────────────────────────
def get_scale(model, image_path: str, real_internode_cm: float, real_diameter_cm: float):
    """
    Given a calibration image (with known real-world internode length and
    culm diameter), return (scale_internode, scale_diameter) in cm/px.

    Parameters
    ----------
    model             : loaded YOLO model
    image_path        : path to calibration image
    real_internode_cm : measured internode length in centimetres
    real_diameter_cm  : measured culm diameter in centimetres

    Returns
    -------
    (scale_internode, scale_diameter) or (None, None) if detection fails
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  Could not read image: {image_path}")
        return None, None

    results = model(image, verbose=False)

    culms, nodes = [], []
    for r in results:
        for box in r.boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "culm":
                culms.append((x1, y1, x2, y2))
            elif label == "node":
                nodes.append((x1, y1, x2, y2))

    # ── Internode pixels ──────────────────────────────────
    node_centers = sorted([(y1 + y2) // 2 for (_, y1, _, y2) in nodes])
    internode_px = [node_centers[i] - node_centers[i - 1]
                    for i in range(1, len(node_centers))]

    if not internode_px:
        print(f"⚠️  No internode pairs found in: {image_path}")
        return None, None

    mean_internode_px = np.mean(internode_px)

    # ── Diameter pixels ───────────────────────────────────
    if not culms:
        print(f"⚠️  No culms found in: {image_path}")
        return None, None

    diam_px      = [(x2 - x1) for (x1, _, x2, _) in culms]
    mean_diam_px = np.mean(diam_px)

    scale_internode = real_internode_cm / mean_internode_px
    scale_diameter  = real_diameter_cm  / mean_diam_px

    return scale_internode, scale_diameter


# ─────────────────────────────────────────────────────────
# PIPELINE: average over multiple calibration images
# ─────────────────────────────────────────────────────────
def run_calibration(model, calib_config=None, save_path=SCALE_CACHE):
    """
    Run calibration over all images in calib_config, average the results,
    persist to save_path, and return (internode_scale, diameter_scale).

    If save_path already exists it is overwritten.
    """
    if calib_config is None:
        calib_config = DEFAULT_CALIB_CONFIG

    si_list, sd_list = [], []

    for cfg in calib_config:
        si, sd = get_scale(model, cfg["path"], cfg["internode_cm"], cfg["diameter_cm"])
        if si is not None and sd is not None:
            si_list.append(si)
            sd_list.append(sd)

    if not si_list:
        raise RuntimeError("Calibration failed: no valid images could be processed.")

    final_internode = float(np.mean(si_list))
    final_diameter  = float(np.mean(sd_list))

    scale_data = {
        "internode_scale": final_internode,
        "diameter_scale":  final_diameter,
    }

    with open(save_path, "w") as f:
        json.dump(scale_data, f, indent=2)

    print(f"Final Scale Internode : {final_internode:.6f} cm/px")
    print(f"Final Scale Diameter  : {final_diameter:.6f} cm/px")
    print(f"Calibration saved to {save_path}")

    return final_internode, final_diameter


# ─────────────────────────────────────────────────────────
# STANDALONE ENTRY-POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ultralytics import YOLO
    _model = YOLO("model/yolov8s_bamboo.pt")
    run_calibration(_model)