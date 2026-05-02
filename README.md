# 🎋 BambooSense

> **Automated Bamboo Growth Monitoring · Biomass Estimation · Carbon Sequestration Assessment**

BambooSense is a computer-vision-powered field tool that detects bamboo culms and nodes from photographs, computes real-world measurements, estimates dry biomass, and calculates carbon sequestration — all through a clean Streamlit web interface.

---

## 🌿 Why Bamboo?

Bamboo is one of the most powerful natural resources on Earth for climate resilience:

- **Fastest-growing plant** — up to 90 cm/day, making it one of the most renewable resources available
- **Carbon powerhouse** — sequesters 2–3× more CO₂ per hectare than equivalent tree plantations
- **Versatile resource** — critical for construction, paper & pulp, and bioenergy production
- **Climate tool** — accurate biomass monitoring directly supports carbon credits, forest management, and IPCC reporting

BambooSense provides the **automated, scalable monitoring** needed to make these assessments fast, repeatable, and field-ready — without tape measures or manual counting.

---

## 📸 Demo

| Upload field images | Detection results | Carbon report |
|---|---|---|
| Drag & drop JPG/PNG | Culms + nodes annotated | CO₂ sequestered per image |

---

## 🗂️ Project Structure

```
bamboosense/
│
├── app.py                  # Streamlit UI — orchestration only
├── calibration.py          # Scale factor derivation from reference images
├── inference.py            # YOLO detection, measurement, carbon logic, charts
│
├── model/
│   └── yolov8s_bamboo.pt   # Fine-tuned YOLOv8-Small model
│
├── calibration_images/
│   ├── calib1.jpeg         # Reference image 1 (known dimensions)
│   └── calib2.jpg          # Reference image 2 (known dimensions)
│
├── bamboo_scale.json       # Auto-generated — cached calibration output
│
├── test_data/              # (standalone mode) input images for batch inference
└── output/                 # (standalone mode) annotated output images
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yourname/bamboosense.git
cd bamboosense
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
streamlit
opencv-python
numpy
pandas
matplotlib
Pillow
ultralytics
reportlab
```

### 3. Add your YOLO model

Place your fine-tuned model at:
```
model/yolov8s_bamboo.pt
```

> If you don't have a custom model, you can start with our trained model and fine-tune it on bamboo imagery.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔬 How It Works

### Step 1 — Calibration (runs once)

`calibration.py` processes reference images where the real-world internode length and culm diameter are known. It computes pixel-to-centimetre scale factors and caches them to `bamboo_scale.json`.

```
scale_internode  =  real_internode_cm  /  mean_internode_pixels
scale_diameter   =  real_diameter_cm   /  mean_diameter_pixels
```

Multiple calibration images are averaged for robustness. If `bamboo_scale.json` already exists, calibration is skipped.

### Step 2 — Detection

YOLOv8 detects two classes per image:

| Class | Description |
|-------|-------------|
| `culm` | The bamboo stalk (bounding box gives height & diameter in pixels) |
| `node` | The joint / internode boundary |

Detections below **0.3 confidence** are discarded.

### Step 3 — Measurement

For each detected culm:

```
Height_cm    =  box_height_px  ×  scale_internode
Diameter_cm  =  box_width_px   ×  scale_diameter
```

### Step 4 — Biomass Estimation

Allometric equation (standard bamboo formula):

```
Biomass_kg  =  K × D² × H
```

Where:
- `K = 0.03` — allometric constant
- `D` = culm diameter (cm)
- `H` = culm height (cm)

### Step 5 — Carbon Sequestration

```
CO₂_kg  =  Biomass_kg × 0.25
```

**Factor: 0.25 kg CO₂ per kg dry bamboo biomass**
Source: [INBAR Technical Report 37](https://www.inbar.int) · IPCC Guidelines Tier 1

> Bamboo biomass is ~47% carbon by dry weight; 1 kg of carbon = 3.67 kg CO₂ equivalent.
> The 0.25 factor is the conservative published Tier 1 value used for carbon accounting.

---

## 📊 Outputs

### In-app

| Output | Description |
|--------|-------------|
| Stat pills | Images processed, culms detected, avg height, total biomass, total CO₂ |
| Carbon banner | Highlighted CO₂ total with tree-equivalent comparison |
| Annotated images | Bounding boxes with height, biomass, and CO₂ labels |
| Biomass chart | Biomass (kg) per image |
| Height distribution | Histogram of culm heights |
| CO₂ chart | Carbon sequestered per image |
| Data table | Per-culm measurements with colour gradients |

### Downloadable

| File | Format | Contents |
|------|--------|----------|
| `bamboo_data.csv` | CSV | Per-culm: height, diameter, biomass, CO₂ |
| `BambooSense_Report.pdf` | PDF | Summary stats, all charts, full data table |
| `biomass_chart.png` | PNG | Biomass per image bar chart |
| `height_distribution.png` | PNG | Height histogram |
| `carbon_chart.png` | PNG | CO₂ sequestered per image |

---

## 🖥️ Standalone / Batch Mode

You can run calibration and inference independently without the Streamlit UI.

### Run calibration

```bash
python calibration.py
```

Processes `calibration_images/` and writes `bamboo_scale.json`.

### Run batch inference

```bash
python inference.py
```

Processes all images in `test_data/`, saves annotated images to `output/`, and generates:
- `output.csv`
- `biomass_chart.png`
- `height_distribution.png`
- `carbon_chart.png`

---


### Detection confidence

Default: `0.3`. Lower this if culms are missed; raise it to reduce false positives.

```python
# In inference.py
rows, ann_img, n_nodes = analyse_image(model, pil_img, scale_i, scale_d, conf_threshold=0.3)
```

### Biomass constant

```python
BIOMASS_K = 0.03  # in inference.py — adjust for your bamboo species
```

### Carbon factor

```python
CARBON_FACTOR = 0.25  # kg CO₂ per kg dry biomass — INBAR TR-37 / IPCC Tier 1
```

---

## 🌍 Carbon Sequestration Methodology

| Parameter | Value | Source |
|-----------|-------|--------|
| Carbon fraction of dry biomass | 47% | IPCC 2006 GL Ch. 4 |
| CO₂ / C ratio | 3.67 | Molecular weight ratio |
| Applied factor (conservative Tier 1) | **0.25 kg CO₂/kg** | INBAR TR-37 |
| Tree equivalent (annual) | 21.77 kg CO₂/tree/yr | US EPA estimate |

The "trees equivalent" figure shown in the app is for communicative purposes and uses the US EPA average annual sequestration for a typical managed tree.

---

## 📋 Requirements

```
Python        >= 3.9
streamlit     >= 1.32
opencv-python >= 4.9
numpy         >= 1.24
pandas        >= 2.0
matplotlib    >= 3.7
Pillow        >= 10.0
ultralytics   >= 8.0
reportlab     >= 4.0
```


---

*BambooSense v2.0 · YOLOv8s · Streamlit · Built for field-ready bamboo monitoring*
