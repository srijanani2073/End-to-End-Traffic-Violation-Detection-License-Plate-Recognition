# 🚦 Towards Safer Roads: Traffic Violation Detection & License Plate Recognition

An end-to-end deep learning framework for real-time traffic violation detection and automatic license plate recognition (ALPR), built with **YOLOv8** and **Keras-OCR**, with automated **e-Challan generation** via email and SMS.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Model Weights](#model-weights)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Database Schema](#database-schema)
- [Challan & Notification Pipeline](#challan--notification-pipeline)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Authors](#authors)

---

## Overview

This system automates traffic violation monitoring using five independently trained YOLOv8 models running in parallel on live video/image input. When a violation is detected, the pipeline:

1. Localizes the vehicle's license plate using a dedicated YOLOv8 model
2. Extracts the plate text using **Keras-OCR**
3. Queries vehicle owner details via the **RegCheck API**
4. Stores violation records in a **Supabase** (PostgreSQL) database
5. Generates a **PDF e-Challan** and dispatches it via **email** (SMTP) and **SMS** (Twilio)

The system achieved an average **mAP@50 of 97.9%**, with **precision 96.9%** and **recall 95.9%**, outperforming existing state-of-the-art models on the same tasks.

---

## Features

| Feature | Details |
|---|---|
|Helmet Detection | Detects riders without helmets on two-wheelers |
|Seatbelt Detection | Detects driver/passenger seatbelt non-usage |
|Mobile Phone Usage | Detects drivers using phones while driving |
|Triple Riding | Detects three or more people on a two-wheeler |
|License Plate Recognition | Localizes and OCR-reads Indian vehicle plates |
|e-Challan Generation | Auto-generates PDF challans with fine details |
|Email Notification | Sends challan PDF to registered vehicle owner |
|SMS Notification | Sends violation summary via Twilio SMS |
|Database Storage | Persists all violation records in Supabase |
|Real-time Processing | ~120 FPS inference on NVIDIA Tesla T4 |

---

## System Architecture

```
Real-Time Video Feed / Image Input
            │
            ▼
   Frame Preprocessing
   (Resize to 640×640, Normalize)
            │
            ▼
┌───────────────────────────────────────┐
│     Parallel YOLOv8 Inference         │
│  ┌──────────┐  ┌──────────────────┐   │
│  │ Seatbelt │  │ Helmet + Phone + │   │
│  │ Model    │  │ Triple Riding    │   │
│  └──────────┘  └──────────────────┘   │
└───────────────────────────────────────┘
            │
     Violation Detected?
            │ YES
            ▼
   License Plate YOLOv8 Model
   → Crop plate region
            │
            ▼
       Keras-OCR
   → Extract plate text
            │
            ▼
   RegCheck API (Vehicle Owner Lookup)
            │
            ▼
   Supabase DB (Store violation record)
            │
            ▼
   PDF Challan Generation (fpdf2)
            │
     ┌──────┴──────┐
     ▼             ▼
 Email (SMTP)   SMS (Twilio)
```

---

## Project Structure

```
traffic-violation-detection/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/                          # Jupyter notebooks (development & experimentation)
│   ├── road_safety_system_final.ipynb  # Main end-to-end pipeline notebook
│   ├── api_and_db.ipynb                # API integration + Supabase DB notebook
│   ├── video_with_crop.ipynb           # Video feed inference with plate cropping
│   └── final/
│       └── final-1-0.ipynb             # Final consolidated training notebook
│
├── models/                             # YOLOv8 model weight files (*.pt)
│   ├── yolov8n.pt                      # Base YOLOv8 nano (vehicle detection backbone)
│   ├── helmet.pt                       # Helmet violation detection model
│   ├── triple.pt                       # Triple riding detection model
│   ├── seatbelt.pt                     # Seatbelt violation detection model (best1.pt)
│   ├── phone.pt                        # Mobile phone usage detection model
│   └── license_plate.pt               # License plate localization model
│
├── datasets/                           # Dataset references and download scripts
│   ├── README_datasets.md              # Instructions to download each dataset
│   └── augmentation_config.yaml        # Augmentation settings used during training
│
├── src/                                # Modular Python source (refactored from notebooks)
│   ├── __init__.py
│   ├── config.py                       # Centralized config (paths, thresholds, keys)
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── violation_detector.py       # Runs all 5 YOLOv8 models in parallel
│   │   ├── plate_detector.py           # License plate localization (YOLOv8)
│   │   └── ocr_reader.py               # Keras-OCR text extraction + post-processing
│   ├── database/
│   │   ├── __init__.py
│   │   └── supabase_client.py          # Supabase insert/query helpers
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── email_sender.py             # SMTP email with challan PDF attachment
│   │   └── sms_sender.py               # Twilio SMS dispatcher
│   ├── challan/
│   │   ├── __init__.py
│   │   └── challan_generator.py        # PDF e-Challan generation with fpdf2
│   └── api/
│       ├── __init__.py
│       └── regcheck_client.py          # RegCheck SOAP API for vehicle owner lookup
│
├── outputs/                            # Generated challans and violation snapshots
│   ├── challans/                       # PDF challan files (gitignored — generated at runtime)
│   └── snapshots/                      # Cropped violation frames (gitignored)
│
├── assets/
│   └── sample_images/                  # Sample test images for demo purposes
│
└── docs/
    └── paper.pdf                       # Research paper (MajP-Traffic.pdf)
```

---

## Datasets

The system uses 7 datasets across 5 violation types. **None of these are included in the repo** — download them from the links below.

| Dataset | Violation Type | Source | Size |
|---|---|---|---|
| Seat Belt Detection by Ff | Seatbelt | Images.CV | ~3,583 images |
| Seat Belt Detection API (SACAIM) | Seatbelt | Roboflow Universe | Included in above |
| FB-YOLOv7 Dataset | Helmet, Triple Riding, Phone | Roboflow | Not disclosed |
| Indian Helmet Detection Dataset | Helmet | Kaggle | ~5,443 images |
| Traffic Violation Detection TVD 2 | Triple Riding, Phone | Roboflow Universe | ~5,998 images |
| Indian Vehicle License Plate Dataset | License Plate | GitHub (DataCluster Labs) | 20+ states |
| Indian License Plates with Labels | License Plate (OCR) | Kaggle | Not disclosed |

All datasets use an **80:10:10 train/val/test split**.

---

## Model Weights

The trained `.pt` weight files are **not committed to Git** due to file size (~17MB+ each). They are distributed separately.

**Option 1 — Download from shared storage:**
> *(Add your Google Drive / HuggingFace / release link here)*

**Option 2 — Use the zipped weights included in the repo release:**
- `best weights.zip` — contains weights for all violation models
- `Best.pt.zip` — final consolidated best weight

After downloading, extract and place `.pt` files into the `models/` directory.

---

## Installation
### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/traffic-violation-detection.git
cd traffic-violation-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt` contents:

```
ultralytics>=8.0.0
keras-ocr
opencv-python
numpy
matplotlib
scikit-learn
supabase
fpdf2
twilio
requests
```

### 3. Download model weights

Place all `.pt` files inside the `models/` directory. See [Model Weights](#model-weights) above.

---

## Configuration

Copy and edit the config file before running:

```bash
cp src/config.example.py src/config.py
```

Edit `src/config.py`:

```python
# --- Model Paths ---
VEHICLE_MODEL_PATH   = "models/yolov8n.pt"
HELMET_MODEL_PATH    = "models/helmet.pt"
TRIPLE_MODEL_PATH    = "models/triple.pt"
SEATBELT_MODEL_PATH  = "models/seatbelt.pt"
PHONE_MODEL_PATH     = "models/phone.pt"
PLATE_MODEL_PATH     = "models/license_plate.pt"

# --- Detection Thresholds ---
CONFIDENCE_THRESHOLD = 0.6

# --- Violation Fines (INR) ---
VIOLATION_FINES = {
    "No Helmet": 1000,
    "No Seatbelt": 1000,
    "Using Mobile Phone": 500,
    "Triple Riding": 1500,
}

# --- Supabase ---
SUPABASE_URL = "https://<your-project>.supabase.co"
SUPABASE_KEY = "<your-anon-key>"

# --- RegCheck API ---
REGCHECK_USERNAME = "<your-username>"

# --- Email (SMTP) ---
SENDER_EMAIL    = "<your-gmail>"
SENDER_PASSWORD = "<your-app-password>"   # Use Gmail App Password, not account password

# --- Twilio SMS ---
TWILIO_ACCOUNT_SID = "<your-account-sid>"
TWILIO_AUTH_TOKEN  = "<your-auth-token>"
TWILIO_NUMBER      = "+1XXXXXXXXXX"
```
---

## Usage

### Run on a single image

Open and run `notebooks/road_safety_system_final.ipynb` — set `image_path` to your image file.

### Run on a live webcam feed

Open and run `notebooks/video_with_crop.ipynb`.

### Generate challans for all stored violations

Run the challan generation cell in `notebooks/api_and_db.ipynb`, which:
- Fetches all unprocessed violations from Supabase
- Generates a PDF challan per vehicle
- Emails and SMSes the owner

### Full pipeline (refactored src — coming soon)

```bash
python src/main.py --input webcam         # Live webcam
python src/main.py --input image.jpg      # Single image
python src/main.py --input video.mp4      # Video file
```
---

## Results & Performance

### Per-class metrics (all YOLOv8 models combined)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| With Helmet | 0.956 | 0.966 | 0.986 | 0.774 |
| Without Helmet | 0.940 | 0.926 | 0.976 | 0.715 |
| Triple Riding | 0.951 | 0.960 | 0.988 | 0.914 |
| Using Mobile | 0.958 | 0.990 | 0.991 | 0.841 |
| Motor-cycle | 0.889 | 0.910 | 0.951 | 0.737 |
| Person-NoSeatbelt | 0.908 | 0.897 | 0.955 | 0.638 |
| Person-Seatbelt | 0.943 | 0.944 | 0.974 | 0.669 |
| Seatbelt | 0.906 | 0.796 | 0.882 | 0.403 |
| Windshield | 0.983 | 0.996 | 0.995 | 0.852 |
| License Plate | 0.988 | 0.982 | 0.994 | 0.857 |
| **Average** | **0.969** | **0.959** | **0.979** | — |

### Benchmark vs. state-of-the-art

| Model | Two-Wheeler | Seatbelt | License Plate | mAP (%) |
|---|:---:|:---:|:---:|---|
| Lin et al. [YOLOv5] | ✅ | ❌ | ❌ | 94.3 |
| Pan et al. | ❌ | ❌ | ✅ | 96.78 |
| Said et al. | ✅ | ❌ | ❌ | 95.2 |
| Gu et al. | ❌ | ✅ | ❌ | 72.3 |
| **Proposed (Ours)** | ✅ | ✅ | ✅ | **97.9** |

### Inference speed

- **~120 FPS** on NVIDIA Tesla T4 GPU (Google Colab)
- **~12 seconds** end-to-end per incident (detection → OCR → challan)

### YOLOv8 variant comparison (License Plate task)

| Variant | Final mAP | Notes |
|---|---|---|
| YOLOv8n (nano) | 0.994 | Best overall mAP; recommended for speed |
| YOLOv8m (medium) | 0.993 | Close second |
| YOLOv8s (small) | 0.990 | Best bounding box precision |
| YOLOv8l (large) | 0.989 | Highest compute cost |

---

## Database Schema

Two tables in Supabase (PostgreSQL):

### `vehicle_info`

| Column | Type | Description |
|---|---|---|
| `plate` | TEXT (PK) | License plate number |
| `owner` | TEXT | Registered owner name |
| `state` | TEXT | Registered state |
| `country` | TEXT | Always "India" |
| `year` | TEXT | Vehicle manufacture year |
| `make` | TEXT | Vehicle manufacturer |
| `model` | TEXT | Vehicle model name |
| `email` | TEXT | Owner email address |
| `phone` | TEXT | Owner phone number |
| `style` | TEXT | "Two Wheeler" or "HCV" |

### `vehicle_violations`

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Auto-generated |
| `plate` | TEXT (FK) | References `vehicle_info.plate` |
| `owner` | TEXT | Owner name at time of violation |
| `violation` | TEXT | Violation type |
| `timestamp` | TIMESTAMPTZ | When violation was recorded |
| `fine` | INTEGER | Fine amount in INR |
| `challan_sent` | BOOLEAN | Whether challan has been dispatched |

---

## Challan & Notification Pipeline

1. **PDF Generation** (`fpdf2`): Creates a structured challan with vehicle details, violation type, fine amount, timestamp, and a QR/reference code.
2. **Email** (`smtplib` + Gmail App Password): Attaches the PDF and sends to the owner's registered email.
3. **SMS** (Twilio): Sends a short violation summary with fine amount to the owner's registered phone number.

Fines are defined as:

| Violation | Fine (INR) |
|---|---|
| No Helmet | ₹1,000 |
| No Seatbelt | ₹1,000 |
| Using Mobile Phone | ₹500 |
| Triple Riding | ₹1,500 |

---

## Limitations

- OCR accuracy drops on damaged, skewed, or non-standard license plates
- Detection degrades under heavy rain, fog, or extreme low-light conditions
- Triple riding and phone usage datasets had class imbalance (mitigated with augmentation)
- RegCheck API may not return owner data for all Indian plates (fallback to mock data used in dev)
- End-to-end latency (~12s/incident) may be too high for very high-throughput intersections

---

## Future Work

- Super-resolution preprocessing to improve OCR on low-quality plates
- Temporal video analysis for signal-jumping and wrong-lane detection
- Real-time analytics dashboard for law enforcement
- Edge deployment (Jetson Nano / Raspberry Pi) for on-camera inference
- Integration with official transport authority APIs (Vahan/Parivahan)
- Multimodal fusion and adaptive thresholding for adverse weather

---

## Authors

| Name | Institution |
|---|---|
| Sri Janani S | Sri Venkateswara College of Engineering |
| Sanjana J | Sri Venkateswara College of Engineering |
| Archana G | Sri Venkateswara College of Engineering |
| Dr. P. Janarthanan | Sri Venkateswara College of Engineering, Sriperumbudur |
| Dr. A. Kavitha | Dwaraka Doss Govardhan Doss Vaishnav College, Chennai |

---

## License

This project is for academic and research purposes. Please cite the associated paper if you use this work.

---

## Citation

```
@article{kavitha2025trafficsafety,
  title={Towards Safer Roads: An End-to-End Deep Learning Framework for Traffic Violation Detection and License Plate Recognition},
  author={Kavitha, A. and Janarthanan, P. and Archana, G. and Sanjana, J. and Sri Janani, S.},
  institution={Sri Venkateswara College of Engineering, Sriperumbudur},
  year={2025}
}
```
