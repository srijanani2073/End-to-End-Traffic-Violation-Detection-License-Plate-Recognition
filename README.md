# 🚦 Towards Safer Roads: Traffic Violation Detection & License Plate Recognition

An end-to-end deep learning framework for real-time traffic violation detection and automatic license plate recognition (ALPR), built with **YOLOv8** and **Keras-OCR**, with automated **e-Challan generation** via email and SMS.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Results & Performance](#results--performance)
- [Database Schema](#database-schema)
- [Challan & Notification Pipeline](#challan--notification-pipeline)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Publication](#publication)
- [Authors](#authors)



## Overview

This system automates traffic violation monitoring using five independently trained YOLOv8 models running in parallel on live video/image input. When a violation is detected, the pipeline:

1. Localizes the vehicle's license plate using a dedicated YOLOv8 model
2. Extracts the plate text using **Keras-OCR**
3. Queries vehicle owner details via the **RegCheck API**
4. Stores violation records in a **Supabase** (PostgreSQL) database
5. Generates a **PDF e-Challan** and dispatches it via **email** (SMTP) and **SMS** (Twilio)

The system achieved an average **mAP@50 of 97.9%**, with **precision 96.9%** and **recall 95.9%**, outperforming existing state-of-the-art models on the same tasks.



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



## Project Structure

```
traffic-violation-detection/
│
├── README.md
├── requirements.txt
│
├── Weights/                             # YOLOv8 model weight files (*.pt)
│   ├── yolov8n.pt                      # Base YOLOv8 nano (vehicle detection backbone)
│   ├── helmet.pt                       # Helmet violation detection model
│   ├── triple.pt                       # Triple riding detection model
│   ├── seatbelt.pt                     # Seatbelt violation detection model (best1.pt)
│   ├── phone.pt                        # Mobile phone usage detection model
│   └── license_plate.pt               # License plate localization model
│
├──Sample-challan-generated.pdf
```



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



## Limitations

- OCR accuracy drops on damaged, skewed, or non-standard license plates
- Detection degrades under heavy rain, fog, or extreme low-light conditions
- Triple riding and phone usage datasets had class imbalance (mitigated with augmentation)
- RegCheck API may not return owner data for all Indian plates (fallback to mock data used in dev)
- End-to-end latency (~12s/incident) may be too high for very high-throughput intersections



## Future Work

- Super-resolution preprocessing to improve OCR on low-quality plates
- Temporal video analysis for signal-jumping and wrong-lane detection
- Real-time analytics dashboard for law enforcement
- Edge deployment (Jetson Nano / Raspberry Pi) for on-camera inference
- Integration with official transport authority APIs (Vahan/Parivahan)
- Multimodal fusion and adaptive thresholding for adverse weather


## Publication
This project is published as a research paper: "Towards Safer Roads: An End-to-End Deep Learning Framework for Traffic Violation Detection and License Plate Recognition" DOI: 10.1177/18758967251397468

[Paper](https://doi.org/10.1177/18758967251397468)

## Authors

| Name | Institution |
|---|---|
| Sri Janani S | Sri Venkateswara College of Engineering |
| Sanjana J | Sri Venkateswara College of Engineering |
| Archana G | Sri Venkateswara College of Engineering |
| Dr. P. Janarthanan | Sri Venkateswara College of Engineering, Sriperumbudur |
| Dr. A. Kavitha | Dwaraka Doss Govardhan Doss Vaishnav College, Chennai |
