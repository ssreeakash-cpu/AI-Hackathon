# 🔥 AI-Based Early Detection of Forest Fire and Smoke from Satellite / Camera Images

### Problem Statement 25 • VIT Tech-a-thon
**School of Computer Science Engineering and Information Systems**

---

## 📌 Project Overview
This repository implements an end-to-end, multi-modal **Spatial-Temporal AI Wildfire Early-Warning Monitoring System** capable of:
1. Detecting early-stage wildfire ignition and smoke plumes from both **satellite imagery** and **ground surveillance cameras**.
2. Distinguishing actual wildfire smoke from confusing environmental distractors (**clouds, fog, dust**).
3. Analyzing **temporal evolution and spread** using spatial-temporal deep learning (ConvLSTM + Attention) across multi-spectral satellite datacubes.
4. Estimating **growth velocity**, **severity levels** (LOW, MEDIUM, HIGH, CRITICAL), and **multi-model risk fusion scores**.
5. Generating geolocated early-warning alerts on an **interactive GIS Leaflet Command Center Dashboard**.

---

## 🛰️ Public Datasets Utilized

### 1. NASA Earthdata (MODIS & VIIRS Fire Data)
Direct integration of real telemetry from NASA's Fire Information for Resource Management System (FIRMS):
* **NASA MODIS (Terra & Aqua)**: 1 km spatial resolution, 4 µm and 11 µm thermal infrared channels, Fire Radiative Power (FRP in MW).
* **NASA VIIRS (Suomi-NPP, NOAA-20, NOAA-21)**: 375 m high-resolution active fire observations across 24-hour and 7-day intervals.
* **South Asia / India Regional Extracts**: Filtered real-time fire points within Indian boundaries for regional early warnings.

### 2. Camera Surveillance Benchmark Dataset
* **Source**: Curated from public-domain HPWREN (High Performance Wireless Research and Education Network) and AI For Mankind datasets.
* **Classes**: `0: Fire`, `1: Smoke`, `2: Clouds/Fog (Negative Distractor)`, `3: Dust/Clear`.
* **Annotations**: Standard YOLO bounding box format for real-time localization.

### 3. Multi-Spectral NetCDF Datacubes (`.nc`)
* 27 remote sensing variables over 10 temporal days x 64x64 km patches (MODIS LST Day/Night, NDVI, Active Fire Mask, ERA5 Wind Vectors, Temperature, Humidity).

---

## 🧠 System Architecture & Hybrid Models

1. **Baseline 2D CNN**: 3-stage convolutional network for rapid 4-class image classification.
2. **Baseline YOLO (YOLOv8)**: Real-time bounding box detection and localization of smoke columns and flame fronts.
3. **ConvLSTM + Temporal Attention**: Preserves spatial feature maps across 10-day multi-spectral sequences, applying attention over temporal steps to predict wildfire spread velocity.
4. **Risk & Severity Fusion Engine**: Integrates detection probabilities, temporal persistence, physical growth rate, and environmental risk into standardized alerts.
5. **FastAPI Backend Server**: Exposes inference endpoints for camera feeds, satellite telemetry, and temporal sequence analytics.
6. **Vite + React GIS Dashboard**: Dark-mode operations console featuring OpenStreetMap Leaflet layers, regional toggles (India vs. Global), and early-warning alert panels.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js 18+ & npm

### 1. Setup Backend & Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ultralytics fastapi uvicorn pydantic torchvision xarray netcdf4
```

### 2. Run the Full Prototype
Use the automated PowerShell launcher:
```powershell
.\run_demo.ps1
```
* **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
* **Backend API Documentation**: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### 3. Download or Update Datasets
```powershell
python data/download_all_datasets.py
```

### 4. Train & Evaluate Models
```powershell
# Train Baseline CNN on real camera imagery
python models/baseline_cnn/train.py

# Train Baseline YOLOv8
python models/baseline_yolo/train.py

# Run ConvLSTM Temporal Attention on Satellite NetCDF
python models/convlstm_attention/train.py
```

---

## 📊 Evaluation Outputs
All evaluation artifacts and metrics are automatically logged into `output/evaluation/`:
* `baseline_cnn_metrics.json`: Precision, Recall, F1, Accuracy on real camera imagery.
* `baseline_yolo_metrics.json`: mAP@50, Precision, Recall, and inference FPS.
* `convlstm_attention_metrics.json`: Sequence attention weights and estimated spread velocity.

---

## 👥 Authors
Developed for **VIT Tech-a-thon 2026-2027** • Problem Statement 25
