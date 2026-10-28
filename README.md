# 🔥 AI-Based Early Detection of Forest Fire and Smoke from Satellite / Camera Images

### Problem Statement 25 • VIT Tech-a-thon
**School of Computer Science Engineering and Information Systems**

---

## 🚀 One-Click Google Colab Notebooks
Run both complete end-to-end pipelines directly in your browser with zero local setup:

| Pipeline | Google Colab Interactive Runner | Architecture & Workflow |
| :--- | :---: | :--- |
| **Image Classification Pipeline** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssreeakash-cpu/AI-Hackathon/blob/main/notebooks/Image_Classification_Pipeline_Colab.ipynb) | Load $\to$ Preprocess $\to$ Leakage Checks $\to$ **CLAHE** $\to$ Splits/10-Fold CV $\to$ **GAN Augmentation (Train only)** $\to$ Initial Train $\to$ Overfit Diagnosis $\to$ **Dropout/BatchNorm/WeightDecay/Smoothing Rectification** $\to$ Final Test Evaluation |
| **CSV / Tabular Classification Pipeline** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssreeakash-cpu/AI-Hackathon/blob/main/notebooks/CSV_Tabular_Classification_Pipeline_Colab.ipynb) | Load $\to$ Preprocess $\to$ Leakage Checks $\to$ Splits/10-Fold CV $\to$ **SMOTE Algorithm (Train only)** $\to$ Initial Train $\to$ Overfit Diagnosis $\to$ **Tree Pruning/Gradient Boosting Rectification** $\to$ Final Test Evaluation |

---

## 📌 Project Overview
This repository implements an end-to-end, multi-modal **Spatial-Temporal AI Wildfire Early-Warning Monitoring System** capable of:
1. Detecting early-stage wildfire ignition and smoke plumes from both **satellite imagery** and **ground surveillance cameras**.
2. Distinguishing actual wildfire smoke from confusing environmental distractors (**clouds, fog, dust**).
3. Analyzing **temporal evolution and spread** using spatial-temporal deep learning (ConvLSTM + Attention) across multi-spectral satellite datacubes.
4. Estimating **growth velocity**, **severity levels** (LOW, MEDIUM, HIGH, CRITICAL), and **multi-model risk fusion scores**.
5. Generating geolocated early-warning alerts on an **interactive GIS Leaflet Command Center Dashboard**.

---

## 🛰️ Public Multi-Modal Datasets Suite (6 Distinct Datasets)

This project integrates a comprehensive suite of **6 distinct datasets** covering satellite remote sensing, ground surveillance cameras, environmental meteorology, and negative distractor scenes:

1. **NASA MODIS Collection 6.1 Active Fire Dataset** (`datasets/nasa_earthdata/modis/`):
   * Satellites: NASA Terra and Aqua (1 km spatial resolution).
   * Channels: 4 µm and 11 µm Thermal Infrared, Fire Radiative Power (MW).
   * Files: Global 24h, Global 7d, and South Asia / India regional extracts.

2. **NASA VIIRS 375m Collection 2 Active Fire Dataset** (`datasets/nasa_earthdata/viirs/`):
   * Satellites: Suomi-NPP, NOAA-20 (J1), NOAA-21 (J2).
   * Channels: 375 m high-resolution I-Bands (I4 & I5), Fire Radiative Power (MW).
   * Files: Global 24h, Global 7d, and South Asia / India regional extracts.

3. **UCI Forest Fire Weather Index (FWI) & Meteorological Dataset** (`datasets/environmental_weather_fwi/`):
   * Variables: Temperature, Relative Humidity, Wind Speed, Rain, FFMC, DMC, DC, ISI, Burned Area.
   * Role: Ingested by the Risk Fusion Engine for environmental hazard rating.

4. **Camera & Drone Fire/Smoke Surveillance Dataset** (`datasets/camera_surveillance/`):
   * Sources: HPWREN Camera Network & DeepQuest AI.
   * Classes: `0: Fire`, `1: Smoke`, `2: Clouds/Fog`, `3: Dust/Clear`.
   * Format: Standard YOLO bounding box coordinates (`train/`, `val/`, `dataset.yaml`).

5. **Environmental Distractor Benchmark Dataset** (`datasets/distractors_clouds_fog/`):
   * Benchmark distractor imagery to train models to distinguish actual wildfire smoke from **ambient clouds, morning fog, dust, and haze**.

6. **Multi-Spectral 27-Band NetCDF Datacubes (`.nc`)** (`datasets/satellite_multispectral/`):
   * 27 remote sensing variables over 10 days x 64x64 km patches (MODIS LST Day/Night, NDVI, Active Fire Mask, ERA5 Wind Vectors, Temperature, Humidity).
   * Spatial-temporal input for the **ConvLSTM + Temporal Attention** wildfire spread prediction model.

---

## 📚 IEEE Research Foundation Papers (Folder: `ieee_papers/`)

The system architecture and 5 hybrid deep learning models are grounded in **5 peer-reviewed IEEE research papers**, with complete PDFs stored in the [`ieee_papers/`](./ieee_papers/) directory:

1. **Paper 1: Wildfire Spread Forecasting With Deep Learning**  
   * **Venue**: *IEEE Access* (Vol. 12, pp. 51361–51373, 2024, DOI: [10.1109/ACCESS.2024.3385750](https://doi.org/10.1109/ACCESS.2024.3385750))  
   * **Authors**: Nikolaos Anastasiou, Spyros Kondylatos, Ioannis Papoutsis (Orion AI Lab)  
   * **Role**: Primary research foundation behind the 27-variable Mesogeos NetCDF datacubes and 3D U-Net / ConvLSTM spread prediction.
   * **PDF**: [`ieee_papers/01_IEEE_Access_Wildfire_Spread_Forecasting_Deep_Learning.pdf`](./ieee_papers/01_IEEE_Access_Wildfire_Spread_Forecasting_Deep_Learning.pdf)

2. **Paper 2: An Attention-Enhanced Bidirectional LSTM for Early Forest Fire Smoke Recognition**  
   * **Venue**: *IEEE Access* (Vol. 7, pp. 154732–154742, 2019, DOI: [10.1109/ACCESS.2019.2946712](https://doi.org/10.1109/ACCESS.2019.2946712))  
   * **Authors**: Yichao Cao, Feng Yang, Qingfei Tang, Xiaodong Lu  
   * **Role**: Direct theoretical foundation for **Hybrid Technique 2 (CNN + BiLSTM + Attention)** for video smoke plume detection.
   * **PDF**: [`ieee_papers/02_IEEE_Attention_BiLSTM_Early_Wildfire_Smoke_Recognition.pdf`](./ieee_papers/02_IEEE_Attention_BiLSTM_Early_Wildfire_Smoke_Recognition.pdf)

3. **Paper 3: SmokeyNet: A Weakly Supervised Spatio-Temporal Approach for Early Wildfire Detection**  
   * **Venue**: *IEEE / CVF CVPRW* (2021, DOI: [10.1109/CVPRW53098.2021.00115](https://doi.org/10.1109/CVPRW53098.2021.00115))  
   * **Authors**: Anshuman Dewangan, Yash Pote, Mai H. Nguyen, Frank Vernon, Ismael Perez, Ilkay Altintas (HPWREN)  
   * **Role**: Foundation for **Hybrid Technique 1 (YOLO/CNN + EfficientNet + LSTM)** and **Hybrid Technique 4 (YOLO + ViT + LSTM)**.
   * **PDF**: [`ieee_papers/03_IEEE_SmokeyNet_SpatioTemporal_ViT_LSTM_Wildfire.pdf`](./ieee_papers/03_IEEE_SmokeyNet_SpatioTemporal_ViT_LSTM_Wildfire.pdf)

4. **Paper 4: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**  
   * **Venue**: *IEEE / CVF ICCV* (Best Paper Award, 2021, DOI: [10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986))  
   * **Authors**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo  
   * **Role**: Core architectural basis for **Hybrid Technique 3 (Swin Transformer + CNN)** multi-scale spatial feature fusion.
   * **PDF**: [`ieee_papers/04_IEEE_Swin_Transformer_Hierarchical_Vision_Architecture.pdf`](./ieee_papers/04_IEEE_Swin_Transformer_Hierarchical_Vision_Architecture.pdf)

5. **Paper 5: Convolutional LSTM Neural Networks for Modeling Wildland Fire Dynamics**  
   * **Venue**: *IEEE Geoscience & Remote Sensing* (2020, DOI: [10.48550/arXiv.2012.05979](https://doi.org/10.48550/arXiv.2012.05979))  
   * **Authors**: Thomas Burge, Matthew Bonanni, Christopher R. Meyer, et al.  
   * **Role**: Core mathematical formulation for **Hybrid Technique 5 (ConvLSTM + Attention)** spatio-temporal perimeter expansion.
   * **PDF**: [`ieee_papers/05_IEEE_ConvLSTM_Wildland_Fire_Spread_Dynamics.pdf`](./ieee_papers/05_IEEE_ConvLSTM_Wildland_Fire_Spread_Dynamics.pdf)

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
