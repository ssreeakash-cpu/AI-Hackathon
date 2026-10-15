# Dataset Strategy

To build an end-to-end AI wildfire early-warning prototype, we must integrate multi-modal data sources (satellite, camera imagery, and environmental variables). This document outlines the minimum realistic dataset combination required to implement all system features while keeping the scope appropriate for an academic research prototype.

## 1. Selected Datasets

### A. Satellite & Environmental Burned Area Dataset
**Dataset**: `mesogeos` (via WildFireSpread)
* **Source**: [Orion-AI-Lab / Hugging Face](https://huggingface.co/datasets/nikos230/WildfireSpread/resolve/main/dataset_64_64_10days.zip)
* **License**: Open/Academic
* **Purpose**: Temporal analysis, wildfire spread prediction, burned area estimation, and ConvLSTM model training.
* **Format**: `.nc` (netCDF)
* **Samples**: ~9,500 samples (64x64 patches, 10 temporal days).
* **Classes/Labels**: Binary burned area mask (0 = no fire, 1 = fire).
* **Variables Provided**: LST, NDVI, meteorological variables (temperature, wind, humidity, precipitation), slope, aspect, etc.
* **Suitability**: Perfectly suited for the 3D U-Net and proposed ConvLSTM temporal spread analysis.
* **Limitations**: Pre-cropped to 64x64, restricted to European/Mediterranean regions, limited to daily temporal resolution.

### B. Fire & Smoke Camera Imagery (Classification & Object Detection)
**Dataset**: FLAME (Fire Luminosity Airborne-based Machine learning Evaluation) & D-Fire
* **Source**: IEEE DataPort / Public Repositories
* **License**: Academic/Creative Commons
* **Purpose**: Training `baseline_cnn`, `baseline_yolo`, `baseline_efficientnet`, and spatial hybrids (YOLO + ViT).
* **Format**: Images (`.jpg`) and annotations (`.txt` YOLO format).
* **Samples**: A curated subset of ~5,000 images balancing fire, smoke, and normal conditions.
* **Classes**: 
  - `0`: Fire
  - `1`: Smoke
* **Suitability**: Excellent for real-time bounding box localization (YOLO) and confidence scoring. Contains aerial and static camera angles.
* **Limitations**: Lacks multispectral bands; completely detached from environmental variables (RGB only).

### C. Temporal Video Dataset
**Dataset**: HPWREN (High Performance Wireless Research and Education Network) Wildfire Cameras
* **Source**: UCSD / HPWREN Public Archive
* **License**: Public Domain/Academic
* **Purpose**: Testing temporal sequence hybrids (CNN + BiLSTM + Attention, YOLO + EfficientNet + LSTM) on real-time visual growth.
* **Format**: `.mp4` / extracted `.jpg` frames.
* **Samples**: ~50 curated time-lapse sequences of ignition and initial spread.
* **Suitability**: Crucial for evaluating visual growth velocity, smoke column evolution, and temporal persistence algorithms.
* **Limitations**: Highly specific to Southern California terrain; visual data only, no precise geolocation labels for the actual fire bounds.

## 2. Methodology & Synthesis

### Minimum Scientifically Valid Combination
To achieve a working prototype without overwhelming computational overhead, we will combine:
1. **D-Fire / FLAME subset** (for all camera-based Fire/Smoke detection & bounding boxes).
2. **mesogeos subset** (for all Satellite-based spread prediction and environmental risk calculations).
3. **HPWREN videos** (for visual temporal analysis and early-warning alert simulation).

### Geolocation & GIS
Since precise GPS metadata is often stripped from public image datasets, geolocation will be handled by:
1. Extracting center-point coordinates from the `mesogeos` `.nc` metadata where available.
2. For camera images, simulating an attached EXIF/GPS coordinate (clearly labeled as `DEMO_MODE` simulation) to demonstrate the GIS Leaflet dashboard functionality, as true coordinate mapping from oblique camera angles requires complex photogrammetry beyond the project scope.

## 3. Data Leakage Prevention
For the image/video datasets (FLAME/HPWREN):
- **Split Strategy**: Data will be partitioned by *fire event/video ID*, not by randomized frames. This ensures that near-identical sequential frames from the same camera feed do not appear in both the training and testing sets, which would artificially inflate accuracy metrics.
- The `mesogeos` dataset already handles temporal splitting inherently by providing isolated regional patches.

## 4. Hardware & Storage Constraints
- The full `mesogeos` dataset is 30GB unzipped. For rapid development, we will utilize a small subset (e.g., 200 samples) during initial pipeline construction.
- The YOLO dataset will be limited to 640x640 resolution to ensure training is feasible on a single consumer GPU or Google Colab instance.
