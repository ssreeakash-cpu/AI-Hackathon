# WildFireSpread Repository Analysis

This document provides a comprehensive analysis of the existing research foundation repository, **WildFireSpread** (https://github.com/Orion-AI-Lab/WildFireSpread), and details the strategy for integrating its components into the new, end-to-end AI wildfire early-warning system.

## 1. Repository Architecture
The repository is structured primarily around training and evaluating deep learning models for wildfire spread prediction (specifically burned area prediction) based on the `mesogeos` dataset. 

Key directories include:
- `configs/`: Configuration files for dataset paths and training/testing setups (`dataset.yaml`, `train_test_config_unet2d.yaml`, `train_test_config_unet3d.yaml`).
- `unet/`: Contains model definitions (`loss.py`, `model.py`, `model_unet3d.py`, `model_unet3d_struct.py`).
- `utils/`: Contains PyTorch `Dataset` loaders and other utilities (`dataset_unet2d.py`, `dataset_unet3d.py`, `utils.py`).
- `scripts/`: Various helper scripts for data analysis, sample generation, shapefile creation, clustering, and dataset statistics extraction.
- `output/` and `loss_csv/`: Output metrics and visualization artifacts.
- Root scripts: Main execution scripts for training, testing, and xAI (`train_unet2d.py`, `train_unet3d.py`, `test_unet2d.py`, `test_unet3d.py`, `xAI_test_unet3d.py`).

## 2. Important Files
- `README.md`: Documents the scope, dataset specifics, models, and evaluation metrics (Dice, IoU, Precision, Recall).
- `unet/model.py`: Contains the `UNet2D` model, its structural variations, and a `ViTSegmentation2` (Vision Transformer) model.
- `unet/model_unet3d.py`: Contains the `UNet3D` model with a Grad-CAM implementation.
- `utils/dataset_unet3d.py` & `utils/dataset_unet2d.py`: Defines the data ingestion pipeline for `.nc` (netCDF) samples, including data normalization and specific channel configurations.
- `configs/dataset.yaml`: Specifies geographic constraints, train/val/test splits (by year and country), and dataset paths.

## 3. Existing Models
The following models are explicitly available in the codebase:
- **UNet2D**: Baseline and temporal sequence models.
- **UNet3D**: Proposed temporal-spatial model using 3D convolutions.
- **Vision Transformer (ViT)**: A variant (`ViTSegmentation2`) is present in `unet/model.py`.
- **ConvLSTM**: *Reported in the README metrics but not found in the source code files.*

## 4. Existing Datasets
The project uses the `mesogeos` dataset (DataCube), which provides $64 \times 64$ km patches across 10 temporal days (5 days prior, 5 days after ignition). 
- Spatial resolution: $1$ pixel = $1 \times 1$ km.
- Temporal resolution: $1$ day.
- Size: ~9500 samples across 31 countries (2006-2022).

## 5. Existing Preprocessing
Preprocessing is handled inside the PyTorch `Dataset` classes (`dataset_unet3d.py` / `dataset_unet2d.py`):
- Data loading from `.nc` (netCDF) files using `xarray`.
- Separation of *dynamic* (weather, indices, LST, fire data) and *static* (DEM, slope, LC fractions) variables.
- Scaling/Normalization (Min-Max normalization per channel).
- Label generation: Binarizing the `burned_areas` on day 4 (ignition day).

## 6. Existing Training Pipeline
The training pipeline consists of standalone scripts (`train_unet2d.py`, `train_unet3d.py`):
- Adam optimizer and custom loss functions (`DiceLoss`, `BCEWithLogitsLoss`).
- Hyperparameters configured via hardcoded settings or YAML files.
- Basic metric tracking (Loss, IoU, Dice).

## 7. Existing Inference Pipeline
The inference pipeline is implemented in `test_unet2d.py` and `test_unet3d.py`:
- Model loading from `.pth` checkpoints.
- Generating predictions, binarizing them, and computing standard segmentation metrics.
- Exporting binary masks to images and Shapefiles (`.shp`).

## 8. Existing ConvLSTM Implementation
While ConvLSTM is listed in the README's performance tables, there is **no ConvLSTM implementation in the repository**. It will need to be newly implemented from scratch.

## 9. Existing U-Net Implementation
- **2D U-Net**: Fully functional, located in `unet/model.py`.
- **3D U-Net**: Fully functional, located in `unet/model_unet3d.py`. Supports arbitrary depths/filters and includes `grad-cam` support.

## 10. Existing ViT Implementation
A `ViTSegmentation2` model exists in `unet/model.py` but seems to be a secondary component. It is not currently active in the default training scripts but is accessible.

## 11. Existing Environmental Features
The repository natively processes 27 remote sensing variables from the `mesogeos` dataset:
- **Dynamic**: `d2m`, `ignition_points`, `lai`, `lst_day`, `lst_night`, `ndvi`, `rh`, `smi`, `sp`, `ssrd`, `t2m`, `tp`, `u` (wind), `v` (wind).
- **Static**: `aspect`, `curvature`, `dem`, `slope`, and various land cover fractions (`lc_agriculture`, `lc_forest`, etc.).

## 12. Existing Wildfire Spread Functionality
The system currently models wildfire spread as a *burned area segmentation task*. Given $t$ days of environmental data and the ignition point/prior burned area, the models predict the final shape and extent of the burned region. Growth velocity and continuous spread predictions over arbitrary horizons are not supported.

## 13. Components Reusable for Our Project
- **3D U-Net Architecture**: Can be extended or used directly as the spatial-temporal backbone.
- **Dataset Loaders**: `utils/dataset_unet3d.py` logic can be adapted for our temporal pipeline when dealing with netCDF datasets.
- **Environmental Processing Logic**: Handling of dynamic/static variables and their scaling.
- **Evaluation Utilities**: Functions for calculating Dice and IoU can be reused.

## 14. Components That Must Be Newly Implemented
The following features from the new architecture are completely missing and must be built:
- **Fire/Smoke Detection (Image/Video Level)**: No camera/YOLO/CNN-based object detection functionality currently exists.
- **Sequence Generation for Video/Camera imagery**: The current pipeline is strictly tied to 1-day step netCDF files.
- **ConvLSTM Backbone**: Needs to be implemented.
- **Attention Mechanisms**: Temporal/spatial attention for BiLSTM and ConvLSTM hybrids.
- **Risk & Severity Fusion Engines**: Logic to combine confidence, spread, and persistence into a single severity/risk score.
- **Growth Velocity Calculators**: Calculating dynamic growth over time from model outputs.
- **Geolocation & GIS Engine**: Translating raw pixels back to actionable latitude/longitude coordinates and creating Leaflet interfaces.
- **API (FastAPI) and Frontend (React)**: No production infrastructure exists.

## 15. Potential Compatibility Problems
- **Dataset Input Formatting**: The existing models strictly expect 27 specific variables from netCDF files in a fixed dimension format (64x64x10). A unified processing pipeline is required to make models agnostic to the source (Satellite vs Camera).
- **Pytorch vs API**: Existing code heavily ties tensor operations directly into evaluation loops. The inference scripts need to be completely abstracted to handle single REST API requests (FastAPI).

## 16. Dataset Requirements
To satisfy the complete system, we need datasets beyond the current `mesogeos` foundation:
- **Image/Video Fire/Smoke Datasets**: For YOLO and baseline CNN training (e.g., FLAME dataset, HPWREN camera dataset).
- The existing `mesogeos` dataset will suffice for the satellite/environmental burned-area segmentation tasks.

## 17. Hardware Requirements
- **Training**: Requires significant GPU acceleration (e.g., NVIDIA RTX 3090/4090 or A100) due to 3D Convolutions and multi-modal transformers.
- **Inference**: Can be run on standard GPUs (e.g., T4/V100) with proper batching. CPU inference may be too slow for the 3D models and YOLO hybrids but is possible for demo modes.

## 18. Recommended Implementation Strategy
1. **Isolate and Abstract**: Extract the useful U-Net and netCDF dataset components from `WildFireSpread` and place them into the new `models/` and `preprocessing/` directories. 
2. **Build the Object Detection Pipeline**: Start with gathering datasets for fire/smoke images and implement `baseline_yolo` and `baseline_cnn` first.
3. **Temporal Processing**: Create sequence generators that can bridge the gap between video frames and the input structures expected by ConvLSTMs.
4. **Implement Missing Hybrids**: Write the ConvLSTM, BiLSTM, and Attention architectures.
5. **Inference & Risk Fusion**: Develop the business logic (Severity, Growth Velocity) as independent Python modules.
6. **API and Dashboard**: Finally, wrap the models in FastAPI and build the React frontend for the complete system showcase.
