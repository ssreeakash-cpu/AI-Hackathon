# Multi-Modal Dataset Suite: Problem Statement 25

This repository contains a comprehensive suite of **6 distinct datasets** curated to fulfill every requirement of **AI-Based Early Detection of Forest Fire and Smoke from Satellite / Camera Images**:

---

### 1. NASA MODIS Collection 6.1 Active Fire Dataset
* **Path**: `datasets/nasa_earthdata/modis/`
* **Satellites**: NASA Terra & Aqua (1 km spatial resolution)
* **Bands**: 4 µm and 11 µm Thermal Infrared Channels, Fire Radiative Power (MW)
* **Files**: `MODIS_C6_1_Global_24h.csv`, `MODIS_C6_1_Global_7d.csv`, `MODIS_SouthAsia_India_24h.csv`

### 2. NASA VIIRS 375m Collection 2 Active Fire Dataset
* **Path**: `datasets/nasa_earthdata/viirs/`
* **Satellites**: Suomi-NPP, NOAA-20 (J1), NOAA-21 (J2)
* **Resolution**: 375 m high resolution (3x sharper than MODIS)
* **Files**: `SUOMI_VIIRS_C2_Global_24h.csv`, `SUOMI_VIIRS_C2_Global_7d.csv`, `J1_VIIRS_C2_Global_24h.csv`, `J2_VIIRS_C2_Global_24h.csv`, `VIIRS_SouthAsia_India_24h.csv`

### 3. UCI Forest Fire Weather Index (FWI) & Meteorological Dataset
* **Path**: `datasets/environmental_weather_fwi/forestfires.csv`
* **Variables**: Temperature, Relative Humidity, Wind Speed, Rain, Fine Fuel Moisture Code (FFMC), Duff Moisture Code (DMC), Drought Code (DC), Initial Spread Index (ISI), Burned Area
* **Role**: Ingested by the Risk Fusion Engine to evaluate environmental fire danger.

### 4. Camera & Drone Fire/Smoke Surveillance Dataset
* **Path**: `datasets/camera_surveillance/`
* **Sources**: HPWREN Camera Network & DeepQuest AI
* **Format**: YOLO bounding box annotations (`train/`, `val/`, `dataset.yaml`)
* **Role**: Ground surveillance camera detection for real-time smoke column recognition.

### 5. Environmental Distractor Benchmark Dataset (Clouds, Fog, Dust)
* **Path**: `datasets/distractors_clouds_fog/`
* **Role**: Directly satisfies the problem statement:
  > *"The system should distinguish actual wildfire smoke from clouds, fog, dust, and other visually similar objects."*

### 6. Multi-Spectral NetCDF Datacubes (`.nc`)
* **Path**: `datasets/satellite_multispectral/sample_modis_viirs_10days_64x64.nc`
* **Schema**: 27 remote sensing variables over 10 days x 64x64 km patches (MODIS LST Day/Night, NDVI, Active Fire Mask, ERA5 Wind Vectors, Temperature, Humidity).
* **Role**: Spatial-Temporal sequence input for ConvLSTM + Attention spread prediction.
