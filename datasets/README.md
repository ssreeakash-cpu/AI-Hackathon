# Project Datasets: Problem Statement 25

This directory contains the complete multi-modal datasets required for **AI-Based Early Detection of Forest Fire and Smoke from Satellite / Camera Images**.

## 1. NASA Earthdata (MODIS and VIIRS Fire Data)
Directory: `datasets/nasa_earthdata/`

* **MODIS (Moderate Resolution Imaging Spectroradiometer)**
  * Satellites: NASA Terra and Aqua
  * Spatial Resolution: 1 km
  * Detection Channels: 4 µm (brightness) & 11 µm (bright_t31) thermal infrared
  * Files:
    * `modis/MODIS_C6_1_Global_24h.csv`
    * `modis/MODIS_C6_1_Global_7d.csv`
    * `south_asia_india/MODIS_SouthAsia_India_24h.csv`

* **VIIRS (Visible Infrared Imaging Radiometer Suite)**
  * Satellites: Suomi-NPP, NOAA-20 (J1), NOAA-21 (J2)
  * Spatial Resolution: 375 m (I-Bands)
  * Detection Channels: I4 (bright_ti4) & I5 (bright_ti5) + Fire Radiative Power (FRP in MW)
  * Files:
    * `viirs/SUOMI_VIIRS_C2_Global_24h.csv`
    * `viirs/SUOMI_VIIRS_C2_Global_7d.csv`
    * `viirs/J1_VIIRS_C2_Global_24h.csv`
    * `viirs/J1_VIIRS_C2_Global_7d.csv`
    * `viirs/J2_VIIRS_C2_Global_24h.csv`
    * `viirs/J2_VIIRS_C2_Global_7d.csv`
    * `south_asia_india/VIIRS_SouthAsia_India_24h.csv`

* **GIS GeoJSON Feature Collection**
  * `nasa_active_fires.geojson`: Ready for Leaflet / OpenStreetMap visualization.

## 2. Camera & Surveillance Fire/Smoke Dataset
Directory: `datasets/camera_surveillance/`
* Includes fire, smoke, and negative distractor objects (clouds, fog, dust) to fulfill the problem statement:
  > *"The system should distinguish actual wildfire smoke from clouds, fog, dust, and other visually similar objects."*
* Format: YOLO bounding boxes (`dataset.yaml`).

## 3. Satellite Multi-Spectral Datacubes
Directory: `datasets/satellite_multispectral/`
* Format: NetCDF (`.nc`)
* 27 variables over 10 days x 64x64 km patches (MODIS LST Day/Night, NDVI, ERA5 Wind, Soil Moisture, FRP).
* Used for the **ConvLSTM + Attention** spatial-temporal wildfire spread prediction.
