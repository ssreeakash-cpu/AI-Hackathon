import numpy as np
import logging
from typing import List, Optional, Tuple
try:
    import xarray as xr
except ImportError:
    xr = None

logger = logging.getLogger(__name__)

def load_satellite_netcdf(nc_path: str, dynamic_vars: List[str], static_vars: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads a mesogeos netCDF file and extracts dynamic and static variables.
    Preserves multispectral environmental information without blindly converting to RGB.
    
    Returns:
        Tuple of (input_tensor, label_tensor) or None on failure.
    """
    if xr is None:
        logger.error("xarray is not installed. Run 'pip install xarray'.")
        return None
        
    try:
        sample = xr.open_dataset(nc_path)
        time_steps = sample['time'].size
        
        # Isolate the ignition points from day 4 as per foundation methodology
        day4_ignition_points = sample['ignition_points'].isel(time=4).copy()
        sample['ignition_points'][:] = 0
        sample['ignition_points'].values[4, :, :] = day4_ignition_points
        
        inputs = []
        
        for var in dynamic_vars:
            inputs.append(sample[var].values)
            
        for var in static_vars:
            data = sample[var].values
            # Broadcast static var across time steps
            data = np.repeat(data[np.newaxis, :, :], time_steps, axis=0)
            inputs.append(data)
            
        input_tensor = np.stack(inputs, axis=0)
        
        # Min-Max Normalization and missing data handling per channel
        for channel in range(input_tensor.shape[0]):
            channel_data = input_tensor[channel]
            data_min = np.nanmin(channel_data)
            data_max = np.nanmax(channel_data)
            
            if data_max - data_min > 0:
                input_tensor[channel] = (channel_data - data_min) / (data_max - data_min)
            else:
                input_tensor[channel] = np.zeros_like(channel_data)
        
        # Replace NaNs/Infs with 0
        input_tensor = np.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Label variable (burned area at day 4)
        label = sample['burned_areas'].values[4]
        label = (label > 0).astype(np.float32)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        
        sample.close()
        return input_tensor, label
        
    except Exception as e:
        logger.error(f"Failed to process satellite data {nc_path}: {e}")
        return None
