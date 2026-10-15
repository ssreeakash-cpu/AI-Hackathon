import pandas as pd
import json
import os
from datetime import datetime

def fetch_nasa_firms_data():
    """
    Downloads LIVE active fire hotspot data for the last 24 hours directly from 
    NASA's FIRMS (Fire Information for Resource Management System) public API.
    Retrieves both MODIS and VIIRS telemetry.
    """
    print("Connecting to NASA Earthdata FIRMS API...")
    
    # NASA FIRMS public CSV URLs for the last 24 hours (Global)
    urls = {
        "VIIRS": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
        "MODIS": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv"
    }
    
    combined_fires = []
    
    for source, url in urls.items():
        try:
            print(f"Downloading active {source} telemetry...")
            df = pd.read_csv(url)
            
            # Filter for higher confidence to reduce noise and false positives
            if source == "VIIRS":
                # VIIRS confidence is 'n' (nominal) or 'h' (high)
                df = df[df['confidence'].isin(['n', 'h'])]
            else:
                # MODIS confidence is 0-100%
                df = df[df['confidence'] > 75]
                
            # Grab the hottest 25 fires from each satellite so we don't crash the React map
            brightness_col = 'brightness' if 'brightness' in df.columns else 'bright_ti4'
            df = df.nlargest(25, brightness_col)
            
            for _, row in df.iterrows():
                combined_fires.append({
                    "source": f"NASA {source} Satellite",
                    "latitude": float(row['latitude']),
                    "longitude": float(row['longitude']),
                    "brightness_kelvin": float(row[brightness_col]),
                    "confidence": str(row['confidence']),
                    "acq_date": str(row['acq_date']),
                    "acq_time": str(row['acq_time'])
                })
        except Exception as e:
            print(f"Error fetching {source} data: {e}")
            
    # Ensure data directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasa_active_fires.json")
    with open(output_path, "w") as f:
        json.dump(combined_fires, f, indent=4)
        
    print(f"Successfully extracted {len(combined_fires)} high-confidence active fire hotspots!")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    fetch_nasa_firms_data()
