from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GeolocationEngine:
    """
    Phase 9: Extracts and manages geolocation coordinates from metadata.
    """
    @staticmethod
    def extract_coordinates(metadata: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Attempts to extract real coordinates. Never fabricates them.
        Returns: (location_available, lat, lon)
        """
        lat = metadata.get("latitude")
        lon = metadata.get("longitude")
        
        if lat is not None and lon is not None:
            return True, float(lat), float(lon)
            
        logger.warning("Coordinates unavailable in metadata.")
        return False, None, None
