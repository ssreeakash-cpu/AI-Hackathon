import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Loads an image from disk gracefully."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Exception loading image {image_path}: {e}")
        return None

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel of LAB color space."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception as e:
        logger.error(f"CLAHE processing failed: {e}")
        return image

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224), enhance: bool = True) -> np.ndarray:
    """Resizes, optionally enhances, and normalizes an image for model input."""
    if enhance:
        image = apply_clahe(image)
    resized = cv2.resize(image, target_size)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def extract_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extracts a Region of Interest given a bounding box (x, y, w, h)."""
    x, y, w, h = bbox
    # Ensure bounds are valid
    y_start, y_end = max(0, y), min(image.shape[0], y+h)
    x_start, x_end = max(0, x), min(image.shape[1], x+w)
    return image[y_start:y_end, x_start:x_end]
