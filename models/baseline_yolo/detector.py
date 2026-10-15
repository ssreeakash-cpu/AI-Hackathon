from ultralytics import YOLO
import time
from typing import Tuple, Dict, Any

class FireSmokeYOLODetector:
    """
    Wrapper class for the YOLO-based Fire and Smoke detector.
    Handles loading the model and running inference with timing metrics.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        
    def predict(self, image_path: str) -> Tuple[Any, Dict[str, float]]:
        """
        Runs inference on an image and calculates FPS/Inference Time.
        Returns the YOLO results object and a dictionary of performance metrics.
        """
        start_time = time.time()
        results = self.model(image_path)
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        
        metrics = {
            "inference_time_seconds": inference_time,
            "fps": fps
        }
        
        return results, metrics
