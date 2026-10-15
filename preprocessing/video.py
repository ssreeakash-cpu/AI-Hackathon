import cv2
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, fps_target: int = 1, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Extracts frames from a video file at a specified frames-per-second rate.
    Handles corrupt videos gracefully by aborting or returning extracted frames.
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return frames

        fps_original = cap.get(cv2.CAP_PROP_FPS)
        if fps_original <= 0 or np.isnan(fps_original):
            fps_original = 30.0  # Fallback to standard 30fps

        frame_interval = max(1, int(fps_original / fps_target))
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                
                if max_frames and len(frames) >= max_frames:
                    break
            count += 1
            
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Exception during video processing {video_path}: {e}")
        return frames
