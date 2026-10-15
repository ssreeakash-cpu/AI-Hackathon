import os
import json
import time
from ultralytics import YOLO

def train_and_evaluate():
    print("==================================================")
    print("--- Training Baseline YOLO on Real Camera Data ---")
    print("==================================================")
    
    yaml_path = os.path.abspath("datasets/camera_surveillance/dataset.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset config not found at: {yaml_path}")

    print(f"Loading real camera dataset configuration: {yaml_path}")
    
    # Initialize YOLOv8 nano model
    model = YOLO("yolov8n.pt") 
    
    print("Training YOLO on real camera smoke/fire surveillance images...")
    # Train for 3 epochs on the camera surveillance dataset
    results = model.train(
        data=yaml_path,
        epochs=3,
        imgsz=640,
        batch=2,
        project="output/yolo",
        name="baseline_camera",
        exist_ok=True,
        verbose=True
    )
    
    print("Evaluating YOLO on real validation images...")
    metrics = model.val()
    
    # Run inference on a real sample camera image
    test_img = "datasets/camera_surveillance/train/images/smoke_hpwren_01.png"
    print(f"Running inference on real camera test image: {test_img}")
    start = time.time()
    infer_results = model(test_img, save=True, project="output/yolo", name="predict", exist_ok=True)
    inference_time = time.time() - start
    fps = 1.0 / inference_time if inference_time > 0 else 0.0
    
    # Extract actual metrics
    output_metrics = {
        "dataset": "HPWREN / AI For Mankind Wildfire Camera Dataset",
        "mAP_50": float(metrics.box.map50),
        "mAP_50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "inference_time_seconds": inference_time,
        "fps": fps,
        "model": "baseline_yolo",
        "real_dataset": True,
        "demo_mode": False
    }
    
    os.makedirs("output/evaluation", exist_ok=True)
    with open("output/evaluation/baseline_yolo_metrics.json", "w") as f:
        json.dump(output_metrics, f, indent=4)
        
    print("\n--- YOLO Evaluation Complete on Real Data ---")
    print(json.dumps(output_metrics, indent=2))
    return output_metrics

if __name__ == "__main__":
    train_and_evaluate()
