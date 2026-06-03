# smoke_detection.py
import cv2
from ultralytics import YOLO
from utils import extract_boxes

def process_smoke(frame, smoke_model):
    results = smoke_model(frame, verbose=False)
    detections = extract_boxes(results)

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = smoke_model.names.get(det['class'], "Smoking")
        conf = det['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        print(f"🚬 Smoking detected: {label} ({conf*100:.1f}%)")

    return detections
