# detectors/hand_detector.py
from ultralytics import YOLO
import os


HAND_MODEL_PATH = "/Users/deepak/Desktop/Driver_Monitoring_System/models/handdsa.pt"

try:
    _hand_model = YOLO(HAND_MODEL_PATH)
except Exception as e:
    # gracefully allow no hand model
    _hand_model = None
    print(f"Warning: hand model not loaded ({HAND_MODEL_PATH}): {e}")

def extract_boxes_from_results(results):
    items = []
    try:
        for r in results:
            for box in r.boxes:
                items.append({
                    'box': tuple(map(int, box.xyxy[0].tolist())),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
    except Exception:
        return []
    return items

def predict(frame, verbose=False):
    """
    Return list of hand detections in same format as extract_boxes.
    If no model is found, returns [].
    """
    if _hand_model is None:
        return []
    res = _hand_model(frame, verbose=verbose)
    return extract_boxes_from_results(res)
