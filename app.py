import base64
import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import mediapipe as mp

# Import detection modules
from utils import extract_boxes
from mask_detection import process_face_mask
from blink_detection import process_blinking
from phone_detection import process_phone
from smoke_detection import process_smoke  

# --- NEW: Import alert system ---
from alert_system import configure_alerts, upload_to_cloudinary, send_whatsapp_alert

# --------------- CONFIG ---------------
MODELS_DIR = "models"
MASK_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")         # mask model
PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")    # COCO model
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "handdsa.pt")      # hand model (optional)
PHONE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")     # use COCO for phone
SMOKE_MODEL_PATH = os.path.join(MODELS_DIR, "detection_module.pt")  # your working smoke model

# thresholds
HAND_ON_MOUTH_THRESHOLD = 0.40
DETECT_HEAVY_EVERY_N = 3
ALERT_COOLDOWN_SECONDS = 60 # --- NEW: Cooldown for alerts (in seconds)

# Flask app
app = Flask(__name__)

# --- NEW: Configure Alert System ---
configure_alerts()

# --- LOAD MODELS ---
print("Loading YOLO models (this can take a while)...")
mask_model = YOLO(MASK_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

# load hand model (optional)
try:
    hand_model = YOLO(HAND_MODEL_PATH)
except Exception as e:
    print(f"Warning: could not load hand model ({HAND_MODEL_PATH}): {e}")
    hand_model = None

# load phone model
try:
    phone_model = YOLO(PHONE_MODEL_PATH)
    phone_class_indices = [i for i, name in phone_model.names.items() if 'phone' in name.lower()]
    if not phone_class_indices:
        print("Warning: phone class not found in phone_model.names — phone detection disabled.")
    else:
        print("Phone class indices:", phone_class_indices, "=>", [phone_model.names[i] for i in phone_class_indices])
except Exception as e:
    print(f"Warning: could not load phone model ({PHONE_MODEL_PATH}): {e}")
    phone_model = None
    phone_class_indices = []

# load smoke model
if os.path.exists(SMOKE_MODEL_PATH):
    try:
        smoke_model = YOLO(SMOKE_MODEL_PATH)
        print("✅ Smoke model loaded:", SMOKE_MODEL_PATH)
    except Exception as e:
        print(f"Warning: could not load smoke model ({SMOKE_MODEL_PATH}): {e}")
        smoke_model = None
else:
    smoke_model = None
    print("Smoke model not found; smoking detection disabled.")

print("Models loaded successfully.")

# --- MediaPipe face mesh for blink detection ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Client State ---
client_states = {}
def get_client_key():
    return request.remote_addr or "unknown"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        payload = request.get_json()
        img_b64 = payload["image"].split(",", 1)[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

        client = get_client_key()
        state = client_states.setdefault(client, {
            "frame_count": 0,
            "eye_closed_count": 0,
            "blink_count": 0,
            "last_eye_status": "open",
            "eye_closed_start_time": None,
            "last_phone_dets": [],
            "last_smoke_dets": [],
            "last_person_boxes": [],
            "last_mask_dets": [],
            "last_hand_boxes": [],
            "last_inference_time": 0.0,
            "last_drowsy_alert_time": 0.0 # --- NEW: Alert timestamp
        })
        state["frame_count"] += 1

        h, w = frame.shape[:2]

        # --- Face / Mask / Hand detection ---
        mask_results = mask_model(frame, verbose=False)

        hand_boxes = []
        if hand_model is not None:
            hand_results = hand_model(frame, verbose=False)
            hand_detections = extract_boxes(hand_results)
            hand_boxes = [hb['box'] for hb in hand_detections]

        final_face_detections = process_face_mask(
            mask_results, hand_boxes, mask_model.names, HAND_ON_MOUTH_THRESHOLD
        )
        state['last_mask_dets'] = final_face_detections
        state['last_hand_boxes'] = hand_boxes

        # --- Person detection ---
        person_results = person_model(frame, classes=[0], verbose=False)
        person_detections = extract_boxes(person_results)
        person_boxes = [p['box'] for p in person_detections]
        state['last_person_boxes'] = person_boxes

        # --- Social distancing calculation (assuming calculate_distance exists in utils or here) ---
        # Mocking this as the function is not provided, but keeping your logic
        social_distancing = []
        # ... (your social distancing logic) ...


        # --- Blink detection ---
        blink_data = process_blinking(frame, face_mesh, state, w, h)

        # --- NEW: Check for Drowsy Alert ---
        if blink_data.get('drowsy_alert'):
            current_time = time.time()
            if (current_time - state.get("last_drowsy_alert_time", 0)) > ALERT_COOLDOWN_SECONDS:
                print("DROWSINESS ALERT DETECTED! Sending alert...")
                
                # Update state *before* sending to prevent parallel requests
                state["last_drowsy_alert_time"] = current_time 
                
                # 1. Upload frame to Cloudinary
                image_url = upload_to_cloudinary(frame)
                
                # 2. Send WhatsApp message
                if image_url:
                    send_whatsapp_alert(image_url)
                else:
                    print("Could not send alert, image upload failed.")
            else:
                # Drowsy, but within cooldown period
                pass 
                # print("Drowsy alert in cooldown...")


        # --- Phone and Smoking detection (throttled) ---
        phone_detections = []
        smoke_detections = []
        run_heavy = (state['frame_count'] % DETECT_HEAVY_EVERY_N == 0)

        if run_heavy:
            now = time.time()

            # --- PHONE detection ---
            if phone_model is not None and phone_class_indices:
                phone_detections = process_phone(
                    frame, phone_model, phone_class_indices, hand_boxes, person_boxes, phone_model.names
                )
                state['last_phone_dets'] = phone_detections
            else:
                phone_detections = state.get('last_phone_dets', [])

            # --- ✅ SMOKE detection ---
            if smoke_model is not None:
                smoke_detections = process_smoke(frame, smoke_model)
                state['last_smoke_dets'] = smoke_detections
            else:
                smoke_detections = state.get('last_smoke_dets', [])

            state['last_inference_time'] = time.time() - now
        else:
            phone_detections = state.get('last_phone_dets', [])
            smoke_detections = state.get('last_smoke_dets', [])

        # --- Final Response ---
        resp = {
            'face_detections': state['last_mask_dets'],
            'person_boxes': state['last_person_boxes'],
            'social_distancing': social_distancing,
            'blink': blink_data,
            'phone_detections': phone_detections,
            'smoke_detections': smoke_detections,
            'perf': {
                'last_heavy_inference_time_sec': round(state.get('last_inference_time', 0.0), 3)
            }
        }

        return jsonify(resp)

    except Exception as e:
        print("Error in /process_image:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # Note: Flask's reloader can cause issues with client state.
    # For production, use a proper WSGI server (like gunicorn).
    # use_reloader=False can help during dev if state resets are an issue.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
