# blink_detection.py
import math
import cv2
import time  # <-- Import time

# --- EAR calculation using MediaPipe landmarks ---
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# thresholds
EAR_BLINK_THRESHOLD = 0.25        # eye aspect ratio threshold for closed
EAR_CONSEC_FRAMES = 2             # how many consecutive frames to consider a BLINK
DROWSY_SECONDS_THRESHOLD = 2.0    # <-- New threshold for drowsiness

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def compute_ear(landmarks, image_width, image_height, side_idxs):
    pts = []
    for idx in side_idxs:
        lm = landmarks[idx]
        pts.append((lm.x * image_width, lm.y * image_height))
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def process_blinking(frame, face_mesh, state, image_width, image_height):
    """
    Detects blinks and drowsiness using MediaPipe Face Mesh.
    Updates the 'state' dictionary with blink counts and timestamps.
    Returns a dictionary with current blink/drowsy status.
    """
    blink_detected = False
    drowsy_alert = False  # <-- New flag for drowsiness
    ear_val = None
    
    frame.flags.writeable = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(rgb)
    frame.flags.writeable = True

    if mp_results.multi_face_landmarks:
        landmarks = mp_results.multi_face_landmarks[0].landmark
        left_ear = compute_ear(landmarks, image_width, image_height, LEFT_EYE_IDX)
        right_ear = compute_ear(landmarks, image_width, image_height, RIGHT_EYE_IDX)
        ear_val = float((left_ear + right_ear) / 2.0)

        # --- Check if eyes are closed ---
        if ear_val < EAR_BLINK_THRESHOLD:
            # 1. Blink counter logic (same as before)
            state['eye_closed_count'] += 1

            # 2. Drowsy timer logic (new)
            if state.get('eye_closed_start_time') is None:
                # If eyes just closed, record the start time
                state['eye_closed_start_time'] = time.time()
            else:
                # If eyes are still closed, check duration
                duration = time.time() - state['eye_closed_start_time']
                if duration > DROWSY_SECONDS_THRESHOLD:
                    drowsy_alert = True  # Trigger the alert

        # --- Eyes are open ---
        else:
            # 1. Blink counter logic
            if state['eye_closed_count'] >= EAR_CONSEC_FRAMES:
                state['blink_count'] += 1
                blink_detected = True
            
            # 2. Reset counters
            state['eye_closed_count'] = 0
            state['eye_closed_start_time'] = None  # Reset drowsy timer

        state['last_eye_status'] = "closed" if ear_val < EAR_BLINK_THRESHOLD else "open"
    
    # --- No face landmarks found ---
    else:
        # Reset all counters if face is lost
        state['eye_closed_count'] = 0
        state['last_eye_status'] = "open"
        state['eye_closed_start_time'] = None

    return {
        'blink_detected': blink_detected,
        'blink_count': state['blink_count'],
        'eye_ear': ear_val,
        'drowsy_alert': drowsy_alert  # <-- Send the alert to the frontend
    }