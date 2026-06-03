# phone_detection.py
from utils import bbox_overlap, extract_boxes

# phone detection tuning
PHONE_CONF = 0.30
PHONE_IOU = 0.45
MIN_PHONE_W_H = (20, 20)         # minimum acceptable phone box size (px)
HAND_PHONE_OVERLAP_TH = 0.05     # require overlap with hand to accept phone detection
PERSON_PHONE_OVERLAP_TH = 0.02   # fallback overlap with person bbox

def process_phone(frame, phone_model, phone_class_indices, hand_boxes, person_boxes, model_names):
    """
    Runs phone detection and filters results based on context (hands, person).
    """
    phone_results = phone_model(frame, imgsz=640, conf=PHONE_CONF, iou=PHONE_IOU,
                               classes=phone_class_indices, verbose=False)
    raw_phone_dets = extract_boxes(phone_results)
    filtered_phones = []

    for pd in raw_phone_dets:
        bx = pd['box']
        conf = pd['confidence']
        w_box = bx[2] - bx[0]; h_box = bx[3] - bx[1]
        
        # reject too-small boxes (noise)
        if w_box < MIN_PHONE_W_H[0] or h_box < MIN_PHONE_W_H[1]:
            continue

        # Check if phone overlaps with any detected hand (preferred)
        keep = False
        for hb in hand_boxes:
            if bbox_overlap(bx, hb) > HAND_PHONE_OVERLAP_TH:
                keep = True
                break

        # If no hand overlap, allow if it overlaps person bbox (fallback)
        if not keep and person_boxes:
            for pb in person_boxes:
                if bbox_overlap(bx, pb) > PERSON_PHONE_OVERLAP_TH:
                    keep = True
                    break
        
        # In a single-person context (like a driver), if no hands or person
        # are found, you might choose to keep the phone detection anyway.
        # For now, we only keep if context is met.
        if not hand_boxes and not person_boxes:
             keep = True # Or False, depending on desired strictness

        # keep only if passes context checks
        if keep:
            filtered_phones.append({
                'box': bx,
                'confidence': conf,
                'class': model_names.get(pd['class'], str(pd['class']))
            })

    return filtered_phones