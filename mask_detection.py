# mask_detection.py
from utils import bbox_overlap, extract_boxes

def process_face_mask(mask_results, hand_boxes, model_names, hand_on_mouth_threshold):
    """
    Processes mask detections and checks for hand-on-mouth overlap.
    """
    final_face_detections = []
    mask_detections = extract_boxes(mask_results)

    for mask_det in mask_detections:
        face_box = mask_det['box']
        label = model_names.get(mask_det['class'], 'unknown')
        confidence = mask_det['confidence']

        # check hand overlap
        is_hand_on_mouth = False
        for hand_box in hand_boxes:
            if bbox_overlap(face_box, hand_box) > hand_on_mouth_threshold:
                is_hand_on_mouth = True
                break
        
        if is_hand_on_mouth:
            label = "Hand on Mouth"

        final_face_detections.append({
            'box': face_box,
            'label': label,
            'confidence': confidence
        })
    
    return final_face_detections