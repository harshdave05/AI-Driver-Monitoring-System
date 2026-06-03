# utils.py


def bbox_iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def bbox_overlap(boxA, boxB):
    """Return overlap ratio relative to the area of the first box (A ∩ B) / area(A)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    if areaA == 0:
        return 0.0
    return interArea / areaA


def extract_boxes(res):
    """Helper to extract boxes from YOLO results."""
    items = []
    try:
        for r in res:
            for box in r.boxes:
                items.append({
                    'box': tuple(map(int, box.xyxy[0].tolist())),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
    except Exception:
        return []
    return items