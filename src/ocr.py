from typing import Tuple, List, Dict
import cv2
import pytesseract
import numpy as np

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Slight blur to reduce noise, then adaptive threshold for mixed backgrounds
    blur = cv2.medianBlur(gray, 3)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return th

def ocr_with_boxes(img, lang: str = "eng") -> Tuple[str, List[Dict]]:
    """
    Returns recognized text and a list of bounding boxes.
    Each box: {left, top, width, height, conf, text}
    """
    prep = preprocess_for_ocr(img)
    data = pytesseract.image_to_data(prep, lang=lang, output_type=pytesseract.Output.DICT)
    n = len(data["text"])
    boxes = []
    lines = []
    for i in range(n):
        txt = data["text"][i].strip()
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        if txt and conf > 40:  # filter low-confidence noise
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append({"left": x, "top": y, "width": w, "height": h, "conf": conf, "text": txt})
            lines.append(txt)
    full_text = " ".join(lines)
    return full_text, boxes

def boxes_to_mask(img, boxes, dilation: int = 4):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for b in boxes:
        x, y, w, h = b["left"], b["top"], b["width"], b["height"]
        if w * h <= 0:
            continue
        # Expand slightly
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        x1 = min(img.shape[1], x + w + 1)
        y1 = min(img.shape[0], y + h + 1)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
