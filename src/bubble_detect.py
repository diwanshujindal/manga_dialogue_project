import cv2
import numpy as np

def detect_text_regions(img):
    """
    A heuristic detector to reinforce OCR boxes by finding likely text regions.
    Works well on high-contrast manga pages.
    Returns a binary mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize contrast
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # Blackhat to highlight dark text on light background
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, rect_kernel)

    # Binary + morphology to connect text lines
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connect = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)), iterations=1)

    # Remove small noise
    contours, _ = cv2.findContours(connect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h > 100 and h < 0.25*gray.shape[0]:  # basic filters
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    return mask
