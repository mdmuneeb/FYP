import cv2
import numpy as np

def pad_and_resize(img, size=224):

    h, w = img.shape[:2]
    m = max(h, w)

    canvas = np.zeros((m, m, 3), dtype=np.uint8)

    y_offset = (m - h) // 2
    x_offset = (m - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LANCZOS4)

    return canvas