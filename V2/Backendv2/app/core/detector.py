from ultralytics import YOLO
import numpy as np
from ..config import YOLO_MODEL
from ..utils.image_utils import pad_and_resize


class GrainDetector:

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.last_boxes = []

    def detect_and_crop(self, image):

        # results = self.model.predict(image, conf=0.5)
        results = self.model.predict(source=image, conf=0.5)

        crops = []
        boxes_all = []

        for r in results:

            img = r.orig_img  # BGR image (OpenCV)

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)

            for x1, y1, x2, y2 in boxes:

                crop = img[y1:y2, x1:x2]

                # skip invalid crops
                if crop.size == 0:
                    continue

                # skip very small crops
                if crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue

                crop = pad_and_resize(crop)

                crops.append(crop)

                boxes_all.append([x1, y1, x2, y2])

        self.last_boxes = boxes_all

        return crops