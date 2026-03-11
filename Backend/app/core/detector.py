# from ultralytics import YOLO
# import numpy as np
# from ..config import YOLO_MODEL
# from ..utils.image_utils import pad_and_resize

# class GrainDetector:

#     def __init__(self):
#         self.model = YOLO(YOLO_MODEL)

#     def detect_and_crop(self, image):

#         results = self.model.predict(image, conf=0.25)

#         crops = []

#         for r in results:

#             img = r.orig_img
#             boxes = r.boxes.xyxy.cpu().numpy()

#             for box in boxes:

#                 x1, y1, x2, y2 = map(int, box)

#                 crop = img[y1:y2, x1:x2]

#                 if crop.size == 0:
#                     continue

#                 crop = pad_and_resize(crop)

#                 crops.append(crop)
            


#         return crops


from ultralytics import YOLO
import numpy as np
from ..config import YOLO_MODEL
from ..utils.image_utils import pad_and_resize


class GrainDetector:

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.last_boxes = []   # store boxes for debug

    def detect_and_crop(self, image):

        results = self.model.predict(image, conf=0.25)

        crops = []
        boxes_all = []

        for r in results:

            img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)

            for x1, y1, x2, y2 in boxes:

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop = pad_and_resize(crop)

                crops.append(crop)

                # save bounding box
                boxes_all.append([x1, y1, x2, y2])

        # store boxes for pipeline debug
        self.last_boxes = boxes_all

        return crops