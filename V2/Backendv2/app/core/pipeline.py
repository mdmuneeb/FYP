import os
import cv2
import shutil

from .detector import GrainDetector
from .classifier import CNNEnsemble
from .meta_model import MetaClassifier
from .osr import OpenSetRecognizer
from .debug_visualizer import draw_predictions


class RicePipeline:

    def __init__(self):
        self.detector = GrainDetector()
        self.cnn = CNNEnsemble()
        self.meta = MetaClassifier()
        self.osr = OpenSetRecognizer()

    def predict(self, image):

        crops = self.detector.detect_and_crop(image)
        boxes = self.detector.last_boxes

        results = []

        folder = "grain_crops/results"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for i, crop in enumerate(crops):

            cnn_out = self.cnn.predict(crop)

            deep_feat = cnn_out["features"]
            softmax_feat = cnn_out["softmax"]

            # 🔥 OSR CHECK
            if self.osr.is_unknown(deep_feat):
                label = "Unknown"
                conf = 0.0
            else:
                label, conf = self.meta.predict(softmax_feat)

            filename = f"grain_{i}_{label}.jpg"
            cv2.imwrite(os.path.join(folder, filename), crop)

            results.append({
                "image": f"/grain_crops/results/{filename}",
                "class": label,
                "confidence": float(conf)
            })

        draw_predictions(
            image,
            boxes,
            results,
            os.path.join(folder, "debug.jpg")
        )

        return results