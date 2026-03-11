import os
import uuid
import cv2

from .detector import GrainDetector
from .classifier import CNNEnsemble
from .meta_model import MetaClassifier
from .debug_visualizer import draw_predictions


class RicePipeline:

    def __init__(self):

        self.detector = GrainDetector()
        self.cnn = CNNEnsemble()
        self.meta = MetaClassifier()

    def predict(self, image):

        crops = self.detector.detect_and_crop(image)

        boxes = self.detector.last_boxes

        results = []

        # Create folder for this request
        folder_name = f"grain_crops/{uuid.uuid4().hex[:8]}"
        os.makedirs(folder_name, exist_ok=True)

        # Create feature log file
        feature_file_path = os.path.join(folder_name, "features.txt")

        with open(feature_file_path, "w") as f:

            for i, crop in enumerate(crops):

                features = self.cnn.predict(crop)

                label, prob = self.meta.predict(features)

                # Save crop image
                save_path = os.path.join(folder_name, f"grain_{i}_{label}.jpg")
                cv2.imwrite(save_path, crop)

                # Save features to text file
                feature_line = f"grain_{i}, class={label}, conf={prob:.4f}, features={features.tolist()}\n"
                f.write(feature_line)

                results.append({
                    "class": label,
                    "confidence": float(prob)
                })

        # Draw debug visualization
        draw_predictions(
            image,
            boxes,
            results,
            os.path.join(folder_name, "debug_prediction.jpg")
        )

        return results