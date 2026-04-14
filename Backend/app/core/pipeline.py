import os
import cv2
import shutil

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

        if image is None:
            raise ValueError("Pipeline received None image")

        print("📸 Pipeline image shape:", image.shape)

        crops = self.detector.detect_and_crop(image)
        boxes = self.detector.last_boxes

        results = []

        # Ensure base folder exists
        base_folder = "grain_crops"
        os.makedirs(base_folder, exist_ok=True)

        # Use single results folder (overwrite mode)
        folder_name = os.path.join(base_folder, "results")

        # Clear old results
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)

        os.makedirs(folder_name, exist_ok=True)

        # Feature log file
        feature_file_path = os.path.join(folder_name, "features.txt")

        with open(feature_file_path, "w") as f:

            for i, crop in enumerate(crops):

                features = self.cnn.predict(crop)
                label, prob = self.meta.predict(features)

                # Save crop image
                filename = f"grain_{i}_{label}.jpg"
                save_path = os.path.join(folder_name, filename)

                success = cv2.imwrite(save_path, crop)
                if not success:
                    print(f"⚠️ Failed to save {save_path}")

                # ✅ IMPORTANT: return RELATIVE path (DOCKER SAFE)
                image_url = f"/grain_crops/results/{filename}"

                # Save features
                feature_line = (
                    f"grain_{i}, class={label}, "
                    f"conf={prob:.4f}, features={features.tolist()}\n"
                )
                f.write(feature_line)

                results.append({
                    "image": image_url,
                    "class": label,
                    "confidence": float(prob)
                })

        # Debug visualization
        draw_predictions(
            image,
            boxes,
            results,
            os.path.join(folder_name, "debug_prediction.jpg")
        )

        return results