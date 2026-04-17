import numpy as np
import joblib
from tqdm import tqdm
import os
from PIL import Image

from app.core.classifier import CNNEnsemble
from app.config import OSR   # ✅ save to correct path

# =========================
# DATA PATH
# =========================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)

DATA_DIR = os.path.join(BASE_DIR, "Datasets", "cnn_dataset", "train")

print("DATA_DIR:", DATA_DIR)

# =========================
# LOAD CNN
# =========================
cnn = CNNEnsemble()

centroids = {}

# =========================
# COMPUTE CENTROIDS
# =========================
for cls in os.listdir(DATA_DIR):

    feats = []

    class_path = os.path.join(DATA_DIR, cls)

    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {cls}"):

        path = os.path.join(class_path, img_name)

        img = Image.open(path).convert("RGB")
        img = np.array(img)

        out = cnn.predict(img)

        feat = out["features"]

        # 🔥 IMPORTANT: Normalize feature BEFORE saving
        feat = feat / np.linalg.norm(feat)

        feats.append(feat)

    # mean centroid
    centroids[cls] = np.mean(feats, axis=0)

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(OSR), exist_ok=True)

joblib.dump(centroids, OSR)

print("✅ Centroids saved at:", OSR)