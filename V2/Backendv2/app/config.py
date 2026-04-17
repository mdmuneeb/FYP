from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL = BASE_DIR / "models/yolo/best.pt"

EFFNET_MODEL = BASE_DIR / "models/cnn/efficientnetv2s_best_finetuned_best.pth"
MOBILENET_MODEL = BASE_DIR / "models/cnn/mobilenet_v2_finetuned_best.pth"
RESNET_MODEL = BASE_DIR / "models/cnn/resnet18_best_finetuned.pth"
OSR = BASE_DIR / "models/osr/centroids.pkl"

XGB_MODEL = BASE_DIR / "models/xgboost/xgboost_model.pkl"

IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.6

CLASS_NAMES = ["1509", "IRRI-6", "Super White"]