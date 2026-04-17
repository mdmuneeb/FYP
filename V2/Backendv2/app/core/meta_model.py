import joblib
import numpy as np
from ..config import XGB_MODEL, CLASS_NAMES


class MetaClassifier:

    def __init__(self):
        self.model = joblib.load(XGB_MODEL)

    def predict(self, features):

        probs = self.model.predict_proba([features])[0]
        class_id = probs.argmax()
        max_prob = probs.max()

        return CLASS_NAMES[class_id], max_prob