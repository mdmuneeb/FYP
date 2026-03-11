import joblib
import numpy as np
from ..config import XGB_MODEL, CLASS_NAMES, UNKNOWN_THRESHOLD


class MetaClassifier:

    def __init__(self):

        self.model = joblib.load(XGB_MODEL)

    def predict(self, features):

        probs = self.model.predict_proba([features])[0]

        max_prob = probs.max()
        class_id = probs.argmax()

        if max_prob < UNKNOWN_THRESHOLD:
            return "Unknown", max_prob

        return CLASS_NAMES[class_id], max_prob