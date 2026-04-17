import numpy as np
import joblib
from ..config import OSR


class OpenSetRecognizer:

    def __init__(self, threshold=0.85):

        print("Loading centroids from:", OSR)

        self.centroids = joblib.load(OSR)
        self.threshold = threshold

    def is_unknown(self, feature):

        feature = feature / np.linalg.norm(feature)

        distances = []

        for c in self.centroids.values():
            c = c / np.linalg.norm(c)
            dist = np.linalg.norm(feature - c)
            distances.append(dist)

        distances = sorted(distances)

        min_dist = distances[0]
        margin = distances[1] - distances[0]

        print("Min:", min_dist, "Margin:", margin)

        # 🔥 FINAL RULE

        # far → unknown
        if min_dist > 0.9:
            return True

        if margin < 0.1:
            return True


        return False