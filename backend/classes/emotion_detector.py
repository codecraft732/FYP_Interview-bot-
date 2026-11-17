"""EmotionDetector: classifies extracted face frames into 7 emotions (FR3)."""
import json
import os

import keras
import numpy as np


class EmotionDetector:
    """Loads the trained CNN model and computes per-emotion percentage breakdowns."""

    def __init__(self, model_path, class_names_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.isfile(class_names_path):
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")

        self.model = keras.models.load_model(model_path)
        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

    def predict_frames(self, frames):
        """Return the top predicted emotion label for each face frame."""
        if frames is None or len(frames) == 0:
            return []

        probabilities = self.model.predict(frames, verbose=0)
        top_indices = np.argmax(probabilities, axis=1)
        return [self.class_names[int(idx)] for idx in top_indices]

    def get_percentages(self, frames):
        """Count top-emotion votes per frame and return percentage breakdown."""
        percentages = {name: 0.0 for name in self.class_names}
        predictions = self.predict_frames(frames)
        total = len(predictions)

        if total == 0:
            return {
                "percentages": percentages,
                "total_frames": 0,
                "dominant_emotion": None,
            }

        for label in predictions:
            percentages[label] += 1

        for name in percentages:
            percentages[name] = round((percentages[name] / total) * 100, 2)

        dominant = max(percentages, key=percentages.get)
        return {
            "percentages": percentages,
            "total_frames": total,
            "dominant_emotion": dominant,
        }
