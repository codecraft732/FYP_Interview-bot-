"""FeedbackGenerator: creates separate-metric interview reports and logs history to CSV (FR6)."""
import json
import os
import uuid
from datetime import datetime

import pandas as pd


class FeedbackGenerator:
    """Combines emotion breakdown and semantic similarity without oversimplifying into one score."""

    HISTORY_COLUMNS = [
        "attempt_id",
        "timestamp",
        "candidate_name",
        "question_id",
        "question",
        "dominant_emotion",
        "emotion_percentages",
        "similarity_score",
        "relevance_label",
        "faces_found",
        "frames_sampled",
        "transcript",
    ]

    def __init__(self, history_csv_path):
        self.history_csv_path = history_csv_path

    def _build_emotion_feedback(self, emotion_result, frames_sampled, faces_found):
        """Generate plain-language tips based on facial expression percentages."""
        tips = []
        total = emotion_result.get("total_frames", 0)
        pct = emotion_result.get("percentages", {})
        dominant = emotion_result.get("dominant_emotion")

        if total == 0:
            tips.append(
                "No face was detected in the video frames. Record in brighter front lighting and face the camera."
            )
            return tips

        if frames_sampled > 0 and faces_found < frames_sampled:
            tips.append(
                f"Face detected in {faces_found} of {frames_sampled} sampled frames. Keep your head steady and well-lit."
            )

        confident_pct = pct.get("happy", 0.0) + pct.get("neutral", 0.0)
        nervous_pct = pct.get("fear", 0.0) + pct.get("sad", 0.0) + pct.get("angry", 0.0)

        if confident_pct >= 60.0:
            tips.append(
                f"Good composure: {confident_pct:.1f}% of frames showed calm or positive expressions (Neutral + Happy)."
            )
        if nervous_pct >= 30.0:
            tips.append(
                f"Elevated tension/nervousness signals ({nervous_pct:.1f}% across Fear/Sad/Angry). Practice pacing and smiling naturally."
            )
        if dominant and not tips:
            tips.append(f"Most frequent expression was '{dominant}' ({pct.get(dominant, 0.0):.1f}%).")
        return tips

    def _build_answer_feedback(self, semantic_result):
        """Generate plain-language tips based on TF-IDF semantic similarity."""
        score = semantic_result.get("similarity_score", 0.0)
        transcript = (semantic_result.get("candidate_transcript") or "").strip()

        if not transcript:
            return ["No clear speech was transcribed. Speak clearly near the microphone and check your internet connection."]
        if score >= 60.0:
            return ["Strong answer relevance: your spoken response aligns well with key HR expectations."]
        if score >= 30.0:
            return ["Moderate relevance: you covered some points, but adding clearer examples and core keywords will improve it."]
        return ["Low relevance: your answer differed from the HR reference. Review the reference answer and structure your key points."]

    def generate_report(self, candidate_name, emotion_result, semantic_result, frames_sampled=0, faces_found=0):
        """Build a structured report keeping facial emotions and answer relevance separate."""
        attempt_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "attempt_id": attempt_id,
            "timestamp": timestamp,
            "candidate_name": candidate_name.strip() or "Candidate",
            "question_id": semantic_result["question_id"],
            "question": semantic_result["question"],
            "reference_answer": semantic_result["reference_answer"],
            "transcript": semantic_result["candidate_transcript"],
            "similarity_score": semantic_result["similarity_score"],
            "relevance_label": semantic_result["relevance_label"],
            "dominant_emotion": emotion_result.get("dominant_emotion"),
            "emotion_percentages": emotion_result.get("percentages", {}),
            "total_emotion_frames": emotion_result.get("total_frames", 0),
            "frames_sampled": frames_sampled,
            "faces_found": faces_found,
            "emotion_feedback": self._build_emotion_feedback(emotion_result, frames_sampled, faces_found),
            "answer_feedback": self._build_answer_feedback(semantic_result),
        }

    def save_to_history(self, report):
        """Append one report summary to history.csv."""
        os.makedirs(os.path.dirname(self.history_csv_path), exist_ok=True)
        row = {
            "attempt_id": report["attempt_id"],
            "timestamp": report["timestamp"],
            "candidate_name": report["candidate_name"],
            "question_id": report["question_id"],
            "question": report["question"],
            "dominant_emotion": report["dominant_emotion"] or "none",
            "emotion_percentages": json.dumps(report["emotion_percentages"]),
            "similarity_score": report["similarity_score"],
            "relevance_label": report["relevance_label"],
            "faces_found": report["faces_found"],
            "frames_sampled": report["frames_sampled"],
            "transcript": report["transcript"],
        }
        df_new = pd.DataFrame([row], columns=self.HISTORY_COLUMNS)
        file_exists = os.path.isfile(self.history_csv_path) and os.path.getsize(self.history_csv_path) > 0
        df_new.to_csv(self.history_csv_path, mode="a", header=not file_exists, index=False)

    def get_history(self, limit=20):
        """Return past attempts (newest first) from history.csv."""
        if not os.path.isfile(self.history_csv_path) or os.path.getsize(self.history_csv_path) == 0:
            return []

        df = pd.read_csv(self.history_csv_path)
        records = df.iloc[::-1].head(limit).to_dict(orient="records")
        for rec in records:
            try:
                rec["emotion_percentages"] = json.loads(rec["emotion_percentages"])
            except (TypeError, ValueError):
                rec["emotion_percentages"] = {}
        return records
