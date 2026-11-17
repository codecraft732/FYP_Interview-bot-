"""Automated tests for core classes, CSV storage, and Flask API endpoints."""
import os
import sys
import tempfile
import unittest

import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import create_app
from classes.emotion_detector import EmotionDetector
from classes.feedback_generator import FeedbackGenerator
from classes.semantic_analyzer import SemanticAnalyzer
from classes.user import User


class TestSemanticAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SemanticAnalyzer(os.path.join(BACKEND_DIR, "data", "hr_qa.csv"))

    def test_questions_loaded(self):
        questions = self.analyzer.get_questions()
        self.assertEqual(len(questions), 15)

    def test_similarity_scores(self):
        res_match = self.analyzer.evaluate_answer(
            1,
            "I am a final year software engineering student interested in computer science, Python, web technologies and teamwork.",
        )
        res_empty = self.analyzer.evaluate_answer(1, "")
        self.assertGreater(res_match["similarity_score"], 25.0)
        self.assertEqual(res_empty["similarity_score"], 0.0)
        self.assertEqual(res_empty["relevance_label"], "Low Relevance")

    def test_invalid_question_raises(self):
        with self.assertRaises(ValueError):
            self.analyzer.get_reference_answer(999)


class TestFeedbackAndUser(unittest.TestCase):
    def test_feedback_history_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fg = FeedbackGenerator(os.path.join(tmpdir, "history.csv"))
            emo = {
                "percentages": {"angry": 0.0, "disgust": 0.0, "fear": 0.0, "happy": 60.0, "neutral": 40.0, "sad": 0.0, "surprise": 0.0},
                "total_frames": 5,
                "dominant_emotion": "happy",
            }
            sem = {
                "question_id": 1,
                "question": "Tell me about yourself.",
                "reference_answer": "ref",
                "candidate_transcript": "I am a student.",
                "similarity_score": 50.0,
                "relevance_label": "Moderate Relevance",
            }
            report = fg.generate_report("Test Candidate", emo, sem, frames_sampled=5, faces_found=5)
            fg.save_to_history(report)
            history = fg.get_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["dominant_emotion"], "happy")
            self.assertEqual(history[0]["similarity_score"], 50.0)

    def test_user_validation_and_agreement_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user = User(os.path.join(tmpdir, "validation.csv"), role="assessor")
            self.assertTrue(user.is_allowed_video("interview.mp4"))
            self.assertFalse(user.is_allowed_video("notes.txt"))

            user.submit_validation("a1", "Assessor 1", "Yes", "")
            user.submit_validation("a2", "Assessor 1", "No", "Synonym mismatch")
            stats = user.get_agreement_stats()
            self.assertEqual(stats["total_reviews"], 2)
            self.assertEqual(stats["agreement_percent"], 50.0)

            with self.assertRaises(ValueError):
                user.submit_validation("a3", "Assessor 1", "Maybe", "")


class TestEmotionDetectorAndApp(unittest.TestCase):
    def test_emotion_detector_handles_empty_and_batch(self):
        ed = EmotionDetector(
            os.path.join(BACKEND_DIR, "models", "best_emotion_model.keras"),
            os.path.join(BACKEND_DIR, "models", "class_names.json"),
        )
        empty_res = ed.get_percentages(np.empty((0, 48, 48, 1), dtype="float32"))
        self.assertEqual(empty_res["total_frames"], 0)
        self.assertIsNone(empty_res["dominant_emotion"])

        sample_batch = np.zeros((2, 48, 48, 1), dtype="float32")
        batch_res = ed.get_percentages(sample_batch)
        self.assertEqual(batch_res["total_frames"], 2)
        self.assertAlmostEqual(sum(batch_res["percentages"].values()), 100.0, places=1)

    def test_flask_endpoints(self):
        app = create_app()
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        q_res = client.get("/api/questions")
        self.assertEqual(q_res.status_code, 200)
        self.assertEqual(len(q_res.get_json()["questions"]), 15)


if __name__ == "__main__":
    unittest.main()
