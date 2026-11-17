"""SemanticAnalyzer: compares transcribed candidate answers with HR reference answers (FR5)."""
import os

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticAnalyzer:
    """Preprocesses text with NLTK and scores relevance using TF-IDF + Cosine Similarity."""

    def __init__(self, qa_csv_path):
        if not os.path.isfile(qa_csv_path):
            raise FileNotFoundError(f"HR Q&A CSV not found: {qa_csv_path}")

        self.qa_df = pd.read_csv(qa_csv_path)
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def get_questions(self):
        """Return list of available HR questions as dicts with id and question."""
        return self.qa_df[["id", "question"]].to_dict(orient="records")

    def get_reference_answer(self, question_id):
        """Return (question, reference_answer) for the given question ID."""
        row = self.qa_df[self.qa_df["id"] == int(question_id)]
        if row.empty:
            raise ValueError(f"Question ID not found: {question_id}")
        return str(row.iloc[0]["question"]), str(row.iloc[0]["reference_answer"])

    def preprocess(self, text):
        """Lowercase, tokenize, remove punctuation/stopwords, and lemmatize."""
        if not text or not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        cleaned = [
            self._lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok.isalpha() and tok not in self._stop_words
        ]
        return " ".join(cleaned)

    def compute_similarity(self, candidate_text, reference_text):
        """Return TF-IDF Cosine Similarity as a 0-100 percentage."""
        clean_ref = self.preprocess(reference_text)
        clean_cand = self.preprocess(candidate_text)
        if not clean_ref or not clean_cand:
            return 0.0

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([clean_ref, clean_cand])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score) * 100, 2)

    def evaluate_answer(self, question_id, candidate_transcript):
        """Compare candidate transcript against the HR reference for question_id."""
        question, reference = self.get_reference_answer(question_id)
        score = self.compute_similarity(candidate_transcript, reference)

        if score >= 60.0:
            label = "High Relevance"
        elif score >= 30.0:
            label = "Moderate Relevance"
        else:
            label = "Low Relevance"

        return {
            "question_id": int(question_id),
            "question": question,
            "reference_answer": reference,
            "candidate_transcript": candidate_transcript,
            "similarity_score": score,
            "relevance_label": label,
        }
