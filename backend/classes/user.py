"""User: manages candidate upload checks and human-in-the-loop assessor validation (FR7)."""
import os
import uuid
from datetime import datetime

import pandas as pd


class User:
    """Represents a candidate or human assessor interacting with the interview bot."""

    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    VALIDATION_COLUMNS = [
        "validation_id",
        "attempt_id",
        "timestamp",
        "assessor_name",
        "agreement",
        "difference_notes",
    ]

    def __init__(self, validation_csv_path, role="candidate"):
        self.validation_csv_path = validation_csv_path
        self.role = role if role in ("candidate", "assessor") else "candidate"

    def is_allowed_video(self, filename):
        """Check whether the uploaded filename has a supported video extension."""
        if not filename or "." not in filename:
            return False
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.ALLOWED_VIDEO_EXTENSIONS

    def submit_validation(self, attempt_id, assessor_name, agreement, difference_notes=""):
        """Record a human assessor's Yes/No agreement and difference notes to validation.csv."""
        normalized_agreement = str(agreement).strip().capitalize()
        if normalized_agreement not in ("Yes", "No"):
            raise ValueError("Agreement must be 'Yes' or 'No'.")

        os.makedirs(os.path.dirname(self.validation_csv_path), exist_ok=True)
        record = {
            "validation_id": uuid.uuid4().hex[:8],
            "attempt_id": str(attempt_id).strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assessor_name": (assessor_name or "Assessor").strip(),
            "agreement": normalized_agreement,
            "difference_notes": (difference_notes or "").strip(),
        }
        df_new = pd.DataFrame([record], columns=self.VALIDATION_COLUMNS)
        file_exists = (
            os.path.isfile(self.validation_csv_path)
            and os.path.getsize(self.validation_csv_path) > 0
        )
        df_new.to_csv(self.validation_csv_path, mode="a", header=not file_exists, index=False)
        return record

    def get_validations(self, limit=50):
        """Return recorded assessor validations (newest first)."""
        if not os.path.isfile(self.validation_csv_path) or os.path.getsize(self.validation_csv_path) == 0:
            return []
        df = pd.read_csv(self.validation_csv_path)
        return df.iloc[::-1].head(limit).fillna("").to_dict(orient="records")

    def get_agreement_stats(self):
        """Compute total reviews and AI-human agreement percentage for viva reporting."""
        if not os.path.isfile(self.validation_csv_path) or os.path.getsize(self.validation_csv_path) == 0:
            return {"total_reviews": 0, "yes_count": 0, "no_count": 0, "agreement_percent": 0.0}

        df = pd.read_csv(self.validation_csv_path)
        total = len(df)
        yes_count = int((df["agreement"] == "Yes").sum())
        no_count = total - yes_count
        pct = round((yes_count / total) * 100, 2) if total > 0 else 0.0
        return {
            "total_reviews": total,
            "yes_count": yes_count,
            "no_count": no_count,
            "agreement_percent": pct,
        }
