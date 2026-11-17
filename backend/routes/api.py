"""API routes connecting the frontend to the 6 core backend classes."""
import os
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from classes.audio_processor import AudioProcessor
from classes.emotion_detector import EmotionDetector
from classes.feedback_generator import FeedbackGenerator
from classes.semantic_analyzer import SemanticAnalyzer
from classes.user import User
from classes.video_processor import VideoProcessor

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Loaded once at startup so the CNN model is not reloaded on every request
_services = {}


def init_services(base_dir):
    """Initialize ML model, NLP analyzer, CSV storage, and user handlers once."""
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    _services["emotion_detector"] = EmotionDetector(
        model_path=os.path.join(models_dir, "best_emotion_model.keras"),
        class_names_path=os.path.join(models_dir, "class_names.json"),
    )
    _services["audio_processor"] = AudioProcessor()
    _services["semantic_analyzer"] = SemanticAnalyzer(
        qa_csv_path=os.path.join(data_dir, "hr_qa.csv")
    )
    _services["feedback_generator"] = FeedbackGenerator(
        history_csv_path=os.path.join(data_dir, "history.csv")
    )
    _services["user"] = User(
        validation_csv_path=os.path.join(data_dir, "validation.csv")
    )


@api_bp.route("/questions", methods=["GET"])
def get_questions():
    """Return available HR interview questions."""
    questions = _services["semantic_analyzer"].get_questions()
    return jsonify({"questions": questions})


@api_bp.route("/analyze", methods=["POST"])
def analyze_interview():
    """Run the full pipeline on one uploaded mock interview video."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video_file = request.files["video"]
    question_id = request.form.get("question_id")
    candidate_name = request.form.get("candidate_name", "Candidate")

    if not question_id:
        return jsonify({"error": "Please select an HR interview question."}), 400

    user = _services["user"]
    if not user.is_allowed_video(video_file.filename):
        return jsonify({"error": "Unsupported video format. Use MP4, AVI, MOV, MKV, or WEBM."}), 400

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(video_file.filename)}"
    video_path = os.path.join(upload_dir, safe_name)
    video_file.save(video_path)

    vp = None
    try:
        vp = VideoProcessor(video_path, frames_per_second=1)
        frames = vp.extract_frames()
        audio_path = vp.extract_audio()

        emotion_result = _services["emotion_detector"].get_percentages(frames)
        speech_result = _services["audio_processor"].transcribe(audio_path)
        semantic_result = _services["semantic_analyzer"].evaluate_answer(
            question_id=question_id,
            candidate_transcript=speech_result["transcript"],
        )

        fg = _services["feedback_generator"]
        report = fg.generate_report(
            candidate_name=candidate_name,
            emotion_result=emotion_result,
            semantic_result=semantic_result,
            frames_sampled=vp.frames_sampled,
            faces_found=vp.faces_found,
        )
        fg.save_to_history(report)
        report["speech_warning"] = speech_result.get("error")
        return jsonify({"report": report}), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    finally:
        # Privacy: always remove uploaded video and extracted WAV after processing
        if vp is not None:
            vp.cleanup()
        elif os.path.isfile(video_path):
            os.remove(video_path)


@api_bp.route("/history", methods=["GET"])
def get_history():
    """Return past interview assessment attempts."""
    history = _services["feedback_generator"].get_history()
    return jsonify({"history": history})


@api_bp.route("/validate", methods=["POST"])
def submit_validation():
    """Record human assessor Yes/No agreement for an attempt (FR7)."""
    payload = request.get_json(silent=True) or {}
    attempt_id = payload.get("attempt_id", "").strip()
    assessor_name = payload.get("assessor_name", "Assessor").strip()
    agreement = payload.get("agreement", "").strip()
    difference_notes = payload.get("difference_notes", "").strip()

    if not attempt_id:
        return jsonify({"error": "Attempt ID is required."}), 400

    try:
        user = _services["user"]
        record = user.submit_validation(attempt_id, assessor_name, agreement, difference_notes)
        stats = user.get_agreement_stats()
        return jsonify({"validation": record, "stats": stats}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@api_bp.route("/validations", methods=["GET"])
def get_validations():
    """Return all human assessor validations and overall AI-human agreement stats."""
    user = _services["user"]
    return jsonify({
        "validations": user.get_validations(),
        "stats": user.get_agreement_stats(),
    })
