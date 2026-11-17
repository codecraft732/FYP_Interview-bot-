"""VideoProcessor: extracts face frames and audio from an uploaded interview video (FR1, FR2)."""
import os
import subprocess

import cv2
import numpy as np


class VideoProcessor:
    """Turns one uploaded video into (a) 48x48 grayscale face frames and (b) a WAV audio file."""

    FACE_SIZE = 48  # the trained model expects 48x48 grayscale images

    def __init__(self, video_path, frames_per_second=1):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.frames_per_second = frames_per_second
        self.audio_path = os.path.splitext(video_path)[0] + "_audio.wav"

        # Haar cascade = a small face detector that ships with OpenCV
        cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._face_detector = cv2.CascadeClassifier(cascade_file)
        if self._face_detector.empty():
            raise RuntimeError("Could not load the OpenCV face detector.")

        # Simple counters so the report can be transparent about poor lighting / angles
        self.frames_sampled = 0
        self.faces_found = 0

    def extract_frames(self):
        """Return face crops as a float32 array of shape (N, 48, 48, 1), pixel values 0-255.

        Values stay 0-255 on purpose: the model already divides by 255 inside itself.
        """
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise ValueError("Could not open the video file.")

        video_fps = capture.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps != video_fps or video_fps <= 0:  # 0 or NaN -> assume 25
            video_fps = 25.0
        step = max(1, int(round(video_fps / self.frames_per_second)))  # keep every step-th frame

        faces = []
        self.frames_sampled = 0
        self.faces_found = 0
        index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break
            if index % step == 0:
                self.frames_sampled += 1
                face = self._crop_face(frame)
                if face is not None:
                    faces.append(face)
                    self.faces_found += 1
            index += 1

        capture.release()

        if not faces:
            return np.empty((0, self.FACE_SIZE, self.FACE_SIZE, 1), dtype="float32")
        return np.array(faces, dtype="float32")

    def _crop_face(self, frame):
        """Find the largest face in a frame, crop it and resize to 48x48. Return None if no face."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(detections) == 0:
            return None

        x, y, w, h = max(detections, key=lambda box: box[2] * box[3])  # largest face only
        face = cv2.resize(gray[y:y + h, x:x + w], (self.FACE_SIZE, self.FACE_SIZE))
        return face.reshape(self.FACE_SIZE, self.FACE_SIZE, 1)

    def extract_audio(self):
        """Save the audio track as 16 kHz mono WAV (best for speech recognition). Return its path."""
        command = [
            "ffmpeg", "-y", "-i", self.video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", self.audio_path,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed or not on PATH.")

        if result.returncode != 0 or not os.path.isfile(self.audio_path):
            raise RuntimeError("Audio extraction failed. Does the video have a sound track?")
        return self.audio_path

    def cleanup(self):
        """Delete the uploaded video and extracted audio (privacy requirement)."""
        for path in (self.video_path, self.audio_path):
            if os.path.isfile(path):
                os.remove(path)
