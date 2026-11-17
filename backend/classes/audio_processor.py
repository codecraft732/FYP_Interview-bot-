"""AudioProcessor: converts extracted interview audio to text using Google Speech-to-Text (FR4)."""
import os
import wave

import speech_recognition as sr


class AudioProcessor:
    """Transcribes a WAV audio file into written text via Google Speech-to-Text API."""

    def __init__(self, language="en-US", chunk_seconds=30):
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.recognizer = sr.Recognizer()

    def _get_duration_seconds(self, audio_path):
        """Return WAV duration in seconds using built-in wave module."""
        with wave.open(audio_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate) if rate > 0 else 0.0

    def transcribe(self, audio_path):
        """Transcribe WAV file in chunks to avoid Google free endpoint length limits."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        total_seconds = self._get_duration_seconds(audio_path)
        if total_seconds <= 0:
            return {"transcript": "", "chunks_processed": 0, "error": "Audio file is empty."}

        transcript_parts = []
        chunks_processed = 0
        offset = 0.0

        with sr.AudioFile(audio_path) as source:
            while offset < total_seconds:
                audio_data = self.recognizer.record(source, duration=self.chunk_seconds)
                chunks_processed += 1
                try:
                    text = self.recognizer.recognize_google(audio_data, language=self.language)
                    if text.strip():
                        transcript_parts.append(text.strip())
                except sr.UnknownValueError:
                    # Silence or unclear speech in this chunk; continue to next chunk
                    pass
                except sr.RequestError as exc:
                    raise RuntimeError(
                        "Google Speech-to-Text API unreachable. Check your internet connection."
                    ) from exc

                offset += self.chunk_seconds

        return {
            "transcript": " ".join(transcript_parts),
            "chunks_processed": chunks_processed,
            "error": None if transcript_parts else "No clear speech detected in audio.",
        }
