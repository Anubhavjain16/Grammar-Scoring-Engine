import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List

from faster_whisper import WhisperModel

LOGGER = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    transcript: str
    language: str
    avg_logprob: float
    no_speech_prob: float


class SpeechRecognition:
    """Speech-to-text powered by faster-whisper large-v3."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto", compute_type: str = "int8_float16") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        LOGGER.info("Loading Whisper model size=%s device=%s compute_type=%s", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root="models")

    def transcribe_bytes(self, audio_bytes: bytes, language: str = "en") -> TranscriptionResult:
        if not audio_bytes:
            raise ValueError("Empty audio payload received.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            segments, info = self.model.transcribe(
                temp_path,
                language=language,
                vad_filter=True,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text_segments: List[str] = []
            avg_logprobs: List[float] = []
            no_speech_probs: List[float] = []

            for segment in segments:
                text_segments.append(segment.text.strip())
                avg_logprobs.append(segment.avg_logprob)
                no_speech_probs.append(segment.no_speech_prob)

            transcript = " ".join(x for x in text_segments if x).strip()
            avg_logprob = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else -10.0
            no_speech_prob = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 1.0

            return TranscriptionResult(
                transcript=transcript,
                language=info.language if info else language,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                LOGGER.warning("Failed to delete temp audio file %s", temp_path)
