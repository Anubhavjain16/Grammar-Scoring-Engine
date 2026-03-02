from typing import Any, Dict

from .asr import SpeechRecognition
from .grammar import GrammarScorer


class CommunicationAnalyzer:
    def __init__(self, speech_recognizer: SpeechRecognition, grammar_scorer: GrammarScorer) -> None:
        self.speech_recognizer = speech_recognizer
        self.grammar_scorer = grammar_scorer

    def analyze_audio(self, audio_bytes: bytes) -> Dict[str, Any]:
        transcribed = self.speech_recognizer.transcribe_bytes(audio_bytes)
        grammar = self.grammar_scorer.score_text(transcribed.transcript)
        return {
            "transcript": transcribed.transcript,
            "language": transcribed.language,
            "asr_avg_logprob": transcribed.avg_logprob,
            "asr_no_speech_prob": transcribed.no_speech_prob,
            "grammar_score": grammar.score,
            "grammar_confidence": grammar.confidence,
            "grammar_label": grammar.label,
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        grammar = self.grammar_scorer.score_text(text)
        return {
            "transcript": text,
            "grammar_score": grammar.score,
            "grammar_confidence": grammar.confidence,
            "grammar_label": grammar.label,
        }
