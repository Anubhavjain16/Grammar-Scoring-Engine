from dataclasses import dataclass

from fastapi.testclient import TestClient

from backend.dependencies import get_analyzer, get_grammar_scorer, get_speech_recognizer
from backend.main import app


@dataclass
class DummyTranscription:
    transcript: str = "hello world"
    language: str = "en"
    avg_logprob: float = -0.3
    no_speech_prob: float = 0.02


@dataclass
class DummyGrammar:
    score: float = 4.5
    confidence: float = 0.91
    label: str = "acceptable"


class DummySpeechRecognizer:
    def transcribe_bytes(self, audio_bytes: bytes):
        assert isinstance(audio_bytes, bytes)
        return DummyTranscription()


class DummyGrammarScorer:
    def score_text(self, text: str):
        assert text
        return DummyGrammar()


class DummyAnalyzer:
    def analyze_audio(self, audio_bytes: bytes):
        return {
            "transcript": "audio transcript",
            "grammar_score": 4.2,
            "grammar_confidence": 0.88,
            "grammar_label": "acceptable",
        }

    def analyze_text(self, text: str):
        return {
            "transcript": text,
            "grammar_score": 3.9,
            "grammar_confidence": 0.8,
            "grammar_label": "acceptable",
        }


app.dependency_overrides[get_speech_recognizer] = lambda: DummySpeechRecognizer()
app.dependency_overrides[get_grammar_scorer] = lambda: DummyGrammarScorer()
app.dependency_overrides[get_analyzer] = lambda: DummyAnalyzer()

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_endpoint():
    resp = client.post("/score", json={"text": "She goes to school every day."})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["grammar_score"] == 4.5
    assert payload["confidence"] == 0.91


def test_transcribe_endpoint():
    resp = client.post(
        "/transcribe",
        files={"audio": ("sample.wav", b"RIFF....WAVE", "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["transcript"] == "hello world"


def test_analyze_text_endpoint():
    resp = client.post("/analyze", data={"text": "this is a test"})
    assert resp.status_code == 200
    assert resp.json()["grammar_score"] == 3.9


def test_analyze_missing_payload():
    resp = client.post("/analyze")
    assert resp.status_code == 422
