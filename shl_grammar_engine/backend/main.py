import logging
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .analyzer import CommunicationAnalyzer
from .asr import SpeechRecognition
from .dependencies import get_analyzer, get_grammar_scorer, get_speech_recognizer
from .grammar import GrammarScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("shl_grammar_engine")

app = FastAPI(title="SHL Grammar Engine", version="1.1.0")


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    speech_recognizer: SpeechRecognition = Depends(get_speech_recognizer),
) -> dict:
    try:
        content = await audio.read()
        result = speech_recognizer.transcribe_bytes(content)
        return {
            "transcript": result.transcript,
            "language": result.language,
            "asr_avg_logprob": result.avg_logprob,
            "asr_no_speech_prob": result.no_speech_prob,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Transcription failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/score")
def score(
    request: ScoreRequest,
    grammar_scorer: GrammarScorer = Depends(get_grammar_scorer),
) -> dict:
    try:
        result = grammar_scorer.score_text(request.text)
        return {
            "grammar_score": result.score,
            "confidence": result.confidence,
            "label": result.label,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Scoring failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze")
async def analyze(
    audio: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    analyzer: CommunicationAnalyzer = Depends(get_analyzer),
) -> dict:
    try:
        if audio is not None:
            content = await audio.read()
            return analyzer.analyze_audio(content)
        if text:
            return analyzer.analyze_text(text)
        raise HTTPException(status_code=422, detail="Provide either audio file or text")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Analysis failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
