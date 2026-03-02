import logging
from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

LOGGER = logging.getLogger(__name__)


@dataclass
class GrammarScoreResult:
    score: float
    confidence: float
    label: str


class GrammarScorer:
    """Grammar scoring using CoLA acceptability classifier."""

    def __init__(self, model_name: str = "textattack/bert-base-uncased-CoLA") -> None:
        self.model_name = model_name
        LOGGER.info("Loading grammar model %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=device,
            truncation=True,
            max_length=256,
        )

    def score_text(self, text: str) -> GrammarScoreResult:
        if not text or not text.strip():
            raise ValueError("No text provided for grammar scoring.")

        outputs = self.pipeline(text)[0]
        prob_map: Dict[str, float] = {entry["label"].upper(): float(entry["score"]) for entry in outputs}
        acceptable = max(prob_map.get("LABEL_1", 0.0), prob_map.get("ACCEPTABLE", 0.0))
        unacceptable = max(prob_map.get("LABEL_0", 0.0), prob_map.get("UNACCEPTABLE", 0.0))

        confidence = max(acceptable, unacceptable)
        label = "acceptable" if acceptable >= unacceptable else "unacceptable"

        score = round(acceptable * 5.0, 2)
        return GrammarScoreResult(score=score, confidence=round(confidence, 4), label=label)
