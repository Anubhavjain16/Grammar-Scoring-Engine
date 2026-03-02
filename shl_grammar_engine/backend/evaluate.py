import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset
from jiwer import wer

from backend.asr import SpeechRecognition
from backend.grammar import GrammarScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("evaluation")
CANONICAL_PROMPT = (
    "Please call Stella. Ask her to bring these things with her from the store: "
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. "
    "We also need a small plastic snake and a big toy frog for the kids. "
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
)


def evaluate_grammar_on_cola(grammar_scorer: GrammarScorer, sample_size: int = 500) -> float:
    dataset = load_dataset("glue", "cola", split=f"validation[:{sample_size}]")
    correct = 0
    for row in dataset:
        pred = grammar_scorer.score_text(row["sentence"])
        predicted_label = 1 if pred.label == "acceptable" else 0
        if predicted_label == int(row["label"]):
            correct += 1
    return correct / len(dataset)


def evaluate_accent_archive_wer(asr: SpeechRecognition, dataset_root: Path, max_samples: int = 50) -> float:
    csv_path = dataset_root / "speakers_all.csv"
    recordings_dir = dataset_root / "recordings"

    if not csv_path.exists() or not recordings_dir.exists():
        raise FileNotFoundError("Expected Speech Accent Archive files: speakers_all.csv and recordings/")

    meta = pd.read_csv(csv_path)
    refs: List[str] = []
    hyps: List[str] = []

    for _, row in meta.head(max_samples).iterrows():
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        wav_path = recordings_dir / f"{filename}.wav"
        if not wav_path.exists():
            continue
        with open(wav_path, "rb") as f:
            transcript = asr.transcribe_bytes(f.read()).transcript
        refs.append(CANONICAL_PROMPT)
        hyps.append(transcript)

    if not refs:
        raise RuntimeError("No audio files found for WER computation")

    return wer(refs, hyps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR + grammar models")
    parser.add_argument("--accent-root", type=str, default="", help="Path to extracted Speech Accent Archive dataset")
    parser.add_argument("--max-audio", type=int, default=20)
    parser.add_argument("--cola-samples", type=int, default=500)
    args = parser.parse_args()

    asr = SpeechRecognition()
    grammar = GrammarScorer()

    results = {
        "grammar_accuracy_cola": evaluate_grammar_on_cola(grammar, sample_size=args.cola_samples),
    }

    if args.accent_root:
        results["wer_accent_archive"] = evaluate_accent_archive_wer(asr, Path(args.accent_root), max_samples=args.max_audio)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
