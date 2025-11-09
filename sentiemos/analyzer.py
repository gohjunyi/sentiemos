"""Utilities for analysing sentiment and emotion from speech audio."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from transformers import pipeline


@dataclass
class SegmentAnalysis:
    """Represents the analysis for a single transcript segment."""

    text: str
    start: Optional[float]
    end: Optional[float]
    sentiment_label: str
    sentiment_score: float
    emotion_label: str
    emotion_score: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "sentiment": {
                "label": self.sentiment_label,
                "score": self.sentiment_score,
            },
            "emotion": {
                "label": self.emotion_label,
                "score": self.emotion_score,
            },
        }


def _load_audio_segment(audio_path: Path, start: Optional[float], end: Optional[float], target_rate: int = 16000) -> dict:
    """Load an audio segment as a dictionary suitable for the transformers audio pipeline."""

    import librosa

    audio, rate = librosa.load(audio_path, sr=target_rate)
    if start is not None or end is not None:
        start = max(start or 0.0, 0.0)
        if end is None:
            end = len(audio) / rate
        end = max(end, start)
        start_idx = int(math.floor(start * rate))
        end_idx = int(math.ceil(end * rate))
        audio = audio[start_idx:end_idx]
    return {"array": audio, "sampling_rate": target_rate}


def analyse_audio(
    audio_path: Path,
    asr_model: str = "openai/whisper-small",
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    emotion_model: str = "superb/wav2vec2-base-superb-er",
    chunk_length_s: float = 30.0,
) -> List[SegmentAnalysis]:
    """Transcribe *audio_path* and annotate each segment with sentiment and emotion."""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        chunk_length_s=chunk_length_s,
    )
    sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
    emotion_classifier = pipeline("audio-classification", model=emotion_model, top_k=1)

    transcript = asr(audio_path.as_posix(), return_timestamps=True)
    chunks: Sequence[dict]
    if isinstance(transcript, dict) and "chunks" in transcript:
        chunks = transcript["chunks"]
    else:
        chunks = [
            {
                "text": transcript["text"] if isinstance(transcript, dict) else str(transcript),
                "timestamp": (None, None),
            }
        ]

    analyses: List[SegmentAnalysis] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        start, end = chunk.get("timestamp", (None, None))

        sentiment_result = sentiment_analyzer(text, truncation=True)[0]

        audio_segment = _load_audio_segment(audio_path, start, end)
        emotion_result = emotion_classifier(audio_segment)[0]

        analyses.append(
            SegmentAnalysis(
                text=text,
                start=start,
                end=end,
                sentiment_label=sentiment_result["label"],
                sentiment_score=float(sentiment_result["score"]),
                emotion_label=emotion_result["label"],
                emotion_score=float(emotion_result["score"]),
            )
        )

    if not analyses:
        raise RuntimeError("No textual content detected in the provided audio file.")

    return analyses


def format_analyses(analyses: Iterable[SegmentAnalysis]) -> str:
    lines = []
    for idx, segment in enumerate(analyses, start=1):
        timing = ""
        if segment.start is not None and segment.end is not None:
            timing = f" ({segment.start:.2f}s – {segment.end:.2f}s)"
        lines.append(f"Segment {idx}{timing}:")
        lines.append(f"  Text: {segment.text}")
        lines.append(
            f"  Sentiment: {segment.sentiment_label} (confidence {segment.sentiment_score:.2%})"
        )
        lines.append(
            f"  Emotion: {segment.emotion_label} (confidence {segment.emotion_score:.2%})"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio and label sentiment and emotion.")
    parser.add_argument("audio", type=Path, help="Path to the audio file to analyse.")
    parser.add_argument("--asr-model", default="openai/whisper-small")
    parser.add_argument("--sentiment-model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--emotion-model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--chunk-length", type=float, default=30.0, help="Chunk size in seconds for ASR.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    args = parser.parse_args(argv)

    analyses = analyse_audio(
        audio_path=args.audio,
        asr_model=args.asr_model,
        sentiment_model=args.sentiment_model,
        emotion_model=args.emotion_model,
        chunk_length_s=args.chunk_length,
    )

    if args.json:
        json_output = [segment.to_dict() for segment in analyses]
        print(json.dumps(json_output, indent=2))
    else:
        print(format_analyses(analyses))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
