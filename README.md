# sentiemos

`sentiemos` is a minimal utility that transcribes an audio file, labels each spoken
segment with a text sentiment prediction, and extracts an overall emotion label
from the audio signal. The defaults rely on open-source Hugging Face models, so
no external API keys are required.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the analyser as a module and pass the path to your audio file:

```bash
python -m sentiemos path/to/audio.wav
```

The command prints a breakdown similar to:

```
Segment 1 (0.00s – 3.25s):
  Text: hello there
  Sentiment: POSITIVE (confidence 98.42%)
  Emotion: happy (confidence 62.11%)
```

Use `--json` if you prefer structured machine-readable output.

### Custom models

You can swap out any of the underlying models:

```bash
python -m sentiemos audio.wav \
  --asr-model openai/whisper-medium \
  --sentiment-model distilbert-base-uncased-finetuned-sst-2-english \
  --emotion-model microsoft/wavlm-base-plus-sv
```

## Notes

- Whisper models provide per-segment timestamps, which we use to attribute
  sentiment labels to the exact spoken fragments.
- Emotion detection runs on the corresponding audio slice for each transcript
  segment. When timestamps are not available, the script falls back to the entire
  audio clip.
- Large models may take time to download on the first run. Consider pinning
  lighter-weight alternatives if you target constrained environments.
