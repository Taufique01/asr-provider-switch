# asr-provider-switch

## Feature

This repository adds a configurable Automatic Speech Recognition (ASR) pipeline, supporting multiple providers: **OpenAI Whisper**, **faster-whisper **, and an **auto-fallback mode**.  

It enables local transcription if OpenAI is unavailable or rate-limited, while keeping the SRT pipeline and downstream code unchanged.

---

## Features

- **Provider switch** via environment variable:
  - `openai` → OpenAI Whisper API
  - `faster-whisper` → Local Whisper model
  - `auto` → Tries OpenAI first; falls back to local on failure
- **OpenAI-style JSON output** (text + segments + words) for compatibility
- **Integration with existing SRT pipeline** without changing downstream callers
- **Unit and integration tests** for reliability
- **CPU-friendly local transcription** using faster-whisper

---

## Prerequisites

- Python 3.12
- ffmpeg (required for local ASR using faster-whisper)

## Installations

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo>
```
2. Create virtual env(optional but recommended)

```bash
python -m venv env
source env/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```


## environment Variables

Set these in .env
```python
# OpenAI API key (for openai provider)
OPENAI_API_KEY=<your_api_key>

# ASR provider: openai | faster-whisper | auto
WHISPER_PROVIDER=openai
```

## Usage

The Flask endpoint `/build_srt_file` accepts an audio file and returns transcription in OpenAI-style JSON.

**Endpoint:**  


**Payload (multipart/form-data):**  

| Field | Type | Description |
|-------|------|-------------|
| audio | file | Audio file to transcribe (supported formats: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm) |

**Example Request (Python/requests):**  

```python
import requests

files = {"audio": open("test.mp3", "rb")}
resp = requests.post("http://localhost:5000/build_srt_file", files=files)
print(resp.json())
```

### Response

```json
{
  "text": "Hello world",
  "segments": [
    {
      "id": 0,
      "start": 0.12,
      "end": 1.87,
      "text": "Hello world",
      "words": [
        {"start": 0.12, "end": 0.32, "word": "Hello"},
        {"start": 0.33, "end": 1.87, "word": "world"}
      ]
    }
  ]
}
```

## Test

```bash
pytest -q
```



