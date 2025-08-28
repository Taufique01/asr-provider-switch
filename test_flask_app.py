import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pytest

import app


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client


FAKE_JSON = {
    "text": "Hello world",
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "Hello world",
            "words": [
                {"start": 0.0, "end": 0.5, "word": "Hello"},
                {"start": 0.5, "end": 1.0, "word": "world"},
            ],
        }
    ],
}


def test_build_srt_openai(monkeypatch, client):
    os.environ["WHISPER_PROVIDER"] = "openai"

    def fake_transcribe_openai(audio_bytes):
        return FAKE_JSON

    monkeypatch.setattr("utils.video_utils.transcribe_openai", fake_transcribe_openai)

    data = {"audio": (io.BytesIO(b"FAKE"), "test.mp3")}
    resp = client.post("/build_srt_file", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["text"] == "Hello world"
    assert payload["segments"][0]["words"][1]["word"] == "world"


def test_build_srt_faster_whisper(monkeypatch, client):
    os.environ["WHISPER_PROVIDER"] = "faster-whisper"

    def fake_transcribe_local(audio_bytes, language=None):
        return FAKE_JSON

    monkeypatch.setattr("utils.video_utils.transcribe_local", fake_transcribe_local)

    data = {"audio": (io.BytesIO(b"FAKE"), "test.mp3")}
    resp = client.post("/build_srt_file", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["text"] == "Hello world"


def test_build_srt_auto_when_openai_works(monkeypatch, client):
    monkeypatch.setenv("WHISPER_PROVIDER", "auto")

    def fake_transcribe_openai(audio_bytes):
        return FAKE_JSON  # simulate success

    def fake_transcribe_local(audio_bytes, language=None):
        raise Exception("Should not be called")

    monkeypatch.setattr("utils.video_utils.transcribe_openai", fake_transcribe_openai)
    monkeypatch.setattr("utils.video_utils.transcribe_local", fake_transcribe_local)

    data = {"audio": (io.BytesIO(b"FAKE"), "test.mp3")}
    resp = client.post("/build_srt_file", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["text"] == "Hello world"


def test_build_srt_auto_when_openai_falls(monkeypatch, client):
    """When provider=auto and OpenAI fails, it should fallback to Local."""
    monkeypatch.setenv("WHISPER_PROVIDER", "auto")

    def fake_transcribe_openai(audio_bytes):
        raise Exception("Simulated OpenAI failure")

    def fake_transcribe_local(audio_bytes, language=None):
        return FAKE_JSON

    monkeypatch.setattr("utils.video_utils.transcribe_openai", fake_transcribe_openai)
    monkeypatch.setattr("utils.video_utils.transcribe_local", fake_transcribe_local)

    data = {"audio": (io.BytesIO(b"FAKE"), "test.mp3")}
    resp = client.post("/build_srt_file", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["text"] == "Hello world"
