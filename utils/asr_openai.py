import os
from openai import OpenAI


def transcribe_openai(audio_bytes: bytes, language: str | None = None) -> dict:
    """
    Transcribe audio with OpenAI Whisper API and return verbose_json.
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Save bytes to temp file (API requires file-like input)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp.name, "rb"),
            response_format="verbose_json",
            language=language
        )

    # response is already in the OpenAI verbose_json format
    return response.to_dict()  # ensures Python dict
