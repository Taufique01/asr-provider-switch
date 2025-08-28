import tempfile
from faster_whisper import WhisperModel
from .asr_adapter import faster_whisper_to_openai_format

def transcribe_local(audio_bytes: bytes, file_ext: str = ".mp3") -> dict:
    """
    Transcribe audio using faster-whisper (supports WAV, MP3, etc.) 
    and return OpenAI-style verbose_json.

    Args:
        audio_bytes: raw audio file bytes
        file_ext: file extension (default '.mp3'), used for temp file
    """
    # Write bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        model = WhisperModel("tiny")
        segments, _ = model.transcribe(tmp.name, word_timestamps=True)
            
        # Convert to OpenAI-style JSON using the adapter
        result = faster_whisper_to_openai_format(segments)

    return result
