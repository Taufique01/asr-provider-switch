from .asr_openai import transcribe_openai
from .asr_local import transcribe_local
import os

def get_whisper_srt(f_name, audio_bytes):
    whisper_provider = os.getenv("WHISPER_PROVIDER")

    if whisper_provider == "openai":
        return transcribe_openai(audio_bytes)
    elif whisper_provider == "faster-whisper":
        return transcribe_local(audio_bytes)
    elif whisper_provider == "auto":
        try:
            return transcribe_openai(audio_bytes)
        except Exception as e:
            return transcribe_local(audio_bytes)

