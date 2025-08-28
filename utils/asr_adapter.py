def faster_whisper_to_openai_format(segments, language=None) -> dict:
    """
    Convert faster-whisper segments (generator or list) to OpenAI-style verbose_json.

    Args:
        segments: list or generator of faster-whisper Segment objects
        language: optional, str

    Returns:
        dict: OpenAI-style JSON
    """
    # Convert generator to list if needed
    segments = list(segments)

    if not segments:
        return {
            "text": "",
            "segments": [],
            "duration": 0,
            "language": language or "english",
            "task": "transcribe"
        }

    # Auto-detect language if not provided
    if language is None:
        language = getattr(segments[0], "language", "english")

    total_duration = segments[-1].end

    result = {
        "text": "".join([seg.text for seg in segments]),
        "segments": [],
        "duration": total_duration,
        "language": language,
        "task": "transcribe"
    }

    for i, seg in enumerate(segments):
        result["segments"].append({
            "id": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })

    return result
