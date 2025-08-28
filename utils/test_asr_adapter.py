import pytest
from .asr_adapter import faster_whisper_to_openai_format

def test_faster_whisper_to_openai_format():
    """
    Test adapter: converts mocked faster-whisper segments to OpenAI-style JSON.
    Supports both list and generator inputs, includes duration and language.
    """
    # Mock segment and word classes
    class MockWord:
        def __init__(self, start, end, word):
            self.start = start
            self.end = end
            self.word = word

    class MockSegment:
        def __init__(self, start, end, text, words):
            self.start = start
            self.end = end
            self.text = text
            self.words = words

    # Create mocked segments
    segments_list = [
        MockSegment(
            start=0.0,
            end=1.0,
            text="Hello world",
            words=[MockWord(0.0, 0.5, "Hello"), MockWord(0.5, 1.0, "world")]
        ),
        MockSegment(
            start=1.0,
            end=2.0,
            text="Goodbye",
            words=[MockWord(1.0, 2.0, "Goodbye")]
        )
    ]

    # Test with generator
    segments_gen = (s for s in segments_list)
    result_gen = faster_whisper_to_openai_format(segments_gen, language="english")
    assert result_gen["text"] == "Hello worldGoodbye"
    assert result_gen["language"] == "english"
    assert result_gen["duration"] == 2.0
    assert len(result_gen["segments"]) == 2
