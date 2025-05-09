import numpy as np
from streamsad.sad import SAD


def test_sad_with_silence():
    """Test SAD model with silence input (should return no segments)"""
    sad = SAD()

    # Create 1 second of silent audio at 16kHz
    silent_audio = np.zeros(16000, dtype=np.float32)

    segments = sad(silent_audio)

    assert isinstance(segments, list), "Output should be a list"
    assert all(
        isinstance(seg, dict) for seg in segments
    ), "Each segment should be a dict"
    assert all(
        "start" in seg and "end" in seg for seg in segments
    ), "Segment missing required keys"

    # Silence should likely produce no segments
    assert len(segments) == 0 or all(
        seg["duration"] < 0.5 for seg in segments
    ), "Unexpected segments detected from silence"
