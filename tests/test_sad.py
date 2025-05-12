import numpy as np
import soundfile as sf

from streamsad.sad import SAD

# feed sad with 0.1 second audio chunks
BUFFER_SIZE = 1600


def buffer_iterator(arr):
    """
    Yield subarrays of length `buffer_size` from `arr`.

    Args:
        arr (np.ndarray): Input 1D array.

    Yields:
        np.ndarray: Subarrays of shape (buffer_size,) or smaller at the end if arr is not evenly divisible.
    """
    for i in range(0, len(arr), BUFFER_SIZE):
        yield arr[i : i + BUFFER_SIZE]


def test_sad_with_silence():
    """Test SAD model with silence input (should return no segments)"""
    sad = SAD()

    # Create 1 second of silent audio at 16kHz
    silent_audio = np.zeros(16000, dtype=np.float32)

    for chunk in buffer_iterator(silent_audio):
        segments = sad(chunk)

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


def test_sad_with_file():
    """Test SAD model with sample file input (should return two segments)"""
    x, fs = sf.read("tests/data/George-crop2.wav")
    silent_audio = np.zeros(16000, dtype=np.float32)

    sad = SAD()
    agg_segments = []
    for chunk in buffer_iterator(x):
        segments = sad(chunk)
        agg_segments.extend(segments)
    assert len(agg_segments) == 1, "No segment is detected"
    assert sad.triggered, "Untriggered state for trailing segment"

    segments = sad(silent_audio)
    agg_segments.extend(segments)
    assert len(agg_segments) == 2, "Trailing segment is undetected"
    assert sad.triggered is False, ""
