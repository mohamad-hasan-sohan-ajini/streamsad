"""SAD module

This module implements streaming-oriented Speech Activity Detection (SAD)
using a lightweight ONNX model and a WebRTC-inspired ring buffer postprocessing algorithm.
"""

from collections import deque
from importlib import resources

import numpy as np
import onnxruntime as ort

from . import models
from .config import Config
from .feature_extractor import FeatureExtractor


class SAD:
    """Streaming Speech Activity Detection (SAD) system.

    This class processes audio input frame-by-frame, applies an ONNX-based SAD model,
    and uses a ring buffer-based postprocessing algorithm to generate speech segments.
    """

    def __init__(self):
        """Initialize the SAD system, including ONNX session, buffers, and postprocessing state."""
        self.input_audio_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0

        # SAD model components
        self.feature_extractor = FeatureExtractor()
        model_path = resources.files(models).joinpath("model_2025-05-09.onnx")
        self.ort_session = ort.InferenceSession(model_path)
        self.state = np.zeros((1, 1, 64), dtype=np.float32)

        # Postprocessing algorithm
        self.ring_buffer = deque(maxlen=Config.ring_buffer_len)
        self.triggered = False
        self.agg_result = []
        self.voiced_frames = []

    def __call__(self, audio_array):
        """Process a new chunk of audio and return detected speech segments (if any).

        Args:
            audio_array (np.ndarray): 1D NumPy array of float32 audio samples.

        Returns:
            list[dict]: List of speech segments with 'start', 'end', and 'duration' keys.
        """
        self.input_audio_buffer = np.concatenate((self.input_audio_buffer, audio_array))
        valid_steps = (
            self.input_audio_buffer.shape[0] - int(self.step * Config.n_hop)
        ) // Config.n_hop
        start_index = int(self.step * Config.n_hop)
        end_index = start_index + int(valid_steps * Config.n_hop)
        tmp_audio_tensor = self.input_audio_buffer[start_index:end_index]

        # Extract features
        spect = self.feature_extractor(tmp_audio_tensor)

        # Run inference
        raw_output, self.state = self.ort_session.run(
            None,
            {"input": spect, "input_state": self.state},
        )
        sad_probs = raw_output[0, :, 1]

        # Apply smoothing and return speech segments
        segments = self.apply_ring_buffer_smoothing(sad_probs)
        return segments

    def get_time(self, steps):
        """Convert step count to time in seconds.

        Args:
            steps (int): Number of steps (frames).

        Returns:
            float: Corresponding time in seconds.
        """
        return steps * Config.n_hop / Config.fs

    def apply_ring_buffer_smoothing(self, sad_probs):
        """Apply postprocessing to SAD probabilities using a ring buffer smoother.

        Args:
            sad_probs (np.ndarray): 1D array of SAD probabilities for each frame.

        Returns:
            list[dict]: Detected speech segments as dictionaries with start, end, and duration.
        """
        segments = []
        binarized_sad_probs = sad_probs > Config.sad_threshold
        iterator = zip(sad_probs, binarized_sad_probs)

        for sad_prob, is_speech in iterator:
            frame = {"index": self.step, "is_speech": is_speech, "sad_prob": sad_prob}
            self.step += 1
            self.ring_buffer.append(frame)
            # make a decision about each frame based on the trigger state
            if not self.triggered:
                num_voiced = len(
                    [frame for frame in self.ring_buffer if frame["is_speech"]]
                )
                if num_voiced > Config.ring_buffer_threshold_num:
                    self.voiced_frames = [frame for frame in self.ring_buffer]
                    self.triggered = True
                    self.ring_buffer.clear()
            else:
                self.voiced_frames.append(frame)
                num_unvoiced = len(
                    [frame for frame in self.ring_buffer if not frame["is_speech"]]
                )
                if num_unvoiced > Config.ring_buffer_threshold_num:
                    segments.append(self.postprocess_rb_result())
                    # TODO: force segmentation in case the current segment exceeds a limit
                    self.voiced_frames = []
                    self.triggered = False
                    self.ring_buffer.clear()

        return segments

    def postprocess_rb_result(self):
        """Convert accumulated voiced frames to a segment dictionary.

        Returns:
            dict: Segment with 'start', 'end', and 'duration' in seconds.
        """
        start_time = self.get_time(self.voiced_frames[0]["index"])
        end_time = self.get_time(self.voiced_frames[-1]["index"] + 1)
        return {
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
        }

    def get_audio(self, segment):
        """Extract raw audio corresponding to a detected segment.

        Args:
            segment (dict): Segment dictionary with 'start' and 'end' keys in seconds.

        Returns:
            np.ndarray: Audio samples from the input buffer corresponding to the segment.
        """
        start_index = int(segment["start"] * Config.fs)
        end_index = int(segment["end"] * Config.fs)
        return self.input_audio_buffer[start_index:end_index]
