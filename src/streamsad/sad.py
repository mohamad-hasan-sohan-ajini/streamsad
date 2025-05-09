"""SAD module"""

from collections import deque

import numpy as np
import onnxruntime as ort

from config import Config
from feature_extractor import FeatureExtractor


class SAD:
    def __init__(self):
        # audio buffer
        self.input_audio_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0

        # SAD model utils
        self.feature_extractor = FeatureExtractor()

        self.ort_session = ort.InferenceSession("sad_model.onnx")
        self.state = np.zeros((1, 1, 64), dtype=np.float32)

        # post processing algorithm
        self.ring_buffer = deque(maxlen=Config.ring_buffer_len)
        self.triggered = False
        self.agg_result = []
        self.voiced_frames = []

    def __call__(self, audio_array):
        # calculate number valid steps and crop valid part of the buffer
        self.input_audio_buffer = np.concatenate((self.input_audio_buffer, audio_array))
        valid_steps = (
            self.input_audio_buffer.shape[0] - int(self.step * Config.n_hop)
        ) // Config.n_hop
        start_index = int(self.step * Config.n_hop)
        end_index = start_index + int(valid_steps * Config.n_hop)
        tmp_audio_tensor = self.input_audio_buffer[start_index:end_index]
        # run model
        spect = self.feature_extractor(tmp_audio_tensor)
        # run model
        raw_output, self.state = self.ort_session.run(
            None, {"input": spect, "input_state": self.state}
        )
        sad_probs = raw_output[0, :, 1]
        # post processing
        segments = self.apply_ring_buffer_smoothing(sad_probs)
        return segments

    def get_time(self, steps):
        return steps * Config.n_hop / Config.fs

    def apply_ring_buffer_smoothing(self, sad_probs):
        segments = []
        binarized_sad_probs = sad_probs > Config.sad_threshold
        iterator = zip(sad_probs, binarized_sad_probs)
        for sad_prob, is_speech in iterator:
            frame = {"index": self.step, "is_speech": is_speech, "sad_prob": sad_prob}
            self.step += 1
            self.ring_buffer.append(frame)
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
        start_time = self.get_time(self.voiced_frames[0]["index"])
        end_time = self.get_time(self.voiced_frames[-1]["index"] + 1)
        return {
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
        }

    def get_audio(self, segment):
        start_index = int(segment["start"] * Config.fs)
        end_index = int(segment["end"] * Config.fs)
        return self.input_audio_buffer[start_index:end_index]
