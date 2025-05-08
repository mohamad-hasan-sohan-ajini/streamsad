"""StreamSAD Feature Extraction Module"""

import numpy as np

from config import Config


class FeatureExtractor:
    """A class for extracting spectrogram features from audio signals using FFT"""

    def __init__(self):
        self.feature_epsilon = Config.feature_epsilon
        self.window = np.hanning(Config.n_fft)

    def compute_fft(self, x_np):
        num_frames = x_np.shape[0] // Config.n_fft
        fft_frames_real = []
        for i in range(num_frames):
            start_idx = i * Config.n_fft
            end_idx = start_idx + Config.n_fft
            frame = x_np[start_idx:end_idx] * self.window
            fft_frame = np.fft.rfft(frame)
            fft_frame = (fft_frame * fft_frame.conj()).real
            fft_frame_real = np.log10(np.abs(fft_frame) + Config.feature_epsilon)
            fft_frames_real.append(fft_frame_real)
        fft_frames_real = np.array(fft_frames_real).T
        return fft_frames_real

    def __call__(self, x_np):
        # x_np of the shape T
        fft_frames_real = self.compute_fft(x_np)
        return np.expand_dims(fft_frames_real, axis=0).astype(np.float32)
