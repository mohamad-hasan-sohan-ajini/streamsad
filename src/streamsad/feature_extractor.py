"""StreamSAD Feature Extraction Module

This module provides a lightweight, NumPy-based feature extractor for computing
log-magnitude spectrograms from audio signals using the real-valued FFT (rFFT).
"""

import numpy as np

from .config import Config


class FeatureExtractor:
    """A class for extracting spectrogram features from audio signals using rFFT.

    This class processes an audio waveform and converts it into a log-scaled
    spectrogram suitable for inference by the SAD model.
    """

    def __init__(self):
        """Initialize the feature extractor with Hann window and epsilon."""
        self.feature_epsilon = Config.feature_epsilon
        self.window = np.hanning(Config.n_fft)

    def compute_fft(self, x_np):
        """Compute log-magnitude spectrogram using real FFT (rFFT).

        Args:
            x_np (np.ndarray): 1D NumPy array of float32 audio samples.

        Returns:
            np.ndarray: 2D array of shape (n_fft // 2 + 1, num_frames) containing
                        log-magnitude spectrogram features.
        """
        num_frames = x_np.shape[0] // Config.n_fft
        fft_frames_real = []

        for i in range(num_frames):
            start_idx = i * Config.n_fft
            end_idx = start_idx + Config.n_fft
            frame = x_np[start_idx:end_idx] * self.window
            fft_frame = np.fft.rfft(frame)
            power_spectrum = (fft_frame * fft_frame.conj()).real
            log_power = np.log10(np.abs(power_spectrum) + self.feature_epsilon)
            fft_frames_real.append(log_power)

        fft_frames_real = np.array(fft_frames_real).T  # Shape: (Freq, Time)
        return fft_frames_real

    def __call__(self, x_np):
        """Extract features from raw audio.

        Args:
            x_np (np.ndarray): 1D NumPy array of float32 audio samples.

        Returns:
            np.ndarray: 3D array of shape (1, Freq, Time), dtype float32,
                        ready for ONNX inference.
        """
        fft_frames_real = self.compute_fft(x_np)
        return np.expand_dims(fft_frames_real, axis=0).astype(np.float32)
