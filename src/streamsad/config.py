"""StreamSAD Configuration Module

Defines configuration parameters for feature extraction, SAD model inference,
and postprocessing used in the streaming SAD pipeline.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Holds global configuration constants for the StreamSAD system."""

    # feature parameters
    fs = 16000
    n_fft = 512
    n_hop = 512
    feature_epsilon = 1e-6

    # output smoothing parameters
    max_segment_duration = 15
    max_recursion_depth = 8
    ring_buffer_len = 7
    ring_buffer_threshold_num = 4
    sad_threshold = 0.4
    force_segmentation_margin_frames = 70
