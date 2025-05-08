"""StreamSAD Configuration Module"""

from dataclasses import dataclass


@dataclass
class Config:
    """StreamSAD Configuration Module"""

    # feature parameters
    fs = 16000
    n_fft = 512
    n_hop = 512
    feature_epsilon = 1e-6

    # raw output smoothing parameters
    max_segment_duration = 15
    max_recursion_depth = 8
    ring_buffer_len = 7
    ring_buffer_threshold_num = 4
    sad_threshold = 0.4
    force_segmentation_margin_frames = 70

    # one-sided segment detection parameters
    segment_terminative_silence = 0.3
