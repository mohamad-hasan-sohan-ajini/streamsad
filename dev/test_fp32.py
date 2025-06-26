import numpy as np
import onnxruntime as ort
import soundfile as sf
import time

from streamsad.feature_extractor import FeatureExtractor

model_fp32 = "../src/streamsad/models/model_2025-06-10.onnx"
ort_session_fp32 = ort.InferenceSession(model_fp32)

x, fs = sf.read("../tests/data/George-crop2.wav")
feature_extractor = FeatureExtractor()
spect = feature_extractor(x)
state = np.zeros((1, 1, 64), dtype=np.float32)

t0 = time.time()
for i in range(1000):
    raw_output_fp32, _ = ort_session_fp32.run(
        None,
        {"input": spect, "input_state": state},
    )
print(time.time() - t0)
