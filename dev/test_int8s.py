import numpy as np
import onnxruntime as ort
import soundfile as sf
import time

from streamsad.feature_extractor import FeatureExtractor

model_int8_static = "model_2025-06-10_static_int8.onnx"
ort_session_int8_dynamic = ort.InferenceSession(model_int8_static)

x, fs = sf.read("../tests/data/George-crop2.wav")
feature_extractor = FeatureExtractor()
spect = feature_extractor(x)
state = np.zeros((1, 1, 64), dtype=np.float32)

t0 = time.time()
for i in range(1000):
    raw_output_int8_dynamic, _ = ort_session_int8_dynamic.run(
        None,
        {"input": spect, "input_state": state},
    )
print(time.time() - t0)
