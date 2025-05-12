# streamsad

`streamsad` is a streaming-oriented Speech Activity Detection (SAD) module that operates frame by frame, without requiring access to the full audio signal (unlike batch processing). Unlike simple energy-based Voice Activity Detection (VAD), it accurately detects human speech while ignoring music, background noise, and silence. Powered by an efficient ONNX model and a post-processing algorithm inspired by WebRTC (using ring buffer smoothing), it runs entirely on the CPU with minimal overhead, making it ideal for real-time voice interfaces, ASR frontends, and low-resource deployments.

# Dependencies

The following third-party dependencies are required to use `streamsad`:

- `numpy`
- `onnxruntime`

# How to Use

Here is an example of how to use the `streamsad` module:

```python
from streamsad.sad import SAD
import numpy as np

# Initialize the SAD model
sad = SAD()

# Create an example audio stream (e.g., 2 seconds of random audio at 16kHz)
audio_np_array = np.random.randn(32000).astype(np.float32)

# Detect speech segments
segments = sad(audio_np_array)

# Print the detected segments
print(segments)
```

# Installation

You can install `streamsad` using pip:

```bash
pip install streamsad
```

# Testing

After installing the module, you can run unit tests using `pytest`:

```bash
pytest -s tests/test_sad.py
```
