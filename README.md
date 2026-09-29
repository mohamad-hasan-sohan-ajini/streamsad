# streamsad

`streamsad` is a streaming-oriented Speech Activity Detection (SAD) module that operates frame by frame, without requiring access to the full audio signal (unlike batch processing). Unlike simple energy-based Voice Activity Detection (VAD), it accurately detects human speech while ignoring music, background noise, and silence. Powered by an efficient ONNX model and a post-processing algorithm inspired by WebRTC (using ring buffer smoothing), it runs entirely on the CPU with minimal overhead, making it ideal for real-time voice interfaces, ASR frontends, and low-resource deployments.

# Requirements

`streamsad` supports Python 3.12 through 3.14. Its runtime dependencies are
installed automatically:

- `numpy==2.5.3`
- `onnxruntime==1.30.0`

# Installation

## Install from PyPI

Create and activate a virtual environment, then install `streamsad`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamsad
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

## Install from source

From the root of a cloned repository, create a virtual environment and install
the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

# How to Use

Here is an example of how to use the `streamsad` module:

```python
import numpy as np
from streamsad import SAD

# Initialize the SAD model
sad = SAD()

# Create an example audio stream (e.g., 2 seconds of random audio at 16kHz)
audio_np_array = np.random.randn(32000).astype(np.float32)

# Detect speech segments
segments = sad(audio_np_array)

# Print the detected segments
print(segments)
```

# Testing

Install the package with its test dependencies, then run the test suite:

```bash
python -m pip install -e '.[test]'
python -m pytest
```
