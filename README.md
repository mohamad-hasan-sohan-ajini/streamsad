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

## Process an audio file in stream mode

`SAD` keeps its model and smoothing state between calls. After loading an audio
file into a NumPy array, feed it to the same `SAD` instance in 1,600-sample
(0.1 second) chunks:

```python
import numpy as np
import soundfile as sf

from streamsad import SAD

SAMPLE_RATE = 16_000
CHUNK_SIZE = 1_600

audio, sample_rate = sf.read("tests/data/George-crop2.wav")

sad = SAD()

for start in range(0, len(audio), CHUNK_SIZE):
    chunk = audio[start : start + CHUNK_SIZE]

    # Pad the final chunk so every call receives exactly 1,600 samples.
    if len(chunk) < CHUNK_SIZE:
        chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

    for segment in sad(chunk):
        print(segment)

# Flush speech that continues to the end of the file, one chunk at a time.
for _ in range(10):
    silence = np.zeros(CHUNK_SIZE)
    for segment in sad(silence):
        print(segment)
```

> **Note:** A chunk size of 1,600 samples is not required; it represents 0.1
> seconds at 16 kHz. The model processes audio in 512-sample frames and buffers
> any remainder between calls. Use at least 512 samples per chunk with the
> current implementation. Smaller chunks reduce latency, while larger chunks
> can improve throughput; multiples of 512 avoid buffered remainders.

# Testing

Install the package with its test dependencies, then run the test suite:

```bash
python -m pip install -e '.[test]'
python -m pytest
```
