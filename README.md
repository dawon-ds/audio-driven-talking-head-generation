# Audio-Driven Talking-Head Generation

**Data Science Capstone Design · 2026**

A capstone project validating MuseTalk inference and exploring lip-region refinement for audio-visual synchronization.

[Architecture](docs/architecture.md) · [Experiments](docs/experiments.md) · [Web Demo](docs/demo.md)

## Project Overview

We validated MuseTalk v1.0 inference, connected media input and output preview through Gradio, and documented a proposed ADLip Generator / SyncLT extension for a **96×96 lip ROI**.

The public repository contains MuseTalk-based training, normal/realtime inference, preprocessing, SyncNet loss utilities, and the Gradio application. **The ADLip Generator and SyncLT extension are documented in the project architecture; their implementation is not included in the current public source tree.**

## Approach

### Lip-Centric Architecture

![Lip-Centric Architecture](docs/images/architecture.png)

The documented design encodes reference lip frames with a frozen VAE and audio with Whisper Tiny. The ADLip Generator combines spatial cross-attention, temporal attention, and feed-forward refinement to predict an audio-conditioned latent change:

`Z_out = Z_ref + αΔ`

The objectives include latent reconstruction, same/different identity losses, delta regularization, and SyncLT contrastive supervision using matched and mismatched audio segments. See [architecture details](docs/architecture.md).

### Public Inference Pipeline

Reference media is processed with face detection and DWPose, then cropped and encoded into VAE latents. Whisper audio features condition MuseTalk UNet inference. Decoded frames are blended back into the face and composed with audio using FFmpeg.

The scripts support MuseTalk v1.0 and v1.5 layouts and normal/realtime inference modes.

## Results

- End-to-end **MuseTalk v1.0 baseline inference** was validated in the project environment.
- A Gradio demo supported inference execution and output preview.
- One recorded baseline sample took approximately **seven hours** end to end.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio inference and output preview |
| `train.py`, `train.sh` | MuseTalk training entry points |
| `inference.sh`, `scripts/` | Inference and preprocessing |
| `configs/` | Training, inference, and preprocessing settings |
| `musetalk/` | Model, data, loss, and processing utilities |
| `download_weights.sh`, `download_weights.bat` | External model download scripts |
| `check_ffmpeg.py` | FFmpeg installation check |
| `docs/` | Architecture, experiment, and demo documentation |

Runtime media, model weights, datasets, and caches are excluded through `.gitignore`.

## Usage

Install dependencies in a compatible Python/CUDA environment:

```bash
pip install -r requirements.txt
```

FFmpeg must be installed separately. Pose/detection dependencies such as MMPose and MMDetection require compatible versions. The demo imports `moviepy.editor`, so it requires a MoviePy version that provides that module.

Download external pretrained weights:

```bash
# Linux / macOS
bash download_weights.sh
```

```bat
:: Windows
download_weights.bat
```

Check FFmpeg:

```bash
python check_ffmpeg.py
# Alternatively: python check_ffmpeg.py /path/to/ffmpeg/bin
```

### Inference

Update the media paths in [normal.yaml](configs/inference/normal.yaml) or [realtime.yaml](configs/inference/realtime.yaml) to your own files before running. The example media is not included.

```bash
bash inference.sh v1.0 normal
bash inference.sh v1.5 normal
bash inference.sh v1.5 realtime
```

### Web Demo

```bash
python app.py
```

For a custom FFmpeg installation, pass `--ffmpeg_path /path/to/ffmpeg/bin`. External model assets are required before launch.

### Training

Configure dataset, model, and checkpoint paths in the training YAML files for your environment before running:

```bash
bash train.sh stage1
bash train.sh stage2
```

These scripts launch the public MuseTalk training code with Hugging Face Accelerate; they do not reproduce the documented ADLip/SyncLT extension.

## Limitations

- There is no controlled baseline-versus-refined comparison establishing improved lip synchronization. The single recorded runtime does not establish general throughput.
- Pretrained weights, datasets, generated media, and the complete original team environment are not included. The documented ADLip/SyncLT experiment cannot be fully reproduced from the current public repository.

## Review

The project emphasized the difference between documenting an architectural extension and validating its effectiveness through controlled experiments.

## Tech Stack

Python · PyTorch · MuseTalk · Whisper · SyncNet · Gradio · OpenCV · FFmpeg · DWPose · Hugging Face Transformers · Accelerate

## Data and Third-party Resources

- **Upstream implementation:** [MuseTalk](https://github.com/TMElyralab/MuseTalk). Upstream copyright and dependency notices are preserved in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- **Other assets:** Whisper, DWPose, VAE, face parsing, pretrained weights and input media have their own source licenses and usage restrictions. The presence of a download script does not establish permission to redistribute such assets.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
