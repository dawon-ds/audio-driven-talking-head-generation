# Audio-Driven Talking-Head Generation

**MuseTalk Lip-Sync Improvement · 2026**

[English] | [한국어](./README_KR.md)

This repository contains the implementation used for an audio-driven talking-head generation project built on **MuseTalk**, including training, normal/realtime inference, model utilities, configuration files, and a **Gradio-based web demo**.

The project focuses on improving audio-visual lip synchronization and building an end-to-end workflow from reference video and speech input to generated talking-head output.

## Project Overview

- **Period:** 2026
- **Task:** Audio-driven talking-head generation
- **Base Model:** MuseTalk
- **Audio Features:** Whisper
- **Synchronization:** SyncNet-based supervision
- **Framework:** PyTorch
- **Demo:** Gradio

## Implementation

The uploaded implementation includes the full project flow used for training and inference:

- MuseTalk model loading and inference
- Whisper-based audio feature extraction
- face landmark / bounding-box preprocessing
- VAE latent encoding and decoding
- audio-conditioned UNet inference
- SyncNet-based synchronization loss during training
- L1, VGG, GAN, mouth-GAN, feature-matching, and synchronization loss support
- normal and realtime inference scripts
- stage-based training configuration
- generated mouth-region blending back into the source frames
- FFmpeg / MoviePy video and audio composition
- Gradio web interface for parameter adjustment, preview, and final generation

The web demo accepts audio and a reference video, provides controls for face-box shift, extra margin, parsing mode, and cheek range, and supports a first-frame inpainting test before generating the final video.

## Pipeline

```text
Reference Video + Audio
        ↓
Frame / Face Preprocessing
        ↓
Whisper Audio Features
        ↓
VAE Latent + MuseTalk UNet
        ↓
Generated Mouth / Face Region
        ↓
Face Parsing & Blending
        ↓
Frame Sequence + Audio Composition
        ↓
Talking-Head Video
        ↓
Gradio Preview
```

## Training

`train.py` provides the training loop using Hugging Face Accelerate. Depending on the configuration, the training pipeline supports:

- L1 reconstruction loss
- VGG perceptual loss
- GAN loss
- mouth-region GAN loss
- feature-matching loss
- SyncNet-based audio-visual synchronization loss
- adapted synchronization weighting
- validation and checkpoint saving

Training is separated into stage configurations under `configs/training/`.

```bash
sh train.sh stage1
sh train.sh stage2
```

## Inference

Normal and realtime inference are available through `inference.sh`.

```bash
sh inference.sh v1.5 normal
sh inference.sh v1.5 realtime
```

The Python implementations are located in `scripts/inference.py` and `scripts/realtime_inference.py`.

## Web Demo

The Gradio demo is implemented in `app.py`.

```bash
python app.py
```

The demo provides:

- audio upload
- reference-video upload
- face bounding-box adjustment
- face parsing / blending parameters
- first-frame inpainting preview
- end-to-end talking-head video generation

## Model Weights

Large pretrained weights are intentionally excluded from GitHub. Download scripts are provided instead.

Linux / macOS:

```bash
bash download_weights.sh
```

Windows:

```bat
download_weights.bat
```

The scripts retrieve the required MuseTalk, SD-VAE, Whisper, DWPose, SyncNet, and face-parsing weights.

## Repository Structure

```text
musetalk-lipsync-improvement/
├── app.py
├── train.py
├── test_ffmpeg.py
├── requirements.txt
├── train.sh
├── inference.sh
├── download_weights.sh
├── download_weights.bat
├── entrypoint.sh
├── configs/
│   ├── inference/
│   └── training/
├── scripts/
│   ├── inference.py
│   ├── realtime_inference.py
│   └── preprocess.py
├── musetalk/
│   ├── data/
│   ├── loss/
│   ├── models/
│   └── utils/
└── docs/
    ├── architecture.md
    ├── experiments.md
    └── demo.md
```

## Installation

```bash
pip install -r requirements.txt
```

FFmpeg is also required for video processing and audio-video composition.

## Tech Stack

`Python` `PyTorch` `MuseTalk` `Whisper` `SyncNet` `Gradio` `OpenCV` `FFmpeg` `Hugging Face Transformers` `Accelerate`

## Notes

Model checkpoints, datasets, generated videos, uploaded media, caches, and other large runtime artifacts are excluded through `.gitignore` and are not stored in this repository.
