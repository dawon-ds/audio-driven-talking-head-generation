# Audio-Driven Talking-Head Generation

**MuseTalk Lip-Sync Improvement · 2026**

[English] | [한국어](./README_KR.md)

This repository contains the implementation used for an audio-driven talking-head generation project built on **MuseTalk**, including training, normal/realtime inference, model utilities, configuration files, preprocessing modules, and a **Gradio-based web demo**.

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

The implementation includes:

- MuseTalk model loading and inference
- Whisper-based audio feature extraction
- face detection and DWPose landmark preprocessing
- face parsing and mouth-region blending
- VAE latent encoding and decoding
- audio-conditioned UNet inference
- SyncNet-based synchronization loss during training
- L1, VGG, GAN, mouth-GAN, feature-matching, and synchronization loss support
- normal and realtime inference scripts
- stage-based training configuration
- Gradio web interface for parameter adjustment, preview, and final generation

## Pipeline

```text
Reference Video + Audio
        ↓
Face Detection / DWPose Preprocessing
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

`train.py` provides the training loop using Hugging Face Accelerate. Depending on the configuration, the training pipeline supports L1 reconstruction loss, VGG perceptual loss, GAN loss, mouth-region GAN loss, feature-matching loss, SyncNet-based audio-visual synchronization loss, adapted synchronization weighting, validation, and checkpoint saving.

```bash
sh train.sh stage1
sh train.sh stage2
```

## Inference

```bash
sh inference.sh v1.5 normal
sh inference.sh v1.5 realtime
```

The Python implementations are located in `scripts/inference.py` and `scripts/realtime_inference.py`.

## Web Demo

```bash
python app.py
```

The demo provides audio and reference-video upload, face bounding-box adjustment, face parsing/blending parameters, first-frame inpainting preview, and end-to-end talking-head video generation.

## Model Weights

Large pretrained weights are not stored in GitHub. Download scripts are provided instead.

```bash
bash download_weights.sh
```

```bat
download_weights.bat
```

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
├── scripts/
├── musetalk/
│   ├── data/
│   │   ├── dataset.py
│   │   └── sample_method.py
│   ├── loss/
│   │   └── vgg_face.py
│   ├── models/
│   └── utils/
│       ├── dwpose/
│       ├── face_detection/
│       ├── face_parsing/
│       └── preprocessing.py
└── docs/
```

## Installation

```bash
pip install -r requirements.txt
```

FFmpeg is also required for video processing and audio-video composition.

## Tech Stack

`Python` `PyTorch` `MuseTalk` `Whisper` `SyncNet` `Gradio` `OpenCV` `FFmpeg` `MMPose` `DWPose` `Hugging Face Transformers` `Accelerate`

## Notes

Model checkpoints, datasets, generated videos, uploaded media, caches, and other large runtime artifacts are excluded through `.gitignore` and are not stored in this repository.
