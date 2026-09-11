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

The implementation includes MuseTalk model loading and inference, Whisper-based audio feature extraction, face detection and DWPose landmark preprocessing, face parsing and mouth-region blending, VAE latent encoding/decoding, audio-conditioned UNet inference, SyncNet-based synchronization loss, stage-based training configuration, normal/realtime inference scripts, and a Gradio web demo.

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

```bash
sh train.sh stage1
sh train.sh stage2
```

## Inference

```bash
sh inference.sh v1.5 normal
sh inference.sh v1.5 realtime
```

## Web Demo

```bash
python app.py
```

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
├── requirements.txt
├── configs/
├── scripts/
├── musetalk/
│   ├── data/
│   ├── loss/
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
