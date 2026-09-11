# Audio-Driven Talking-Head Generation

**MuseTalk Lip-Sync Improvement · 2026**

This repository contains the implementation used for an audio-driven talking-head generation project built on **MuseTalk**, including training, normal/realtime inference, preprocessing, synchronization losses, configuration files, and a **Gradio-based web demo**.

The project focuses on improving audio-visual lip synchronization and building an end-to-end workflow from reference video and speech input to generated talking-head output.

## Project Overview

- **Period:** 2026
- **Task:** Audio-driven talking-head generation
- **Base Model:** MuseTalk
- **Inference Support:** MuseTalk v1.0 / v1.5
- **Audio Features:** Whisper
- **Synchronization:** SyncNet-based supervision
- **Framework:** PyTorch
- **Demo:** Gradio

## Implementation

The repository includes:

- MuseTalk training and inference
- Whisper-based audio feature extraction
- face detection and DWPose landmark preprocessing
- face parsing and mouth-region blending
- VAE latent encoding / decoding
- audio-conditioned UNet inference
- SyncNet-based synchronization loss
- stage-based training configurations
- normal and realtime inference scripts
- Gradio web demo

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

`train.py` is launched through Hugging Face Accelerate and uses the stage-specific configurations under `configs/training/`.

## Inference

Normal and realtime inference support MuseTalk v1.0 and v1.5 model layouts.

```bash
sh inference.sh v1.5 normal
sh inference.sh v1.5 realtime
```

The Python entry points are:

```text
scripts/inference.py
scripts/realtime_inference.py
```

Normal inference uses `configs/inference/normal.yaml`; realtime inference uses `configs/inference/realtime.yaml`.

## Web Demo

```bash
python app.py
```

The Gradio interface provides an end-to-end workflow for reference media and audio input, preprocessing, generation, and output preview.

## Model Weights

Large pretrained weights are not stored in GitHub. Download scripts are provided instead.

```bash
bash download_weights.sh
```

```bat
download_weights.bat
```

## FFmpeg Check

FFmpeg is required for video processing and audio-video composition. A small environment check is provided:

```bash
python check_ffmpeg.py <ffmpeg-bin-path>
```

## Experimental Lip-Centric Design

Project materials also explored a 96×96 lip-ROI refinement design using concepts such as ADLip Generator, LipCrossAttention, residual latent updates, and additional synchronization supervision. These are documented separately as **experimental design elements** and are not presented as standalone implemented modules unless corresponding code is present in this repository.

See `docs/architecture.md` for the distinction between the implemented MuseTalk pipeline and the experimental refinement design.

## Repository Structure

```text
musetalk-lipsync-improvement/
├── app.py
├── train.py
├── check_ffmpeg.py
├── requirements.txt
├── train.sh
├── inference.sh
├── download_weights.sh
├── download_weights.bat
├── entrypoint.sh
├── configs/
│   ├── inference/
│   │   ├── normal.yaml
│   │   └── realtime.yaml
│   └── training/
├── scripts/
│   ├── inference.py
│   ├── preprocess.py
│   └── realtime_inference.py
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
    ├── architecture.md
    ├── demo.md
    └── experiments.md
```

## Installation

```bash
pip install -r requirements.txt
```

FFmpeg is also required separately at the system level.

## Tech Stack

`Python` `PyTorch` `MuseTalk` `Whisper` `SyncNet` `Gradio` `OpenCV` `FFmpeg` `MMPose` `DWPose` `Hugging Face Transformers` `Accelerate`

## Notes

- Model checkpoints, datasets, generated videos, uploaded media, caches, and other large runtime artifacts are excluded through `.gitignore`.
- Reproducing training or inference additionally requires the appropriate MuseTalk weights, compatible CUDA/runtime dependencies, and system-level FFmpeg.
- Some runtime dependencies in the pose/detection stack may require environment-specific version compatibility beyond `pip install -r requirements.txt`.
