# Audio-Driven Talking-Head Generation

**MuseTalk Lip-Sync Improvement · 2026**

[English](./README.md) | [한국어]

이 저장소는 **MuseTalk**을 기반으로 진행한 Audio-Driven Talking-Head Generation 프로젝트의 실제 구현 코드를 정리한 포트폴리오 저장소입니다. 학습, 일반/실시간 inference, 모델 유틸리티, 설정 파일, 전처리 모듈과 **Gradio 기반 웹 데모**를 포함합니다.

프로젝트는 음성과 입 움직임의 정합성을 개선하고, 레퍼런스 영상과 음성 입력부터 최종 talking-head 영상 생성까지 이어지는 End-to-End 흐름을 구축하는 데 초점을 두었습니다.

## Project Overview

- **Period:** 2026
- **Task:** Audio-driven talking-head generation
- **Base Model:** MuseTalk
- **Audio Features:** Whisper
- **Synchronization:** SyncNet-based supervision
- **Framework:** PyTorch
- **Demo:** Gradio

## Implementation

저장소에는 다음 구현이 포함되어 있습니다.

- MuseTalk 모델 로딩 및 inference
- Whisper 기반 음성 feature 추출
- face detection 및 DWPose landmark 전처리
- face parsing 및 mouth-region blending
- VAE latent encoding / decoding
- 오디오 조건 기반 UNet inference
- 학습 과정의 SyncNet 기반 synchronization loss
- L1, VGG, GAN, mouth-GAN, feature-matching, synchronization loss 지원
- 일반 / realtime inference script
- stage 기반 training configuration
- Gradio Web Demo

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

`train.py`는 Hugging Face Accelerate를 활용한 학습 loop를 포함합니다. 설정에 따라 L1 reconstruction loss, VGG perceptual loss, GAN loss, mouth-region GAN loss, feature-matching loss, SyncNet 기반 audio-visual synchronization loss, adapted synchronization weighting, validation 및 checkpoint 저장을 지원합니다.

```bash
sh train.sh stage1
sh train.sh stage2
```

## Inference

```bash
sh inference.sh v1.5 normal
sh inference.sh v1.5 realtime
```

Python 구현은 `scripts/inference.py`, `scripts/realtime_inference.py`에 있습니다.

## Web Demo

```bash
python app.py
```

웹 데모는 음성·레퍼런스 영상 업로드, 얼굴 bounding-box 조정, face parsing/blending 파라미터 조정, 첫 frame inpainting preview, End-to-End 영상 생성을 지원합니다.

## Model Weights

대용량 pretrained weight는 GitHub에 직접 포함하지 않고 다운로드 script로 관리합니다.

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

영상 처리 및 음성·영상 결합을 위해 FFmpeg도 필요합니다.

## Tech Stack

`Python` `PyTorch` `MuseTalk` `Whisper` `SyncNet` `Gradio` `OpenCV` `FFmpeg` `MMPose` `DWPose` `Hugging Face Transformers` `Accelerate`

## Notes

모델 checkpoint, dataset, 생성 영상, 업로드 media, cache 및 기타 대용량 runtime artifact는 `.gitignore`를 통해 제외하며 저장소에 포함하지 않습니다.
