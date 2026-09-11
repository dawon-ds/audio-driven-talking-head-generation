# Audio-Driven Talking-Head Generation

**MuseTalk Lip-Sync Improvement · Capstone Project · 2026**

This repository documents a capstone project focused on improving lip synchronization in **audio-driven talking-head generation** using MuseTalk as the baseline system.

The project explored a lip-centric refinement strategy around a **96×96 mouth ROI**, combining an additional lip generator with audio-visual synchronization supervision, while also building an end-to-end inference flow and a Python-based web demo.

## Project Overview

- **Period:** 2026
- **Task:** Audio-driven talking-head generation
- **Base Model:** MuseTalk v1.0
- **Approach:** Lip-Centric refinement with ADLip Generator
- **Audio Features:** Whisper
- **Synchronization:** SyncNet / SyncLT supervision
- **Framework:** PyTorch

## Problem

MuseTalk can generate high-quality talking-head videos, but the project focused on a remaining challenge: improving the alignment between **mouth motion and input audio**.

Rather than modifying the full face-generation pipeline, the project concentrated on the mouth region and designed a lip-centric refinement strategy intended to make local mouth motion more responsive to audio features.

## Baseline

The project first established and validated a **MuseTalk v1.0 inference baseline**.

In the project environment, one sample end-to-end inference run required approximately **7 hours**, which also motivated work on a more practical inference workflow and demo interface.

## Lip-Centric Improvement

The proposed refinement stage focused on a **96×96 lip ROI**.

The refinement module receives:

- a reference lip frame,
- a corresponding audio segment,
- latent visual features from the baseline generation pipeline.

The **ADLip Generator** predicts a latent lip-motion adjustment, represented as `Δ`, and the refined latent is formed as:

```text
Z_out = Z_ref + αΔ
```

This residual formulation is designed to preserve the reference visual structure while allowing audio-conditioned mouth movement to be adjusted locally.

## ADLip Generator Design

The ADLip Generator was designed as the core lip-refinement module.

Its purpose is to learn an audio-conditioned update in latent space rather than regenerate the entire face. The design combines:

- reference lip latent features,
- speech features extracted from Whisper,
- cross-modal fusion through LipCrossAttention,
- residual prediction of lip-motion change `Δ`.

The visual latent acts as the **query**, while the audio features are used as **key/value** in cross-attention. This enables the model to condition local mouth motion directly on speech information while preserving the surrounding facial structure.

## Model Architecture

The experimental architecture used:

- **Frozen VAE** for latent image representation,
- **Whisper audio features**,
- **LipCrossAttention** as the trainable audio-visual fusion component,
- visual latent features as the query,
- audio features as key/value,
- residual latent lip update through the ADLip Generator.

The overall design isolates the lip-synchronization component without unnecessarily updating the full image-generation backbone.

## Training Objectives

The project used a combination of objectives including:

- latent L1 reconstruction loss,
- silence loss,
- delta regularization,
- SyncNet-based synchronization loss,
- TTA-related synchronization supervision.

These losses jointly encourage visual reconstruction quality, stable residual updates, and stronger correspondence between speech and generated lip motion.

## SyncNet / SyncLT Design

A frozen SyncNet-style network was used as an audio-visual synchronization supervisor.

The synchronization objective compares matched and mismatched audio-video pairs so that the model learns whether the generated mouth motion is temporally consistent with the input speech.

The SyncLT supervision strategy was incorporated into training to provide an explicit lip-sync signal in addition to reconstruction-based objectives. This helps prevent the refinement model from optimizing only visual similarity while ignoring whether mouth movement actually matches the spoken audio.

## Inference Pipeline

The project integrated the full pipeline as:

```text
Reference Image + Audio
        ↓
MuseTalk Baseline Inference
        ↓
Lip ROI / Latent Extraction
        ↓
ADLip Generator
  + Whisper Audio Features
  + LipCrossAttention
        ↓
Sync-Aware Lip Refinement
        ↓
Generated Talking-Head Video
        ↓
Web Demo Preview
```

The end-to-end workflow connected reference-image and audio inputs to final video generation and allowed the result to be reviewed through a Python-based web interface.

## Project Work

The project covered the complete development flow, including:

- MuseTalk v1.0 baseline setup and inference validation,
- analysis of the lip-sync limitation,
- 96×96 lip-ROI based refinement design,
- ADLip Generator architecture design,
- Whisper-based audio feature integration,
- LipCrossAttention-based audio-visual fusion,
- residual latent update design,
- SyncNet / SyncLT synchronization supervision,
- reconstruction and regularization loss design,
- model experiments and ablation analysis,
- end-to-end inference integration,
- Python-based web demo development,
- result analysis and presentation.

## Repository Scope

This public repository is currently a **portfolio-oriented technical summary** of the capstone project.

Large model weights, generated media, external pretrained assets, and the complete development environment are not included here.

## Repository Structure

```text
musetalk-lipsync-improvement/
├── README.md
├── .gitignore
└── docs/
    ├── architecture.md
    ├── experiments.md
    └── demo.md
```

## Tech Stack

`Python` `PyTorch` `MuseTalk` `Whisper` `VAE` `Cross-Attention` `SyncNet` `Computer Vision` `Multimodal AI`
