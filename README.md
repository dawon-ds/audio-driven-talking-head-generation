# Audio-Driven Talking-Head Generation

**MuseTalk Lip-Sync Improvement · Capstone Project · 2026**

This repository documents a team capstone project focused on improving lip synchronization in **audio-driven talking-head generation** using MuseTalk as the baseline system.

The project explored a lip-centric refinement strategy around a **96×96 mouth ROI**, combining an additional lip generator with audio-visual synchronization supervision, while also building an end-to-end inference flow and a Python-based web demo.

## Project Overview

- **Period:** March–June 2026
- **Team:** 4 members
- **Base model:** MuseTalk
- **Task:** Audio-driven talking-head generation
- **Focus:** Lip-sync improvement and inference usability
- **Core components:** MuseTalk baseline, Lip-Centric refinement, ADLip Generator, SyncNet supervision, end-to-end inference, web demo

## Problem

MuseTalk can generate high-quality talking-head videos, but the project focused on a remaining challenge: improving the alignment between **mouth motion and input audio**.

Rather than modifying the full face-generation pipeline, the team concentrated on the mouth region and tested a lip-centric refinement strategy designed to make local mouth motion more responsive to audio features.

## Baseline

The project first established and validated a **MuseTalk v1.0 inference baseline**.

In the project environment, one sample end-to-end inference run required approximately **7 hours**, which also motivated work on a more practical inference workflow and demo interface.

## Lip-Centric Improvement

The proposed refinement stage focused on a **96×96 lip ROI**.

Conceptually, the refinement module receives:

- a reference lip frame,
- a corresponding audio segment,
- latent visual features from the baseline generation pipeline.

The **ADLip Generator** predicts a latent lip-motion adjustment, represented as `Δ`, and the refined latent is formed as:

```text
Z_out = Z_ref + αΔ
```

This residual formulation was intended to preserve the reference visual structure while allowing audio-conditioned mouth movement to be adjusted locally.

## Model Architecture

The experimental architecture used:

- **Frozen VAE** for latent image representation,
- **Whisper audio features**,
- **LipCrossAttention** as the trainable audio-visual fusion component,
- visual latent features as the query,
- audio features as key/value,
- residual latent lip update through the ADLip Generator.

The overall goal was to isolate and train the lip-synchronization component without unnecessarily updating the full image-generation backbone.

## Training Objectives

The project materials describe a combination of losses including:

- latent L1 reconstruction loss,
- silence loss,
- delta regularization,
- SyncNet-based synchronization loss,
- TTA-related synchronization supervision.

### SyncLT / SyncNet Supervision

A frozen SyncNet-style model was used to provide audio-visual synchronization guidance through matched and mismatched audio-video pairs.

This supervision was designed to encourage generated mouth motion to better correspond to the input speech signal.

## Inference Pipeline

The project integrated an end-to-end flow from:

```text
Reference Image + Audio
        ↓
MuseTalk Baseline Inference
        ↓
Lip-Centric Refinement
        ↓
Generated Talking-Head Video
        ↓
Web Demo Preview
```

The final demo allowed users to run inference and review the generated result in a Python-based interface.

## My Contribution

My verifiable responsibilities in the project included:

- setting up and validating the MuseTalk baseline,
- integrating the end-to-end inference flow,
- developing the Python-based web demo,
- analyzing experiment results,
- preparing presentation materials and participating in the midterm presentation.

The ADLip / SyncNet-based model design is documented here as a **team project result** rather than claimed as an individual implementation.

## Repository Scope

This public repository is currently a **portfolio-oriented technical summary** of the capstone project.

Large model weights, generated media, external pretrained assets, and the complete team development environment are not included here.

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

## Notes

- This was a **4-person capstone project**.
- The repository documents the overall verified technical pipeline and the repository owner's confirmed responsibilities.
- Team-level components are not presented as individually implemented unless the available project records support that claim.
