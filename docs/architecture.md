# Architecture

## Implemented MuseTalk Pipeline

The repository contains the MuseTalk-based training and inference pipeline used for audio-driven talking-head generation, including:

- Whisper-based audio feature extraction,
- VAE latent encoding and decoding,
- audio-conditioned UNet generation,
- face detection and DWPose preprocessing,
- face parsing and blending,
- SyncNet-based synchronization supervision,
- normal and realtime inference.

Inference utilities support both MuseTalk v1.0 and v1.5 model layouts.

## Lip-Centric Refinement Design

In addition to the implemented MuseTalk pipeline, the project explored a lip-centric refinement design targeting a **96×96 lip ROI**.

The experimental design combined:

- reference visual lip information,
- Whisper-based audio features,
- latent image representation from a frozen VAE,
- cross-attention between visual and audio features,
- a residual latent update predicted by an ADLip Generator concept.

The proposed residual formulation was:

```text
Z_out = Z_ref + αΔ
```

where `Δ` represents an audio-conditioned latent adjustment for the lip region.

## LipCrossAttention Design

The proposed LipCrossAttention component used:

- visual latent features as **query**,
- audio features as **key/value**.

This design was intended to let the lip representation attend to temporally relevant speech information.

## Synchronization Design

The implemented training code includes SyncNet-based synchronization losses. Project materials also explored an additional frozen SyncNet-style contrastive supervision design using matched and mismatched audio-video pairs.

## Training Objectives

The repository training code supports reconstruction, perceptual, adversarial, feature-matching, and synchronization-related losses through the project training configurations and loss modules.

The lip-centric refinement materials additionally describe objectives such as latent L1 loss, silence loss, delta regularization, and TTA-related synchronization supervision. These are documented here as experimental design elements and should not be interpreted as standalone implemented modules unless corresponding code is present in the repository.
