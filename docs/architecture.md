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

The experimental design combines:

- reference lip frames encoded into a latent representation by a frozen VAE,
- Whisper-based audio features,
- an ADLip Generator that predicts an audio-driven latent change,
- Spatial Cross-Attention between the reference latent and audio features,
- Temporal Attention across the lip sequence,
- a residual latent update followed by frozen VAE decoding.

The proposed residual formulation is:

```text
Z_out = Z_ref + αΔ
```

where `Δ` represents the predicted audio-driven latent change and `α` controls its scale.

## Spatial Cross-Attention Design

The proposed Spatial Cross-Attention component uses:

- the reference visual latent `Z_ref` as the **query**,
- Whisper audio features as **key/value** inputs.

This is followed by Temporal Attention and a feed-forward network inside the ADLip Generator.

## Synchronization Design

The implemented MuseTalk training code includes SyncNet-based synchronization supervision.

The experimental refinement design additionally uses a SyncLT-style contrastive objective with:

- generated lip video,
- matched positive audio,
- mismatched negative audio.

The synchronization network is used to encourage stronger audio-lip alignment through contrastive supervision.

## Experimental Training Objectives

The final lip-centric design shown in `images/architecture.png` includes:

- **Latent Reconstruction Loss**,
- **Same Identity Loss**,
- **Different Identity Loss**,
- **Delta Regularization**,
- **SyncLT Contrastive Loss**.

These losses are combined into the total training objective and backpropagated to update the trainable refinement components.

The diagram documents the project's proposed refinement architecture. The current repository primarily contains the MuseTalk training/inference implementation and supporting modules, so experimental components should not be interpreted as standalone implemented modules unless corresponding source code is present.
