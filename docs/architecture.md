# Architecture

## MuseTalk-Based Pipeline

The project uses MuseTalk as the base for audio-driven talking-head generation. The end-to-end pipeline includes:

- Whisper-based audio feature extraction,
- VAE latent encoding and decoding,
- audio-conditioned generation,
- face detection and DWPose preprocessing,
- face parsing and blending,
- SyncNet / SyncLT-based synchronization supervision,
- normal and realtime inference.

Inference utilities support both MuseTalk v1.0 and v1.5 model layouts.

## Lip-Centric Architecture

The lip-centric model targets a **96×96 lip ROI** and performs audio-conditioned refinement in latent space.

Reference lip frames are encoded by a frozen VAE to obtain the reference latent `Z_ref`. An audio segment is processed by the Whisper Tiny encoder to obtain audio features. These representations are passed to the ADLip Generator, which predicts an audio-driven latent change `Δ`.

The ADLip Generator consists of:

- **Spatial Cross-Attention** between the reference visual latent and audio features,
- **Temporal Attention** across the lip sequence,
- a **Feed-Forward Network** for latent refinement.

The residual latent update is:

```text
Z_out = Z_ref + αΔ
```

where `Δ` is the predicted audio-conditioned latent change and `α` controls the update scale. `Z_out` is decoded by the frozen VAE decoder to generate the lip video, which is then pasted back into the face.

## Spatial Cross-Attention

Spatial Cross-Attention uses:

- the reference visual latent `Z_ref` as the **query**,
- Whisper audio features as **key/value** inputs.

This allows the visual latent representation to be conditioned directly on the speech features before Temporal Attention models dependencies across the lip-frame sequence.

## Synchronization Supervision

The project uses synchronization supervision to improve audio-lip alignment. SyncLT receives the generated lip video together with matched positive audio and mismatched negative audio segments and applies a contrastive synchronization objective.

This encourages the generated lip motion to align more strongly with the corresponding speech while distinguishing mismatched audio-video pairs.

## Training Objectives

The lip-centric architecture shown in `images/architecture.png` uses:

- **Latent Reconstruction Loss**,
- **Same Identity Loss**,
- **Different Identity Loss**,
- **Delta Regularization**,
- **SyncLT Contrastive Loss**.

These objectives are combined into the total loss. Backpropagation updates the trainable lip-generation and synchronization components while the VAE encoder and decoder remain frozen.

## Architecture Flow

```text
Reference Lip Frames                         Audio Segment
        ↓                                         ↓
Frozen VAE Encoder                       Whisper Tiny Encoder
        ↓                                         ↓
      Z_ref                                 Audio Feature
        └──────────────────┬──────────────────────┘
                           ↓
                    ADLip Generator
              Spatial Cross-Attention
                           ↓
                  Temporal Attention
                           ↓
                 Feed-Forward Network
                           ↓
                           Δ
                           ↓
                    Scale by α
                           ↓
                Z_out = Z_ref + αΔ
                           ↓
                  Frozen VAE Decoder
                           ↓
                 Generated Lip Video
                           ↓
                    Paste Back to Face

Generated Lip Video + Positive/Negative Audio
                           ↓
                         SyncLT
                           ↓
                   Contrastive Loss
```
