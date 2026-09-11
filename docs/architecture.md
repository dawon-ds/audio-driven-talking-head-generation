# Architecture

## Baseline

The project used **MuseTalk v1.0** as the baseline for audio-driven talking-head generation.

The improvement stage focused on the mouth area rather than redesigning the complete generation pipeline.

## Lip-Centric Refinement

A **96×96 lip ROI** was used as the primary refinement target.

The experimental design combined:

- reference visual lip information,
- Whisper-based audio features,
- latent image representation from a frozen VAE,
- cross-attention between visual and audio features,
- a residual latent update predicted by the ADLip Generator.

The core residual formulation was:

```text
Z_out = Z_ref + αΔ
```

where `Δ` represents an audio-conditioned latent adjustment for the lip region.

## Cross-Attention

The LipCrossAttention component used:

- visual latent features as **query**,
- audio features as **key/value**.

This structure was intended to let the lip representation selectively attend to temporally relevant speech information.

## Synchronization Supervision

A frozen SyncNet-style model was used as an auxiliary supervision mechanism.

Matched and mismatched audio-video pairs were used to provide contrastive synchronization guidance, encouraging stronger correspondence between generated lip motion and the speech signal.

## Training Losses

The project materials describe the following objectives:

- latent L1 loss,
- silence loss,
- delta regularization,
- SyncNet synchronization loss,
- TTA-related synchronization supervision.

The exact weight configuration is not reproduced here unless directly supported by the archived project materials.
