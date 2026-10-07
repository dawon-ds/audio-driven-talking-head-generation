# Experiments

## Baseline Validation

The team validated end-to-end **MuseTalk v1.0** inference in the project environment. One recorded sample required approximately **seven hours** for the full process. The record covers a single sample and does not establish general throughput.

## Lip-Centric Architecture Experiment

The project records describe a **96×96 lip ROI**, an ADLip Generator for audio-conditioned latent refinement, and SyncLT supervision with matched and mismatched audio segments.

The public source tree contains MuseTalk UNet training and inference with SyncNet utilities. The ADLip Generator and SyncLT extension are documented in [architecture.md](architecture.md), but their implementation and experiment checkpoints are not included here.

## Results & Evaluation

The documented outcomes are baseline inference validation, the lip-centric architecture experiment, and a Gradio preview workflow.

The available records do not provide a quantitative baseline-versus-refined lip-synchronization comparison. They therefore do not establish a measured synchronization gain or a reproducible numerical ablation result.

## Limitations & Future Work

A future evaluation should compare baseline and refined outputs using consistent synchronization metrics, inputs, and runtime conditions. Reproducing the extension also requires the missing implementation, checkpoints, and original experiment settings.

The current repository is a public MuseTalk implementation and project documentation; it does not reproduce the complete original team experiment.
