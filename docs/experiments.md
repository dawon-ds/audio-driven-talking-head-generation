# Experiments

## Baseline Validation

The first stage of the capstone established a working **MuseTalk v1.0** baseline and confirmed end-to-end inference behavior in the project environment.

One recorded sample required approximately **7 hours** for the full inference process, highlighting both computational cost and the practical need for a more focused workflow.

## Lip-Centric Refinement

The project then implemented a **lip-centric** architecture centered on the 96×96 mouth region.

The model preserves the reference visual representation while learning an audio-conditioned residual update in latent space. The ADLip Generator combines Spatial Cross-Attention, Temporal Attention, and a Feed-Forward Network to predict the latent change applied to the reference lip representation.

## Synchronization Experiments

Audio-visual synchronization supervision was integrated through SyncNet / SyncLT-style training. The setup uses matched and mismatched audio-video pairs so that the synchronization component can distinguish aligned lip motion from incorrect audio pairings.

The resulting synchronization loss is combined with the lip-generation objectives during training.

## Evaluation

The capstone presentation included:

- an ablation comparison,
- generated result examples,
- qualitative lip-sync inspection,
- a web-demo demonstration.

This portfolio repository does not reproduce numeric experimental values that are not available in a stable source artifact.

## Interpretation

The experiments were model-improvement work built on top of the MuseTalk baseline rather than evidence of a universally superior replacement for MuseTalk.

The main experimental contribution was implementing and testing localized lip-region latent refinement together with explicit synchronization supervision while keeping pretrained components such as the VAE frozen.
