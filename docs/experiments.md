# Experiments

## Baseline Validation

The first stage of the capstone was to establish a working **MuseTalk v1.0** baseline and confirm end-to-end inference behavior in the project environment.

One recorded sample required approximately **7 hours** for the full inference process, highlighting both computational cost and the practical need for a streamlined workflow.

## Refinement Direction

The project then explored a **lip-centric** strategy centered on the 96×96 mouth region.

The refinement idea was to preserve most of the baseline visual representation while learning a localized residual adjustment conditioned on speech audio.

## Synchronization Experiments

Audio-visual synchronization supervision was introduced with a frozen SyncNet-style model. The training setup used matched and mismatched pairs so that the refinement module could learn whether a lip sequence corresponded to the input audio.

## Evaluation

The capstone presentation included:

- an ablation comparison,
- generated result examples,
- qualitative lip-sync inspection,
- a web-demo demonstration.

This portfolio repository does not reproduce numeric experimental values that are not available in a stable source artifact.

## Interpretation

The experiments were treated as exploratory model-improvement work rather than evidence of a universally superior replacement for MuseTalk.

The main contribution of the experiment phase was testing whether local lip-region refinement and explicit synchronization supervision could improve the behavior of the baseline generation pipeline while leaving most pretrained components frozen.
