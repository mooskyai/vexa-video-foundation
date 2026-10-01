# Synthetic Motion Experiment Record

This document preserves the research trail that led to the canonical synthetic-motion trainer. Version labels below describe experiments only; they are not runtime architecture or public API concepts.

## Frozen gate

The generated-video exit criteria were fixed throughout the corrective work:

- direction accuracy >= 0.75;
- color accuracy >= 0.75;
- static rate <= 0.10;
- mean motion > 0.02;
- no obvious class collapse;
- materially outperform the same-protocol random baseline.

The authoritative protocol uses 64 balanced test samples, fixed seeds, classifier-free guidance 3.0, and 50 deterministic DDIM steps.

## Experiment history

| Experiment | Direction | Color | Mean motion | Static rate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial generative path | 0.312500 | 0.265625 | 0.136491 | 0.000000 | Failed prompt control |
| Conditioning correction | 0.265625 | 0.703125 | 0.163834 | 0.000000 | Direction remained at chance |
| Decoded semantic correction | 0.250000 | 0.781250 | 0.139715 | 0.000000 | Color/motion passed; direction failed |
| Short sampler-aligned direction rollout | 0.156250 | 0.781250 | 0.131964 | 0.015625 | Short-horizon direction did not survive 50 steps |
| First full-horizon correction | 1.000000 | 0.343750 | 0.044545 | 0.156250 | Direction solved; color/static regressed |
| Balanced full-horizon correction | 0.828125 | 1.000000 | 0.066824 | 0.000000 | Frozen gate passed |

## Horizon diagnosis

The short-rollout experiment showed that direction control existed but was overwritten later in denoising. On the validation split, direction accuracy was 1.0 at 4 and 6 DDIM steps, 0.953125 at 8, 0.421875 at 12, and near chance from 16 through 50. Color followed the opposite trend and improved over longer trajectories.

That diagnosis motivated the final balanced objective: periodic differentiation through the exact 50-step sampling path, with direction preservation, caption-color preservation, and a stronger minimum-motion floor. Checkpoint selection uses fixed validation-split generations and normalized deficits against all four frozen gates.

## Accepted evidence

Accepted checkpoint frozen evaluation:

```text
direction_accuracy=0.828125
color_accuracy=1.000000
mean_motion=0.066824
static_rate=0.000000
direction_confusion=[[16,0,0,0],[2,13,1,0],[0,0,16,0],[8,0,0,8]]
color_confusion=[[16,0,0,0],[0,16,0,0],[0,0,16,0],[0,0,0,16]]
gate_passed=True
```

Same-protocol random baseline:

```text
direction_accuracy=0.250000
color_accuracy=0.250000
mean_motion=0.012409
static_rate=1.000000
gate_passed=False
```

The upward direction remains the weakest class in the accepted checkpoint, which should be tracked in later scaling work, but the frozen aggregate gate is satisfied without lowering thresholds.
