# Model Card — Development Line

## Status

No frontier model checkpoint is included in this starter kit. The code initializes all neural components randomly.

## Intended use

Research and development of a from-scratch generative video foundation model.

## Current capability

Milestone 1 is active. Stage A passed its supervised renderer sanity gate. The first Stage-B GPU
run also established a useful partial result: VAE validation reconstruction reached `0.024338`,
generated mean motion increased from the random baseline `0.009729` to `0.136491`, and static rate
fell from `1.0` to `0.0`. However direction accuracy was only `0.312500` and color accuracy only
`0.265625`, so controlled generation did not pass. The active Stage-B corrective keeps the
project-owned byte/text path, adds token-level text attention, text counterfactual/null objectives
and deterministic classifier-free guidance. Stage-B-v2 then reached color `0.703125`, mean motion
`0.163834` and static rate `0.0`, but direction remained at chance (`0.265625`). The active Stage-B-v3
correction adds decoded predicted-clean direction/color semantic losses while keeping caption text as the
only model conditioning input. M1 remains incomplete and M2 remains blocked until the frozen
generated-video gate passes.

## Not currently supported

Do not describe the starter as supporting production text-to-video, photorealistic humans, 30-second narratives, native audio, 1080p, editing or multimodal references. Those are roadmap targets.

## Training data

The development line uses deterministic synthetic clips only. M1 train/validation/test samples occupy disjoint deterministic seed spaces; future real-data checkpoints must link to a versioned provenance manifest.
