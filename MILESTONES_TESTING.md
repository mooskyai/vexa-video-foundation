# Milestones Testing

Every milestone has an exit gate. Passing unit tests alone is not enough once model quality becomes part of the milestone.

## Global test layers

1. **Unit:** deterministic CPU tests; seconds, not minutes.
2. **Integration:** tiny training/inference runs; may use one GPU.
3. **Regression:** fixed seeds/configs/checkpoints and metric tolerances.
4. **Quality:** dataset-backed quantitative evaluation.
5. **Human:** blinded rubric-based comparison for subjective quality.
6. **Scale:** multi-GPU/resume/fault tests for later milestones.

## M0 gate

Required:

- `uv run pytest` passes on CPU;
- tokenizer round-trip passes for ASCII/UTF-8 byte sequences supported by the tokenizer contract;
- VAE output shape equals input shape for supported dimensions;
- DiT output shape equals latent shape;
- diffusion `add_noise` is deterministic with fixed inputs/noise;
- gradients reach trainable parameters;
- config loader rejects invalid dimensions;
- `vexa-video smoke` runs without downloading any model assets.

## M1 gate — synthetic motion

### Stage A — renderer and supervised temporal sanity — passed

Required before generative M1 training:

- same seed -> identical clip/caption/trajectory/control;
- cardinal right/left/down/up trajectory signs match requested controls;
- balanced direction/color coverage in the deterministic control cycle;
- train/validation/test seed spaces are disjoint;
- the `MotionProbe` backpropagates on CPU;
- `vexa-video m1-probe` evaluates on the held-out `test` seed space;
- direction accuracy >= 0.95;
- color accuracy >= 0.95;
- the probe checkpoint records config, optimizer, seed-space identifier and baseline/final metrics.

Recorded RTX 5050 Stage-A evidence:

```text
steps=300
baseline_direction_accuracy=0.246094
baseline_color_accuracy=0.250000
direction_accuracy=0.984375
color_accuracy=1.000000
gate_passed=True
```

The supervised probe is deliberately not used as evidence that the generative model follows controls.

### Stage B — generative synthetic motion — active

Correctness requirements:

- Stage-B uses one object, four directions, four colors, square/circle and fixed medium linear speed;
- VAE reconstruction is measured separately and must pass the foreground-aware reconstruction safety gate before diffusion starts;
- diffusion training predicts epsilon from noisy VAE latents conditioned through the project-owned byte tokenizer/text Transformer;
- text conditioning includes pooled text plus parameter-free token-level attention using the existing learned text projection;
- training compares the correct caption against one-word color and direction counterfactual captions and a null caption; no structured direction/color labels are fed into the denoiser;
- corrective training deliberately oversamples the high-noise half of the diffusion schedule so prompt semantics cannot be bypassed by relying only on nearly-clean visual latents;
- Video DiT receives explicit deterministic temporal/spatial patch positions;
- training reports reconstruction loss, diffusion loss, total loss, gradient norm, counterfactual/null losses, color/direction prompt-loss gaps and decoded semantic direction/color losses;
- Stage-B-v3 computes a differentiable soft-centroid motion loss on decoded predicted-clean video at a fixed noisy timestep, including a one-word direction-counterfactual target; semantic labels supervise the loss only and are not model inputs;
- reverse diffusion starts from Gaussian latent noise and is deterministic for fixed model/config/seed;
- final sampling uses the configured deterministic classifier-free guidance scale;
- generated output has shape `[B, 3, T, H, W]` and is decoded through `TinyVideoVAE`;
- `latest.pt` and `best.pt` contain VAE/text/DiT states, optimizer states, global/phase progress, config, seed, split/curriculum metadata, metrics and RNG/data-generator state;
- resume restores model/optimizer/progress/RNG state rather than silently restarting;
- CPU tests cover one optimization step, finite losses, gradients, sampler shape/determinism, checkpoint round-trip, resume and cardinal generated-video metric behavior.

Evaluation protocol:

- first record an untrained/random model using exactly the same generation/evaluation path;
- use 64 generated videos for the frozen final candidate protocol;
- balance the four directions and four colors;
- use fixed test-split prompts/seeds;
- use 50 deterministic reverse-diffusion steps for the final candidate;
- measure direction from generated temporal motion;
- measure color from generated RGB appearance;
- report direction accuracy, color accuracy, mean motion, static rate and confusion matrices.
- report the guidance scale and an explicit `gate_passed` result from the frozen thresholds.

Requested labels are ground truth only. The evaluator must infer its answer from generated pixels and temporal movement.

Frozen final M1 thresholds:

```text
direction_accuracy >= 0.75
color_accuracy     >= 0.75
static_rate        <= 0.10
mean_motion        > 0.02
```

Stage-B v1 recorded a random baseline of direction `0.265625`, color `0.250000`, mean motion
`0.009729`, static rate `1.000000`. Its trained `best.pt` reached direction `0.312500`, color
`0.265625`, mean motion `0.136491`, static rate `0.000000`: motion/static collapse improved, but
prompt control failed the frozen gate. Because the corrective v2 path changes text conditioning and
uses classifier-free guidance, it must record a fresh random baseline with the same 64-sample/50-step
v2 protocol before trained evaluation. The absolute thresholds above do not move. Training loss,
VAE reconstruction, visually interesting samples or the Stage-A probe cannot complete M1. M2
remains blocked until `gate_passed=True` on generated-video evidence and the result materially beats
the corresponding v2 random baseline. Stage-B-v2 trained evaluation reached direction `0.265625`,
color `0.703125`, mean motion `0.163834`, static rate `0.000000`; direction remained at chance, so v3
adds decoded semantic direction supervision while preserving the same frozen absolute gate.

## M2 gate — VAE

Correctness:

- encode/decode supports configured frame/resolution buckets;
- chunked/tiled output matches untiled output within tolerance;
- checkpoint round-trip is numerically stable.

Quality:

- reconstruction metrics beat the frozen M2 baseline;
- temporal reconstruction error is tracked separately from frame reconstruction;
- fast motion and scene cuts have dedicated validation slices.

## M3 gate — latent diffusion

- training loss decreases on a tiny overfit dataset;
- sampler is deterministic given seed/config/checkpoint;
- no NaN/Inf under supported mixed precision;
- generated latent statistics remain inside decoder-safe bounds;
- generated samples beat random-noise decode on frozen synthetic metrics.

## M4 gate — text conditioning

Use compositional synthetic prompts with held-out combinations.

Required metrics:

- color;
- object type;
- count;
- direction;
- speed bucket;
- action;
- camera motion where available.

The model must beat an unconditional checkpoint on prompt-conditioned tasks.

## M5 gate — scalable DiT

For every architecture candidate record:

- parameter count;
- training tokens/video frames seen;
- throughput;
- peak VRAM;
- loss curve;
- quality metrics;
- failure rate/NaN rate.

A larger model is not accepted if gains are not reproducible relative to increased compute.

## M6 gate — real-video data

- 100% of training assets have provenance status;
- duplicate/leakage report generated;
- train/validation/test manifests frozen and hashed;
- corrupt decode rate measured;
- clip duration/resolution/aspect distributions reported;
- removal procedure exists for a source that must be withdrawn.

## M7 gate — short real-world T2V

Evaluation set must separate:

- static scenes;
- camera motion;
- rigid object motion;
- articulated/human motion;
- multi-object interaction;
- lighting/weather changes;
- difficult textures.

Track prompt adherence, temporal consistency, artifact rate and blinded human preference.

## M8 gate — I2V / first-last frame

- first-frame reconstruction similarity;
- last-frame target similarity;
- motion smoothness between constraints;
- identity drift;
- failure under large pose/camera change;
- controllable motion-strength response.

## M9 gate — references

Test independently:

- one reference image;
- multiple views of same subject;
- multiple subjects;
- prop reference;
- environment reference;
- reference video.

Also test conflicting references and unrelated distractors.

## M10 gate — editing

Each edit task requires:

- changed-region accuracy;
- preserved-region stability;
- temporal boundary quality;
- instruction adherence;
- no unintended identity reset.

## M11 gate — audio representation

- reconstruction fidelity;
- bandwidth/latent rate;
- speech intelligibility evaluation;
- transient/event reconstruction;
- music/ambience slices;
- deterministic codec behavior.

## M12 gate — joint A/V

- synchronized event onset error;
- speech/visual timing metric where applicable;
- audio-content correspondence;
- video quality regression check against video-only model;
- audio quality regression check against audio-only baseline.

## M13 gate — 15–30 second generation

Measure metrics as a function of time, not only over the whole clip:

- identity drift at early/middle/late segments;
- prop persistence;
- scene geometry consistency;
- shot transition correctness;
- story-state adherence;
- extension continuity at boundaries;
- audio continuity.

A 30-second output with severe late-stage drift fails the milestone.

## M14 gate — 720p/1080p

- true output resolution verified;
- high-frequency detail improves versus upscaled lower-resolution baseline;
- temporal flicker does not regress beyond the frozen tolerance;
- tiled inference has no visible seams in evaluation crops;
- text/UI rendering gets a dedicated legibility suite.

## M15 gate — director controls

Every control must have a measurable response curve or discrete success metric. Controls that are accepted syntactically but ignored by the model do not count.

## M16 gate — efficient inference

Compare against the undistilled/unquantized project checkpoint:

- latency;
- throughput;
- peak VRAM/RAM;
- quality deltas by evaluation slice;
- determinism/reproducibility;
- supported hardware.

## M17 gate — frontier comparison

Before making parity claims:

- freeze prompt/reference test sets;
- document competitor/model versions and evaluation date;
- use identical output constraints where possible;
- run blinded human comparisons;
- report per-dimension results, confidence/uncertainty, and known mismatches;
- avoid collapsing all capability into one opaque score.

The release report must distinguish measured facts from interpretation.

### Stage-B-v5 corrective evidence

The v4 frozen test result remained a failure: direction `0.156250`, color `0.781250`, mean motion `0.131964`, static rate `0.015625`. A separate validation-split sweep showed direction accuracy `1.0` at 4/6 steps, `0.953125` at 8, `0.421875` at 12, and near chance from 16 to 50. v5 must therefore preserve direction through the full 50-step path. Training periodically differentiates through 50 CFG-DDIM steps, and checkpoint selection includes 16 fixed validation-split 50-step generations. The final M1 gate is still the frozen 64-sample test protocol and M2 remains blocked.
