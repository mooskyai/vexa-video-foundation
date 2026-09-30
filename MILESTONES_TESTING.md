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

Dataset tests:

- same seed -> identical clip/caption/trajectory;
- split determinism;
- ground-truth trajectory agrees with rendered frames;
- no train/test seed overlap.

Model quality tests:

- direction accuracy above a documented baseline;
- object/color/count accuracy measured from ground-truth synthetic renderer metadata;
- temporal trajectory error improves materially over an untrained/random baseline;
- generated motion does not collapse to static frames.

Metric thresholds are set only after baseline runs and then frozen in the milestone report.

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
