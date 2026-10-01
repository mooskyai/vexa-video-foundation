# Milestones

This roadmap is ordered by dependency, not marketing value. Each milestone must pass the corresponding gate in `MILESTONES_TESTING.md` before the next milestone is treated as active.

## M0 — Reproducible research foundation

**Goal:** deterministic Python 3.14/uv project with tiny end-to-end tensor path.

Deliverables:

- byte tokenizer;
- from-scratch text Transformer;
- tiny Video VAE;
- tiny Video DiT;
- DDPM schedule;
- deterministic synthetic clips;
- CLI doctor/smoke commands;
- checkpoint metadata contract;
- CPU unit tests.

## M1 — Synthetic motion world

**Status:** active. Stage A passed; Stage B active; M1 incomplete; M2 blocked.

**Goal:** prove controlled temporal generation before real video.

### Stage A — complete

The deterministic renderer exposes cardinal motion, four colors, square/circle shapes and speed buckets with disjoint split seed spaces. The project-trained `MotionProbe` passed the frozen held-out gate on the RTX 5050 at 300 steps:

```text
baseline_direction_accuracy=0.246094
baseline_color_accuracy=0.250000
direction_accuracy=0.984375
color_accuracy=1.000000
gate_passed=True
```

This only establishes renderer/label learnability.

### Stage B — active

The first generative curriculum is deliberately narrower: one object, four directions, four colors, square/circle, constant medium linear velocity, static camera, 8 frames and 32x32 RGB. The existing `TinyVideoVAE`, `TransformerTextEncoder`, `VideoDiT` and diffusion schedule remain the model path. Stage B adds:

- foreground-aware VAE reconstruction warmup and a safety gate before diffusion training;
- deterministic 3D Fourier patch positions in `VideoDiT` so temporal/spatial tokens are distinguishable;
- epsilon-prediction training with reconstruction/diffusion/total loss and gradient norm tracking;
- deterministic reduced-step reverse diffusion from Gaussian latent noise;
- resumable composite checkpoints with VAE/text/DiT/optimizer/RNG/config/dataset state;
- generated-RGB direction, color, mean-motion, static-rate and confusion-matrix evaluation;
- a frozen random-model evaluation path using the same prompts/seeds/protocol as trained checkpoints.

The first GPU baseline proved motion learning but not prompt adherence. Random -> trained metrics were
direction `0.265625 -> 0.312500`, color `0.250000 -> 0.265625`, mean motion
`0.009729 -> 0.136491`, static rate `1.000000 -> 0.000000`. Stage B therefore remains active.

The corrective conditioning experiment stays inside Stage B and adds:

- parameter-free token-level attention in the existing projected text space, preserving Stage-B v1 checkpoint compatibility;
- correct-vs-color-counterfactual and correct-vs-direction-counterfactual text objectives so the denoiser cannot minimize epsilon loss while ignoring prompt words;
- an explicit null-caption denoising objective for classifier-free guidance;
- high-noise timestep oversampling during corrective training so captions are useful when the latent itself is ambiguous;
- deterministic classifier-free guided sampling (`guidance_scale=3.0` in `tiny.toml`);
- held-out color/direction prompt-gap penalties in `best.pt` selection;
- frozen generated-video thresholds in configuration and an explicit `gate_passed` metric.

No structured control labels are fed to the denoiser; captions remain the conditioning interface.
The v2 random baseline must be recorded before corrective training because the generation protocol
changed. The frozen absolute gate remains direction >= 0.75, color >= 0.75, static <= 0.10 and mean
motion > 0.02. Complete generated videos are authoritative.

Stage-B-v2 improved color accuracy to `0.703125` and retained strong motion (`0.163834`, static rate
`0.000000`) but direction stayed at chance (`0.265625`). Stage-B-v3 remains inside M1 and adds a
decoded predicted-clean semantic objective: soft-centroid motion must follow the caption direction, and
a direction-counterfactual caption must reverse/reorient motion under the same noisy latent. A smaller
decoded color objective protects the nearly-passing color control. This changes training loss only;
caption text remains the sole conditioning input and the v2 checkpoint remains model/optimizer compatible.

The broader synthetic curriculum (bounce, acceleration, rotation/scale, occlusion, multiple objects and camera motion) remains deferred until this narrow generator works.

## M2 — Video autoencoder v1

**Goal:** learn a compact latent video representation.

Deliverables:

- temporal/spatial compression;
- reconstruction training;
- reconstruction metrics;
- tiled/chunked decode path;
- versioned latent format.

## M3 — Latent video diffusion baseline

**Goal:** unconditional latent video generation.

Deliverables:

- diffusion/flow objective abstraction;
- latent denoiser training;
- sampler;
- EMA evaluation weights if experiments support them;
- reproducible sample grids.

## M4 — Text-conditioned video generation

**Goal:** train text and video representations jointly enough to follow synthetic prompts.

Prompt curriculum includes color, shape, count, direction, speed, action and camera language.

Deliverables:

- token-level text conditioning;
- classifier-free conditioning path or alternative;
- prompt adherence evaluation on synthetic ground truth.

## M5 — Video DiT v2 and scaling laws

**Goal:** replace toy architecture decisions with a scalable DiT.

Research tracks:

- temporal/spatial factorization;
- efficient attention;
- learned/rotary positional encoding;
- adaptive normalization/modulation;
- gradient checkpointing;
- Flash/SDPA paths implemented through PyTorch primitives;
- model sizes: tiny, small, base and large research tiers.

Record scaling curves for loss, motion metrics, data and compute.

## M6 — Licensed real-video corpus v1

**Goal:** move from synthetic semantics to real visual dynamics.

Deliverables:

- provenance-first manifest format;
- decode/clip sampling pipeline;
- deduplication and split policy;
- caption/script metadata strategy;
- aspect/resolution bucketing;
- data quality filters that do not require third-party pretrained AI models.

## M7 — 256/512-class text-to-video

**Goal:** useful short real-world clips.

Focus:

- 2–5 second clips;
- subject persistence;
- camera motion;
- human/object motion quality;
- prompt alignment;
- reduced flicker and texture drift.

## M8 — Image-to-video + first/last-frame control

**Goal:** controllable generation from user-owned visual references.

Deliverables:

- image reference encoder trained from scratch;
- first-frame conditioning;
- first/last-frame conditioning;
- motion-strength controls;
- identity/appearance preservation evaluation.

## M9 — Reference-to-video and identity state

**Goal:** multiple reference assets with persistent subjects/props/scenes.

Deliverables:

- typed reference slots;
- multi-image references;
- reference video conditioning;
- subject/prop identity tokens;
- scene-state memory representation;
- conflict-resolution policy when references disagree.

## M10 — Video editing model

**Goal:** generation and editing share one foundation stack.

Tasks:

- masked inpainting;
- object replacement/removal;
- background/lighting/style edits;
- motion edits;
- video-to-video transformation;
- prompt-controlled continuation/extension.

## M11 — Native audio representation

**Goal:** train project-owned audio latent/token representation.

Deliverables:

- audio autoencoder/codec or spectrogram latent model;
- dialogue/ambient/effect/music dataset representation;
- timestamp alignment to video;
- audio reconstruction and generation tests.

## M12 — Joint audio-video generation

**Goal:** native synchronized A/V rather than post-hoc soundtrack generation.

Deliverables:

- shared timeline conditioning;
- cross-modal attention/modulation;
- event synchronization losses;
- dialogue/lip timing research track;
- sound-event correspondence evaluation.

## M13 — Multi-shot and long-form state

**Goal:** coherent 15–30 second narratives.

Deliverables:

- shot/timeline representation;
- persistent story state;
- transition modeling;
- chunk overlap or global refinement strategy;
- extension without subject reset;
- multi-round continuation.

The project should demonstrate 30-second generation only after continuity metrics pass, not merely because the model can allocate that many frames.

## M14 — Production resolution

**Goal:** 720p, then 1080p output.

Deliverables:

- high-resolution latent/video decoder;
- project-trained temporal super-resolution/refinement if used;
- tiled inference;
- memory-bounded decode;
- fine-detail and text/UI rendering research track;
- perceptual quality evaluation that does not hide temporal regressions.

## M15 — Director controls and multimodal production input

**Goal:** rich control comparable to modern production-oriented video systems.

Planned controls:

- camera path and lens language;
- actor blocking/trajectory;
- timing and shot boundaries;
- masks/layout/depth/pose where generated or provided without external model dependencies;
- image/video/audio reference combinations;
- dialogue script and audio reference;
- edit/extend/regenerate-region workflows.

## M16 — Efficient inference variants

**Goal:** make our own model family deployable.

Deliverables:

- distillation using only our teacher checkpoints;
- quantization calibration from approved data;
- CPU/offload paths;
- memory/latency profiles;
- short-clip local edition;
- server/distributed edition.

## M17 — Frontier evaluation and release gates

**Goal:** compare capability classes without claiming unsupported parity.

Evaluation dimensions:

- visual fidelity;
- prompt adherence;
- motion/physics;
- temporal identity consistency;
- reference fidelity;
- editing accuracy;
- audio quality and synchronization;
- long-form story continuity;
- camera/director control;
- text/UI rendering;
- inference efficiency;
- human preference.

A 'Seedance 2.5/Wan 3.0 level' claim is allowed only after a documented, reproducible evaluation suite supports the relevant dimensions. Capability parity is multidimensional; one attractive demo is not parity.

Stage-B-v5 remains inside M1. The v4 horizon sweep localized the remaining failure to late denoising: direction is nearly perfect through 8 DDIM steps and collapses by 12-16, while color improves over longer trajectories. v5 adds periodic 50-step differentiable direction preservation and full-horizon validation-generation checkpoint selection. M2 stays blocked until the unchanged frozen gate reports `gate_passed=True`.
