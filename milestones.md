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

**Status:** active.

**Goal:** prove temporal learning before real video.

M1 starts with a supervised sanity gate while keeping the M0 tokenizer/VAE/DiT/diffusion architecture intact. Stage A uses balanced cardinal controls (right/left/down/up), four colors, square/circle shapes, speed buckets, deterministic captions/trajectories and disjoint split seed spaces. A tiny project-trained motion probe must recover direction and color at >= 0.95 accuracy before generative optimization begins.

Dataset curriculum then expands through:

- moving shapes;
- acceleration and bounce;
- rotation/scale;
- occlusion;
- multiple objects;
- simple depth ordering;
- camera pan/zoom simulation;
- deterministic captions and ground-truth trajectories.

Exit result: generated clips show measurable motion learning rather than independent-frame noise. The supervised probe is a prerequisite, not the M1 completion criterion.

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
