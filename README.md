# Vexa Video Foundation

A from-scratch PyTorch research codebase for building a generative video foundation model without third-party AI model weights or inference APIs.

The long-term capability target is the class of workflows publicly demonstrated by current frontier systems such as Seedance 2.5 and Wan 3.0: long-form video, native audio-video generation, multimodal references, editing, extension, strong temporal consistency, multi-shot storytelling, and production-resolution output. **This repository is a starter research stack, not a claim of current parity.** Reaching frontier quality requires large licensed datasets, distributed training infrastructure, substantial experimentation, and repeated model scaling.

## Non-negotiable project rules

- Python **3.14**.
- Package and environment management with **uv**.
- PyTorch is the core ML framework.
- No pretrained video model, image model, text encoder, VAE, audio codec, CLIP/T5/BERT model, or third-party inference API is required by the architecture.
- Foundation components are trained from random initialization unless a future experiment explicitly opts into a separately documented baseline.
- Training data must be owned, licensed, public-domain, synthetic, or otherwise permitted for the intended use.
- Every milestone has tests and an exit gate in `MILESTONES_TESTING.md`.

## What is included now

The starter kit contains executable skeletons for:

- deterministic synthetic video generation with balanced cardinal direction/color controls;
- a tiny from-scratch byte tokenizer;
- a trainable Transformer text encoder;
- a 3D convolutional Video VAE;
- a latent Video DiT with temporal/spatial patching;
- a DDPM-style noise schedule;
- training utilities and checkpoint format;
- a CLI smoke test and supervised motion-sanity probe;
- CPU unit tests that validate shapes, determinism, gradients, and diffusion math;
- architecture and scaling plans through the frontier-capability milestones.

The initial model is intentionally tiny. Its purpose is to prove the pipeline before scaling compute.

## Repository layout

```text
vexa-video-foundation/
├── src/vexa_video/
│   ├── data/              # datasets and controlled synthetic motion
│   ├── diffusion/         # forward/reverse diffusion primitives
│   ├── inference/         # samplers and generation entry points
│   ├── models/            # tokenizer, text encoder, VAE, Video DiT
│   ├── training/          # trainer, evaluation, probe, checkpoints
│   ├── utils/             # determinism and video utilities
│   ├── cli.py
│   └── config.py
├── configs/               # tiny/small/frontier configuration examples
├── scripts/               # Windows-friendly development commands
├── tests/
├── ARCHITECTURE.md
├── CODING_GUIDELINE.md
├── MILESTONES_TESTING.md
├── milestones.md
└── DATA_POLICY.md
```

## Setup on Windows 11 with uv

Install/pin Python 3.14 and create the environment:

```powershell
uv python install 3.14
uv python pin 3.14
uv sync --group dev
```

Verify:

```powershell
uv run python --version
uv run vexa-video doctor
uv run pytest
```

For an NVIDIA CUDA build of PyTorch, use the PyTorch wheel index appropriate for the CUDA build installed on your machine. For the CUDA 13.2 setup used by this project during development, a typical command is:

```powershell
uv add torch --index pytorch-cu132=https://download.pytorch.org/whl/cu132
```

If the exact wheel/index changes, follow the current PyTorch installation matrix rather than weakening the Python 3.14 project requirement.

## First smoke run

```powershell
uv run vexa-video smoke --config configs/tiny.toml
```

Expected behavior: the command builds the text encoder, Video VAE, Video DiT and diffusion schedule from random initialization, runs a forward/backward-capable pass, and prints tensor shapes plus parameter counts. It does **not** produce a meaningful video yet.

Generate a deterministic synthetic training clip as a tensor checkpoint:

```powershell
uv run vexa-video synth --output outputs/sample.pt --frames 16 --size 64 --seed 42
```

## Synthetic motion training

The current development baseline is one canonical synthetic-motion training path. Historical v2/v3/v4/v5 experiment labels are not part of the runtime architecture; their results are preserved in `docs/experiments/synthetic-motion.md`.

The active curriculum is intentionally narrow: one object, four cardinal directions, four colors, square/circle, fixed medium linear speed, static camera, 8 frames at 32x32. Training uses the project-owned byte tokenizer, Transformer text encoder, TinyVideoVAE, VideoDiT, classifier-free-guided DDIM sampling, caption counterfactuals, decoded semantic losses, and balanced full-horizon direction/color preservation.

The accepted 64-sample, 50-step frozen evaluation is:

```text
direction_accuracy=0.828125
color_accuracy=1.000000
mean_motion=0.066824
static_rate=0.000000
gate_passed=True
```

Manual review of a generated MP4 exposed a quality gap that the original four metrics did not
measure: a clip can move in the requested direction and retain the requested color while the object
is diffuse, deformed, or disappears across frames. M1 is therefore reopened before M2. The evaluator
now also reports shape accuracy, object-like frame rate, persistent-video rate, foreground-area
ratio, and a shape confusion matrix. These new measurements are diagnostic first; their final exit
thresholds are frozen only after recording the current checkpoint and same-protocol random baseline.

`gate_passed=True` currently means the original direction/color/motion/static control gate passed.
It is not sufficient by itself to close M1 until the visual-fidelity diagnostics have frozen
thresholds and passed them.

The follow-up VAE isolation test localized the first visual-fidelity failure. Reconstruction of the
64 deterministic test videos retained direction/color and object persistence perfectly, but shape
accuracy was only `0.515625` and the mean detected foreground occupied `2.075480x` the renderer's
expected object area. At 32x32 the renderer's default object is 4x4 pixels; the previous 4x spatial
VAE compression reduced that entire shape to roughly one latent cell. M1 therefore now uses 2x
spatial and 2x temporal compression so square/circle geometry has a larger latent footprint.

Before any corrected diffusion training, the VAE must independently pass a frozen reconstruction
gate: direction/color/shape accuracy >= `0.95`, object-like frame rate >= `0.95`, persistent-video
rate >= `0.95`, and mean foreground-area ratio in `[0.75, 1.50]`. The reconstruction objective also
includes a balanced silhouette term that penalizes missing foreground and diffuse background bleed.

The first 2x-spatial recovery run preserved direction, color, visibility and persistence, but it still
collapsed circle geometry: trajectory-aligned template evaluation classified every square correctly
and every circle incorrectly (`0.500000` aggregate shape accuracy, silhouette IoU `0.673023`). At
32x32 the original controlled object was only 4x4 pixels, so square and circle differed by just four
corner pixels. The controlled M1 curriculum now uses an 8x8 object footprint at 32x32 while keeping
the same direction/color/shape/speed vocabulary, static camera, frame count and resolution. Generic
M0 synthetic rendering remains unchanged.

Extending the same 8x8/2x-spatial VAE from 400 to 800 warmup steps cleared the frozen VAE gate without changing the objective or thresholds: reconstruction `0.005050`, direction `1.000000`, color `1.000000`, shape `0.984375`, object-like frame rate `1.000000`, persistent-video rate `1.000000`, and foreground-area ratio `1.044366`. The canonical tiny configuration therefore uses 800 VAE warmup steps before corrected diffusion training.

Validate the replacement latent representation without spending a diffusion run:

```powershell
uv run vexa-video train --config configs/tiny.toml `
  --run-dir runs/synthetic-motion-shape-recovery --vae-only --cuda
```

Only a checkpoint produced by this new 2x-spatial latent contract may be used for the corrected
diffusion run. The previous 4x-spatial M1 checkpoint remains historical evidence and is intentionally
rejected by the new checkpoint compatibility check.

The same-protocol random baseline is:

```text
direction_accuracy=0.250000
color_accuracy=0.250000
mean_motion=0.012409
static_rate=1.000000
gate_passed=False
```

The frozen gate remains direction >= 0.75, color >= 0.75, static <= 0.10, and mean motion > 0.02. Requested labels are used only as evaluation ground truth; generated RGB pixels and temporal motion determine the measured result.

Run the supervised sanity probe:

```powershell
uv run vexa-video probe `
  --config configs/tiny.toml `
  --run-dir runs/probe `
  --steps 300 `
  --cuda
```

Train the canonical synthetic-motion generator:

```powershell
uv run vexa-video train `
  --config configs/tiny.toml `
  --run-dir runs/synthetic-motion `
  --cuda
```

Evaluate a checkpoint with the frozen protocol:

```powershell
uv run vexa-video evaluate `
  --config configs/tiny.toml `
  --checkpoint runs/synthetic-motion/checkpoints/best.pt `
  --samples 64 `
  --sampling-steps 50 `
  --output runs/synthetic-motion/generated-metrics.json `
  --cuda
```

Evaluate the same-protocol random baseline:

```powershell
uv run vexa-video evaluate `
  --config configs/tiny.toml `
  --random-baseline `
  --samples 64 `
  --sampling-steps 50 `
  --output runs/synthetic-motion/random-baseline.json `
  --cuda
```

Generate a sample:

```powershell
uv run vexa-video generate `
  --config configs/tiny.toml `
  --checkpoint runs/synthetic-motion/checkpoints/best.pt `
  --prompt "a red square moves right at medium speed" `
  --output outputs/red-right.mp4 `
  --seed 42 `
  --cuda
```

The earlier 4x-spatial synthetic-motion checkpoint is no longer compatible after the shape-recovery
latent-contract correction. Preserve it as experiment evidence; train a clean 2x-spatial candidate
and pass the VAE gate before starting corrected diffusion training. M2 remains blocked.

## Development checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=vexa_video
```

Or:

```powershell
.\scripts\check.ps1
```

## Research path

The project starts with controlled synthetic motion because it lets us measure whether the model actually learns object identity, direction, velocity, occlusion, camera motion, and temporal continuity. We then move to latent diffusion, text conditioning, real licensed video, reference conditioning, native audio, editing, long-form generation and resolution scaling.

See:

- `ARCHITECTURE.md` for the intended model stack.
- `milestones.md` for the full research roadmap.
- `MILESTONES_TESTING.md` for milestone exit criteria.
- `CODING_GUIDELINE.md` for implementation rules.
- `DATA_POLICY.md` for dataset provenance requirements.

## Capability target, not baseline capability

The final roadmap targets:

- text-to-video and image-to-video;
- first/last-frame control;
- reference-to-video with multiple images/video/audio inputs;
- prompt-driven video editing and extension;
- 30-second coherent multi-shot generation;
- native dialogue, ambience, music/effects generation with A/V synchronization;
- persistent subject, prop and scene identity;
- camera and blocking control;
- readable text/UI rendering as a dedicated research track;
- 720p and 1080p production output;
- efficient local inference variants distilled from our own trained models.

Those are multi-year/frontier-scale research goals unless significant compute and data resources are available. The repository is structured so early milestones remain useful on a single development GPU while later milestones can move to distributed training.
