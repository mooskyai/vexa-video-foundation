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
- a CLI smoke test and M1 supervised motion-sanity probe;
- CPU unit tests that validate shapes, determinism, gradients, and diffusion math;
- architecture and scaling plans through the frontier-capability milestones.

The initial model is intentionally tiny. Its purpose is to prove the pipeline before scaling compute.

## Repository layout

```text
vexa-video-foundation/
├── src/vexa_video/
│   ├── data/              # datasets and synthetic curriculum
│   ├── diffusion/         # forward/reverse diffusion primitives
│   ├── inference/         # samplers and generation entry points
│   ├── models/            # tokenizer, text encoder, VAE, Video DiT
│   ├── training/          # training/checkpoint utilities
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

## M1 — Synthetic motion sanity gate

M1 begins without changing the starter architecture. The first stage expands the deterministic synthetic renderer to balanced right/left/down/up motion, red/green/blue/yellow appearance, square/circle shapes and slow/medium/fast motion with disjoint train/validation/test seed spaces. Before generative training, a small supervised 3D-convolutional probe must prove that direction and color are recoverable from the rendered video signal.

Run the M1 probe on the development GPU:

```powershell
uv run vexa-video m1-probe `
  --config configs/tiny.toml `
  --run-dir runs/m1-probe `
  --cuda
```

The frozen Stage-A gate in `configs/tiny.toml` requires both direction and color accuracy >= 0.95 on the held-out test seed space. Passing this probe does **not** complete M1; it only authorizes the next M1 step: generative temporal training and evaluation using the existing VAE/DiT/diffusion starter stack.

## Development checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src
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
