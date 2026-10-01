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

## M1 — Synthetic motion

M1 is still active. Stage A passed on the RTX 5050 development GPU; Stage B is the current generative experiment. M1 is **not complete** and M2 remains blocked until complete generated videos pass the frozen M1 quality gate.

### Stage A — passed

The supervised sanity probe established that direction and color are recoverable from the deterministic renderer without lowering the gate:

```text
steps=300
baseline_direction_accuracy=0.246094
baseline_color_accuracy=0.250000
direction_accuracy=0.984375
color_accuracy=1.000000
gate_passed=True
```

The probe is only evidence that the synthetic signal is learnable. It is not evidence that the generative model follows prompts.

### Stage B — active

The first Stage-B GPU run proved that the generator learned motion but did **not** pass
controlled generation. With the original pooled-text conditioning path, the frozen random
baseline and trained `best.pt` measured:

```text
                         random       trained
direction_accuracy      0.265625     0.312500
color_accuracy          0.250000     0.265625
mean_motion             0.009729     0.136491
static_rate             1.000000     0.000000
```

The VAE was healthy (`vae_validation_reconstruction_loss=0.024338`) and motion/static collapse
was solved, but direction and color remained near chance. Stage B therefore stays active and M2
stays blocked.

Stage B connects the existing project-owned components into the first real generative training path:

```text
synthetic RGB video
  -> TinyVideoVAE warmup + reconstruction safety gate
  -> frozen latent representation
  -> diffusion noise
  -> ByteTokenizer + TransformerTextEncoder
  -> token-level text attention + pooled text conditioning
  -> VideoDiT with deterministic 3D Fourier patch positions
  -> correct-vs-counterfactual caption objective + null-prompt objective
  -> epsilon/noise prediction
  -> deterministic reduced-step reverse diffusion with classifier-free guidance
  -> TinyVideoVAE.decode
  -> generated RGB video evaluation
```

The Stage-B curriculum is intentionally narrow: one object, four directions, four colors, square/circle, fixed medium linear speed, static camera, 8 frames at 32x32. Bounce, acceleration, rotation, camera movement, multiple objects and real video remain disabled.

Before diffusion starts, the VAE is trained on reconstruction and must pass the configured foreground-aware reconstruction safety gate. This prevents a broken latent representation from silently contaminating diffusion training. The VAE safety threshold is not the M1 generative quality gate.

The corrective conditioning path still uses only the project-owned caption path. Direction/color
control labels are **not** injected into the model. During training, valid captions are compared
against captions with only the color word or direction word changed; this forces the denoiser to
be sensitive to the text it already receives. A null caption is trained in parallel so deterministic
classifier-free guidance can strengthen prompt adherence at sampling time. Half of corrective
training timesteps are drawn from the high-noise half of the diffusion schedule, where the caption
carries more information than the corrupted latent. The token-attention path adds no new
parameters, so the first Stage-B `best.pt` remains weight/optimizer compatible.

The completed Stage-B-v2 GPU evaluation improved color control substantially but still failed the
frozen gate: direction `0.265625`, color `0.703125`, mean motion `0.163834`, static rate `0.000000`.
The direction result remained at chance even though color nearly reached its gate. Stage-B-v3 therefore
adds a decoded semantic auxiliary at a fixed noisy timestep. It reconstructs the model's predicted clean
latent through the frozen VAE and penalizes motion whose soft centroid disagrees with the caption
direction. The same noisy latent is also evaluated with a one-word direction counterfactual, so the model
must change the generated motion when only `left/right/up/down` changes. A smaller decoded color loss
is retained to push color across its existing `0.75` gate. Captions remain the only model conditioning
input; semantic targets are training losses only and are never injected into the denoiser.

Stage-B-v3 resumes the v2 `best.pt` without changing model parameters or optimizer structure. Only the
validation selector resets because the score now includes held-out decoded direction/color semantic
losses. The corrective target is extended to 8000 diffusion steps.

The completed Stage-B-v3 generated-video evaluation passed color (`0.781250`), mean-motion
(`0.139715`) and static-rate (`0.000000`) gates, but direction remained at chance (`0.250000`).
Stage-B-v4 therefore adds sampler-aligned direction supervision: a small training sub-batch starts
from Gaussian latent noise, runs the same classifier-free-guided DDIM implementation used by
inference for a short differentiable rollout, decodes the result, and applies the existing soft
cardinal-motion loss. All four direction captions share the same starting noise, so the caption must
cause the motion difference. This is a training-only correction; the frozen 50-step evaluation path,
guidance scale and M1 thresholds do not change. The v4 target is 10000 diffusion steps.

Stage-B-v4 proved that direction control is horizon-specific rather than absent. Its frozen 50-step
result was direction `0.156250`, color `0.781250`, mean motion `0.131964`, static rate `0.015625`.
A validation-only horizon sweep then measured direction `1.000000` at 4 and 6 steps, `0.953125` at
8, `0.421875` at 12, and near chance from 16 through 50 steps. The first Stage-B-v5 correction added
a periodic differentiable 50-step direction loss and solved direction completely, but its frozen
50-step test result regressed color/static behavior: direction `1.000000`, color `0.343750`, mean
motion `0.044545`, static rate `0.156250`. This remains a Stage-B-v5 regression correction rather
than a new milestone.

The balanced v5 correction restarts from the v4 `best.pt`, where color/static already passed. The
same 50-step four-direction rollout now supervises both cardinal motion and the original caption color,
so no additional expensive rollout is required. The full-horizon direction term is reduced to avoid
dominating the clipped gradient, and the long-horizon minimum displacement is raised to protect the
static-rate gate. Validation checkpoint selection uses normalized deficits against all four frozen gates
so a solved direction score cannot hide a color/static regression.

Run the balanced v5 correction in a fresh directory:

```powershell
uv run vexa-video train `
  --config configs/tiny.toml `
  --run-dir runs/m1-stage-b-v5-balanced `
  --resume runs/m1-stage-b-v4/checkpoints/best.pt `
  --cuda
```

The target remains 13000 diffusion steps. Every fourth optimization step uses the balanced 50-step
preservation objective; every checkpoint interval runs the fixed 16-sample validation-split 50-step
selector. The frozen 64-sample test protocol and all absolute M1 thresholds remain unchanged.

Because the corrective path changes the generation protocol (token attention + guidance), record
a fresh random baseline before corrective training. The absolute M1 thresholds do not change:

```powershell
uv run vexa-video evaluate-m1 `
  --config configs/tiny.toml `
  --random-baseline `
  --samples 64 `
  --sampling-steps 50 `
  --output runs/m1-stage-b-v2/random-baseline.json `
  --cuda
```

Resume the successful VAE/denoiser state from the original Stage-B `best.pt` into a separate run
directory. The old checkpoint resumes at its recorded diffusion step, while the best-checkpoint
selector is reset because validation now includes prompt-sensitivity gaps:

```powershell
uv run vexa-video train `
  --config configs/tiny.toml `
  --run-dir runs/m1-stage-b-v2 `
  --resume runs/m1-stage-b/checkpoints/best.pt `
  --cuda
```

Checkpoint resume normalizes saved CUDA RNG byte tensors back to CPU before handing them to PyTorch's CUDA RNG API. This keeps the original Stage-B CUDA checkpoints compatible with corrective training while preserving model, optimizer, RNG, and diffusion-step state.

Resume without resetting model/optimizer/RNG/training progress:

```powershell
uv run vexa-video train `
  --config configs/tiny.toml `
  --run-dir runs/m1-stage-b-v2 `
  --resume runs/m1-stage-b-v2/checkpoints/latest.pt `
  --cuda
```

`latest.pt` is the most recent resumable checkpoint. `best.pt` minimizes held-out reconstruction,
conditional diffusion loss and a penalty when color/direction counterfactual captions are not worse
than the correct caption by the configured margin. Final M1 completion is still decided by complete
generated-video metrics, not by this validation score.

Evaluate the trained generator:

```powershell
uv run vexa-video evaluate-m1 `
  --config configs/tiny.toml `
  --checkpoint runs/m1-stage-b-v2/checkpoints/best.pt `
  --samples 64 `
  --sampling-steps 50 `
  --output runs/m1-stage-b-v2/generated-metrics.json `
  --cuda
```

The evaluator derives direction from generated temporal motion and color from generated RGB pixels.
It reports direction accuracy, color accuracy, mean motion, static rate, confusion matrices, the
guidance scale and `gate_passed`. The frozen gate is direction >= 0.75, color >= 0.75, static <=
0.10 and mean motion > 0.02. Requested labels are used only as ground truth for comparison.

Generate a sample after training:

```powershell
uv run vexa-video generate `
  --config configs/tiny.toml `
  --checkpoint runs/m1-stage-b-v2/checkpoints/best.pt `
  --prompt "a red square moves right at medium speed" `
  --output outputs/red-right.mp4 `
  --seed 42 `
  --cuda
```

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
