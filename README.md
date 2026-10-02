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

The active curriculum is intentionally narrow: one object, four cardinal directions, four colors, square/circle, fixed medium linear speed, static camera, 8 frames at 32x32. Training uses the project-owned byte tokenizer, Transformer text encoder, TinyVideoVAE, VideoDiT, classifier-free-guided DDIM sampling, caption counterfactuals, decoded semantic losses, and balanced full-horizon direction/color/shape preservation.

The historical 64-sample, 50-step control-only evaluation that triggered the later visual-fidelity review was:

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

The final generated-video gate keeps the original thresholds and adds the visual-fidelity requirements before corrected diffusion training: direction >= 0.75, color >= 0.75, shape >= 0.75, static <= 0.10, mean motion > 0.02, object-like frame rate >= 0.90, persistent-video rate >= 0.90, and mean foreground-area ratio in `[0.75, 1.50]`. Requested labels are ground truth only; generated RGB pixels and temporal motion determine every measured result.

Run the supervised sanity probe:

```powershell
uv run vexa-video probe `
  --config configs/tiny.toml `
  --run-dir runs/probe `
  --steps 300 `
  --cuda
```

Train the corrected synthetic-motion generator from the independently passing VAE checkpoint. Shape is supervised through caption counterfactuals, decoded differentiable shape loss, and the same full-horizon rollout already used for motion/color; no structured shape control is injected into the denoiser:

```powershell
uv run vexa-video train `
  --config configs/tiny.toml `
  --run-dir runs/synthetic-motion-shape-control `
  --resume runs/synthetic-motion-shape-footprint/checkpoints/latest.pt `
  --cuda
```

Checkpoint selection now includes generated direction, color, shape, motion, static rate, object persistence, and renderer-relative foreground area at the full 50-step horizon, so a direction/color-only checkpoint cannot become canonical.

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

The first shape-aware diffusion run exposed an objective mismatch: generated direction/color converged, but shape stayed near chance while persistence collapsed and foreground area expanded to roughly four times the renderer target. The correction keeps caption-only shape conditioning and adds differentiable per-frame foreground-area supervision to decoded semantic predictions and short/full CFG-DDIM rollout shape losses. Start this corrected diffusion objective from the independently passing VAE-only checkpoint at `diffusion_step=0`; do not resume the failed shape-control diffusion checkpoints.

The 1,200-step foreground-area probe corrected the expansion shortcut but exposed a remaining conditioning asymmetry: direction reached `0.843750` while color stayed at `0.250000` and shape at `0.500000`. Sampler-aligned training had enumerated direction captions only, so full-horizon color and shape losses never received same-noise color/shape counterfactual rollouts. The next correction rolls out independent direction, color, and shape caption families from shared initial noise. The VAE, frozen gates, evaluator, CFG-DDIM sampler, and caption-only denoiser interface remain unchanged.

The 1,200-step attribute-symmetric probe validated the color correction: direction reached `1.000000` and color improved to `0.687500`, but shape remained `0.500000`. Geometry also regressed because foreground-area loss had moved from the full rollout batch to the smaller shape-only family, ending at object-like frame rate `0.753906`, persistence `0.312500`, and foreground-area ratio `2.391071`. The follow-up keeps same-noise attribute rollouts, restores area supervision across every direction/color/shape rollout, and adds an object-centered square/circle template loss aligned to the evaluator's foreground geometry.

The 1,200-step shape-template probe restored foreground scale by the end (`0.767433` area ratio), but the legacy normalized corner-moment shape proxy remained in the optimization path. When generated foreground became sparse, its variance-product denominator approached zero: semantic shape loss spiked as high as `2300.089111` and the raw gradient norm reached `4567552.000000`. Those shape gradients dominated globally clipped updates, erasing the previous color gain and leaving the run at direction `0.531250`, color `0.250000`, and shape `0.500000`. The next correction removes the corner-moment proxy from semantic and sampler-aligned optimization while retaining the bounded object-centered template loss, foreground-area supervision, and same-noise direction/color/shape rollout families.

The 1,200-step stable-shape probe confirmed the corner-moment removal fixed the numerical failure: foreground-area ratio ended at `0.804091`, object-like frame rate at `0.941406`, persistence at `0.937500`, and direction recovered to `0.812500`, but color reached only `0.343750` and shape remained `0.500000`. Whole-template MSE still permits an average silhouette because square and circle share most foreground pixels. The next correction supervises only the square-only corner pixels and adds a same-noise square-vs-circle occupancy margin, while the existing area objective remains responsible for object scale and persistence.

The 1,200-step discriminative-corner probe remained stable and improved conditioning, reaching direction `0.906250` and color `0.468750` at step 1100 with foreground-area ratio `1.172894`, but generated shape stayed exactly `0.500000` throughout validation. The frozen evaluator classifies shape from foreground fill inside the detected bounding box; for the 8x8 curriculum its square/circle fills are `1.000000` and `0.812500`, with decision boundary `0.906250`. The next correction therefore optimizes a bounded differentiable approximation of that fill-ratio geometry and requires same-noise square/circle rollouts to span the evaluator's `0.187500` target gap. The VAE, caption-only control interface, sampler, area objective, and frozen gates remain unchanged.

## M1 research-rescue training

The evaluator-aligned fill probe confirmed that proxy loss reduction is not sufficient to make the denoiser use the shape word. At step 1100 it reached direction `1.000000`, color `0.500000`, motion `0.043313`, and static rate `0.062500`, while generated shape stayed exactly `0.500000`; the final step retained direction `0.968750` and color `0.500000` but shape remained at chance and foreground area regressed above the frozen range. M1 therefore stays open and M2 stays blocked.

The first 300-step research-rescue probe localized the blocker before decoding. The frozen VAE remained healthy, but its square/circle latent difference was distributed (`top1=0.089188`, `top4=0.298075`, `top8=0.493130`, effective rank `25.746357`), so no low-rank shape projection is used. More importantly, `latent_shape_cos` stayed near zero (`0.027183` at step 300), `latent_shape_ratio` collapsed to `0.420006`, and generated shape remained exactly `0.500000`. The global diagnostics also measured genuine objective conflict at step 300: diffusion-vs-shape `-0.121657` and direction-vs-shape `-0.242229`.

The next research-rescue objective keeps the passed VAE, caption-only denoiser interface, CFG-DDIM sampler, frozen evaluator, and all M1 thresholds unchanged. It adds seven targeted mechanisms:

- same-seed square/circle counterfactual videos provide a causal latent target; both captions see the same noisy midpoint latent, and the predicted square-minus-circle latent displacement is matched to the frozen VAE's true displacement;
- a normalized latent target-ranking loss requires each caption-conditioned prediction to be closer to its own frozen-VAE target than to the square/circle counterfactual, directly penalizing midpoint/collapsed solutions;
- shape-protected gradient surgery computes primary and shape gradients over the full trainable text/DiT path and projects only the primary component that has negative global dot product with the protected shape gradient; aligned primary updates are left untouched;
- GeomLoss Sinkhorn divergence compares centered `8x8` decoded foreground measures with exact square/circle measures, but its gradient is causally gated off while latent shape cosine is below `0.10`, ramps between `0.10` and `0.30`, and reaches full weight at `0.30`;
- Min-SNR-gamma weighting rebalances the ordinary epsilon-prediction objective across diffusion timesteps;
- frozen-VAE latent SVD and checkpoint-step task-gradient cosines remain diagnostics only;
- checkpoint selection is feasibility-first: normalized violations of every frozen generated-video gate dominate the ordinary validation losses, which are used only as a tie-breaker.

The research controls default to `causal_shape_weight=1.0`, `latent_rank_weight=1.0`, `latent_rank_margin=0.25`, `sinkhorn_shape_weight=0.5`, causal Sinkhorn gate `0.10 -> 0.30`, and `min_snr_gamma=5.0`. They can be overridden without changing the frozen config or gates:

```powershell
$env:VEXA_M1_CAUSAL_SHAPE_WEIGHT = "1.0"
$env:VEXA_M1_LATENT_RANK_WEIGHT = "1.0"
$env:VEXA_M1_LATENT_RANK_MARGIN = "0.25"
$env:VEXA_M1_SINKHORN_SHAPE_WEIGHT = "0.5"
$env:VEXA_M1_SINKHORN_BLUR = "0.12"
$env:VEXA_M1_SINKHORN_MARGIN = "0.02"
$env:VEXA_M1_SINKHORN_GATE_START = "0.10"
$env:VEXA_M1_SINKHORN_GATE_FULL = "0.30"
$env:VEXA_M1_SHAPE_GRADIENT_SURGERY = "1"
$env:VEXA_M1_MIN_SNR_GAMMA = "5.0"
```

Run one controlled 300-step rescue probe from the independently passing VAE-only checkpoint:

```powershell
uv run vexa-video train `
  --config configs/tiny.toml `
  --run-dir runs/m1-research-probe `
  --resume runs/synthetic-motion-shape-footprint/checkpoints/latest.pt `
  --steps 300 `
  --cuda
```

Do not extend a rescue probe past 300 steps before reviewing `latent_rank`, `latent_rank_margin`, `shape_grad_cos`, `shape_grad_projected`, `sinkhorn_gate`, `latent_shape_cos`, `latent_shape_ratio`, generated shape, and the frozen gate violations.

For a bounded hyperparameter search, Optuna runs independent 300-step trials from the same VAE-ready checkpoint and minimizes the feasibility-first checkpoint score:

```powershell
uv run python scripts/m1_optuna.py `
  --config configs/tiny.toml `
  --resume runs/synthetic-motion-shape-footprint/checkpoints/latest.pt `
  --run-dir runs/m1-research-search `
  --trials 8 `
  --steps 300 `
  --cuda
```

`geomloss==0.3.1` is the only new research runtime dependency used by this path; Optuna is a development dependency. Kornia, MONAI, LibMTL, and TorchOpt are intentionally not required.

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
