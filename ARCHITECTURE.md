# Architecture

## 1. Goal

Build a video foundation model from random initialization using PyTorch. The architecture must support incremental scaling from synthetic 64px clips on one GPU to distributed multimodal audio-video training without forcing a rewrite of public interfaces or checkpoint semantics.

We do not import pretrained AI weights. PyTorch/CUDA, FFmpeg, operating-system libraries and standard development utilities are infrastructure, not model intelligence.

## 2. Capability progression

The end-state system is intended to support:

```text
Text ───────────────┐
Images ─────────────┤
Video references ──┤
Audio references ──┤──> Multimodal conditioning ──┐
Masks/layout ───────┤                               │
Timing/camera ──────┘                               v
                                             Video/Audio DiT
                                                   │
                              ┌────────────────────┴────────────────────┐
                              v                                         v
                        Video latent                              Audio latent
                              │                                         │
                         Video decoder                              Audio decoder
                              │                                         │
                              └──────────── synchronized timeline ──────┘
                                                   │
                                                   v
                                             final A/V clip
```

The first milestones implement only a subset of this graph.

## 3. Core components

### 3.1 Tokenizer

Milestone 0 uses a byte tokenizer because it requires no external vocabulary or pretrained assets. Later milestones train our own subword tokenizer on the approved caption/script corpus.

Required properties:

- deterministic encode/decode;
- explicit BOS/EOS/PAD/UNK semantics;
- versioned vocabulary artifacts;
- no hidden dependency on an external language model.

### 3.2 Text encoder

A Transformer encoder trained jointly or in a dedicated language-conditioning stage. Early versions use pooled sequence embeddings. Later versions expose token-level conditioning to every DiT block using cross-attention or modulation.

### 3.3 Video VAE / video autoencoder

Input tensor convention:

```text
[B, C, T, H, W]
```

Initial implementation uses Conv3D blocks. Later versions add:

- causal temporal compression;
- stronger residual blocks;
- perceptual/discriminator losses trained from scratch;
- tiled encode/decode;
- temporal chunking;
- higher spatial compression;
- precision-aware decode.

The VAE is independently benchmarked. Diffusion quality cannot exceed a decoder that destroys detail or motion.

### 3.4 Video Diffusion Transformer

The latent video is patchified with a 3D convolution. Tokens receive:

- deterministic 3D Fourier features for temporal/spatial patch position;
- diffusion timestep conditioning;
- text conditioning;
- later: image/video/audio/reference conditioning;
- later: frame/time/shot/camera metadata.

The Stage-B positional features fix a correctness gap in the M0 scaffold: flattened video patches must not be permutation-ambiguous when learning left/right/up/down motion. The public `VideoDiT.forward` interface remains unchanged.

The first Stage-B GPU run showed a second limitation: pooled text was sufficient for learning
generic motion but not for reliable direction/color adherence. The corrective Stage-B path keeps
the same model parameters and checkpoint layout, but reuses the existing `text_proj` space for
parameter-free token-level attention before and after the Video DiT Transformer stack. This makes
individual caption tokens directly available to video tokens without introducing an external text
encoder or structured control channel. Stage-B-v2 subsequently improved color but left direction at
chance. Stage-B-v3 therefore adds a training-only semantic path: at a fixed noisy timestep, the
predicted clean latent is decoded through the frozen VAE, soft foreground centroids estimate motion,
and direction/color semantic losses compare that decoded result with the caption and one-word
counterfactual captions. These targets affect the loss only; the DiT still receives text tokens, timestep
and noisy video latents as its conditioning inputs.

A production-scale block is expected to evolve toward adaptive LayerNorm/modulation, factorized or windowed attention, RoPE or equivalent positional treatment, efficient temporal attention, and optional mixture-of-experts at large scale. Those are research directions, not assumptions baked into Milestone 0.

### 3.5 Diffusion / flow objective

The starter implements epsilon-prediction with a DDPM-style beta schedule because it is easy to verify. Later milestones should compare:

- epsilon prediction;
- v-prediction;
- rectified flow / flow matching;
- timestep weighting strategies;
- noise distributions suited to video latents.

Changing objectives must be isolated behind a stable training-target interface.

### 3.6 Native audio

Native A/V generation is a later milestone, not an audio track pasted on after video generation. The target architecture has:

- our own audio tokenizer/codec or spectrogram autoencoder;
- an audio latent timeline aligned to video time;
- shared or cross-modal Transformer blocks;
- explicit synchronization losses for speech/lip movement and events;
- dialogue, ambience, effects and music channels represented separately when useful.

### 3.7 Multimodal reference conditioning

Reference assets are normalized into typed condition streams rather than concatenated ad hoc. Planned types include:

- subject identity image;
- style image;
- first frame;
- last frame;
- reference clip;
- motion/reference trajectory;
- audio reference;
- mask/region;
- depth/pose/layout;
- shot plan and timestamps.

Each condition has an encoder trained inside this project or learned jointly.

## 4. Long-form generation

Thirty-second coherent generation is not treated as simply increasing `T`.

The design separates three levels:

1. **shot latent**: local motion and appearance;
2. **story/scene state**: subject identity, props, environment, intent and continuity;
3. **timeline plan**: shot boundaries, timestamps, transitions, dialogue and camera instructions.

Long-form milestones will investigate hierarchical generation:

```text
prompt/script
   -> timeline planner representation
   -> global scene/state tokens
   -> chunk/shot latent generation
   -> overlap-aware temporal stitching or joint refinement
   -> high-resolution decode
```

The planner must eventually be trained from project-owned data; it must not depend on a third-party LLM.

## 5. Resolution scaling

Do not train the first useful model directly at 1080p. Planned path:

```text
64 -> 128 -> 256 -> 512-class -> 720p -> 1080p
```

High-resolution output can use a separately trained video super-resolution/refinement model, provided that model is also trained from scratch in this project.

## 6. Training architecture

```text
Dataset shards
   ↓
Decode / sample / augment
   ↓
Batch assembly
   ↓
VAE encode (or cached approved latents)
   ↓
Condition encoders
   ↓
Noise/objective target
   ↓
Video DiT forward
   ↓
Conditional epsilon loss + text counterfactual/null-prompt losses
   ↓
Autograd
   ↓
Optimizer
   ↓
Checkpoint + metrics + sample generation
```

Scaling requirements:

- single-GPU debug mode;
- gradient accumulation;
- mixed precision;
- gradient checkpointing;
- distributed data parallel;
- FSDP/ZeRO-style sharding when model size requires it;
- deterministic resumable data sampling;
- versioned dataset manifests;
- asynchronous checkpoint/sample writing only when correctness is preserved.

## 7. Checkpoint contract

Every checkpoint must record at minimum:

- model state;
- optimizer state when resumable;
- global step and epoch/sample counters;
- config snapshot;
- tokenizer/version identifiers;
- dataset manifest hash/version;
- random seeds/RNG states where practical;
- source commit hash when available;
- precision/distributed settings.

Never save only a naked model state and call it a reproducible training checkpoint.

## 8. Evaluation architecture

Evaluation is divided into layers:

- **unit correctness**: shapes, gradients, serialization;
- **reconstruction**: VAE PSNR/SSIM-like internal metrics plus temporal error;
- **motion**: direction, speed, trajectory, collision/occlusion on synthetic ground truth;
- **prompt alignment**: project-trained evaluators and human evaluation;
- **temporal consistency**: identity/appearance drift and flicker;
- **audio-video synchronization**: event and phoneme/viseme timing;
- **long-form continuity**: subject/prop/scene state across shots;
- **human preference**: blinded pairwise evaluation with documented rubric.

External benchmark models may be used only if explicitly allowed by the project's no-third-party-model policy for evaluation. The default evaluation path should not require them.

## 9. Security and provenance

Training ingestion must attach provenance metadata to every asset. Data that cannot be traced to an approved source does not enter training. Model releases should document dataset classes and known limitations without exposing private source material.

## 10. Current implementation

The M0 interfaces remain in place. M1 Stage A has passed its renderer sanity gate and Stage B is active. The current Stage-B path is:

```text
StageBSyntheticDataset (fixed medium-speed cardinal curriculum)
        ↓
TinyVideoVAE reconstruction warmup + foreground-aware safety gate
        ↓
frozen latent video
        ↓
noise + timestep
        ↓
ByteTokenizer -> TransformerTextEncoder
        ↓
pooled text + parameter-free token-level text attention
        ↓
VideoDiT + deterministic 3D patch positions
        ↓
epsilon prediction + decoded predicted-clean semantic direction/color loss
        ↓
deterministic reduced-step reverse diffusion + classifier-free guidance
        ↓
TinyVideoVAE.decode
        ↓
generated RGB direction/color/motion evaluation
```

Stage-B checkpoints are composite resumable training artifacts rather than naked model state dicts. They record VAE/text/DiT and optimizer states, progress, config, synthetic curriculum metadata, metrics and RNG/data-sampler state.

The conditioning corrective adds no trainable parameters to `VideoDiT`, so Stage-B v1 checkpoints
remain state-dict and optimizer compatible. Resuming a v1 checkpoint deliberately resets only the
`best.pt` validation selector because v2 selection adds color/direction prompt-gap penalties; model,
optimizer, VAE progress, diffusion progress and RNG/data-generator state are preserved. Stage-B-v3
keeps the same parameter/optimizer layout, so the v2 checkpoint resumes directly; only the best-score
selector resets again because v3 adds held-out decoded semantic losses.

This does not complete M1. Generated-video quality remains the gate, and M2 remains blocked until the frozen evaluation protocol passes.

### Stage-B-v5 full-horizon preservation

The v4 validation sweep established that caption-controlled direction is correct through roughly eight DDIM steps but is overwritten by later denoising, while color improves at longer horizons. v5 therefore keeps the shared inference/training CFG-DDIM helper, adds a periodic 50-step differentiable direction objective, and uses fixed validation-split 50-step generated metrics in checkpoint selection. This changes training and model selection only; the public 50-step inference protocol and frozen M1 thresholds remain unchanged.
