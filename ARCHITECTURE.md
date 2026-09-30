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

- diffusion timestep conditioning;
- text conditioning;
- later: image/video/audio/reference conditioning;
- later: frame/time/shot/camera metadata.

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
Losses
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

## 10. Current starter implementation

Milestone 0 intentionally provides:

```text
ByteTokenizer
TransformerTextEncoder
TinyVideoVAE
VideoDiT
LinearNoiseSchedule
SyntheticMotionDataset
```

These are scaffolding components. Their interfaces are expected to survive while their internals become substantially more capable.
