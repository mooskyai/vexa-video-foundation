# M1 conditioning, reverse dynamics, and spatial binding audit

Date: 2026-10-02 (Asia/Calcutta). This is an analysis record, not a training correction.
M1 remains open; M2 remains blocked. Model code, objectives, sampler, VAE, evaluator,
dependencies, and frozen acceptance thresholds are unchanged.

## Exact starting state and evidence provenance

The first successful Git inspection established:

```text
branch:           main
HEAD:             fb3972d388482b8b52cb7778746d8806dab2baf3
local origin/main:f872fd286a9b2d7ce911ea146f3d9b12419136dc
tracking:         ahead 1, behind 0
index:            clean
working tree:     clean, including untracked files
```

These tracking values refer to the local remote-tracking ref; no fetch was performed.
`fb3972d` already commits the causal scan, dynamics decomposition, math helpers, tests,
and accompanying documentation. Its diff from `f872fd2` contains no changes to models,
`training/trainer.py`, or configs. There are no uncommitted diagnostics to preserve.

The handoff's passing VAE, failed 300-step text-conditioning probe, and causal-scan
numbers are historical observations. The user deleted the corresponding run artifacts.
IDE tabs do not establish that their files still exist. No deleted checkpoint was
searched for, and no trained-checkpoint decomposition was run in this audit.
A rebuilt VAE or fresh diffusion run will not recover the deleted learned DiT state.

Current local baseline checks passed: Ruff, Ruff format check, MyPy (`src tests`),
63 pytest tests, and `git diff --check`. Commands used `uv run --offline --no-sync`
to retain the installed environment. Runtime: Python 3.14.7, PyTorch 2.14.0+cu132.
The controls below used CPU and performed no optimizer step or parameter update.

## The conditioning circuit that actually exists

Relevant sources:

- [Byte tokenizer and text encoder](../../src/vexa_video/models/text_encoder.py), lines 16-99.
- [DiT](../../src/vexa_video/models/dit.py), lines 60-95 and 111-177.
- [Text-separation objective](../../src/vexa_video/training/m1_research.py), lines 203-263.
- [Tiny configuration](../../configs/tiny.toml).

The tokenizer retains every controlled caption. Captions contain 37-43 ASCII bytes,
within the 64-token budget including BOS/EOS. `square` and `circle` each occupy six
bytes at identical offsets, so changing shape does not shift the remaining caption.
The text encoder is a two-layer, width-64 Transformer with a learned position table
and final LayerNorm. No pretrained weights or structured semantic inputs are involved.

Write projected text tokens as E, latent patch embeddings as P(x), position features
as p, timestep embedding as q(t), global pooling as L(E), and token text attention as
T(h,E). The current forward pass is:

```text
h0 = P(x) + p + q(t) + L(E) + T(P(x), E)
h  = two ordinary pre-LayerNorm Transformer encoder blocks(h0)
epsilon = output_projection(LayerNorm(h + T(h, E)))
```

The first text query is the raw patch embedding: position, timestep, and global text
have not yet been added. Equal patches receive equal first-branch conditioning. At
high noise, initial text allocation therefore depends on noisy local patch contents.
The final text attention can use positions, timestep, and contextualized content.
The network can express local binding; a broadcast input does not prove incapacity,
because the Transformer can combine it with token-specific content and position.

Text attention uses a single dot-product head, raw queries, and the same projected
text for keys and values. It has no independent learned Q/K/V, output projection,
residual gate, or per-block text reinjection. Ordinary self-attention inside the
Transformer DOES have learned Q/K/V. Both text-attention outputs are LayerNormed.
The pooled text is norm-weighted and then LayerNormed. Its importance logits equal
`norm(E_j)/96`, which can remain nearly uniform at ordinary norms. Learned entropy,
not the name of the pooling function, determines how selective it is.

`f872fd2` removed per-token projection normalization and replaced bounded cosine
attention logits with raw dot products divided by sqrt(96). Projected magnitude can
change selectivity, but output LayerNorm largely normalizes branch amplitude.
Increasing raw separation is therefore not equivalent to strengthening useful
conditioning at the object boundary.

CPU parameter counts are DiT 307,504 and text encoder 120,832. DiT contains 223,680
block parameters, 74,208 timestep-MLP parameters, and 6,240 text-projection parameters.
Its latent tensor is `[4,4,16,16]` per video; patch size `[1,2,2]` produces 256 tokens
on a `[4,8,8]` grid. Each patch embeds 16 latent scalars into width 96 and predicts
16 scalars. The 8x8 object covers about two to three patch positions per spatial
axis, depending on phase. There is no dimension-reducing patch projection proving
that shape is impossible. Width/depth sufficiency remains an empirical question.

## Text metrics have an exact blind spot

The text loss measures raw projected shape-span means and raw whole-caption means.
The DiT consumes different quantities: norm-weighted pooling plus LayerNorm, and
attention-weighted sums plus LayerNorm. Raw `text_shape_global_ratio` is not the
counterfactual ratio in either consumed pathway.

A checkpoint-free counterexample uses a width-96 alternating +1/-1 vector v.
Set every projected square token to `v + 2*ones` and every circle token to
`v - 2*ones`, with actual tokenizer masks for the paired captions. Then:

| Measurement | Current-code result |
| --- | ---: |
| Text-separation loss | 0.000000 |
| Raw span cosine | -0.600000 |
| Relative span delta | 1.788854 |
| Raw whole-caption/span ratio | 1.000000 |
| Consumed pool maximum difference | 2.38e-7 |
| Consumed attention maximum difference, 256 queries | 1.19e-7 |

In exact arithmetic the two consumed differences are zero: LayerNorm removes the
channel-constant shift, and attention logits shift equally across all unmasked keys.
This disproves the inference that passing the raw text loss proves useful consumed
conditioning. It does not prove the deleted checkpoint used this particular shortcut.
Its historical trajectory divergence establishes some surviving caption influence.

Required measurements are counterfactual deltas in the ACTUAL pool and first/final
attention outputs; projected token norms; query norms; logit dispersion; attention
entropy and shape-byte mass; and attended-vector variance before normalization.
Attention maps alone are not sufficient: test branch ablations and resulting epsilon
and decoded-boundary responses. Contextual text can also carry shape outside its byte span.

## Position features have endpoint aliases

`spatiotemporal_position_embedding` maps coordinates in `[-1,1]` using sine/cosine
at frequencies `pi*2**k`. Opposite endpoints alias in exact arithmetic: sine is zero
and cosine is the same. On the configured `[4,8,8]` grid, the ideal float64 position
matrix has rank 14: temporal 2, vertical 6, horizontal 6 (verified in float64 with
absolute tolerance 1e-7 and relative tolerance zero). Float32 features have tiny
extra components from evaluating large trigonometric arguments. Their endpoint-delta
RMS is 0.000675 versus position-feature RMS 0.707107. At quantization 0.01 there are
147 distinct vectors for 256 positions; energy beyond the first 14 singular values
is about 2.88e-7.

Low rank alone is not a defect: positions need distinguishable coordinates, not one
orthogonal dimension each. Endpoint aliasing and high-frequency redundancy are
concrete limitations, especially for first/last latent time positions. Content,
VAE-derived boundary cues, and learned spatial behavior can still distinguish tokens.
The DiT patch convolution itself has no padding. This is a
candidate contributor, not evidence authorizing a positional correction now.
Use endpoint/interior and translated-object controls to determine its relevance.

## Paired training targets are registered; free-running comparisons are not

The training pair builder (`trainer.py`, lines 203-229) preserves seed, color,
direction, speed, and footprint while changing only shape. Renderer start positions
and motion depend on those unchanged quantities (`synthetic.py`, lines 103-139).
The paired trajectories are therefore identical. Flattening the latent tensor is a
bijective reshape that retains coordinate correspondence. It is not a registration bug.

The dynamics script instead encodes a validation-renderer target delta, then begins
a generated trajectory from unrelated Gaussian noise (lines 214-227). Captions do
not specify start position. A correct generated boundary change at another location
need not align with that absolute-coordinate renderer delta. Permutation CKA cannot
repair missing coordinate correspondence.

A renderer-only control demonstrates the size of this confound. For all 16
color/direction contexts, render exact square/circle pairs with seeds
`42 + 10000000 + i` and `42 + 83001 + i`, 8 frames, size 32, footprint 8. Each pair
is internally trajectory-matched and contains perfect shapes. Compare their RGB
square-minus-circle deltas:

```text
mean absolute-coordinate cosine: 0.036458
mean object-cropped cosine:       1.000001 (float32 rounding)
```

Thus a cosine near the historical 0.05 can arise from perfect geometry at a different
position. This control concerns RGB registration, not the deleted VAE's latent response.
The original local scan uses the renderer's noisy latent midpoint, avoiding the
free-running registration mismatch. Its historical low cosine remains meaningful
within that teacher-forced experiment, subject to position uncertainty and midpoint
input distribution at each noise level.

For generated paths, report centroid displacement, motion, color, foreground scale,
contrast, and background response BEFORE registration. Do not center each prompt's
output independently and silently erase shape-induced translation. Use a common
pair frame or a registered rendered/re-encoded target. Centered RGB boundary measures
are a cleaner starting point than assuming the VAE's latent grid is exactly equivariant.
Stride, padding, temporal compression, and GroupNorm complicate that assumption.

## Timestep dynamics: parameterization and transport must be separated

The current schedule and deterministic DDIM implementation have the expected algebra.
Let a_t be cumulative alpha, nominal SNR r_t = a_t/(1-a_t), and Delta epsilon be the
square-minus-circle response at IDENTICAL noisy state. Then:

```text
Delta predicted_clean = -Delta epsilon / sqrt(r_t)
```

Changing timestep changes this multiplier even if epsilon sensitivity is constant.
The historical high-noise magnitude ratio near 60 therefore does not itself indicate
strong useful shape forcing. Shrinking late predicted-clean response alone does not
prove semantic erasure. Multiplication by a scalar does not improve cosine alignment.

Current-source schedule values, recomputed on CPU:

| t | Nominal SNR | 1/sqrt(SNR) | Next sampled t | DDIM A | DDIM B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 999 | 0.00004036 | 157.4073 | 979 | 1.22152 | -0.22153 |
| 510 | 0.07554351 | 3.6383 | 489 | 1.11200 | -0.11665 |
| 500 | 0.08435955 | 3.4430 | not on 50-step grid | - | - |
| 489 | 0.09511339 | 3.2425 | 469 | 1.10184 | -0.10710 |
| 347 | 0.40698244 | 1.5675 | 326 | 1.07442 | -0.08965 |
| 326 | 0.50130941 | 1.4124 | 306 | 1.06636 | -0.08271 |
| 245 | 1.14944835 | 0.9327 | 224 | 1.05161 | -0.07806 |
| 0 | 9997.340882 | 0.0100 | final clean | 1.00005 | -0.01000 |

At t=500 nominal SNR is -10.7387 dB. Actual latent/object information SNR is not
necessarily this value. The deterministic VAE returns unstandardized encoder outputs;
per-channel centered signal SNR is `r_t*Var(z_channel)`. Measure latent means,
variances, shape-delta energy, and spatial localization uncertainty before identifying
a geometry-forming band from nominal schedule SNR alone. No VAE modification follows
from measuring its scale.

Write guided epsilon as `G_c(x)=g*epsilon_c(x)+(1-g)*epsilon_empty(x)`, and a transition
as `F_c(x)=A*x+B*G_c(x)`, where:

```text
A = sqrt(a_previous/a_t)
B = sqrt(1-a_previous) - A*sqrt(1-a_t)
```

B is negative in reverse sampling. With square/circle states x_s and x_c, the existing
symmetric split is exact:

```text
C = 0.5 * [(F_s(x_s)-F_c(x_s)) + (F_s(x_c)-F_c(x_c))]
S = 0.5 * [(F_s(x_s)-F_s(x_c)) + (F_c(x_s)-F_c(x_c))]
C + S = F_s(x_s)-F_c(x_c)
```

But S contains `A*(x_s-x_c)`, not just learned feedback. Over the full schedule,
analytic transport alone has product `1/sqrt(a_999)=157.4104`. This is not the real
trajectory gain, because model epsilon terms may cancel it. It proves that state gain
above one and RMS divergence growth are insufficient evidence of unstable learned
feedback or the moment geometry forms. Report analytic transport and the remaining
model-state term separately. The exact split attributes one transition, not a sum of
independent final-output causal effects; nonlinear interactions remain.

At fixed state, unconditional epsilon cancels from the shape difference:

```text
G_square(x) - G_circle(x) = g*(epsilon_square(x)-epsilon_circle(x))
```

CFG cannot rotate this direct square/circle epsilon difference at the same state.
It can magnify a wrong response and change the evolving states. At g=3, state feedback
contains `3*J_conditional - 2*J_empty`, so empty-prompt dynamics may still matter.
Matched read-only guidance 1/2/3 runs and timestep-window caption interventions can
separate direct forcing from downstream amplification. Acceptance remains CFG=3.

The dynamics script currently decodes post-step noisy states directly and labels
rows with the consumed timestep. These are off-manifold inputs to the clean-latent
decoder until completion. Decode predicted-clean estimates alongside noisy states,
record both t and previous_t, and include foreground validity/area/persistence.
The hardcoded decode indices also need the actual final index if step count differs.
An intermediate shape label on noise is not a geometry-formation measurement.

## Objectives and statistical evidence have further limits

1. **Ranking is not independent geometric evidence.** The logged ranking margin is
   algebraically `dot(predicted_delta,target_delta)/norm(target_delta)**2`, apart from
   its small-denominator clamp. A control with target `[1,0]` and response `[0.25,5]`
   gives margin 0.25 but cosine 0.049938 and aligned energy 0.002494. The ranking hinge
   can pass while orthogonal energy dominates. The full ranking loss DOES retain a
   correct-distance penalty; it is 1.597656 here, not zero.

2. **The fill proxy is not the frozen evaluator.** Training uses continuous occupancy
   inside a fixed 8x8 box around a detached/snapped centroid (`trainer.py`, lines 435-465).
   Evaluation uses binary foreground and its detected bounding box (`evaluation.py`,
   lines 67-84). A crisp 6x6 square has soft fill 0.5625 and evaluator fill 1.0; an 8x8
   square has both 1.0. At constant RGB -0.95, current shape-fill loss is 1.0 with zero
   gradient everywhere because strength is below 0.10. This demonstrates size coupling
   and a dead region, not a proposed evaluator change. Existing ideal-renderer tests
   and finite-gradient checks do not establish alignment on generated artifacts.

3. **Shape objectives do not constrain the same mechanism.** Counterfactual epsilon
   contrast rewards a scalar error gap, not a specific boundary-directed intervention.
   Causal delta and ranking train at t=500. There is ALSO a four-step CFG-DDIM rollout
   on every update inside `_semantic_conditioning_losses` (lines 945-954), adding
   direction and shape endpoint losses; its sampled timesteps are 999, 666, 333, 0.
   Full 50-step rollout runs every four updates. Each rollout has ten variants of one
   base context: direction loss uses four direction variants, color loss four color
   variants, fill/margin two shape variants, and area all ten. The short rollout's
   color loss is computed but discarded. These rollouts do not directly supervise
   each transition's local boundary response. Ordinary epsilon training does cover
   all timesteps, with 75% sampling probability in 500-999 and 25% in 0-499 under the
   configured high-noise mixture. Min-SNR weighting additionally reduces epsilon
   weight where SNR exceeds gamma. The question is where direct causal geometry is
   learned, not whether off-500 timesteps ever appear in training. Sinkhorn supplies
   no geometry gradient while its cosine gate is off. None of these facts recommends
   increasing loss weights.

4. **Low-noise full-delta matching is not automatically a valid oracle.** At identical
   state, forcing the full target delta D requires `Delta epsilon=-sqrt(SNR)*D`.
   At t=0 this is about -100*D. A healthy near-clean denoiser can instead be state
   dominated, and an averaged/swapped-caption input can be atypical. Test true noisy
   square, true noisy circle, and midpoint anchors separately. Do not demand equal
   intervention magnitude at every timestep or extrapolate midpoint targets blindly.

5. **Task gradient diagnostics are restricted.** `trainer.py:1314` passes
   `trainable[-2:]`, which are terminal DiT `norm.weight` and `norm.bias`.
   Historical diffusion/shape and direction/shape task cosines are not whole-model
   cosines. Surgery itself uses all text/DiT parameters, but its Euclidean raw-gradient
   protection is not a guarantee about the actual AdamW update after adaptive scaling,
   momentum, and weight decay. Diagnose per-module gradients and actual update direction.

6. **CKA does not test signed or spatial correctness.** Current biased CKA for two
   independent random 16x4096 matrices was 0.996509; permutation null mean 0.996549,
   p=0.540856. Comparing a matrix with its negative gives CKA about 1 and significant
   p=0.003891 while cosine is -1. Calibration removes a null-baseline illusion; it
   does not remove sign/rotation invariance or position mismatch. Use signed gain
   together with the unsigned aligned-energy fraction. Sixteen initial scan contexts
   also mix color/direction with trajectory; repeated seeds and context-conditioned
   nulls are needed, plus treatment of the 50 timestep comparisons.

7. **SVD rank depends on samples and nuisance variation.** A 16-sample centered response
   has rank at most 15. Earlier target effective rank about 26 is not comparable without
   identical samples. Unregistered target spectra include translation, color, motion,
   stride phase, and VAE behavior. They justify avoiding tiny projections of those
   particular targets, not a claim that intrinsic centered shape has rank 26.
   Centering also removes a shared response; report mean energy separately.

## Answers to the handoff's architectural questions

| Question | Current conclusion |
| --- | --- |
| Can this DiT localize shape? | Yes in representational terms; learned boundary allocation is unverified. Its first query lacks position/time, and later routing is shallow and tied. |
| Which timesteps carry useful geometry? | Unknown without registered predicted-clean decoded evidence. Nominal SNR and divergence growth do not establish it. Include exact t=500, absent from the sampler scan. |
| Is empty-prompt CFG the cause? | It cancels from fixed-state direct shape differences, but can alter state feedback. Compare guided trajectories and model-state Jacobians. |
| Is latent flattening misregistered? | No for same-seed training pairs. Free-running comparison against renderer-location targets is confounded. |
| Is centered/equivariant supervision better? | Plausible if common-frame registration reveals boundary alignment while translation/background drift remains. Centered supervision already exists in some proxies; it has not solved binding. |
| Is width 96/two blocks sufficient? | Not established. No hard dimensional bottleneck proves insufficiency; a controlled frozen-state representability probe is needed before scaling. |
| Will cross-attention/adaLN fix it? | Both are candidates, not conclusions. Separate read/write projections and per-block access may help routing; adaLN is global and does not itself prescribe a boundary. |
| Is timestep supervision more important? | Fixed-t concentration is real, but historical local cosines stayed low across the scanned band. Moving one timestep alone is not supported as the remedy. |
| Are more packages needed? | No. PyTorch, finite differences, hooks, grid sampling, and directional JVP/VJP cover the required diagnostics. |

## Diagnostic contract before selecting one correction

No next correction is selected in this audit. The strongest defensible project-specific
hypothesis remains that caption changes fail to produce the desired boundary response,
with conditioning routing, noise-dependent position uncertainty, and reverse feedback
still entangled. The following measurements distinguish them:

1. Reproduce the existing 800-step VAE gate unchanged, preserving the accepted artifact
   outside `runs`, with source SHA, config, checkpoint SHA-256, VAE/diffusion steps,
   seed/runtime, and validation metrics. Stop before diffusion if the VAE gate fails.
   This is a subsequent execution stage; no VAE training was launched during this audit.
   A new baseline learned DiT specimen is required for learned-path tests because the
   deleted failed checkpoint cannot be recovered. Use unchanged source/objectives and
   record the full research settings; a reproduction is not proof of old-weight behavior.

2. On an immutable specimen, test noisy square/circle/midpoint anchors at all sampled
   timesteps PLUS t=500, with repeated noise seeds and translated object positions.
   Report epsilon, predicted-clean, and DDIM forcing separately; measure empirical
   latent power. Preserve absolute and common-frame boundary responses together.

3. Trace the actual pooled and attention conditioning, then ablate each existing branch
   without updating weights. Relate consumed shape differences to epsilon spatial
   support and decoded predicted-clean boundary changes, color, motion, and background.

4. Repair diagnostic interpretation before using its numbers as an experiment gate:
   common-frame targets, analytic-vs-model state terms, clean predictions, complete
   frozen object-validity metrics, actual timestep labels, and final-step indexing.
   The existing scripts' algebra is usable; their present outputs cannot alone authorize
   a geometry/timestep correction. Source repairs should be a separate diagnostic patch.

5. Compare guidance 1/2/3 and shape-caption switches restricted to noise bands, following
   each intervention through the remaining frozen reverse path. These are diagnostic
   protocols only. Use directional finite differences/JVPs along matched shape and
   orthogonal controls rather than constructing a dense 4096x4096 Jacobian. A high local
   scalar gain alone does not characterize a non-normal multi-step state dynamics product.

6. Preregister ONE correction only after the mechanism separates:

   | Discriminating evidence | Candidate experiment |
   | --- | --- |
   | Consumed text is separated, but boundary routing fails across registered anchors | Additional learned per-block text attention residual |
   | Registered response works in one band but off-band forcing destroys it | Timestep/SNR-band causal supervision with distribution-valid targets |
   | Absolute response is low while common-frame shape is correct, with nuisance drift | Object-frame/equivariance constraints that retain motion/position reporting |
   | Endpoint ambiguity specifically predicts failures relative to interior controls | Isolated positional-encoding correction |
   | Correct learned response survives locally but guidance-dependent state terms dominate | Reverse-dynamics/guidance-aware training study, preserving final CFG=3 |

   For an additional attention residual, zero-initialize only its output projection
   and retain existing pathways, so initialization preserves the old function. Keep
   independent Q/K/V nonzero; zeroing both output projection and scalar gate can make
   the branch unable to learn. Replacing old blocks with adaLN-Zero does not preserve
   their existing function automatically. A future architecture patch needs explicit
   function-equivalence and gradient checks, checkpoint/optimizer versioning, and a
   controlled comparison against an unchanged specimen.

Pilot success requires increased signed/common-frame boundary alignment and reduced
orthogonal/background effects on held-out contexts/seeds, with direction, color,
object validity, area, and motion tracked together. Proxy loss reduction or one positive
ranking gap is insufficient. Pilot criteria are not replacements for final acceptance:
64 balanced TEST videos, 50 steps, CFG=3, all frozen generated-video gates, materially
worse random baseline, clean code checks, and manual MP4 validation remain mandatory.

## Published evidence and its limits

- [Peebles and Xie, Scalable Diffusion Models with Transformers](https://arxiv.org/html/2212.09748v2)
  compare conditioning blocks and report adaLN-Zero advantages on their class-conditional
  ImageNet setup. This supports investigating conditioning and identity-initialized
  residuals. It does not prove that a tiny byte-conditioned video model needs adaLN,
  or that an added attention branch will bind its shape words correctly.
- [Hang et al., Min-SNR weighting](https://arxiv.org/abs/2303.09556) study conflicting
  timestep optimization and SNR-based weighting. The repo already implements
  `min(SNR,gamma)/SNR` for epsilon loss. That weighting does not by itself provide
  causal boundary supervision at missing timesteps.
- [Ho and Salimans, classifier-free guidance](https://arxiv.org/abs/2207.12598) motivate
  combining conditional/unconditional predictions. The fixed-state cancellation and
  state-feedback distinction above are derived from this repository's implemented formula.
- [Song et al., DDIM](https://arxiv.org/abs/2010.02502) provide the deterministic reverse
  sampling framework. A and B above were derived and evaluated from current source;
  published sampling theory does not identify this checkpoint's faulty vector field.
- [Kornblith et al., CKA](https://proceedings.mlr.press/v97/kornblith19a.html) compare
  representational similarity structures. The bias and sign counterexamples above are
  direct controls on the repo implementation, not a claim that CKA measures boundary binding.

## Reproducing the checkpoint-free controls

Use the installed environment, CPU, and no optimizer. For the schedule, construct
`LinearNoiseSchedule` with `configs/tiny.toml`; evaluate its `alpha_cumprod` and
`sampling_timesteps(50)` using the equations above. For registration, use the stated
seeds/contexts and crop each renderer delta at its recorded `trajectory_xy` top-left
coordinate to `[3,8,8,8]`; compare flattened absolute/cropped vectors with cosine.

The CKA control used `torch.set_num_threads(1)`, `torch.manual_seed(20261002)`, then
`build_stage_b_components(load_config('configs/tiny.toml'), device=cpu)` before drawing
two 16x4096 standard-normal matrices; 256 permutations with seed 7. The text control
used alternating `[1,-1]` repeated 48 times, offsets +/-2, actual paired tokenizer masks,
thresholds 0.25/0.35/0.15, and 256 random queries seeded 7. Float32 results may differ
slightly with platform/runtime. None of these controls evaluates learned quality.


## Diagnostic implementation follow-up

The subsequent diagnostic-only patch implements the conditioning/dynamics measurements requested by
this audit without selecting a training correction. `m1_shape_dynamics_decomposition.py` now includes
exact `t=500` local probes in addition to the frozen sampler grid; square, circle, and midpoint noisy
anchors; repeated noise seeds; deterministic alternate renderer positions; empirical latent power;
direct epsilon, predicted-clean, and DDIM forcing responses; normalized consumed pool/attention traces;
and read-only branch ablations. Reverse-step state feedback is split into the analytic `A * delta_x`
transport term and the remaining model epsilon term, with algebraic residuals reported. Selected
predicted-clean estimates are decoded with the frozen generated-video metrics instead of interpreting
noisy intermediate states as clean geometry.

The diagnostic forward is a parameter-free mirror of the current DiT and must match production within
`1e-5` before branch-ablation results are accepted. This implementation still does not recover the
deleted learned checkpoint, alter training, or establish that cross-attention, timestep-band
supervision, positional encoding, or object-frame constraints are the correct remedy. Optimizer-level
gradient/update diagnostics remain necessary if gradient conflict is reconsidered.
