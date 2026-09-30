# Coding Guideline

## Runtime and tooling

- Python 3.14 only for the current development line.
- Use `uv` for Python installation, environments, dependencies and command execution.
- Use modern type syntax (`list[str]`, `X | None`).
- Public functions, methods and dataclasses must be typed.
- `mypy --strict` is the target for package code.
- `ruff` owns formatting and linting. Do not hand-format around it.
- Tests use `pytest`.

## Model code

- Tensor layout is `[B, C, T, H, W]` unless a function documents a different layout.
- Put layout conversions next to the operation that requires them.
- Assert divisibility/shape invariants before silent reshaping.
- No global device assumptions. Infer from tensors or accept an explicit device.
- Avoid `.cuda()` in library code. Use `.to(device)`.
- Do not hardcode `float16`; precision is a configuration concern.
- Keep model forward methods side-effect free.
- New architectures must have a CPU shape/gradient test with a tiny configuration.
- Parameter initialization must be explicit when it materially differs from PyTorch defaults.

## Reproducibility

Every experiment must have:

- config file checked into source control or preserved with the run;
- seed;
- dataset manifest/version;
- code commit identifier when available;
- checkpoint step;
- model parameter count;
- precision and device topology.

A result without enough metadata to rerun it is an observation, not a benchmark.

## Data rules

- Never silently crawl or ingest arbitrary web video.
- Every dataset item must resolve to provenance metadata.
- Train/validation/test splits must be stable and leakage-checked.
- Synthetic generators must be deterministic from a seed.
- Personal/private media must not enter a shared training corpus without explicit permission.

See `DATA_POLICY.md`.

## Dependency policy

Core ML dependencies should stay small. Adding a package requires answering:

1. Can the standard library or PyTorch do this safely?
2. Is the dependency maintained for Python 3.14?
3. Does it download or invoke pretrained AI assets implicitly?
4. Is its license compatible with the project?
5. Will it become part of the checkpoint/runtime contract?

No dependency may silently download pretrained model weights.

## Third-party AI boundary

The default project must not depend on:

- hosted model APIs;
- pretrained video/image/audio models;
- pretrained text encoders;
- pretrained tokenizers with learned vocabularies;
- pretrained perceptual/evaluation networks.

A research comparison may live in a separately labeled optional experiment, but it must not contaminate the from-scratch checkpoint lineage.

## Modules

- `data/`: acquisition manifests, decoding, sampling, augmentation, synthetic data.
- `models/`: neural network definitions only.
- `diffusion/`: objectives, schedules and samplers.
- `training/`: optimization and orchestration.
- `inference/`: generation APIs and user-facing sampling logic.
- `utils/`: narrowly scoped infrastructure helpers.

Do not place training loops inside model classes.

## Errors and logging

- Fail early on invalid shapes/configuration.
- Error messages must say what was expected and what was received.
- Do not catch `Exception` unless adding useful context and re-raising.
- Training logs should report machine-readable scalars plus concise human-readable progress.
- Never log raw private dataset content by default.

## Testing

A change is incomplete without tests when it changes:

- tensor shape semantics;
- checkpoint schema;
- tokenizer behavior;
- diffusion math;
- dataset determinism;
- conditioning behavior;
- distributed/resume semantics.

Tests should be tiny and deterministic. Expensive quality evaluation belongs in milestone suites, not normal unit tests.

## Git conventions

Recommended commit prefixes:

```text
feat:     new capability
fix:      correctness fix
model:    architecture/model change
data:     dataset pipeline change
train:    training change
infer:    inference/sampling change
test:     tests/evaluation
docs:     documentation
perf:     measured performance change
```

Avoid commits that combine architecture changes, unrelated refactors and formatting noise.

## Performance changes

Do not merge an optimization because it 'looks faster'. Record:

- hardware;
- input shape;
- precision;
- warmup;
- measured throughput/latency;
- peak memory where relevant;
- quality/correctness comparison.

Numerical changes require an explicit tolerance or quality evaluation.

## Documentation

When a milestone changes setup, architecture, commands, config or checkpoint behavior, update `README.md`, `ARCHITECTURE.md` and/or `MILESTONES_TESTING.md` in the same change.
