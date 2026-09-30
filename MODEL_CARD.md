# Model Card — Development Line

## Status

No frontier model checkpoint is included in this starter kit. The code initializes all neural components randomly.

## Intended use

Research and development of a from-scratch generative video foundation model.

## Current capability

Milestone 1 is active. The code now includes a balanced deterministic synthetic-motion curriculum and a supervised direction/color sanity probe on top of the original tokenizer/text encoder, tiny video autoencoder, Video DiT and diffusion primitives.

## Not currently supported

Do not describe the starter as supporting production text-to-video, photorealistic humans, 30-second narratives, native audio, 1080p, editing or multimodal references. Those are roadmap targets.

## Training data

The development line uses deterministic synthetic clips only. M1 train/validation/test samples occupy disjoint deterministic seed spaces; future real-data checkpoints must link to a versioned provenance manifest.
