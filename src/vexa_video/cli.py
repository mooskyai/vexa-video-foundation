from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from vexa_video.config import load_config
from vexa_video.data.synthetic import render_moving_square
from vexa_video.diffusion import LinearNoiseSchedule
from vexa_video.inference import sample_video
from vexa_video.models import ByteTokenizer, TinyVideoVAE, TransformerTextEncoder, VideoDiT
from vexa_video.training import (
    build_stage_b_components,
    evaluate_m1_generation,
    load_stage_b_weights,
    train_m1_probe,
    train_stage_b,
)
from vexa_video.utils.seed import seed_everything


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but CUDA is not available")
    return torch.device("cuda" if args.cuda else "cpu")


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda={torch.version.cuda}")
        print(f"gpu={torch.cuda.get_device_name(0)}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    tokenizer = ByteTokenizer()
    tokens = tokenizer.batch(["a red square moves right"], cfg.text.max_length, device=device)
    text_encoder = TransformerTextEncoder(
        vocab_size=cfg.text.vocab_size,
        max_length=cfg.text.max_length,
        d_model=cfg.text.d_model,
        layers=cfg.text.layers,
        heads=cfg.text.heads,
        ff_mult=cfg.text.ff_mult,
    ).to(device)
    vae = TinyVideoVAE(
        in_channels=cfg.vae.in_channels,
        latent_channels=cfg.vae.latent_channels,
        base_channels=cfg.vae.base_channels,
    ).to(device)
    dit = VideoDiT(
        latent_channels=cfg.vae.latent_channels,
        text_dim=cfg.text.d_model,
        hidden_size=cfg.dit.hidden_size,
        layers=cfg.dit.layers,
        heads=cfg.dit.heads,
        patch_size=(cfg.dit.patch_t, cfg.dit.patch_h, cfg.dit.patch_w),
    ).to(device)
    schedule = LinearNoiseSchedule(
        cfg.diffusion.timesteps,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
    )

    sample = render_moving_square(
        frames=cfg.data.frames,
        size=cfg.data.height,
        seed=cfg.seed,
    )
    video = sample.video.unsqueeze(0).to(device)
    text = text_encoder(tokens.input_ids, tokens.attention_mask)
    reconstruction, latents = vae(video)
    noise = torch.randn_like(latents)
    timestep = torch.tensor([cfg.diffusion.timesteps // 2], device=device)
    noisy = schedule.add_noise(latents, noise, timestep)
    predicted = dit(noisy, timestep, text, tokens.attention_mask)
    loss = torch.nn.functional.mse_loss(predicted, noise)
    torch.autograd.backward(loss)

    print(f"device={device}")
    print(f"config={asdict(cfg)}")
    print(f"video_shape={tuple(video.shape)}")
    print(f"latent_shape={tuple(latents.shape)}")
    print(f"reconstruction_shape={tuple(reconstruction.shape)}")
    print(f"dit_output_shape={tuple(predicted.shape)}")
    print(f"text_params={parameter_count(text_encoder):,}")
    print(f"vae_params={parameter_count(vae):,}")
    print(f"dit_params={parameter_count(dit):,}")
    print(f"loss={loss.item():.6f}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    sample = render_moving_square(frames=args.frames, size=args.size, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video": sample.video,
            "caption": sample.caption,
            "trajectory_xy": sample.trajectory_xy,
            "seed": args.seed,
        },
        output,
    )
    print(f"saved={output}")
    print(f"caption={sample.caption}")
    print(f"shape={tuple(sample.video.shape)}")
    return 0


def cmd_m1_probe(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = _device_from_args(args)
    result = train_m1_probe(
        cfg,
        device=device,
        run_dir=args.run_dir,
        steps=args.steps,
    )
    print(f"device={device}")
    print(f"baseline_direction_accuracy={result.baseline_direction_accuracy:.6f}")
    print(f"baseline_color_accuracy={result.baseline_color_accuracy:.6f}")
    print(f"direction_accuracy={result.direction_accuracy:.6f}")
    print(f"color_accuracy={result.color_accuracy:.6f}")
    print(f"gate_passed={result.gate_passed}")
    print(f"checkpoint={result.checkpoint}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = _device_from_args(args)
    result = train_stage_b(
        cfg,
        device=device,
        run_dir=args.run_dir,
        steps=args.steps,
        resume=args.resume,
    )
    print(f"device={device}")
    print(f"diffusion_step={result.diffusion_step}")
    print(f"vae_validation_reconstruction_loss={result.vae_validation_reconstruction_loss:.6f}")
    print(f"best_validation_score={result.best_validation_score:.6f}")
    print(f"latest_checkpoint={result.latest_checkpoint}")
    print(f"best_checkpoint={result.best_checkpoint}")
    return 0


def cmd_evaluate_m1(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = _device_from_args(args)
    components = build_stage_b_components(cfg, device=device)
    evaluation_kind = "random_baseline"
    if args.checkpoint is not None:
        load_stage_b_weights(args.checkpoint, components=components, device=device)
        evaluation_kind = "checkpoint"
    output = args.output
    result = evaluate_m1_generation(
        cfg=cfg,
        components=components,
        device=device,
        samples=args.samples,
        sampling_steps=args.sampling_steps,
        output=output,
    )
    print(f"evaluation={evaluation_kind}")
    print(f"device={device}")
    print(f"samples={result.samples}")
    print(f"sampling_steps={result.sampling_steps}")
    print(f"guidance_scale={result.guidance_scale:.6f}")
    print(f"direction_accuracy={result.direction_accuracy:.6f}")
    print(f"color_accuracy={result.color_accuracy:.6f}")
    print(f"mean_motion={result.mean_motion:.6f}")
    print(f"static_rate={result.static_rate:.6f}")
    print(f"direction_confusion={result.direction_confusion}")
    print(f"color_confusion={result.color_confusion}")
    print(f"gate_passed={result.gate_passed}")
    if output is not None:
        print(f"metrics={output}")
    return 0


def _save_mp4(video: Tensor, output: Path, *, fps: int) -> None:
    rgb = ((video.detach().cpu() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    frames = rgb.permute(1, 2, 3, 0).contiguous()
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        input=frames.numpy().tobytes(),
        check=True,
    )


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = _device_from_args(args)
    components = build_stage_b_components(cfg, device=device)
    load_stage_b_weights(args.checkpoint, components=components, device=device)
    video = sample_video(
        cfg=cfg,
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        vae=components.vae,
        dit=components.dit,
        schedule=components.schedule,
        prompts=[args.prompt],
        seed=args.seed,
        sampling_steps=args.sampling_steps,
        device=device,
    )[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4":
        _save_mp4(video, output, fps=args.fps)
    else:
        torch.save({"video": video.cpu(), "prompt": args.prompt, "seed": args.seed}, output)
    print(f"saved={output}")
    print(f"shape={tuple(video.shape)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vexa-video")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Show Python/PyTorch/CUDA environment")
    doctor.set_defaults(func=cmd_doctor)

    smoke = sub.add_parser("smoke", help="Run an end-to-end tiny model smoke test")
    smoke.add_argument("--config", default="configs/tiny.toml")
    smoke.add_argument("--cuda", action="store_true", help="Use CUDA when available")
    smoke.set_defaults(func=cmd_smoke)

    m1_probe = sub.add_parser(
        "m1-probe",
        help="Train the supervised M1 temporal/color sanity probe",
    )
    m1_probe.add_argument("--config", default="configs/tiny.toml")
    m1_probe.add_argument("--run-dir", default="runs/m1-probe")
    m1_probe.add_argument("--steps", type=int, help="Override configured probe steps")
    m1_probe.add_argument("--cuda", action="store_true", help="Require CUDA")
    m1_probe.set_defaults(func=cmd_m1_probe)

    train = sub.add_parser("train", help="Train M1 Stage-B generative synthetic motion")
    train.add_argument("--config", default="configs/tiny.toml")
    train.add_argument("--run-dir", default="runs/m1-stage-b")
    train.add_argument("--steps", type=int, help="Override configured diffusion steps")
    train.add_argument("--resume", help="Resume a Stage-B checkpoint")
    train.add_argument("--cuda", action="store_true", help="Require CUDA")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate-m1", help="Evaluate complete generated M1 videos")
    evaluate.add_argument("--config", default="configs/tiny.toml")
    evaluation_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluation_source.add_argument("--checkpoint", help="Stage-B checkpoint to evaluate")
    evaluation_source.add_argument(
        "--random-baseline",
        action="store_true",
        help="Evaluate the frozen random model baseline",
    )
    evaluate.add_argument("--samples", type=int, help="Balanced sample count (multiple of 16)")
    evaluate.add_argument("--sampling-steps", type=int, help="Reverse diffusion steps")
    evaluate.add_argument("--output", help="Optional metrics JSON path")
    evaluate.add_argument("--cuda", action="store_true", help="Require CUDA")
    evaluate.set_defaults(func=cmd_evaluate_m1)

    generate = sub.add_parser("generate", help="Generate a video from a Stage-B checkpoint")
    generate.add_argument("--config", default="configs/tiny.toml")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--sampling-steps", type=int, help="Reverse diffusion steps")
    generate.add_argument("--fps", type=int, default=8)
    generate.add_argument("--cuda", action="store_true", help="Require CUDA")
    generate.set_defaults(func=cmd_generate)

    synth = sub.add_parser("synth", help="Create a deterministic synthetic motion sample")
    synth.add_argument("--output", default="outputs/sample.pt")
    synth.add_argument("--frames", type=int, default=16)
    synth.add_argument("--size", type=int, default=64)
    synth.add_argument("--seed", type=int, default=42)
    synth.set_defaults(func=cmd_synth)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
