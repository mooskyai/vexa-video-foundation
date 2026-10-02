from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import optuna
import torch

from vexa_video.config import load_config
from vexa_video.training.m1_research import M1ResearchSettings
from vexa_video.training.trainer import train_stage_b
from vexa_video.utils.seed import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short constrained M1 research search")
    parser.add_argument("--config", default="configs/tiny.toml")
    parser.add_argument("--resume", required=True, help="VAE-ready checkpoint")
    parser.add_argument("--run-dir", default="runs/m1-research-search")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--cuda", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.trials <= 0 or args.steps <= 0:
        raise ValueError("trials and steps must be positive")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but CUDA is not available")

    cfg = load_config(args.config)
    device = torch.device("cuda" if args.cuda else "cpu")
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(root / 'study.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name="m1-research-rescue",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        settings = M1ResearchSettings(
            causal_shape_weight=trial.suggest_float("causal_shape_weight", 0.25, 2.5, log=True),
            latent_rank_weight=trial.suggest_float("latent_rank_weight", 0.5, 2.0, log=True),
            latent_rank_margin=trial.suggest_float("latent_rank_margin", 0.10, 0.50),
            sinkhorn_shape_weight=trial.suggest_float("sinkhorn_shape_weight", 0.10, 1.5, log=True),
            sinkhorn_blur=trial.suggest_float("sinkhorn_blur", 0.06, 0.24, log=True),
            sinkhorn_margin=trial.suggest_float("sinkhorn_margin", 0.005, 0.08, log=True),
            sinkhorn_gate_start=0.10,
            sinkhorn_gate_full=0.30,
            min_snr_gamma=trial.suggest_float("min_snr_gamma", 2.0, 8.0, log=True),
            shape_gradient_surgery=True,
            gradient_diagnostics=True,
            svd_samples=32,
        )
        settings.validate()
        trial_dir = root / f"trial-{trial.number:03d}"
        if trial_dir.exists():
            raise RuntimeError(f"trial directory already exists: {trial_dir}")
        seed_everything(cfg.seed)
        result = train_stage_b(
            cfg,
            device=device,
            run_dir=trial_dir,
            steps=args.steps,
            resume=args.resume,
            research=settings,
        )
        trial.set_user_attr("settings", asdict(settings))
        trial.set_user_attr("best_checkpoint", str(result.best_checkpoint))
        return result.best_validation_score

    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    summary = {
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": best.params,
        "trials": len(study.trials),
        "steps_per_trial": args.steps,
    }
    (root / "best-trial.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
