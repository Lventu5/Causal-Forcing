import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, ODETrainer, ScoreDistillationTrainer, ConsistencyDistillationTrainer
from utils.training_checkpoint import (
    DEFAULT_CHECKPOINT_DIR_NAME,
    rolling_model_checkpoint_path,
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_auto_resume_from_last(config) -> None:
    if not _as_bool(getattr(config, "auto_resume_from_last", False)):
        return
    if not getattr(config, "logdir", ""):
        return

    checkpoint_path = rolling_model_checkpoint_path(
        config.logdir,
        checkpoint_dir_name=getattr(
            config,
            "checkpoint_dir_name",
            DEFAULT_CHECKPOINT_DIR_NAME,
        ),
    )
    if checkpoint_path is None or not checkpoint_path.exists():
        return

    previous_ckpt = str(getattr(config, "generator_ckpt", ""))
    config.generator_ckpt = str(checkpoint_path.resolve())
    config.checkpoint_mode = "resume"
    print(
        "Auto-resuming from rolling checkpoint "
        f"{config.generator_ckpt} (initial generator_ckpt was {previous_ckpt or '<unset>'})."
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")

    args, overrides = parser.parse_known_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf 
    # get the filename of config_path
    config_name = os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME")
    if not config_name:
        config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir or os.environ.get("WANDB_SAVE_DIR", "")
    config.disable_wandb = args.disable_wandb or bool(getattr(config, "disable_wandb", False))
    _maybe_auto_resume_from_last(config)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "consistency_distillation":
        trainer = ConsistencyDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
