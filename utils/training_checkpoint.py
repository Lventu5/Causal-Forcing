from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_FORMAT = "cf_ui_training_v1"
CHECKPOINT_MODES = {"auto", "initialize", "resume"}
DEFAULT_CHECKPOINT_DIR_NAME = "last"
STEP_CHECKPOINT_DIR_NAMES = {"", "0", "false", "none", "step", "step_numbered"}
KNOWN_STAGES = (
    "action_node_stage3_dmd",
    "action_node_ca_warmup",
    "action_node_stage2",
    "action_node_stage1",
    "action_stage1_actionxattn",
    "action_stage1_actionmap",
    "action_stage3_dmd",
    "action_stage2",
    "action_stage1",
)
_STEP_PATTERN = re.compile(r"checkpoint_model_(\d+)")


@dataclass(frozen=True)
class CheckpointLoad:
    model_path: Path
    payload: Mapping[str, Any]
    mode: str
    checkpoint_stage: str | None
    step: int
    initialization_checkpoint: str | None

    @property
    def is_resume(self) -> bool:
        return self.mode == "resume"


def resolve_model_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    return checkpoint_path.resolve()


def trainer_checkpoint_path(model_path: str | os.PathLike[str]) -> Path:
    return resolve_model_checkpoint_path(model_path).with_name("trainer.pt")


def normalize_checkpoint_dir_name(
    checkpoint_dir_name: str | os.PathLike[str] | None,
) -> str | None:
    if checkpoint_dir_name is None:
        return None
    name = str(checkpoint_dir_name).strip()
    if name.lower() in STEP_CHECKPOINT_DIR_NAMES:
        return None
    name_path = Path(name)
    if name in {".", ".."} or name_path.is_absolute() or name_path.name != name:
        raise ValueError(
            "checkpoint_dir_name must be a single directory name, "
            f"got {checkpoint_dir_name!r}"
        )
    return name


def checkpoint_dir_for_step(
    output_path: str | os.PathLike[str],
    step: int,
    *,
    checkpoint_dir_name: str | os.PathLike[str] | None = None,
) -> Path:
    checkpoint_dir_name = normalize_checkpoint_dir_name(checkpoint_dir_name)
    if checkpoint_dir_name is not None:
        return Path(output_path).expanduser() / checkpoint_dir_name
    return Path(output_path).expanduser() / f"checkpoint_model_{int(step):06d}"


def rolling_model_checkpoint_path(
    output_path: str | os.PathLike[str],
    *,
    checkpoint_dir_name: str | os.PathLike[str] | None = DEFAULT_CHECKPOINT_DIR_NAME,
) -> Path | None:
    checkpoint_dir_name = normalize_checkpoint_dir_name(checkpoint_dir_name)
    if checkpoint_dir_name is None:
        return None
    return Path(output_path).expanduser() / checkpoint_dir_name / "model.pt"


def infer_checkpoint_stage(
    payload: Mapping[str, Any],
    checkpoint_path: str | os.PathLike[str],
) -> str | None:
    stage = payload.get("training_stage")
    if isinstance(stage, str) and stage:
        return stage

    normalized_path = str(resolve_model_checkpoint_path(checkpoint_path))
    for known_stage in KNOWN_STAGES:
        if re.search(rf"(^|/){re.escape(known_stage)}(/|$)", normalized_path):
            return known_stage
    return None


def infer_checkpoint_step(
    payload: Mapping[str, Any],
    checkpoint_path: str | os.PathLike[str],
) -> int:
    step = payload.get("step")
    if isinstance(step, int) and step >= 0:
        return step

    match = _STEP_PATTERN.search(str(resolve_model_checkpoint_path(checkpoint_path)))
    return int(match.group(1)) if match else 0


def load_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    current_stage: str,
    checkpoint_mode: str = "auto",
) -> CheckpointLoad:
    mode = checkpoint_mode.strip().lower()
    if mode not in CHECKPOINT_MODES:
        raise ValueError(
            f"checkpoint_mode must be one of {sorted(CHECKPOINT_MODES)}, got {checkpoint_mode!r}"
        )

    model_path = resolve_model_checkpoint_path(checkpoint_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {model_path} must contain a mapping.")

    checkpoint_stage = infer_checkpoint_stage(payload, model_path)
    if mode == "auto":
        mode = (
            "resume"
            if current_stage and checkpoint_stage == current_stage
            else "initialize"
        )
    elif mode == "resume":
        if checkpoint_stage is not None and checkpoint_stage != current_stage:
            raise ValueError(
                "Cannot resume a different training stage: "
                f"current={current_stage!r}, checkpoint={checkpoint_stage!r}. "
                "Use checkpoint_mode=initialize for a stage handoff."
            )

    initialization_checkpoint = payload.get("initialization_checkpoint")
    if not isinstance(initialization_checkpoint, str) or not initialization_checkpoint:
        initialization_checkpoint = None

    return CheckpointLoad(
        model_path=model_path,
        payload=payload,
        mode=mode,
        checkpoint_stage=checkpoint_stage,
        step=infer_checkpoint_step(payload, model_path),
        initialization_checkpoint=initialization_checkpoint,
    )


def load_trainer_payload(checkpoint: CheckpointLoad) -> Mapping[str, Any] | None:
    path = trainer_checkpoint_path(checkpoint.model_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Trainer checkpoint {path} must contain a mapping.")
    trainer_step = infer_checkpoint_step(payload, path)
    trainer_stage = infer_checkpoint_stage(payload, path)
    if trainer_step != checkpoint.step or (
        trainer_stage is not None
        and checkpoint.checkpoint_stage is not None
        and trainer_stage != checkpoint.checkpoint_stage
    ):
        return None
    return payload


def extract_generator_state(
    payload: Mapping[str, Any],
    *,
    for_resume: bool,
) -> Mapping[str, Any]:
    if for_resume and "generator" in payload:
        state = payload["generator"]
    elif not for_resume and "generator_ema" in payload:
        state = payload["generator_ema"]
    elif "generator" in payload:
        state = payload["generator"]
    elif "model" in payload:
        state = payload["model"]
    elif "generator_ema" in payload:
        state = payload["generator_ema"]
    else:
        state = payload

    fixed = {}
    for key, value in state.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


def checkpoint_metadata(
    *,
    training_stage: str,
    step: int,
    initialization_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "training_stage": training_stage,
        "step": int(step),
        "initialization_checkpoint": initialization_checkpoint,
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary_path)
    os.replace(temporary_path, path)
