from pathlib import Path
import sys

import pytest
import torch


CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils.training_checkpoint import (  # noqa: E402
    extract_generator_state,
    load_checkpoint,
    trainer_checkpoint_path,
)


def test_auto_resumes_same_stage_and_restores_recorded_step(tmp_path: Path) -> None:
    model_path = tmp_path / "action_stage1" / "checkpoint_model_002000" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_stage": "action_stage1",
            "step": 2000,
            "generator": {"weight": torch.tensor([1.0])},
        },
        model_path,
    )

    checkpoint = load_checkpoint(
        model_path,
        current_stage="action_stage1",
        checkpoint_mode="auto",
    )

    assert checkpoint.is_resume
    assert checkpoint.step == 2000
    assert trainer_checkpoint_path(model_path).name == "trainer.pt"


def test_auto_initializes_new_stage_and_prefers_ema(tmp_path: Path) -> None:
    model_path = tmp_path / "action_stage1" / "checkpoint_model_003000" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_stage": "action_stage1",
            "step": 3000,
            "generator": {"weight": torch.tensor([1.0])},
            "generator_ema": {"weight": torch.tensor([2.0])},
        },
        model_path,
    )

    checkpoint = load_checkpoint(
        model_path,
        current_stage="action_stage2",
        checkpoint_mode="auto",
    )
    state = extract_generator_state(
        checkpoint.payload,
        for_resume=checkpoint.is_resume,
    )

    assert not checkpoint.is_resume
    assert checkpoint.step == 3000
    assert state["weight"].item() == 2.0


def test_legacy_checkpoint_infers_stage_and_step_from_path(tmp_path: Path) -> None:
    model_path = tmp_path / "action_node_stage2" / "checkpoint_model_004500" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save({"generator": {"weight": torch.tensor([1.0])}}, model_path)

    checkpoint = load_checkpoint(
        model_path,
        current_stage="action_node_stage2",
        checkpoint_mode="auto",
    )

    assert checkpoint.is_resume
    assert checkpoint.checkpoint_stage == "action_node_stage2"
    assert checkpoint.step == 4500


def test_explicit_resume_rejects_previous_stage(tmp_path: Path) -> None:
    model_path = tmp_path / "action_stage1" / "checkpoint_model_001000" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_stage": "action_stage1",
            "generator": {"weight": torch.tensor([1.0])},
        },
        model_path,
    )

    with pytest.raises(ValueError, match="different training stage"):
        load_checkpoint(
            model_path,
            current_stage="action_stage2",
            checkpoint_mode="resume",
        )
