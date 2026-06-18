from pathlib import Path
import sys

import pytest
import torch


CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils.training_checkpoint import (  # noqa: E402
    checkpoint_dir_for_step,
    extract_generator_state,
    load_checkpoint,
    load_trainer_payload,
    rolling_model_checkpoint_path,
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


def test_rolling_checkpoint_paths_use_last_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "action_stage1"

    assert (
        checkpoint_dir_for_step(output_path, 2000, checkpoint_dir_name="last")
        == output_path / "last"
    )
    assert (
        rolling_model_checkpoint_path(output_path, checkpoint_dir_name="last")
        == output_path / "last" / "model.pt"
    )


def test_step_numbered_checkpoint_paths_can_still_be_requested(tmp_path: Path) -> None:
    output_path = tmp_path / "action_stage1"

    assert (
        checkpoint_dir_for_step(output_path, 2000, checkpoint_dir_name="step")
        == output_path / "checkpoint_model_002000"
    )
    assert rolling_model_checkpoint_path(output_path, checkpoint_dir_name="step") is None


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


def test_actionmap_stage_initializes_from_old_action_stage1(tmp_path: Path) -> None:
    model_path = tmp_path / "action_stage1" / "last" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_stage": "action_stage1",
            "step": 25000,
            "generator": {"weight": torch.tensor([1.0])},
        },
        model_path,
    )

    checkpoint = load_checkpoint(
        model_path,
        current_stage="action_stage1_actionmap",
        checkpoint_mode="auto",
    )

    assert not checkpoint.is_resume
    assert checkpoint.checkpoint_stage == "action_stage1"
    assert checkpoint.step == 25000


def test_legacy_actionxattn_checkpoint_infers_stage_from_path(tmp_path: Path) -> None:
    model_path = tmp_path / "action_stage1_actionxattn" / "last" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save({"generator": {"weight": torch.tensor([1.0])}}, model_path)

    checkpoint = load_checkpoint(
        model_path,
        current_stage="action_stage1_actionxattn",
        checkpoint_mode="auto",
    )

    assert checkpoint.is_resume
    assert checkpoint.checkpoint_stage == "action_stage1_actionxattn"


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


def test_mismatched_trainer_payload_is_ignored(tmp_path: Path) -> None:
    model_path = tmp_path / "action_stage1" / "last" / "model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_stage": "action_stage1",
            "step": 2000,
            "generator": {"weight": torch.tensor([1.0])},
        },
        model_path,
    )
    torch.save(
        {
            "training_stage": "action_stage1",
            "step": 1000,
            "generator_optimizer": {"state": {}},
        },
        trainer_checkpoint_path(model_path),
    )

    checkpoint = load_checkpoint(
        model_path,
        current_stage="action_stage1",
        checkpoint_mode="auto",
    )

    assert load_trainer_payload(checkpoint) is None


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
