from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils.ui_sim_dataset import UISimLatentDataset  # noqa: E402
from utils.ui_sim_prompts import (  # noqa: E402
    build_ui_block_prompt,
    load_graph_frame_states,
    prompt_for_block,
)


def test_graph_block_prompt_uses_state_sequence_and_deduplicates(tmp_path: Path) -> None:
    token_path = tmp_path / "traj_node_tokens.json"
    token_path.write_text(
        """
        {
          "frame_states": ["desktop:///", "/home/wmui", "/home/wmui", "/home/wmui/Documents"],
          "frames": [["ignored fallback"]]
        }
        """,
        encoding="utf-8",
    )

    states = load_graph_frame_states(token_path)
    prompt = prompt_for_block(
        mode="graph_paths",
        fixed_prompt="fallback",
        frame_states=states,
        start=0,
        num_frames=4,
    )

    assert prompt.count("Home folder") == 1
    assert "the desktop then the Home folder" in prompt
    assert 'the folder "Documents" at "/home/wmui/Documents"' in prompt
    assert "crisp and legible" in prompt


def test_graph_block_prompt_falls_back_when_states_are_missing() -> None:
    assert build_ui_block_prompt(["", " "], fixed_prompt="fallback") == "fallback"
    assert prompt_for_block(
        mode="fixed",
        fixed_prompt="fallback",
        frame_states=None,
        start=0,
        num_frames=4,
    ) == "fallback"


def test_graph_frame_states_support_existing_sidecars(tmp_path: Path) -> None:
    token_path = tmp_path / "legacy_node_tokens.json"
    token_path.write_text(
        '{"frames": [["/home/wmui"], ["/home/wmui/Documents", "folder file"]]}',
        encoding="utf-8",
    )

    assert load_graph_frame_states(token_path) == [
        "/home/wmui",
        "/home/wmui/Documents",
    ]


def test_ui_dataset_rejects_legacy_fixed_prompt_cache_in_graph_mode(tmp_path: Path) -> None:
    sample_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "clean_latent": torch.zeros(4, 16, 1, 1),
            "actions": torch.zeros(3, 3),
            "prompt": "desktop file manager UI transition",
        },
        sample_path,
    )
    config = SimpleNamespace(
        data_path=str(sample_path),
        image_or_video_shape=[1, 4, 16, 1, 1],
        ui_random_clip=False,
        ui_prompt_mode="graph_paths",
        width=832,
        height=480,
    )

    with pytest.raises(ValueError, match="Rebuild the latent cache"):
        UISimLatentDataset(sample_path, config)[0]


def test_ui_dataset_rejects_graph_prompt_cache_in_fixed_mode(tmp_path: Path) -> None:
    sample_path = tmp_path / "graph.pt"
    torch.save(
        {
            "clean_latent": torch.zeros(4, 16, 1, 1),
            "actions": torch.zeros(3, 3),
            "prompt": "A graph-derived caption.",
            "metadata": {"prompt_mode": "graph_paths"},
        },
        sample_path,
    )
    config = SimpleNamespace(
        data_path=str(sample_path),
        image_or_video_shape=[1, 4, 16, 1, 1],
        ui_random_clip=False,
        ui_prompt_mode="fixed",
        width=832,
        height=480,
    )

    with pytest.raises(ValueError, match="prompt mode mismatch"):
        UISimLatentDataset(sample_path, config)[0]
