from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data._utils.collate import default_collate

CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils.ui_sim_conditioning import (  # noqa: E402
    attach_ui_batch_conditioning,
    normalize_action_coordinates,
    unpack_packed_graph_tokens,
    ui_conditioning_dropout_kwargs,
)
from utils.ui_sim_dataset import UISimLatentDataset  # noqa: E402
from utils.ui_sim_element_loss import (  # noqa: E402
    build_element_loss_weighter,
    build_element_loss_weight_map,
)
from utils.wan_wrapper import UIActionNodeConditioner  # noqa: E402
from wan.modules.model import WanCrossAttention  # noqa: E402


def test_normalize_action_coordinates_letterboxes_legacy_source() -> None:
    actions = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [5.0, 0.0, 0.0]])

    mapped = normalize_action_coordinates(
        actions,
        source_width=1024,
        source_height=768,
        target_width=832,
        target_height=480,
        coordinate_mode="legacy_normalized_source",
    )

    assert torch.isclose(mapped[0, 1], torch.tensor(96.0 / 832.0))
    assert torch.isclose(mapped[0, 2], torch.tensor(0.5))
    assert torch.isclose(mapped[1, 1], torch.tensor(0.5))
    assert torch.isclose(mapped[1, 2], torch.tensor(0.5))
    assert torch.equal(mapped[2, 1:3], torch.zeros(2))


def test_unpack_packed_graph_tokens_with_mask() -> None:
    packed = torch.tensor(
        [
            [
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                1.0, 0.0,
            ]
        ]
    )

    tokens, mask = unpack_packed_graph_tokens(
        packed,
        tokens_per_frame=2,
        token_dim=3,
        has_mask=True,
    )

    assert tokens.shape == (1, 2, 3)
    assert mask is not None
    assert mask.tolist() == [[True, False]]
    assert torch.equal(tokens[0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_attach_ui_batch_conditioning_shifts_i2v_source_rows() -> None:
    batch = {
        "actions": torch.tensor([[[1.0, 0.1, 0.2], [2.0, 0.3, 0.4], [3.0, 0.5, 0.6]]]),
        "node_tokens": torch.arange(1 * 3 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 4),
        "node_mask": torch.tensor([[[True, False], [True, True], [False, True]]]),
    }
    prompt = {"prompt_embeds": torch.ones(1, 2, 8)}

    cond, uncond = attach_ui_batch_conditioning(
        batch,
        prompt,
        prompt,
        device="cpu",
        dtype=torch.float32,
        num_latent_frames=4,
        i2v=True,
    )

    assert cond["action_cond"].shape == (1, 4, 3)
    assert torch.equal(cond["action_cond"][0, 0], torch.zeros(3))
    assert torch.equal(cond["action_cond"][0, 1], batch["actions"][0, 0])
    assert cond["node_tokens"].shape == (1, 4, 2, 4)
    assert cond["node_mask"][0, 0].tolist() == [False, False]
    assert torch.equal(uncond["action_cond"], torch.zeros_like(cond["action_cond"]))
    assert torch.equal(uncond["node_mask"], torch.zeros_like(cond["node_mask"]))


def test_attach_ui_batch_conditioning_drops_node_condition_signal() -> None:
    batch = {
        "actions": torch.ones(1, 3, 3),
        "node_tokens": torch.ones(1, 3, 2, 4),
        "node_mask": torch.ones(1, 3, 2, dtype=torch.bool),
    }
    prompt = {"prompt_embeds": torch.ones(1, 2, 8)}

    cond, uncond = attach_ui_batch_conditioning(
        batch,
        prompt,
        prompt,
        device="cpu",
        dtype=torch.float32,
        num_latent_frames=4,
        i2v=True,
        condition_dropout_enabled=True,
        node_dropout=1.0,
    )

    assert torch.equal(cond["node_tokens"], torch.zeros_like(cond["node_tokens"]))
    assert torch.equal(cond["node_mask"], torch.zeros_like(cond["node_mask"]))
    assert torch.equal(uncond["node_tokens"], torch.zeros_like(cond["node_tokens"]))
    assert torch.equal(uncond["node_mask"], torch.zeros_like(cond["node_mask"]))
    assert torch.equal(cond["action_cond"][0, 1:], batch["actions"][0])


def test_ui_conditioning_dropout_kwargs_only_uses_node_dropout_for_block_crossattn() -> None:
    config = SimpleNamespace(
        model_kwargs=SimpleNamespace(
            ui_conditioning=SimpleNamespace(
                enabled=True,
                block_cross_attn=True,
                action_dropout=0.10,
                node_dropout=0.25,
            )
        )
    )

    kwargs = ui_conditioning_dropout_kwargs(config)

    assert kwargs == {
        "condition_dropout_enabled": True,
        "action_dropout": 0.0,
        "node_dropout": 0.25,
    }


def test_ui_sim_latent_dataset_slices_source_rows(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.pt"
    clean_latent = torch.arange(5 * 16 * 1 * 1, dtype=torch.float32).reshape(5, 16, 1, 1)
    actions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.25, 0.25],
            [2.0, 0.50, 0.50],
            [3.0, 0.75, 0.75],
        ]
    )
    node_values = torch.arange(4 * (2 * 4 + 2), dtype=torch.float32).reshape(4, 10)
    node_values[:, 8:] = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    torch.save(
        {
            "clean_latent": clean_latent,
            "actions": actions,
            "node_emb": node_values,
            "metadata": {
                "source_width": 1024,
                "source_height": 768,
                "action_coordinate_mode": "already_letterboxed_normalized",
            },
        },
        sample_path,
    )

    config = SimpleNamespace(
        data_path=str(sample_path),
        image_or_video_shape=[1, 4, 16, 1, 1],
        ui_random_clip=False,
        ui_graph_tokens_per_frame=2,
        ui_graph_token_dim=4,
        ui_graph_token_has_mask=True,
        width=832,
        height=480,
    )

    dataset = UISimLatentDataset(sample_path, config)
    item = dataset[0]

    assert item["clean_latent"].shape == (4, 16, 1, 1)
    assert item["actions"].shape == (3, 3)
    assert torch.equal(item["actions"], actions[:3])
    assert item["node_tokens"].shape == (3, 2, 4)
    assert item["node_mask"].shape == (3, 2)
    assert item["clip_metadata"]["start_frame"] == 0
    assert item["clip_metadata"]["end_frame"] == 4


def test_ui_sim_latent_dataset_loads_referenced_node_rows(tmp_path: Path) -> None:
    node_path = tmp_path / "traj_node_emb.pt"
    node_values = torch.arange(6 * (2 * 4 + 2), dtype=torch.float32).reshape(6, 10)
    node_values[:, 8:] = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    torch.save(node_values, node_path)

    sample_path = tmp_path / "sample_ref.pt"
    torch.save(
        {
            "clean_latent": torch.zeros(4, 16, 1, 1),
            "actions": torch.zeros(3, 3),
            "node_emb_path": str(node_path),
            "node_emb_start": 2,
            "metadata": {
                "source_width": 1024,
                "source_height": 768,
                "action_coordinate_mode": "already_letterboxed_normalized",
            },
        },
        sample_path,
    )

    config = SimpleNamespace(
        data_path=str(sample_path),
        image_or_video_shape=[1, 4, 16, 1, 1],
        ui_random_clip=False,
        ui_graph_tokens_per_frame=2,
        ui_graph_token_dim=4,
        ui_graph_token_has_mask=True,
        width=832,
        height=480,
    )

    item = UISimLatentDataset(sample_path, config)[0]

    assert item["node_tokens"].shape == (3, 2, 4)
    assert torch.equal(item["node_tokens"][0, 0], node_values[2, :4])
    assert item["node_mask"].tolist() == [[False, True], [True, False], [True, True]]


def test_cf_element_loss_uses_clip_metadata_and_letterbox(tmp_path: Path) -> None:
    cache_dir = tmp_path / "element_cache"
    cache_dir.mkdir()
    source_video = tmp_path / "traj.mp4"
    (cache_dir / "traj.json").write_text(
        """
        {
          "trajectory_path": "traj.mp4",
          "frames": [
            [{"type": "push button", "label": "Open", "bbox": [0.0, 0.0, 0.5, 1.0]}],
            [{"type": "push button", "label": "Open", "bbox": [0.0, 0.0, 0.5, 1.0]}],
            [{"type": "push button", "label": "Open", "bbox": [0.0, 0.0, 0.5, 1.0]}],
            [{"type": "push button", "label": "Open", "bbox": [0.0, 0.0, 0.5, 1.0]}]
          ]
        }
        """,
        encoding="utf-8",
    )

    sample_path = tmp_path / "sample.pt"
    torch.save(
        {
            "clean_latent": torch.zeros(4, 16, 6, 10),
            "actions": torch.zeros(3, 3),
            "metadata": {
                "source_video": str(source_video),
                "source_width": 100,
                "source_height": 100,
                "target_width": 200,
                "target_height": 100,
                "action_coordinate_mode": "already_letterboxed_normalized",
            },
        },
        sample_path,
    )
    dataset = UISimLatentDataset(
        sample_path,
        SimpleNamespace(
            data_path=str(sample_path),
            image_or_video_shape=[1, 4, 16, 6, 10],
            ui_random_clip=False,
            width=200,
            height=100,
        ),
    )
    batch = default_collate([dataset[0]])
    config = SimpleNamespace(
        element_loss=SimpleNamespace(
            enabled=True,
            cache_dir=str(cache_dir),
            base_weight=1.0,
            element_boost=7.0,
            text_cache_dir="",
            text_boost=25.0,
            text_min_confidence=0.5,
            text_padding_px=1,
        )
    )

    weighter = build_element_loss_weighter(config)
    weight = build_element_loss_weight_map(
        weighter,
        batch,
        batch["clean_latent"],
        device="cpu",
    )

    assert weight is not None
    assert weight.shape == (1, 4, 1, 6, 10)
    assert torch.equal(weight[0, 0, 0, :, 2:5], torch.full((6, 3), 7.0))
    assert torch.equal(weight[0, 0, 0, :, 0], torch.ones(6))


def test_wan_cross_attention_handles_masked_frame_conditions() -> None:
    module = WanCrossAttention(dim=8, num_heads=2, condition_dim=4)
    x = torch.randn(1, 6, 8)
    tokens = torch.randn(1, 3, 2, 4)
    mask = torch.tensor([[[True, False], [True, True], [False, False]]])

    residual = module(
        x,
        tokens,
        mask,
        num_frames=3,
        frame_seqlen=2,
    )

    assert residual.shape == x.shape
    assert torch.isfinite(residual).all()


def test_wan_cross_attention_aligns_teacher_forcing_frames() -> None:
    module = WanCrossAttention(dim=8, num_heads=2, condition_dim=4)
    x = torch.randn(1, 12, 8)
    tokens = torch.randn(1, 3, 2, 4)
    mask = torch.ones(1, 3, 2, dtype=torch.bool)

    residual = module(
        x,
        tokens,
        mask,
        num_frames=6,
        frame_seqlen=2,
    )

    assert residual.shape == x.shape
    assert torch.isfinite(residual).all()


def test_action_conditioner_freezes_unused_node_adapter_for_block_attention() -> None:
    conditioner = UIActionNodeConditioner(use_node_cross_attn=False)

    conditioner.enable_trainable_parameters()

    assert conditioner.action_gate.requires_grad
    assert conditioner.action_proj[1].weight.requires_grad
    assert not conditioner.q_proj.weight.requires_grad
    assert not conditioner.k_proj.weight.requires_grad
    assert not conditioner.v_proj.weight.requires_grad
    assert not conditioner.node_out.weight.requires_grad
    assert not conditioner.node_gate.requires_grad


def test_action_conditioner_ignores_nodes_when_disabled() -> None:
    conditioner = UIActionNodeConditioner(use_node_cross_attn=False)
    latents = torch.randn(1, 3, 16, 2, 2)
    node_tokens = torch.randn(1, 3, 2, 1024)
    node_mask = torch.ones(1, 3, 2, dtype=torch.bool)

    output = conditioner(latents, node_tokens=node_tokens, node_mask=node_mask)

    assert torch.equal(output, latents)
