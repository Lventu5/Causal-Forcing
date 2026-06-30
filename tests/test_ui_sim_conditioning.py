from __future__ import annotations

import math
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
    load_graph_token_positions,
    normalize_action_coordinates,
    unpack_packed_graph_tokens,
    ui_conditioning_dropout_kwargs,
)
from utils.ui_sim_dataset import UISimLatentDataset  # noqa: E402
from utils.ui_sim_element_loss import (  # noqa: E402
    build_element_loss_weighter,
    build_element_loss_weight_map,
)
from utils.ui_sim_visualization import UISimTrainingVisualizer  # noqa: E402
from utils.wan_wrapper import (  # noqa: E402
    UIActionNodeConditioner,
    UIActionPositionEncoding,
    UICenteredPolarPositionEncoding,
    UIPatchDiscretePositionEncoding,
    WanDiffusionWrapper,
    WanVAEWrapper,
)
from pipeline import CausalDiffusionInferencePipeline  # noqa: E402
from wan.modules.causal_model import CausalWanModel  # noqa: E402
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
        "node_positions": torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [0.2, 0.1]]]]
        ),
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
    assert torch.equal(cond["action_cond"][0, 0], torch.tensor([5.0, 0.0, 0.0]))
    assert torch.equal(cond["action_cond"][0, 1], batch["actions"][0, 0])
    assert cond["node_tokens"].shape == (1, 4, 2, 4)
    assert cond["node_mask"][0, 0].tolist() == [False, False]
    assert cond["node_positions"].shape == (1, 4, 2, 2)
    assert torch.equal(cond["node_positions"][0, 0], torch.zeros(2, 2))
    assert torch.equal(cond["node_positions"][0, 1], batch["node_positions"][0, 0])
    assert torch.equal(uncond["action_cond"], torch.zeros_like(cond["action_cond"]))
    assert torch.equal(uncond["node_mask"], torch.zeros_like(cond["node_mask"]))
    assert torch.equal(uncond["node_positions"], torch.zeros_like(cond["node_positions"]))


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
    clean_latent = torch.arange(5 * 16 * 1 * 1, dtype=torch.float16).reshape(5, 16, 1, 1)
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
    assert item["clean_latent"].dtype == clean_latent.dtype
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
    (tmp_path / "traj_node_tokens.json").write_text(
        """
        {
          "frames": [
            ["button A @0,0,10,10", "button B @10,10,20,20"],
            ["button C @20,20,30,30", "button D @30,30,40,40"],
            ["button E @10,20,30,40", "button F @40,50,60,70"],
            ["button G @0,0,100,100", "button H @70,80,90,100"],
            ["button I @50,0,70,20", "button J"],
            ["button K @0,50,20,70", "button L @80,0,100,20"]
          ]
        }
        """,
        encoding="utf-8",
    )

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
    assert item["node_positions"].shape == (3, 2, 2)
    assert torch.allclose(item["node_positions"][0, 0], torch.tensor([0.2, 0.3]))
    assert torch.allclose(item["node_positions"][2, 1], torch.tensor([0.5, 0.5]))


def test_load_graph_token_positions_reads_bbox_centers(tmp_path: Path) -> None:
    token_path = tmp_path / "traj_node_tokens.json"
    token_path.write_text(
        """
        {
          "frames": [
            ["button A @0,0,20,20", "button B @20,30,40,50"],
            ["button C @10,20,30,40", "button D"],
            ["button E @90,90,100,100"]
          ]
        }
        """,
        encoding="utf-8",
    )

    positions = load_graph_token_positions(
        token_path,
        start=1,
        num_frames=3,
        tokens_per_frame=2,
    )

    assert positions.shape == (2, 2, 2)
    assert torch.allclose(positions[0, 0], torch.tensor([0.2, 0.3]))
    assert torch.allclose(positions[0, 1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(positions[1, 0], torch.tensor([0.95, 0.95]))
    assert torch.equal(positions[1, 1], torch.zeros(2))


def test_wan_vae_framewise_decode_preserves_latent_frame_count() -> None:
    vae = WanVAEWrapper.__new__(WanVAEWrapper)
    torch.nn.Module.__init__(vae)
    observed_shapes = []

    def fake_decode(latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        observed_shapes.append((tuple(latent.shape), use_cache))
        return latent[:, :, :3]

    vae.decode_to_pixel = fake_decode
    latent = torch.randn(2, 4, 16, 3, 5)

    decoded = vae.decode_framewise_to_pixel(latent)

    assert observed_shapes == [((8, 1, 16, 3, 5), False)]
    assert decoded.shape == (2, 4, 3, 3, 5)


def test_ui_training_visualizer_intervals() -> None:
    visualizer = UISimTrainingVisualizer.__new__(UISimTrainingVisualizer)
    visualizer.enabled = True
    visualizer.denoising_interval = 1000
    visualizer.rollout_interval = 5000

    assert visualizer.should_log_denoising(1000)
    assert not visualizer.should_log_denoising(1500)
    assert visualizer.should_log_rollout(5000)
    assert not visualizer.should_log_rollout(4000)


def test_ui_training_visualizer_keeps_node_positions() -> None:
    visualizer = UISimTrainingVisualizer.__new__(UISimTrainingVisualizer)
    visualizer.device = torch.device("cpu")
    visualizer.dtype = torch.float32
    visualizer.num_frames = 3

    batch = {
        "clean_latent": torch.zeros(1, 4, 16, 2, 2),
        "actions": torch.zeros(1, 3, 3),
        "node_tokens": torch.zeros(1, 3, 2, 4),
        "node_mask": torch.ones(1, 3, 2, dtype=torch.bool),
        "node_positions": torch.rand(1, 3, 2, 2),
    }

    clean_latent, ui_batch = visualizer._clean_latent_and_condition(batch)

    assert clean_latent.shape[1] == 3
    assert set(ui_batch) == {"actions", "node_tokens", "node_mask", "node_positions"}
    assert torch.equal(ui_batch["node_positions"], batch["node_positions"])


def test_causal_wan_rebuilds_block_mask_for_visualization_shape(
    monkeypatch,
) -> None:
    model = CausalWanModel.__new__(CausalWanModel)
    torch.nn.Module.__init__(model)
    model.block_mask = None
    model._block_mask_key = None
    model.num_frame_per_block = 1
    model.independent_first_frame = False
    model.local_attn_size = 21
    builds = []

    def build_teacher_mask(
        device,
        num_frames,
        frame_seqlen,
        num_frame_per_block,
    ):
        mask = (num_frames, frame_seqlen, num_frame_per_block)
        builds.append(mask)
        return mask

    monkeypatch.setattr(
        CausalWanModel,
        "_prepare_teacher_forcing_mask",
        staticmethod(build_teacher_mask),
    )

    training_mask = model._ensure_block_mask(
        device="cpu",
        num_frames=21,
        frame_seqlen=1560,
        teacher_forcing=True,
    )
    reused_mask = model._ensure_block_mask(
        device="cpu",
        num_frames=21,
        frame_seqlen=1560,
        teacher_forcing=True,
    )
    preview_mask = model._ensure_block_mask(
        device="cpu",
        num_frames=9,
        frame_seqlen=1560,
        teacher_forcing=True,
    )

    assert reused_mask is training_mask
    assert preview_mask != training_mask
    assert builds == [(21, 1560, 1), (9, 1560, 1)]


def test_ui_training_visualizer_offloads_vae_after_decode() -> None:
    class FakeVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.model = SimpleNamespace(clear_cache=self._clear_cache)
            self.clear_count = 0

        def _clear_cache(self) -> None:
            self.clear_count += 1

        def decode_framewise_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
            return latent[:, :, :3]

    visualizer = UISimTrainingVisualizer.__new__(UISimTrainingVisualizer)
    visualizer.config = SimpleNamespace(vae_decode_mode="single_frame")
    visualizer.model = SimpleNamespace(vae=FakeVAE())
    visualizer.device = torch.device("cpu")
    visualizer.dtype = torch.float32
    latent = torch.randn(1, 2, 16, 3, 5)

    target, prediction = visualizer._decode_pair(latent, latent + 1)

    assert target.device.type == "cpu"
    assert prediction.device.type == "cpu"
    assert visualizer.model.vae.anchor.device.type == "cpu"
    assert visualizer.model.vae.clear_count == 2


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


def test_dfot_exact_action_position_encoding_matches_formula() -> None:
    conditioner = UIActionNodeConditioner(
        use_node_cross_attn=False,
        action_embedding_recipe="dfot_exact",
        action_fourier_freqs=2,
    )
    assert isinstance(conditioner.action_position_encoding, UIActionPositionEncoding)
    xy = torch.tensor([[[0.25, 0.50], [1.0, 0.0]]])

    encoded = conditioner.action_position_encoding(xy)

    freqs = torch.tensor([math.pi, 2.0 * math.pi])
    radius = torch.sqrt((xy ** 2).sum(dim=-1)).clamp(max=math.sqrt(2.0))
    theta = torch.atan2(xy[..., 1], xy[..., 0])
    radius_angle = math.pi * radius / math.sqrt(2.0)
    s3 = torch.stack(
        [
            torch.cos(theta),
            torch.sin(theta),
            torch.cos(radius_angle),
            torch.sin(radius_angle),
        ],
        dim=-1,
    ) / math.sqrt(2.0)
    expected_angles = s3.unsqueeze(-1) * freqs
    expected = torch.cat([expected_angles.sin(), expected_angles.cos()], dim=-1).flatten(-2)

    assert torch.allclose(encoded, expected)


def test_patch_discrete_position_encoding_maps_normalized_clicks() -> None:
    encoding = UIPatchDiscretePositionEncoding(
        grid_height=4,
        grid_width=8,
        emb_dim=6,
    )
    xy = torch.tensor([[0.0, 0.0], [0.999, 0.999], [0.5, 0.5], [1.2, -0.2]])

    y_idx, x_idx = encoding.patch_indices(xy)
    encoded = encoding(xy)

    assert y_idx.tolist() == [0, 3, 2, 0]
    assert x_idx.tolist() == [0, 7, 4, 7]
    assert encoded.shape == (4, 6)


def test_centered_polar_position_encoding_uses_angle_and_bounded_radius_features() -> None:
    encoding = UICenteredPolarPositionEncoding(
        n_freqs=2,
        target_width=4,
        target_height=2,
    )
    xy = torch.tensor([[0.5, 0.5], [1.0, 1.0]])

    encoded = encoding(xy)

    assert encoded.shape == (2, 5)
    assert torch.allclose(encoded[0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    assert torch.isclose(encoded[1, 0], torch.tensor(math.atan2(1.0, 2.0) / math.pi))
    assert torch.all(encoded[:, 1:].abs() <= 1.0)


def test_action_conditioner_builds_positioned_action_tokens_and_mask() -> None:
    conditioner = UIActionNodeConditioner(
        use_node_cross_attn=False,
        action_token_dim=12,
        hidden_dim=16,
    )
    actions = torch.tensor([[[5.0, 0.0, 0.0], [1.0, 0.25, 0.75]]])

    tokens, mask, positions = conditioner.build_action_tokens(
        actions,
        height=8,
        width=8,
        dtype=torch.float32,
        drop_action=False,
    )

    assert tokens.shape == (1, 2, 1, 12)
    assert mask.tolist() == [[[False], [True]]]
    assert torch.equal(tokens[0, 0], torch.zeros_like(tokens[0, 0]))
    assert torch.allclose(positions[0, 1, 0], torch.tensor([0.25, 0.75]))


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
    assert torch.equal(residual[:, 4:], torch.zeros_like(residual[:, 4:]))


def test_wan_cross_attention_stores_reduced_attention_maps() -> None:
    module = WanCrossAttention(dim=8, num_heads=2, condition_dim=4)
    module.store_attn_weights = True
    x = torch.randn(1, 6, 8)
    tokens = torch.randn(1, 3, 2, 4)
    mask = torch.tensor([[[True, False], [True, True], [False, False]]])

    residual = module(
        x,
        tokens,
        mask,
        num_frames=3,
        frame_seqlen=2,
        grid_sizes=torch.tensor([[3, 1, 2]]),
    )

    assert residual.shape == x.shape
    assert module.last_attn_slot_map.shape == (3, 2)
    assert module.last_attn_patch_map.shape == (3, 2, 2)
    assert module.last_attn_num_frames == 3
    assert module.last_attn_tokens_per_frame == 2
    assert module.last_attn_frame_seqlen == 2
    assert module.last_attn_grid_hw == (1, 2)
    assert torch.equal(module.last_attn_slot_map[2], torch.zeros(2))
    assert torch.equal(module.last_attn_patch_map[2], torch.zeros(2, 2))


def test_wan_cross_attention_changes_with_condition_positions() -> None:
    torch.manual_seed(0)
    module = WanCrossAttention(dim=8, num_heads=2, condition_dim=4)
    with torch.no_grad():
        module.gate.fill_(1.0)
    x = torch.randn(1, 4, 8)
    tokens = torch.randn(1, 2, 2, 4)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    positions = torch.tensor(
        [[[[0.1, 0.1], [0.9, 0.9]], [[0.2, 0.8], [0.8, 0.2]]]],
        dtype=torch.float32,
    )

    residual_a = module(
        x,
        tokens,
        mask,
        positions,
        num_frames=2,
        frame_seqlen=2,
        grid_sizes=torch.tensor([[2, 1, 2]]),
    )
    residual_b = module(
        x,
        tokens,
        mask,
        positions.flip(dims=[2]),
        num_frames=2,
        frame_seqlen=2,
        grid_sizes=torch.tensor([[2, 1, 2]]),
    )

    assert residual_a.shape == x.shape
    assert not torch.allclose(residual_a, residual_b)


def test_wan_cross_attention_rope_2d_changes_with_condition_positions() -> None:
    torch.manual_seed(1)
    module = WanCrossAttention(
        dim=8,
        num_heads=2,
        condition_dim=4,
        position_encoding="rope_2d",
    )
    with torch.no_grad():
        module.gate.fill_(1.0)
    x = torch.randn(1, 4, 8)
    tokens = torch.randn(1, 2, 2, 4)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    positions = torch.tensor(
        [[[[0.1, 0.1], [0.9, 0.9]], [[0.2, 0.8], [0.8, 0.2]]]],
        dtype=torch.float32,
    )

    residual_a = module(
        x,
        tokens,
        mask,
        positions,
        num_frames=2,
        frame_seqlen=2,
        grid_sizes=torch.tensor([[2, 1, 2]]),
    )
    residual_b = module(
        x,
        tokens,
        mask,
        positions.flip(dims=[2]),
        num_frames=2,
        frame_seqlen=2,
        grid_sizes=torch.tensor([[2, 1, 2]]),
    )

    assert residual_a.shape == x.shape
    assert not torch.allclose(residual_a, residual_b)


def test_wan_cross_attention_rope_2d_requires_query_grid_positions() -> None:
    module = WanCrossAttention(
        dim=8,
        num_heads=2,
        condition_dim=4,
        position_encoding="rope_2d",
    )
    x = torch.randn(1, 4, 8)
    tokens = torch.randn(1, 2, 2, 4)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    positions = torch.rand(1, 2, 2, 2)

    try:
        module(
            x,
            tokens,
            mask,
            positions,
            num_frames=2,
            frame_seqlen=2,
        )
    except ValueError as exc:
        assert "requires grid_sizes" in str(exc)
    else:
        raise AssertionError("Expected rope_2d without grid_sizes to fail.")


def test_wan_cross_attention_rejects_unknown_position_encoding() -> None:
    try:
        WanCrossAttention(
            dim=8,
            num_heads=2,
            condition_dim=4,
            position_encoding="bad_mode",
        )
    except ValueError as exc:
        assert "position_encoding" in str(exc)
    else:
        raise AssertionError("Expected invalid position encoding to fail.")


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

    assert conditioner.action_residual_gate.requires_grad
    assert conditioner.action_type_embedding.weight.requires_grad
    assert conditioner.action_spatial_proj[0].weight.requires_grad
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


def test_action_conditioner_uses_spatial_action_maps() -> None:
    conditioner = UIActionNodeConditioner(use_node_cross_attn=False)
    latents = torch.zeros(1, 2, 16, 8, 8)
    actions = torch.tensor(
        [
            [
                [5.0, 0.0, 0.0],
                [1.0, 0.5, 0.5],
            ]
        ]
    )

    output = conditioner(latents, action_cond=actions)

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, :1], latents[:, :1])
    assert not torch.equal(output[:, 1:], latents[:, 1:])


def test_action_conditioner_dropout_zeroes_action_signal() -> None:
    conditioner = UIActionNodeConditioner(
        use_node_cross_attn=False,
        action_dropout=1.0,
    )
    conditioner.train()
    latents = torch.zeros(1, 1, 16, 4, 4)
    actions = torch.tensor([[[1.0, 0.5, 0.5]]])

    output = conditioner(latents, action_cond=actions)

    assert torch.equal(output, latents)


def test_action_conditioner_rejects_misaligned_action_frames() -> None:
    conditioner = UIActionNodeConditioner(use_node_cross_attn=False)
    latents = torch.zeros(1, 2, 16, 4, 4)
    actions = torch.zeros(1, 1, 3)

    try:
        conditioner(latents, action_cond=actions)
    except ValueError as exc:
        assert "action_cond frame shape" in str(exc)
    else:
        raise AssertionError("Expected misaligned action frames to fail.")


def test_action_conditioner_partial_load_ignores_old_action_mlp_keys() -> None:
    conditioner = UIActionNodeConditioner(use_node_cross_attn=False)
    old_state = {
        "action_gate": torch.ones(1),
        "action_proj.1.weight": torch.randn(256, 3),
        "action_proj.1.bias": torch.randn(256),
        "action_proj.3.weight": torch.randn(16, 256),
        "action_proj.3.bias": torch.randn(16),
    }

    result = conditioner.load_state_dict(old_state, strict=False)

    assert "action_gate" in result.unexpected_keys
    assert "action_proj.1.weight" in result.unexpected_keys
    assert "action_residual_gate" in result.missing_keys


def test_pipeline_marks_ui_conditions_with_local_frame_start() -> None:
    condition = {
        "prompt_embeds": torch.ones(1, 2, 8),
        "action_cond": torch.randn(1, 16, 3),
    }

    shifted = CausalDiffusionInferencePipeline._condition_dict_for_local_frame(
        condition,
        7,
    )

    assert shifted["ui_frame_start"] == 7
    assert "ui_frame_start" not in condition
    assert shifted["action_cond"] is condition["action_cond"]


def test_wan_wrapper_ui_frame_start_overrides_global_token_start() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.action_cond = None

        def __call__(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
            self.action_cond = kwargs["action_cond"].detach().clone()
            return latents

    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    recorder = Recorder()
    wrapper.ui_conditioner = recorder
    wrapper.block_cross_attn_enabled = False
    wrapper.node_conditioning_enabled = False
    wrapper.frame_seq_length = 2

    latents = torch.zeros(1, 1, 16, 1, 1)
    actions = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)

    wrapper._prepare_ui_conditioning(
        latents,
        {
            "action_cond": actions,
            "ui_frame_start": 2,
        },
        current_start=99 * wrapper.frame_seq_length,
    )

    assert recorder.action_cond is not None
    assert torch.equal(recorder.action_cond, actions[:, 2:3])
