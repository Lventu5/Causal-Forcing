#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.io import read_video, write_video
from tqdm import tqdm

from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.misc import set_seed
from utils.ui_sim_conditioning import normalize_action_coordinates, unpack_packed_graph_tokens
from utils.ui_sim_dataset import DEFAULT_PROMPT, letterbox_image


def _load_generator_state(path: str, *, use_ema: bool) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "generator_ema" in state and use_ema:
        return state["generator_ema"]
    if "generator" in state:
        return state["generator"]
    if "model" in state:
        return state["model"]
    return state


def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
    return Image.fromarray(frame.cpu().numpy()).convert("RGB")


def _iter_mp4s(processed_dir: Path, split: str) -> List[Path]:
    split_dir = processed_dir / split
    if split_dir.exists():
        return sorted(split_dir.glob("*.mp4"))
    return sorted(processed_dir.glob("*.mp4"))


def _load_actions(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as f:
        return torch.tensor(json.load(f), dtype=torch.float32)


def _load_node_rows(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return torch.load(str(path), map_location="cpu", weights_only=True).float()


def _as_model_batch(
    *,
    actions: torch.Tensor,
    node_rows: torch.Tensor | None,
    config: Any,
) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {"actions": actions.unsqueeze(0)}
    if node_rows is None:
        return batch

    tokens_per_frame = int(getattr(config, "ui_graph_tokens_per_frame", 32))
    token_dim = int(getattr(config, "ui_graph_token_dim", 1024))
    has_mask = bool(getattr(config, "ui_graph_token_has_mask", True))
    tokens, mask = unpack_packed_graph_tokens(
        node_rows,
        tokens_per_frame=tokens_per_frame,
        token_dim=token_dim,
        has_mask=has_mask,
    )
    batch["node_tokens"] = tokens.unsqueeze(0)
    if mask is not None:
        batch["node_mask"] = mask.unsqueeze(0)
    return batch


def _encode_initial_latent(
    *,
    pipeline: torch.nn.Module,
    frame: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Dict[str, float]]:
    image, letterbox_meta = letterbox_image(
        _frame_to_pil(frame),
        target_width=width,
        target_height=height,
    )
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    pixel = to_tensor(image).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = pipeline.vae.encode_to_latent(pixel).to(device=device, dtype=dtype)
    if latent.shape[1] != 1:
        raise RuntimeError(f"Expected one initial latent frame, got shape {tuple(latent.shape)}")
    return latent, letterbox_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run long autoregressive CF++ UI simulator I2V rollouts from processed trajectories."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-output-frames", type=int, default=301, help="Includes the initial frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if args.num_output_frames < 2:
        raise ValueError("--num-output-frames must include the initial frame and at least one generated frame.")

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    config.i2v = True

    pipeline_cls = CausalInferencePipeline if hasattr(config, "denoising_step_list") else CausalDiffusionInferencePipeline
    pipeline = pipeline_cls(config, device=device).to(dtype=dtype)
    load_result = pipeline.generator.load_state_dict(
        _load_generator_state(args.checkpoint_path, use_ema=args.use_ema),
        strict=False,
    )
    print(f"Generator load result: {load_result}")
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    pipeline.eval()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_paths = _iter_mp4s(processed_dir, args.split)[:max(args.num_trajectories, 0)]
    metrics: List[Dict[str, Any]] = []

    for traj_idx, mp4_path in tqdm(list(enumerate(mp4_paths)), desc="cf-ui-ar-eval"):
        frames, _, _ = read_video(str(mp4_path), pts_unit="sec", output_format="THWC")
        if frames.numel() == 0:
            raise ValueError(f"No frames decoded from {mp4_path}")
        if args.start_frame >= frames.shape[0] - 1:
            raise ValueError(f"start frame {args.start_frame} leaves no future frames for {mp4_path}")

        actions_path = mp4_path.with_suffix(".json")
        if not actions_path.exists():
            raise FileNotFoundError(f"Missing action sidecar: {actions_path}")
        raw_actions = _load_actions(actions_path)

        requested_future = args.num_output_frames - 1
        available_future = min(
            requested_future,
            int(raw_actions.shape[0] - args.start_frame),
            int(frames.shape[0] - args.start_frame - 1),
        )
        if available_future <= 0:
            continue

        initial_latent, _ = _encode_initial_latent(
            pipeline=pipeline,
            frame=frames[args.start_frame],
            width=int(getattr(config, "width", 832)),
            height=int(getattr(config, "height", 480)),
            device=device,
            dtype=dtype,
        )

        source_rows = raw_actions[args.start_frame:args.start_frame + available_future]
        actions = normalize_action_coordinates(
            source_rows,
            source_width=int(getattr(config, "ui_source_width", 1024)),
            source_height=int(getattr(config, "ui_source_height", 768)),
            target_width=int(getattr(config, "width", 832)),
            target_height=int(getattr(config, "height", 480)),
            coordinate_mode=str(getattr(config, "ui_action_coordinate_mode", "legacy_normalized_source")),
        )

        node_rows = None
        node_path = mp4_path.with_name(mp4_path.stem + "_node_emb.pt")
        if node_path.exists():
            packed = _load_node_rows(node_path)
            assert packed is not None
            node_rows = packed[args.start_frame:args.start_frame + available_future]

        noise = torch.randn(
            [1, available_future, 16, 60, 104],
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[args.prompt],
                initial_latent=initial_latent,
                ui_batch=_as_model_batch(actions=actions, node_rows=node_rows, config=config),
                return_latents=True,
            )

        row = {
            "idx": int(traj_idx),
            "trajectory": str(mp4_path),
            "start_frame": int(args.start_frame),
            "requested_frames": int(args.num_output_frames),
            "generated_latent_frames": int(latents.shape[1]),
            "generated_future_steps": int(available_future),
            "has_node_rows": bool(node_rows is not None),
        }
        metrics.append(row)

        if not args.no_video:
            pixels = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()).clamp(0, 255).to(torch.uint8)
            write_video(str(output_dir / f"traj_{traj_idx:04d}.mp4"), pixels[0], fps=args.fps)
        pipeline.vae.model.clear_cache()

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
