#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.misc import set_seed
from utils.ui_sim_dataset import UISimLatentDataset


def _load_generator_state(path: str, *, use_ema: bool) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "generator_ema" in state and use_ema:
        return state["generator_ema"]
    if "generator" in state:
        return state["generator"]
    if "model" in state:
        return state["model"]
    return state


def _batchify(sample: Dict[str, Any]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for key in ("actions", "node_tokens", "node_mask"):
        if key in sample:
            value = sample[key]
            batch[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else torch.tensor(value).unsqueeze(0)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CF++ UI simulator I2V rollouts from cached latent samples.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-path", default=None, help="Overrides config.data_path when provided.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-output-frames", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    if args.data_path is not None:
        config.data_path = args.data_path
    config.i2v = True

    dataset = UISimLatentDataset(config.data_path, config)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for idx in tqdm(range(min(args.num_samples, len(dataset))), desc="ui-i2v-eval"):
        sample = dataset[idx]
        clean_latent = sample["clean_latent"].unsqueeze(0).to(device=device, dtype=dtype)
        initial_latent = clean_latent[:, :1]
        frames_to_generate = args.num_output_frames - 1
        noise = torch.randn(
            [1, frames_to_generate, 16, 60, 104],
            device=device,
            dtype=dtype,
        )
        prompt = sample["prompts"]
        video, latents = pipeline.inference(
            noise=noise,
            text_prompts=[prompt],
            initial_latent=initial_latent,
            ui_batch=_batchify(sample),
            return_latents=True,
        )
        target = clean_latent[:, :latents.shape[1]]
        latent_mse = torch.mean((latents.float() - target.float()) ** 2).item()
        row = {
            "idx": int(idx),
            "sample_path": sample.get("sample_path", ""),
            "prompt": prompt,
            "latent_mse": latent_mse,
            "frames": int(latents.shape[1]),
        }
        metrics.append(row)

        if not args.no_video:
            pixels = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()).clamp(0, 255).to(torch.uint8)
            write_video(str(output_dir / f"sample_{idx:04d}.mp4"), pixels[0], fps=16)
        pipeline.vae.model.clear_cache()

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
