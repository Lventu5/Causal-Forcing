#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torchvision.io import read_video
from torchvision import transforms
from tqdm import tqdm

from utils.ui_sim_dataset import letterbox_image
from utils.wan_wrapper import WanVAEWrapper


def _load_actions(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as f:
        return torch.tensor(json.load(f), dtype=torch.float32)


def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
    return Image.fromarray(frame.cpu().numpy()).convert("RGB")


def _load_letterboxed_video(mp4_path: Path, *, width: int, height: int) -> tuple[torch.Tensor, Dict[str, float]]:
    frames, _, _ = read_video(str(mp4_path), pts_unit="sec", output_format="THWC")
    if frames.numel() == 0:
        raise ValueError(f"No frames decoded from {mp4_path}")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    encoded_frames: List[torch.Tensor] = []
    letterbox_meta: Dict[str, float] | None = None
    for frame in frames:
        image, meta = letterbox_image(_frame_to_pil(frame), target_width=width, target_height=height)
        letterbox_meta = meta
        encoded_frames.append(to_tensor(image))
    assert letterbox_meta is not None
    return torch.stack(encoded_frames, dim=1), letterbox_meta  # [C, T, H, W]


def _load_optional_node(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return torch.load(str(path), map_location="cpu", weights_only=True).float()


def _iter_mp4s(processed_dir: Path, split: str) -> List[Path]:
    split_dir = processed_dir / split
    if split_dir.exists():
        return sorted(split_dir.glob("*.mp4"))
    return sorted(processed_dir.glob("*.mp4"))


def _save_manifest(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CF UI simulator latent-cache samples.")
    parser.add_argument("--processed-dir", required=True, help="Processed DFoT-style root containing split/*.mp4 + .json.")
    parser.add_argument("--output-dir", required=True, help="Output cache directory. Keep this on HPC storage, not in git.")
    parser.add_argument("--split", default="training", help="Split subdirectory to process.")
    parser.add_argument("--num-frames", type=int, default=21, help="Frames per cached training window, including frame 0.")
    parser.add_argument("--stride", type=int, default=20, help="Window stride in frames.")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--include-node-emb", action="store_true", help="Copy sibling *_node_emb.pt packed graph rows.")
    parser.add_argument("--limit", type=int, default=0, help="Optional trajectory limit for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Decode/check inputs but do not load VAE or write samples.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_paths = _iter_mp4s(processed_dir, args.split)
    if args.limit > 0:
        mp4_paths = mp4_paths[:args.limit]
    if not mp4_paths:
        raise FileNotFoundError(f"No .mp4 trajectories found for split {args.split!r} under {processed_dir}")

    vae = None if args.dry_run else WanVAEWrapper().eval().cuda().to(dtype=torch.bfloat16)
    manifest_rows: List[Dict[str, Any]] = []

    for mp4_path in tqdm(mp4_paths, desc="ui-sim-cache"):
        actions_path = mp4_path.with_suffix(".json")
        if not actions_path.exists():
            raise FileNotFoundError(f"Missing action sidecar for {mp4_path}: {actions_path}")

        pixel, letterbox_meta = _load_letterboxed_video(mp4_path, width=args.width, height=args.height)
        actions = _load_actions(actions_path)
        node_emb = _load_optional_node(mp4_path.with_name(mp4_path.stem + "_node_emb.pt")) if args.include_node_emb else None

        total_frames = pixel.shape[1]
        if total_frames < args.num_frames:
            continue
        starts = range(0, total_frames - args.num_frames + 1, max(args.stride, 1))
        for start in starts:
            end = start + args.num_frames
            out_name = f"{mp4_path.stem}_s{start:05d}.pt"
            out_path = output_dir / out_name
            if args.dry_run:
                manifest_rows.append({"path": out_name, "source_video": str(mp4_path), "start": start})
                continue

            window = pixel[:, start:end].unsqueeze(0).cuda().to(dtype=torch.bfloat16)
            with torch.no_grad():
                clean_latent = vae.encode_to_latent(window).float().cpu()[0]
            if clean_latent.shape[0] != args.num_frames:
                raise RuntimeError(
                    f"VAE returned {clean_latent.shape[0]} latent frames for {args.num_frames} input frames."
                )

            sample: Dict[str, Any] = {
                "clean_latent": clean_latent,
                "actions": actions[start:start + args.num_frames - 1].contiguous(),
                "prompt": "desktop file manager UI transition",
                "metadata": {
                    **letterbox_meta,
                    "source_video": str(mp4_path),
                    "start_frame": start,
                    "frame_skip": 1,
                    "action_coordinate_mode": "legacy_normalized_source",
                },
            }
            if node_emb is not None:
                sample["node_emb"] = node_emb[start:start + args.num_frames - 1].contiguous()
            torch.save(sample, out_path)
            manifest_rows.append({"path": out_name, "source_video": str(mp4_path), "start": start})

    _save_manifest(output_dir, manifest_rows)
    print(f"Wrote {len(manifest_rows)} manifest rows to {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
