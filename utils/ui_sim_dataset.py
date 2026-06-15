from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils.ui_sim_conditioning import (
    LetterboxSpec,
    normalize_action_coordinates,
    unpack_packed_graph_tokens,
)


DEFAULT_PROMPT = "desktop file manager UI transition"


def letterbox_image(
    image: Image.Image,
    *,
    target_width: int = 832,
    target_height: int = 480,
    fill: int | tuple[int, int, int] = 0,
) -> tuple[Image.Image, Dict[str, float]]:
    image = image.convert("RGB")
    spec = LetterboxSpec(
        source_width=image.width,
        source_height=image.height,
        target_width=target_width,
        target_height=target_height,
    )
    resized = image.resize((spec.resized_width, spec.resized_height), Image.BICUBIC)
    pad_left = int(round(spec.pad_left))
    pad_top = int(round(spec.pad_top))
    pad_right = target_width - spec.resized_width - pad_left
    pad_bottom = target_height - spec.resized_height - pad_top
    padded = ImageOps.expand(
        resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=fill,
    )
    return padded, {
        "source_width": float(image.width),
        "source_height": float(image.height),
        "target_width": float(target_width),
        "target_height": float(target_height),
        "scale": float(spec.scale),
        "pad_left": float(pad_left),
        "pad_top": float(pad_top),
    }


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            payload = json.load(f)
            rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must contain a list of samples: {path}")
    return rows


def _discover_samples(data_path: Path) -> List[Dict[str, Any]]:
    if data_path.is_file():
        if data_path.suffix in {".json", ".jsonl"}:
            root = data_path.parent
            rows = _read_manifest(data_path)
            for row in rows:
                if "path" in row:
                    row["path"] = str((root / row["path"]).resolve() if not Path(row["path"]).is_absolute() else Path(row["path"]))
            return rows
        return [{"path": str(data_path)}]

    for name in ("manifest.jsonl", "manifest.json"):
        manifest = data_path / name
        if manifest.exists():
            rows = _read_manifest(manifest)
            for row in rows:
                if "path" in row:
                    row["path"] = str((data_path / row["path"]).resolve() if not Path(row["path"]).is_absolute() else Path(row["path"]))
            return rows

    samples = [{"path": str(path)} for path in sorted(data_path.rglob("*.pt"))]
    if not samples:
        raise FileNotFoundError(f"No UI simulator latent samples found under {data_path}")
    return samples


def _as_tensor(
    value: Any,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _slice_source_rows(rows: torch.Tensor, *, start: int, num_frames: int) -> torch.Tensor:
    needed = max(num_frames - 1, 0)
    if rows.shape[0] < start + needed:
        raise ValueError(
            f"Condition rows too short for clip start={start}, frames={num_frames}: "
            f"got {rows.shape[0]}, need {start + needed}"
        )
    return rows[start:start + needed]


def _load_referenced_rows(path_value: str | Path, *, start: int, num_frames: int) -> torch.Tensor:
    path = Path(path_value)
    rows = torch.load(str(path), map_location="cpu", weights_only=True)
    rows = _as_tensor(rows)
    return _slice_source_rows(rows, start=start, num_frames=num_frames)


class UISimLatentDataset(Dataset):
    """Latent-cache dataset for CF++ UI simulator training.

    Each sample is a `.pt` dictionary or manifest row pointing to one. Expected
    keys are:

    - `clean_latent`: `[T, C, H, W]` or `[1, T, C, H, W]`;
    - `actions`: source-frame rows, usually DFoT `[action_type, x, y]`;
    - optional `node_tokens` + `node_mask`, or packed `node_emb`;
    - optional `prompt` and `metadata`.
    """

    def __init__(self, data_path: str | Path, config: Any):
        self.data_path = Path(data_path)
        self.samples = _discover_samples(self.data_path)
        self.num_frames = int(getattr(config, "num_training_frames", 0) or getattr(config, "num_frames", 21))
        if hasattr(config, "image_or_video_shape"):
            self.num_frames = int(config.image_or_video_shape[1])
        self.random_clip = bool(getattr(config, "ui_random_clip", True))
        self.prompt = str(getattr(config, "ui_prompt", DEFAULT_PROMPT))
        self.target_width = int(getattr(config, "width", 832))
        self.target_height = int(getattr(config, "height", 480))
        self.action_coordinate_mode = str(getattr(config, "ui_action_coordinate_mode", "legacy_normalized_source"))
        self.source_width = int(getattr(config, "ui_source_width", 1024))
        self.source_height = int(getattr(config, "ui_source_height", 768))
        self.tokens_per_frame = int(getattr(config, "ui_graph_tokens_per_frame", 32))
        self.graph_token_dim = int(getattr(config, "ui_graph_token_dim", 1024))
        self.graph_token_has_mask = bool(getattr(config, "ui_graph_token_has_mask", True))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_payload(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "path" not in sample:
            return dict(sample)
        path = Path(sample["path"])
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"UI simulator sample must be a dict: {path}")
        merged = dict(payload)
        for key, value in sample.items():
            if key != "path":
                merged.setdefault(key, value)
        merged.setdefault("sample_path", str(path))
        return merged

    def _clip_start(self, total_frames: int) -> int:
        if total_frames < self.num_frames:
            raise ValueError(f"Latent sample has {total_frames} frames, need {self.num_frames}")
        if not self.random_clip or total_frames == self.num_frames:
            return 0
        return random.randint(0, total_frames - self.num_frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        payload = self._load_payload(self.samples[idx])

        clean_latent = _as_tensor(payload["clean_latent"], dtype=None)
        if not clean_latent.is_floating_point():
            clean_latent = clean_latent.float()
        if clean_latent.ndim == 5 and clean_latent.shape[0] == 1:
            clean_latent = clean_latent[0]
        if clean_latent.ndim != 4:
            raise ValueError(f"clean_latent must be [T, C, H, W], got {tuple(clean_latent.shape)}")

        total_frames = int(clean_latent.shape[0])
        start = self._clip_start(total_frames)
        end = start + self.num_frames
        clean_latent = clean_latent[start:end].contiguous()

        out: Dict[str, Any] = {
            "prompts": str(payload.get("prompt") or payload.get("prompts") or self.prompt),
            "clean_latent": clean_latent,
            "idx": idx,
            "clip_start": start,
            "sample_path": str(payload.get("sample_path", "")),
        }

        metadata = payload.get("metadata") or {}
        base_start = int(
            payload.get(
                "start_frame",
                payload.get("start", metadata.get("start_frame", 0)),
            )
        )
        frame_skip = int(payload.get("frame_skip", metadata.get("frame_skip", 1)))
        source_video = (
            metadata.get("source_video")
            or payload.get("source_video")
            or payload.get("video_path")
            or payload.get("sample_path")
            or ""
        )
        start_frame = base_start + start * frame_skip
        out["clip_metadata"] = {
            "video_path": str(source_video),
            "start_frame": start_frame,
            "end_frame": start_frame + self.num_frames * frame_skip,
            "frame_skip": frame_skip,
            "source_width": int(metadata.get("source_width", self.source_width)),
            "source_height": int(metadata.get("source_height", self.source_height)),
            "target_width": int(metadata.get("target_width", self.target_width)),
            "target_height": int(metadata.get("target_height", self.target_height)),
        }

        if "actions" in payload:
            source_width = int(metadata.get("source_width", self.source_width))
            source_height = int(metadata.get("source_height", self.source_height))
            actions = _as_tensor(payload["actions"])
            actions = _slice_source_rows(actions, start=start, num_frames=self.num_frames)
            actions = normalize_action_coordinates(
                actions,
                source_width=source_width,
                source_height=source_height,
                target_width=self.target_width,
                target_height=self.target_height,
                coordinate_mode=str(metadata.get("action_coordinate_mode", self.action_coordinate_mode)),
            )
            out["actions"] = actions

        if "node_tokens" in payload:
            node_tokens = _as_tensor(payload["node_tokens"])
            out["node_tokens"] = _slice_source_rows(node_tokens, start=start, num_frames=self.num_frames)
            if "node_mask" in payload:
                node_mask = torch.as_tensor(payload["node_mask"]).bool()
                out["node_mask"] = _slice_source_rows(node_mask, start=start, num_frames=self.num_frames)
        elif "node_emb" in payload or "node_cond" in payload:
            packed = _as_tensor(payload.get("node_emb", payload.get("node_cond")))
            packed = _slice_source_rows(packed, start=start, num_frames=self.num_frames)
            tokens, mask = unpack_packed_graph_tokens(
                packed,
                tokens_per_frame=self.tokens_per_frame,
                token_dim=self.graph_token_dim,
                has_mask=self.graph_token_has_mask,
            )
            out["node_tokens"] = tokens
            if mask is not None:
                out["node_mask"] = mask
        elif "node_emb_path" in payload or "node_cond_path" in payload:
            node_path = payload.get("node_emb_path", payload.get("node_cond_path"))
            node_start = int(payload.get("node_emb_start", start_frame))
            packed = _load_referenced_rows(node_path, start=node_start, num_frames=self.num_frames)
            tokens, mask = unpack_packed_graph_tokens(
                packed,
                tokens_per_frame=self.tokens_per_frame,
                token_dim=self.graph_token_dim,
                has_mask=self.graph_token_has_mask,
            )
            out["node_tokens"] = tokens
            if mask is not None:
                out["node_mask"] = mask

        return out


def build_training_dataset(config: Any) -> Dataset:
    from utils.dataset import LatentLMDBDataset, ShardingLMDBDataset, TextDataset

    dataset_type = str(getattr(config, "dataset_type", "latent_lmdb"))
    if dataset_type == "ui_sim_latent":
        return UISimLatentDataset(config.data_path, config)
    if dataset_type == "latent_lmdb":
        return LatentLMDBDataset(config.data_path, max_pair=int(1e8))
    if dataset_type == "sharded_lmdb":
        return ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
    if dataset_type == "text":
        return TextDataset(config.data_path)
    raise ValueError(f"Unsupported dataset_type: {dataset_type!r}")


def iter_sample_paths(data_path: str | Path) -> Iterable[Path]:
    for row in _discover_samples(Path(data_path)):
        if "path" in row:
            yield Path(row["path"])
