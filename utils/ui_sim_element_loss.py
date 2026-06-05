from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _ensure_project_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    if src_root.exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def unbatch_clip_metadata(
    clip_metadata: Any,
    batch_size: int,
) -> Optional[List[Dict[str, Any]]]:
    """Convert default-collated clip metadata into DFoT weighter metadata."""
    if clip_metadata is None:
        return None

    if isinstance(clip_metadata, list):
        if not clip_metadata:
            return None
        if len(clip_metadata) == batch_size and all(
            isinstance(item, Mapping) for item in clip_metadata
        ):
            return [dict(item) for item in clip_metadata]

    if isinstance(clip_metadata, Mapping):
        rows: List[Dict[str, Any]] = []
        for idx in range(batch_size):
            row: Dict[str, Any] = {}
            for key, value in clip_metadata.items():
                if isinstance(value, torch.Tensor):
                    row[key] = _to_python(value[idx])
                elif isinstance(value, (list, tuple)):
                    row[key] = _to_python(value[idx])
                else:
                    row[key] = _to_python(value)
            rows.append(row)
        return rows

    return None


def build_element_loss_weighter(
    config: Any,
    *,
    is_main_process: bool = False,
) -> Any | None:
    """Build the shared UI element loss weighter when enabled in CF configs."""
    element_cfg = _cfg_get(config, "element_loss", None)
    if element_cfg is None or not bool(_cfg_get(element_cfg, "enabled", False)):
        return None

    cache_dir = _cfg_get(element_cfg, "cache_dir", None)
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("element_loss.enabled=true requires element_loss.cache_dir")

    _ensure_project_src_on_path()
    from ui_sim.dfot_simulator.element_loss import ElementLossWeighter

    weighter = ElementLossWeighter(
        cache_dir=cache_dir,
        base_weight=float(_cfg_get(element_cfg, "base_weight", 1.0)),
        element_boost=float(_cfg_get(element_cfg, "element_boost", 5.0)),
        text_cache_dir=_cfg_get(element_cfg, "text_cache_dir", None),
        text_boost=float(_cfg_get(element_cfg, "text_boost", 50.0)),
        text_min_confidence=float(_cfg_get(element_cfg, "text_min_confidence", 0.5)),
        text_padding_px=int(_cfg_get(element_cfg, "text_padding_px", 2)),
    )
    if is_main_process:
        print(
            "[element_loss] enabled for CF UI training: "
            f"cache={cache_dir}, boost={float(_cfg_get(element_cfg, 'element_boost', 5.0))}"
        )
    return weighter


def build_element_loss_weight_map(
    weighter: Any | None,
    batch: Mapping[str, Any],
    clean_latent: torch.Tensor,
    *,
    device: torch.device | int | str,
) -> torch.Tensor | None:
    if weighter is None:
        return None
    metadata = unbatch_clip_metadata(
        batch.get("clip_metadata"),
        batch_size=int(clean_latent.shape[0]),
    )
    if metadata is None:
        return None
    return weighter.build_batch_weight_map(
        metadata,
        H=int(clean_latent.shape[-2]),
        W=int(clean_latent.shape[-1]),
        device=clean_latent.device,
    ).to(device=clean_latent.device, dtype=torch.float32)
