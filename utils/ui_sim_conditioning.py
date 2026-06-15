from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class LetterboxSpec:
    source_width: int
    source_height: int
    target_width: int = 832
    target_height: int = 480

    @property
    def scale(self) -> float:
        return min(
            self.target_width / self.source_width,
            self.target_height / self.source_height,
        )

    @property
    def resized_width(self) -> int:
        return round(self.source_width * self.scale)

    @property
    def resized_height(self) -> int:
        return round(self.source_height * self.scale)

    @property
    def pad_left(self) -> float:
        return (self.target_width - self.resized_width) / 2.0

    @property
    def pad_top(self) -> float:
        return (self.target_height - self.resized_height) / 2.0


def normalize_action_coordinates(
    actions: torch.Tensor,
    *,
    source_width: int = 1024,
    source_height: int = 768,
    target_width: int = 832,
    target_height: int = 480,
    coordinate_mode: str = "legacy_normalized_source",
    no_action_type: int = 5,
) -> torch.Tensor:
    """Map action coordinates into the letterboxed WAN canvas.

    Action layout is `[action_type, x, y, ...]`. The first three dimensions are
    kept; extra dimensions are passed through unchanged.
    """
    if actions.numel() == 0 or actions.shape[-1] < 3:
        return actions
    if coordinate_mode in {"already_letterboxed", "already_letterboxed_normalized"}:
        return actions

    out = actions.clone().float()
    spec = LetterboxSpec(
        source_width=int(source_width),
        source_height=int(source_height),
        target_width=int(target_width),
        target_height=int(target_height),
    )

    if coordinate_mode in {"legacy_normalized_source", "normalized_source"}:
        x_src = out[..., 1] * float(source_width)
        y_src = out[..., 2] * float(source_height)
    elif coordinate_mode == "pixel_source":
        x_src = out[..., 1]
        y_src = out[..., 2]
    else:
        raise ValueError(f"Unsupported action coordinate_mode: {coordinate_mode!r}")

    x_dst = (x_src * spec.scale + spec.pad_left) / float(target_width)
    y_dst = (y_src * spec.scale + spec.pad_top) / float(target_height)

    valid = torch.isfinite(x_dst) & torch.isfinite(y_dst)
    no_action = (
        (out[..., 0].round().to(torch.long) == int(no_action_type))
        & (out[..., 1].abs() < 1e-8)
        & (out[..., 2].abs() < 1e-8)
    )
    valid = valid & ~no_action
    out[..., 1] = torch.where(valid, x_dst.clamp(0.0, 1.0), out[..., 1])
    out[..., 2] = torch.where(valid, y_dst.clamp(0.0, 1.0), out[..., 2])
    out[..., 1:3] = torch.where(no_action.unsqueeze(-1), torch.zeros_like(out[..., 1:3]), out[..., 1:3])
    return out


def unpack_packed_graph_tokens(
    packed: torch.Tensor,
    *,
    tokens_per_frame: int,
    token_dim: int,
    has_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpack `[K * D flattened tokens | K mask]` rows."""
    if tokens_per_frame <= 1:
        mask = torch.ones(packed.shape[:-1] + (1,), dtype=torch.bool, device=packed.device)
        return packed.unsqueeze(-2), mask

    token_values_dim = tokens_per_frame * token_dim
    expected_dim = token_values_dim + (tokens_per_frame if has_mask else 0)
    if packed.shape[-1] != expected_dim:
        raise ValueError(
            "Graph-token packed dim mismatch: "
            f"got {packed.shape[-1]}, expected {expected_dim} "
            f"(K={tokens_per_frame}, D={token_dim}, mask={has_mask})."
        )

    tokens = packed[..., :token_values_dim].reshape(
        *packed.shape[:-1],
        tokens_per_frame,
        token_dim,
    )
    if not has_mask:
        mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=packed.device)
        return tokens, mask

    mask = packed[..., token_values_dim:].reshape(*packed.shape[:-1], tokens_per_frame) > 0.5
    empty = ~mask.any(dim=-1)
    if empty.any():
        token_values = tokens.abs().sum(dim=(-1, -2)) > 0
        if (empty & token_values).any():
            raise ValueError("Packed graph-token row has values but no valid mask bits.")
        mask = mask.clone()
        mask[..., 0] |= empty
    return tokens, mask


def prepend_null_frame(tensor: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[dim] = 1
    return torch.cat([torch.zeros(shape, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim)


def align_source_rows_to_latent_frames(
    source_rows: torch.Tensor,
    *,
    num_latent_frames: int,
    i2v: bool,
) -> torch.Tensor:
    """Convert source-frame transition rows into per-latent-frame rows.

    For I2V, generated frame `t + 1` uses source row `t`, so frame 0 receives a
    null row and rows are shifted right by one.
    """
    if source_rows.ndim < 3:
        raise ValueError(f"Expected source rows with shape [B, T, ...], got {tuple(source_rows.shape)}")

    if i2v:
        needed = max(num_latent_frames - 1, 0)
        if source_rows.shape[1] < needed:
            raise ValueError(
                f"Need at least {needed} source rows for {num_latent_frames} I2V latent frames, "
                f"got {source_rows.shape[1]}."
            )
        aligned = prepend_null_frame(source_rows[:, :needed], dim=1)
    else:
        if source_rows.shape[1] < num_latent_frames:
            raise ValueError(
                f"Need at least {num_latent_frames} source rows, got {source_rows.shape[1]}."
            )
        aligned = source_rows[:, :num_latent_frames]
    return aligned


def _get_config_value(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def ui_conditioning_dropout_kwargs(config: Any) -> dict:
    """Return train-time UI condition dropout kwargs for `attach_ui_batch_conditioning`.

    Action dropout is still handled inside `UIActionNodeConditioner`, where the
    action adapter lives. For block cross-attention, node tokens bypass that
    adapter, so `node_dropout` is applied here as true condition dropout.
    """
    model_kwargs = _get_config_value(config, "model_kwargs", None)
    ui_conditioning = _get_config_value(model_kwargs, "ui_conditioning", None)
    if not bool(_get_config_value(ui_conditioning, "enabled", False)):
        return {"condition_dropout_enabled": False}

    block_cross_attn = bool(_get_config_value(ui_conditioning, "block_cross_attn", False))
    node_dropout = float(_get_config_value(ui_conditioning, "node_dropout", 0.0)) if block_cross_attn else 0.0
    return {
        "condition_dropout_enabled": True,
        "action_dropout": 0.0,
        "node_dropout": node_dropout,
    }


def _apply_frame_condition_dropout(
    value: torch.Tensor,
    *,
    p: float,
    enabled: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not enabled or p <= 0.0:
        return value, None
    if p >= 1.0:
        keep = torch.zeros(value.shape[:2], dtype=torch.bool, device=value.device)
    else:
        keep = torch.rand(value.shape[:2], device=value.device) >= float(p)
    keep_view = keep
    while keep_view.ndim < value.ndim:
        keep_view = keep_view.unsqueeze(-1)
    return value * keep_view.to(dtype=value.dtype), keep


def attach_ui_batch_conditioning(
    batch: Mapping[str, object],
    conditional_dict: Mapping[str, torch.Tensor],
    unconditional_dict: Mapping[str, torch.Tensor],
    *,
    device: torch.device | int | str,
    dtype: torch.dtype,
    num_latent_frames: int,
    i2v: bool,
    condition_dropout_enabled: bool = False,
    action_dropout: float = 0.0,
    node_dropout: float = 0.0,
) -> Tuple[MutableMapping[str, torch.Tensor], MutableMapping[str, torch.Tensor]]:
    """Attach action/node tensors from a dataloader batch to CF condition dicts."""
    cond: MutableMapping[str, torch.Tensor] = dict(conditional_dict)
    uncond: MutableMapping[str, torch.Tensor] = dict(unconditional_dict)

    if "actions" in batch:
        actions = torch.as_tensor(batch["actions"]).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        actions = align_source_rows_to_latent_frames(
            actions,
            num_latent_frames=num_latent_frames,
            i2v=i2v,
        )
        actions, _ = _apply_frame_condition_dropout(
            actions,
            p=action_dropout,
            enabled=condition_dropout_enabled,
        )
        cond["action_cond"] = actions
        uncond["action_cond"] = torch.zeros_like(actions)

    if "node_tokens" in batch:
        node_tokens = torch.as_tensor(batch["node_tokens"]).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        node_tokens = align_source_rows_to_latent_frames(
            node_tokens,
            num_latent_frames=num_latent_frames,
            i2v=i2v,
        )

        if "node_mask" in batch:
            node_mask = torch.as_tensor(batch["node_mask"]).to(
                device=device,
                non_blocking=True,
            )
            node_mask = align_source_rows_to_latent_frames(
                node_mask,
                num_latent_frames=num_latent_frames,
                i2v=i2v,
            ).bool()
        else:
            node_mask = torch.ones(node_tokens.shape[:-1], dtype=torch.bool, device=device)
            node_mask[:, 0] = False if i2v else node_mask[:, 0]

        node_tokens, node_keep = _apply_frame_condition_dropout(
            node_tokens,
            p=node_dropout,
            enabled=condition_dropout_enabled,
        )
        if node_keep is not None:
            node_mask = node_mask & node_keep.unsqueeze(-1)

        cond["node_tokens"] = node_tokens
        cond["node_mask"] = node_mask
        uncond["node_tokens"] = torch.zeros_like(node_tokens)
        uncond["node_mask"] = torch.zeros_like(node_mask)

    return cond, uncond
