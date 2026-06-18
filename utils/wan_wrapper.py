import math
import os
import types
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        ).to(dtype=x.dtype)


class UIActionPositionEncoding(nn.Module):
    """Fourier features over a smooth spherical encoding of normalized x/y."""

    _R_MAX = math.sqrt(2.0)

    def __init__(self, n_freqs: int = 8) -> None:
        super().__init__()
        self.n_freqs = int(n_freqs)
        if self.n_freqs <= 0:
            raise ValueError("action_fourier_freqs must be positive.")
        freqs = torch.tensor(
            [2 ** idx * math.pi for idx in range(self.n_freqs)],
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 4 * 2 * self.n_freqs

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0].clamp(0.0, 1.0)
        y = xy[..., 1].clamp(0.0, 1.0)
        radius = torch.sqrt(x * x + y * y).clamp(max=self._R_MAX)
        theta = torch.atan2(y, x)
        radius_angle = math.pi * radius / self._R_MAX
        s3 = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(radius_angle),
                torch.sin(radius_angle),
            ],
            dim=-1,
        ) / math.sqrt(2.0)
        angles = s3.unsqueeze(-1) * self.freqs.to(device=xy.device, dtype=xy.dtype)
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


class UICenteredPolarPositionEncoding(nn.Module):
    """Angle plus Fourier radius features around the screen center."""

    def __init__(
        self,
        n_freqs: int = 8,
        *,
        target_width: int = 832,
        target_height: int = 480,
    ) -> None:
        super().__init__()
        self.n_freqs = int(n_freqs)
        if self.n_freqs <= 0:
            raise ValueError("action_fourier_freqs must be positive.")
        self.target_width = float(target_width)
        self.target_height = float(target_height)
        corner_dx = self.target_width * 0.5
        corner_dy = self.target_height * 0.5
        self.radius_max = math.sqrt(corner_dx * corner_dx + corner_dy * corner_dy)
        freqs = torch.tensor(
            [2 ** idx * math.pi for idx in range(self.n_freqs)],
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 1 + 2 * self.n_freqs

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0].clamp(0.0, 1.0)
        y = xy[..., 1].clamp(0.0, 1.0)
        dx = (x - 0.5) * self.target_width
        dy = (y - 0.5) * self.target_height
        angle = torch.atan2(dy, dx) / math.pi
        radius = torch.sqrt(dx * dx + dy * dy) / max(self.radius_max, 1e-6)
        radius = radius.clamp(0.0, 1.0)
        radius_angles = radius.unsqueeze(-1) * self.freqs.to(device=xy.device, dtype=xy.dtype)
        radius_features = torch.cat([radius_angles.sin(), radius_angles.cos()], dim=-1)
        return torch.cat([angle.unsqueeze(-1), radius_features], dim=-1)


class UIPatchDiscretePositionEncoding(nn.Module):
    """Learned x/y embeddings after mapping normalized coordinates to WAN patches."""

    def __init__(
        self,
        *,
        grid_height: int = 30,
        grid_width: int = 52,
        emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.emb_dim = int(emb_dim)
        if self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError("patch discrete grid dimensions must be positive.")
        if self.emb_dim <= 1:
            raise ValueError("patch discrete embedding dim must be greater than 1.")
        x_dim = self.emb_dim // 2
        y_dim = self.emb_dim - x_dim
        self.x_embedding = nn.Embedding(self.grid_width, x_dim)
        self.y_embedding = nn.Embedding(self.grid_height, y_dim)

    @property
    def out_dim(self) -> int:
        return self.emb_dim

    def patch_indices(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xy[..., 0].clamp(0.0, 1.0)
        y = xy[..., 1].clamp(0.0, 1.0)
        x_idx = torch.floor(x * self.grid_width).long().clamp(0, self.grid_width - 1)
        y_idx = torch.floor(y * self.grid_height).long().clamp(0, self.grid_height - 1)
        return y_idx, x_idx

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        y_idx, x_idx = self.patch_indices(xy)
        return torch.cat([self.x_embedding(x_idx), self.y_embedding(y_idx)], dim=-1)


def _resolve_wan_model_dir(model_name: str, model_root: Optional[str] = None) -> Path:
    root = Path(model_root or os.environ.get("WAN_MODEL_DIR", "wan_models"))
    if root.name == model_name or (root / "Wan2.1_VAE.pth").exists():
        return root
    return root / model_name


class UIActionNodeConditioner(nn.Module):
    """Frame-aligned UI conditioning adapter for WAN latents."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        action_dim: int = 3,
        action_num_types: int = 6,
        action_noop_type: int = 5,
        action_type_emb_dim: int = 64,
        action_fourier_freqs: int = 8,
        action_embedding_recipe: str = "dfot_exact",
        action_coord_encoding: str = "dfot_s3_fourier",
        action_coord_space: str = "wan_patch_grid",
        action_patch_grid_height: int = 30,
        action_patch_grid_width: int = 52,
        action_coord_emb_dim: int = 64,
        action_token_dim: int = 1024,
        action_spatial_hidden_dim: int = 32,
        action_spatial_sigma: float = 0.035,
        target_width: int = 832,
        target_height: int = 480,
        node_token_dim: int = 1024,
        hidden_dim: int = 256,
        action_dropout: float = 0.0,
        node_dropout: float = 0.0,
        use_node_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_num_types = int(action_num_types)
        self.action_noop_type = int(action_noop_type)
        self.action_spatial_sigma = float(action_spatial_sigma)
        self.action_embedding_recipe = str(action_embedding_recipe)
        self.action_coord_encoding_name = self._resolve_action_coord_encoding(
            self.action_embedding_recipe,
            str(action_coord_encoding),
        )
        self.action_coord_space = str(action_coord_space)
        self.action_token_dim = int(action_token_dim)
        self.node_token_dim = int(node_token_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dropout = float(action_dropout)
        self.node_dropout = float(node_dropout)
        self.use_node_cross_attn = bool(use_node_cross_attn)
        if self.action_dim < 3:
            raise ValueError("UI action conditioning requires action_dim >= 3.")
        if self.action_num_types <= 1:
            raise ValueError("action_num_types must be greater than 1.")
        if not 0 <= self.action_noop_type < self.action_num_types:
            raise ValueError("action_noop_type must be in [0, action_num_types).")
        if self.action_spatial_sigma <= 0.0:
            raise ValueError("action_spatial_sigma must be positive.")

        self.action_type_embedding = nn.Embedding(
            self.action_num_types,
            int(action_type_emb_dim),
        )
        self.action_position_encoding = self._build_action_position_encoding(
            n_freqs=int(action_fourier_freqs),
            grid_height=int(action_patch_grid_height),
            grid_width=int(action_patch_grid_width),
            coord_emb_dim=int(action_coord_emb_dim),
            target_width=int(target_width),
            target_height=int(target_height),
        )
        action_feature_dim = int(action_type_emb_dim) + self.action_position_encoding.out_dim
        self.action_global_proj = nn.Sequential(
            FP32LayerNorm(action_feature_dim),
            nn.Linear(action_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, latent_channels),
        )
        self.action_token_proj = nn.Sequential(
            FP32LayerNorm(action_feature_dim),
            nn.Linear(action_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_token_dim),
        )
        self.action_spatial_proj = nn.Sequential(
            nn.Conv2d(self.action_num_types + 1, int(action_spatial_hidden_dim), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(int(action_spatial_hidden_dim), latent_channels, kernel_size=1),
        )
        self.action_residual_gate = nn.Parameter(torch.full((1,), 1e-2))

        self.q_proj = nn.Linear(latent_channels, self.hidden_dim)
        self.k_proj = nn.Linear(self.node_token_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.node_token_dim, self.hidden_dim)
        self.node_out = nn.Linear(self.hidden_dim, latent_channels)
        self.node_gate = nn.Parameter(torch.full((1,), 1e-3))

    @staticmethod
    def _resolve_action_coord_encoding(recipe: str, coord_encoding: str) -> str:
        if recipe == "dfot_exact":
            return "dfot_s3_fourier"
        if recipe == "patch_discrete":
            return "patch_discrete"
        if recipe == "centered_polar":
            return "centered_polar_fourier"
        if recipe != "custom":
            raise ValueError(
                "action_embedding_recipe must be one of "
                "'dfot_exact', 'patch_discrete', 'centered_polar', or 'custom'."
            )
        if coord_encoding not in {
            "dfot_s3_fourier",
            "patch_discrete",
            "centered_polar_fourier",
        }:
            raise ValueError(f"Unsupported action_coord_encoding: {coord_encoding!r}")
        return coord_encoding

    def _build_action_position_encoding(
        self,
        *,
        n_freqs: int,
        grid_height: int,
        grid_width: int,
        coord_emb_dim: int,
        target_width: int,
        target_height: int,
    ) -> nn.Module:
        if self.action_coord_space != "wan_patch_grid":
            raise ValueError("Only action_coord_space='wan_patch_grid' is supported.")
        if self.action_coord_encoding_name == "dfot_s3_fourier":
            return UIActionPositionEncoding(n_freqs=n_freqs)
        if self.action_coord_encoding_name == "patch_discrete":
            return UIPatchDiscretePositionEncoding(
                grid_height=grid_height,
                grid_width=grid_width,
                emb_dim=coord_emb_dim,
            )
        if self.action_coord_encoding_name == "centered_polar_fourier":
            return UICenteredPolarPositionEncoding(
                n_freqs=n_freqs,
                target_width=target_width,
                target_height=target_height,
            )
        raise ValueError(f"Unsupported action coordinate encoding: {self.action_coord_encoding_name!r}")

    def enable_trainable_parameters(self) -> None:
        self.requires_grad_(True)
        if not self.use_node_cross_attn:
            for module in (self.q_proj, self.k_proj, self.v_proj, self.node_out):
                module.requires_grad_(False)
            self.node_gate.requires_grad_(False)

    def _drop_frame_condition(self, value: torch.Tensor, p: float) -> torch.Tensor:
        if not self.training or p <= 0.0:
            return value
        keep = torch.rand(value.shape[:2], device=value.device) >= p
        while keep.ndim < value.ndim:
            keep = keep.unsqueeze(-1)
        return value * keep.to(dtype=value.dtype)

    def prepare_action_condition(
        self,
        action_cond: torch.Tensor,
        *,
        device: torch.device,
        drop_action: bool = True,
    ) -> torch.Tensor:
        action_dtype = self.action_type_embedding.weight.dtype
        action_cond = action_cond.to(device=device, dtype=action_dtype)
        if drop_action:
            action_cond = self._drop_frame_condition(action_cond, self.action_dropout)
        return action_cond

    def _action_components(
        self,
        action_cond: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = torch.nan_to_num(
            action_cond[..., :3].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        raw_type = clean[..., 0]
        xy = clean[..., 1:3].clamp(0.0, 1.0)
        action_type = raw_type.round().long().clamp(0, self.action_num_types - 1)

        null_row = clean.abs().amax(dim=-1) < 1e-8
        noop_row = action_type == self.action_noop_type
        valid = ~(null_row | noop_row)
        valid_f = valid.to(dtype=dtype)
        return clean, action_type, xy, valid_f

    def _action_feature(
        self,
        action_type: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        action_feature_dtype = self.action_type_embedding.weight.dtype
        return torch.cat(
            [
                self.action_type_embedding(action_type).to(dtype=action_feature_dtype),
                self.action_position_encoding(xy.float()).to(dtype=action_feature_dtype),
            ],
            dim=-1,
        )

    def _encode_action_condition(
        self,
        action_cond: torch.Tensor,
        *,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action_cond.ndim != 3 or action_cond.shape[-1] < 3:
            raise ValueError(
                "action_cond must have shape [B, F, >=3], got "
                f"{tuple(action_cond.shape)}."
            )

        device = action_cond.device
        _, action_type, xy, valid_f = self._action_components(action_cond, dtype=dtype)
        action_feature = self._action_feature(action_type, xy)
        global_residual = self.action_global_proj(action_feature).to(dtype=dtype)
        global_residual = global_residual * valid_f[..., None]

        grid_y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
        grid_x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        dx = xx[None, None] - xy[..., 0, None, None]
        dy = yy[None, None] - xy[..., 1, None, None]
        sigma = float(self.action_spatial_sigma)
        heatmap = torch.exp(-0.5 * ((dx / sigma) ** 2 + (dy / sigma) ** 2))
        heatmap = heatmap * valid_f[..., None, None].float()

        type_maps = F.one_hot(
            action_type,
            num_classes=self.action_num_types,
        ).to(dtype=torch.float32)
        type_maps = type_maps[..., :, None, None] * heatmap.unsqueeze(-3)
        spatial_input = torch.cat([heatmap.unsqueeze(-3), type_maps], dim=-3)
        bsz, frames = spatial_input.shape[:2]
        spatial_residual = self.action_spatial_proj(
            spatial_input.reshape(
                bsz * frames,
                self.action_num_types + 1,
                height,
                width,
            ).to(dtype=self.action_spatial_proj[0].weight.dtype)
        )
        spatial_residual = spatial_residual.reshape(
            bsz,
            frames,
            spatial_residual.shape[1],
            height,
            width,
        ).to(dtype=dtype)
        spatial_residual = spatial_residual * valid_f[:, :, None, None, None]

        return spatial_residual + global_residual[..., None, None]

    def build_action_tokens(
        self,
        action_cond: torch.Tensor,
        *,
        height: int,
        width: int,
        dtype: torch.dtype,
        drop_action: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_cond.ndim != 3 or action_cond.shape[-1] < 3:
            raise ValueError(
                "action_cond must have shape [B, F, >=3], got "
                f"{tuple(action_cond.shape)}."
            )
        del height, width
        action_cond = self.prepare_action_condition(
            action_cond,
            device=action_cond.device,
            drop_action=drop_action,
        )
        _, action_type, xy, valid_f = self._action_components(action_cond, dtype=dtype)
        action_feature = self._action_feature(action_type, xy)
        token = self.action_token_proj(action_feature).to(dtype=dtype).unsqueeze(-2)
        mask = valid_f.bool().unsqueeze(-1)
        positions = xy.to(dtype=dtype).unsqueeze(-2)
        token = token * valid_f[..., None, None]
        return token, mask, positions

    def forward(
        self,
        latents: torch.Tensor,
        *,
        action_cond: Optional[torch.Tensor] = None,
        node_tokens: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        drop_action: bool = True,
    ) -> torch.Tensor:
        # latents: [B, F, C, H, W]
        out = latents
        dtype = latents.dtype

        if action_cond is not None:
            if action_cond.shape[:2] != latents.shape[:2]:
                raise ValueError(
                    "action_cond frame shape must match latents: "
                    f"got {tuple(action_cond.shape[:2])}, expected {tuple(latents.shape[:2])}."
                )
            action_cond = self.prepare_action_condition(
                action_cond,
                device=latents.device,
                drop_action=drop_action,
            )
            action_residual = self._encode_action_condition(
                action_cond,
                height=latents.shape[-2],
                width=latents.shape[-1],
                dtype=dtype,
            )
            out = out + torch.tanh(self.action_residual_gate).to(dtype=dtype) * action_residual

        if self.use_node_cross_attn and node_tokens is not None:
            bsz, frames, channels, height, width = out.shape
            node_dtype = self.q_proj.weight.dtype
            tokens = self._drop_frame_condition(
                node_tokens.to(device=latents.device, dtype=node_dtype),
                self.node_dropout,
            )
            tokens = tokens.reshape(bsz * frames, tokens.shape[-2], tokens.shape[-1])

            if node_mask is None:
                mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=out.device)
            else:
                mask = node_mask.reshape(bsz * frames, node_mask.shape[-1]).to(device=out.device).bool()
                empty = ~mask.any(dim=-1)
                if empty.any():
                    mask = mask.clone()
                    mask[empty, 0] = True
                    tokens = tokens.clone()
                    tokens[empty] = 0

            spatial = out.permute(0, 1, 3, 4, 2).reshape(bsz * frames, height * width, channels)
            query = self.q_proj(spatial.to(dtype=node_dtype))
            key = self.k_proj(tokens)
            value = self.v_proj(tokens)
            score = torch.bmm(query.float(), key.float().transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
            score = score.masked_fill(~mask[:, None, :], torch.finfo(score.dtype).min)
            attn = torch.softmax(score, dim=-1).to(dtype=node_dtype)
            attended = torch.bmm(attn, value)
            residual = self.node_out(attended).to(dtype=dtype)
            residual = residual.reshape(bsz, frames, height, width, channels).permute(0, 1, 4, 2, 3)
            out = out + torch.tanh(self.node_gate).to(dtype=dtype) * residual

        return out


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_root: Optional[str] = None) -> None:
        super().__init__()
        model_dir = _resolve_wan_model_dir(model_name, model_root)

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=str(model_dir / "google" / "umt5-xxl"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_root: Optional[str] = None):
        super().__init__()
        model_dir = _resolve_wan_model_dir(model_name, model_root)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=str(model_dir / "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_framewise_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode independently encoded UI latents without temporal expansion."""
        batch_size, num_frames, channels, height, width = latent.shape
        flattened = latent.reshape(batch_size * num_frames, 1, channels, height, width)
        decoded = self.decode_to_pixel(flattened, use_cache=False)
        if decoded.shape[1] != 1:
            raise RuntimeError(
                "Frame-wise WAN VAE decode returned "
                f"{decoded.shape[1]} frames per latent; expected 1."
            )
        return decoded.reshape(
            batch_size,
            num_frames,
            decoded.shape[2],
            decoded.shape[3],
            decoded.shape[4],
        )


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            model_root: Optional[str] = None,
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            ui_conditioning: Optional[dict] = None,
    ):
        super().__init__()
        model_dir = _resolve_wan_model_dir(model_name, model_root)

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                str(model_dir), local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(str(model_dir))
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.frame_seq_length = 1560
        self.ui_conditioner = None
        self.node_conditioning_enabled = False
        self.block_cross_attn_enabled = False
        self.action_block_cross_attn_enabled = False
        self.require_graph_token_positions = False
        self.train_ui_conditioner_only = False
        self.train_condition_cross_attn_only = False
        if ui_conditioning is not None and bool(ui_conditioning.get("enabled", False)):
            node_token_dim = int(ui_conditioning.get("node_token_dim", 1024))
            node_dropout = float(ui_conditioning.get("node_dropout", 0.0))
            node_attention_dropout = float(ui_conditioning.get("node_attention_dropout", 0.0))
            action_attention_dropout = float(ui_conditioning.get("action_attention_dropout", 0.0))
            action_token_dim = int(ui_conditioning.get("action_token_dim", node_token_dim))
            condition_position_encoding = str(
                ui_conditioning.get("condition_position_encoding", "learned_projection")
            )
            self.block_cross_attn_enabled = bool(ui_conditioning.get("block_cross_attn", False))
            self.action_block_cross_attn_enabled = bool(ui_conditioning.get("action_block_cross_attn", False))
            self.require_graph_token_positions = bool(ui_conditioning.get("require_graph_token_positions", False))
            has_node_token_dim = (
                hasattr(ui_conditioning, "__contains__")
                and "node_token_dim" in ui_conditioning
            )
            self.node_conditioning_enabled = (
                self.block_cross_attn_enabled
                or bool(ui_conditioning.get("node_conditioning", False))
                or has_node_token_dim
            )
            if self.block_cross_attn_enabled:
                self.model.add_condition_cross_attention(
                    condition_dim=node_token_dim,
                    dropout=node_attention_dropout,
                    position_encoding=condition_position_encoding,
                )
            if self.action_block_cross_attn_enabled:
                add_action_cross_attn = getattr(self.model, "add_action_condition_cross_attention", None)
                if add_action_cross_attn is None:
                    raise ValueError("WAN model does not support action condition cross-attention.")
                add_action_cross_attn(
                    condition_dim=action_token_dim,
                    dropout=action_attention_dropout,
                    position_encoding=condition_position_encoding,
                )
            self.ui_conditioner = UIActionNodeConditioner(
                latent_channels=int(ui_conditioning.get("latent_channels", 16)),
                action_dim=int(ui_conditioning.get("action_dim", 3)),
                action_num_types=int(ui_conditioning.get("action_num_types", 6)),
                action_noop_type=int(ui_conditioning.get("action_noop_type", 5)),
                action_type_emb_dim=int(ui_conditioning.get("action_type_emb_dim", 64)),
                action_fourier_freqs=int(ui_conditioning.get("action_fourier_freqs", 8)),
                action_embedding_recipe=str(ui_conditioning.get("action_embedding_recipe", "dfot_exact")),
                action_coord_encoding=str(ui_conditioning.get("action_coord_encoding", "dfot_s3_fourier")),
                action_coord_space=str(ui_conditioning.get("action_coord_space", "wan_patch_grid")),
                action_patch_grid_height=int(ui_conditioning.get("action_patch_grid_height", 30)),
                action_patch_grid_width=int(ui_conditioning.get("action_patch_grid_width", 52)),
                action_coord_emb_dim=int(ui_conditioning.get("action_coord_emb_dim", 64)),
                action_token_dim=action_token_dim,
                action_spatial_hidden_dim=int(ui_conditioning.get("action_spatial_hidden_dim", 32)),
                action_spatial_sigma=float(ui_conditioning.get("action_spatial_sigma", 0.035)),
                target_width=int(ui_conditioning.get("target_width", 832)),
                target_height=int(ui_conditioning.get("target_height", 480)),
                node_token_dim=node_token_dim,
                hidden_dim=int(ui_conditioning.get("hidden_dim", 256)),
                action_dropout=float(ui_conditioning.get("action_dropout", 0.0)),
                node_dropout=node_dropout,
                use_node_cross_attn=self.node_conditioning_enabled and not self.block_cross_attn_enabled,
            )
            self.train_condition_cross_attn_only = bool(
                ui_conditioning.get("condition_cross_attn_only", False)
            )
            if self.train_condition_cross_attn_only and not self.block_cross_attn_enabled:
                raise ValueError("condition_cross_attn_only requires block_cross_attn=true.")
            self.train_ui_conditioner_only = (
                bool(ui_conditioning.get("freeze_backbone", False))
                or self.train_condition_cross_attn_only
            )
        self.post_init()

    def enable_trainable_ui_conditioning_only(self) -> None:
        self.model.requires_grad_(False)
        if self.ui_conditioner is not None:
            if self.train_condition_cross_attn_only:
                self.ui_conditioner.requires_grad_(False)
            else:
                self.ui_conditioner.enable_trainable_parameters()
        set_condition_grad = getattr(self.model, "set_condition_cross_attention_requires_grad", None)
        if set_condition_grad is not None:
            set_condition_grad(True)
        set_action_condition_grad = getattr(self.model, "set_action_condition_cross_attention_requires_grad", None)
        if set_action_condition_grad is not None:
            set_action_condition_grad(True)

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            FP32LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _frame_start_from_tokens(self, current_start: Optional[int]) -> int:
        if current_start is None:
            return 0
        if isinstance(current_start, torch.Tensor):
            current_start = int(current_start.item())
        return int(current_start) // self.frame_seq_length

    def _slice_condition(self, value: torch.Tensor, frame_start: int, frames: int) -> torch.Tensor:
        if value.shape[1] == frames:
            return value
        if value.shape[1] >= frame_start + frames:
            return value[:, frame_start:frame_start + frames]
        if value.shape[1] == 1:
            return value.expand(-1, frames, *value.shape[2:])
        raise ValueError(
            f"UI condition has {value.shape[1]} frames, cannot slice "
            f"frame_start={frame_start}, frames={frames}."
        )

    def _prepare_ui_conditioning(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        current_start: Optional[int],
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        action_block_cross_attn_enabled = bool(getattr(self, "action_block_cross_attn_enabled", False))
        block_cross_attn_enabled = bool(getattr(self, "block_cross_attn_enabled", False))
        if self.ui_conditioner is None and not (block_cross_attn_enabled or action_block_cross_attn_enabled):
            return noisy_image_or_video, None, None, None, None, None, None
        if "action_cond" not in conditional_dict and "node_tokens" not in conditional_dict:
            return noisy_image_or_video, None, None, None, None, None, None

        frames = noisy_image_or_video.shape[1]
        frame_start = int(conditional_dict.get("ui_frame_start", self._frame_start_from_tokens(current_start)))
        action_cond = conditional_dict.get("action_cond")
        node_tokens = conditional_dict.get("node_tokens")
        node_mask = conditional_dict.get("node_mask")
        node_positions = conditional_dict.get("node_positions")
        if action_cond is not None:
            action_cond = self._slice_condition(action_cond.to(device=noisy_image_or_video.device), frame_start, frames)
        if node_tokens is not None:
            node_tokens = self._slice_condition(node_tokens.to(device=noisy_image_or_video.device), frame_start, frames)
        if node_mask is not None:
            node_mask = self._slice_condition(node_mask.to(device=noisy_image_or_video.device), frame_start, frames)
        if node_positions is not None:
            node_positions = self._slice_condition(node_positions.to(device=noisy_image_or_video.device), frame_start, frames)

        action_tokens = None
        action_mask = None
        action_positions = None
        if self.ui_conditioner is not None and action_cond is not None:
            prepare_action_condition = getattr(self.ui_conditioner, "prepare_action_condition", None)
            if prepare_action_condition is not None:
                action_cond = prepare_action_condition(
                    action_cond,
                    device=noisy_image_or_video.device,
                    drop_action=True,
                )

        if self.ui_conditioner is not None:
            noisy_image_or_video = self.ui_conditioner(
                noisy_image_or_video,
                action_cond=action_cond,
                node_tokens=node_tokens if self.node_conditioning_enabled and not self.block_cross_attn_enabled else None,
                node_mask=node_mask if self.node_conditioning_enabled and not self.block_cross_attn_enabled else None,
                drop_action=False,
            )
            if action_block_cross_attn_enabled and action_cond is not None:
                action_tokens, action_mask, action_positions = self.ui_conditioner.build_action_tokens(
                    action_cond,
                    height=noisy_image_or_video.shape[-2],
                    width=noisy_image_or_video.shape[-1],
                    dtype=noisy_image_or_video.dtype,
                    drop_action=False,
                )
        if not self.node_conditioning_enabled or not block_cross_attn_enabled:
            node_tokens = None
            node_mask = None
            node_positions = None
        elif bool(getattr(self, "require_graph_token_positions", False)) and node_tokens is not None and node_positions is None:
            raise ValueError("Graph-token positions are required but missing from the UI condition dict.")
        return noisy_image_or_video, node_tokens, node_mask, node_positions, action_tokens, action_mask, action_positions

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        
        classify_mode: Optional[bool] = False, # DF
        concat_time_embeddings: Optional[bool] = False, #DF
        clean_x: Optional[torch.Tensor] = None, # TF
        aug_t: Optional[torch.Tensor] = None, # for TF clean GT, if it's also noisy and needs denoising by the model, aug_t is its timestep
        
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        (
            noisy_image_or_video,
            condition_tokens,
            condition_mask,
            condition_positions,
            action_tokens,
            action_mask,
            action_positions,
        ) = self._prepare_ui_conditioning(
            noisy_image_or_video,
            conditional_dict,
            current_start=current_start,
        )

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                condition_tokens=condition_tokens,
                condition_mask=condition_mask,
                condition_positions=condition_positions,
                action_tokens=action_tokens,
                action_mask=action_mask,
                action_positions=action_positions,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    aug_t=aug_t,
                    condition_tokens=condition_tokens,
                    condition_mask=condition_mask,
                    condition_positions=condition_positions,
                    action_tokens=action_tokens,
                    action_mask=action_mask,
                    action_positions=action_positions,
                ).permute(0, 2, 1, 3, 4)
            else:
                # diffusion forcing or bidirectional
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        condition_tokens=condition_tokens,
                        condition_mask=condition_mask,
                        condition_positions=condition_positions,
                        action_tokens=action_tokens,
                        action_mask=action_mask,
                        action_positions=action_positions,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        condition_tokens=condition_tokens,
                        condition_mask=condition_mask,
                        condition_positions=condition_positions,
                        action_tokens=action_tokens,
                        action_mask=action_mask,
                        action_positions=action_positions,
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
