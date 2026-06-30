# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        ).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanCrossAttention(nn.Module):
    """Generic frame-aligned cross-attention for extra condition tokens."""

    def __init__(
        self,
        dim,
        num_heads,
        condition_dim=None,
        qk_norm=True,
        eps=1e-6,
        dropout=0.0,
        position_encoding="learned_projection",
        position_rope_scale=2.0 * math.pi,
        position_rope_theta=10000.0,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.condition_dim = dim if condition_dim is None else int(condition_dim)
        self.dropout = float(dropout)
        self.position_encoding = str(position_encoding)
        if self.position_encoding not in {"learned_projection", "rope_2d"}:
            raise ValueError(
                "position_encoding must be 'learned_projection' or 'rope_2d', "
                f"got {position_encoding!r}."
            )
        if self.position_encoding == "rope_2d" and self.head_dim < 4:
            raise ValueError("rope_2d condition position encoding requires head_dim >= 4.")
        self.position_rope_scale = float(position_rope_scale)
        self.position_rope_theta = float(position_rope_theta)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(self.condition_dim, dim)
        self.v = nn.Linear(self.condition_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.query_pos = nn.Linear(2, dim, bias=False)
        self.condition_pos = nn.Linear(2, dim, bias=False)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.gate = nn.Parameter(torch.full((1,), 1e-3))
        self.store_attn_weights = False
        self.first_attn_slot_map = None
        self.first_attn_patch_map = None
        self.last_attn_slot_map = None
        self.last_attn_patch_map = None
        self.last_attn_num_frames = 0
        self.last_attn_tokens_per_frame = 0
        self.last_attn_frame_seqlen = 0
        self.last_attn_grid_hw = None
        self.last_attn_frame_has_condition = None

        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.o.weight)
        nn.init.xavier_uniform_(self.query_pos.weight)
        nn.init.xavier_uniform_(self.condition_pos.weight)
        for layer in (self.q, self.k, self.v, self.o):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _align_frames(value, num_frames, *, fill_value=0, unframed_ndim=3):
        if value is None:
            return None
        if value.ndim == unframed_ndim:
            value = value.unsqueeze(1)
        if value.shape[1] == num_frames:
            return value
        if value.shape[1] == 1:
            return value.expand(-1, num_frames, *value.shape[2:])
        if value.shape[1] * 2 == num_frames:
            pad_shape = list(value.shape)
            pad_shape[1] = value.shape[1]
            pad = value.new_full(pad_shape, fill_value)
            return torch.cat([pad, value], dim=1)
        raise ValueError(
            f"Condition has {value.shape[1]} frames, cannot align to {num_frames} frames."
        )

    @staticmethod
    def _frame_query_positions(
        *,
        bsz,
        num_frames,
        frame_seqlen,
        grid_sizes,
        device,
        dtype,
    ):
        if grid_sizes is None:
            return None
        positions = []
        for _, height, width in grid_sizes.tolist():
            height = int(height)
            width = int(width)
            if height * width != int(frame_seqlen):
                return None
            y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
            x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            frame_pos = torch.stack([xx, yy], dim=-1).reshape(1, frame_seqlen, 2)
            positions.append(frame_pos.expand(num_frames, -1, -1))
        if len(positions) != int(bsz):
            return None
        return torch.cat(positions, dim=0)

    @staticmethod
    def _apply_axis_rope(part, coord, inv_freq, scale):
        part_dtype = part.dtype
        angles = (
            coord.float().unsqueeze(1).unsqueeze(-1)
            * float(scale)
            * inv_freq.to(device=part.device).float().view(1, 1, 1, -1)
        )
        even = part.float()[..., 0::2]
        odd = part.float()[..., 1::2]
        cos = angles.cos()
        sin = angles.sin()
        rotated = torch.stack(
            [even * cos - odd * sin, even * sin + odd * cos],
            dim=-1,
        ).flatten(-2)
        return rotated.to(dtype=part_dtype)

    def _rope_axis_dims(self):
        x_dim = (self.head_dim // 2) // 2 * 2
        y_dim = ((self.head_dim - x_dim) // 2) * 2
        if x_dim <= 0 or y_dim <= 0:
            raise ValueError(
                f"rope_2d requires at least one rotary pair per axis, got head_dim={self.head_dim}."
            )
        return x_dim, y_dim

    def _rope_inv_freq(self, axis_dim, *, device):
        return 1.0 / (
            self.position_rope_theta
            ** (torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32) / float(axis_dim))
        )

    def _apply_2d_rope(self, value, positions):
        x_dim, y_dim = self._rope_axis_dims()
        x_part = value[..., :x_dim]
        y_part = value[..., x_dim:x_dim + y_dim]
        rest = value[..., x_dim + y_dim:]
        x_part = self._apply_axis_rope(
            x_part,
            positions[..., 0],
            self._rope_inv_freq(x_dim, device=value.device),
            self.position_rope_scale,
        )
        y_part = self._apply_axis_rope(
            y_part,
            positions[..., 1],
            self._rope_inv_freq(y_dim, device=value.device),
            self.position_rope_scale,
        )
        if rest.numel() == 0:
            return torch.cat([x_part, y_part], dim=-1)
        return torch.cat([x_part, y_part, rest], dim=-1)

    def forward(
        self,
        x,
        condition_tokens,
        condition_mask=None,
        condition_positions=None,
        *,
        num_frames=None,
        frame_seqlen=None,
        grid_sizes=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C].
            condition_tokens(Tensor): Shape [B, F, K, D] or [B, K, D].
            condition_mask(Tensor, optional): Shape [B, F, K] or [B, K].
        """
        if condition_tokens is None:
            return x.new_zeros(x.shape)

        bsz, seq_len, channels = x.shape
        if num_frames is None:
            num_frames = condition_tokens.shape[1] if condition_tokens.ndim == 4 else 1
        num_frames = int(num_frames)
        if frame_seqlen is None:
            if seq_len % num_frames != 0:
                raise ValueError(
                    f"Sequence length {seq_len} is not divisible by {num_frames} frames."
                )
            frame_seqlen = seq_len // num_frames
        frame_seqlen = int(frame_seqlen)
        active_len = num_frames * frame_seqlen
        if active_len > seq_len:
            raise ValueError(
                f"Condition span {active_len} exceeds sequence length {seq_len}."
            )

        dtype = x.dtype
        tokens = condition_tokens.to(device=x.device, dtype=dtype)
        tokens = self._align_frames(tokens, num_frames)
        if tokens.shape[0] != bsz:
            raise ValueError(
                f"Condition batch {tokens.shape[0]} does not match latent batch {bsz}."
            )

        if condition_mask is None:
            mask = torch.ones(tokens.shape[:3], dtype=torch.bool, device=x.device)
        else:
            mask = condition_mask.to(device=x.device).bool()
            mask = self._align_frames(
                mask,
                num_frames,
                fill_value=False,
                unframed_ndim=2,
            )
        positions = None
        if condition_positions is not None:
            positions = condition_positions.to(device=x.device, dtype=dtype)
            positions = self._align_frames(positions, num_frames)

        flat_tokens = tokens.reshape(bsz * num_frames, tokens.shape[-2], tokens.shape[-1])
        flat_mask = mask.reshape(bsz * num_frames, mask.shape[-1])
        flat_positions = None
        if positions is not None:
            if positions.shape[:3] != tokens.shape[:3] or positions.shape[-1] != 2:
                raise ValueError(
                    "condition_positions must have shape [B, F, K, 2] aligned "
                    f"with condition tokens, got {tuple(positions.shape)} for "
                    f"tokens {tuple(tokens.shape)}."
                )
            flat_positions = positions.reshape(bsz * num_frames, positions.shape[-2], positions.shape[-1])
        frame_has_condition = flat_mask.any(dim=-1)
        empty = ~frame_has_condition
        if empty.any():
            flat_mask = flat_mask.clone()
            flat_mask[empty, 0] = True
            flat_tokens = flat_tokens.clone()
            flat_tokens[empty] = 0
            if flat_positions is not None:
                flat_positions = flat_positions.clone()
                flat_positions[empty] = 0.5

        x_active = x[:, :active_len]
        query_tokens = x_active.reshape(bsz * num_frames, frame_seqlen, channels)
        q_linear = self.q(query_tokens)
        k_linear = self.k(flat_tokens)
        query_positions = None
        if flat_positions is not None:
            query_positions = self._frame_query_positions(
                bsz=bsz,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                grid_sizes=grid_sizes,
                device=x.device,
                dtype=dtype,
            )
            if self.position_encoding == "learned_projection":
                if query_positions is not None:
                    q_pos = self.query_pos(
                        query_positions.to(dtype=self.query_pos.weight.dtype)
                    ).to(dtype=q_linear.dtype)
                    q_linear = q_linear + q_pos
                k_pos = self.condition_pos(
                    flat_positions.to(dtype=self.condition_pos.weight.dtype)
                ).to(dtype=k_linear.dtype)
                k_linear = k_linear + k_pos
            elif query_positions is None:
                raise ValueError("rope_2d condition position encoding requires grid_sizes.")
        q = self.norm_q(q_linear).view(
            bsz * num_frames, frame_seqlen, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.norm_k(k_linear).view(
            bsz * num_frames, flat_tokens.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        if flat_positions is not None and self.position_encoding == "rope_2d":
            if query_positions is None:
                raise ValueError("rope_2d condition position encoding requires grid_sizes.")
            q = self._apply_2d_rope(q, query_positions)
            k = self._apply_2d_rope(k, flat_positions)
        v = self.v(flat_tokens).view(
            bsz * num_frames, flat_tokens.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_mask = torch.zeros(
            flat_mask.shape[0], 1, 1, flat_mask.shape[1],
            dtype=q.dtype, device=x.device,
        )
        attn_mask = attn_mask.masked_fill(~flat_mask[:, None, None, :], torch.finfo(q.dtype).min)
        if self.store_attn_weights:
            scale = q.shape[-1] ** -0.5
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            scores = scores + attn_mask.float()
            attn_weights = scores.softmax(dim=-1)
            attn_for_value = (
                F.dropout(attn_weights, p=self.dropout, training=True)
                if self.training and self.dropout > 0
                else attn_weights
            )
            attended = torch.matmul(attn_for_value, v.float()).to(dtype=q.dtype)

            frame_keep = frame_has_condition.detach().to(device=attn_weights.device)
            stored = attn_weights.detach().float()
            stored = stored * frame_keep[:, None, None, None].float()
            slot_map = stored.mean(dim=(1, 2)).cpu()
            patch_map = stored.mean(dim=1).cpu()
            if self.first_attn_slot_map is None:
                self.first_attn_slot_map = slot_map
                self.first_attn_patch_map = patch_map
            self.last_attn_slot_map = slot_map
            self.last_attn_patch_map = patch_map
            self.last_attn_num_frames = int(num_frames)
            self.last_attn_tokens_per_frame = int(flat_tokens.shape[1])
            self.last_attn_frame_seqlen = int(frame_seqlen)
            self.last_attn_frame_has_condition = frame_has_condition.detach().cpu()
            self.last_attn_grid_hw = None
            if grid_sizes is not None and len(grid_sizes) > 0:
                _, grid_h, grid_w = grid_sizes[0].detach().cpu().tolist()
                grid_h = int(grid_h)
                grid_w = int(grid_w)
                if grid_h * grid_w == int(frame_seqlen):
                    self.last_attn_grid_hw = (grid_h, grid_w)
        else:
            attended = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        attended = attended.transpose(1, 2).flatten(2)
        residual = self.o(attended)
        residual = residual * frame_has_condition[:, None, None].to(dtype=residual.dtype)
        residual = residual.reshape(bsz, active_len, channels)

        out = x.new_zeros(x.shape)
        out[:, :active_len] = torch.tanh(self.gate).to(dtype=dtype) * residual.to(dtype=dtype)
        return out


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.condition_cross_attn = None
        self.action_condition_cross_attn = None
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        condition_tokens=None,
        condition_mask=None,
        condition_positions=None,
        action_tokens=None,
        action_mask=None,
        action_positions=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            if self.action_condition_cross_attn is not None and action_tokens is not None:
                num_frames = int(action_tokens.shape[1]) if action_tokens.ndim == 4 else 1
                frame_seqlen = x.shape[1] // num_frames
                x = x + self.action_condition_cross_attn(
                    self.norm3(x),
                    action_tokens,
                    action_mask,
                    action_positions,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    grid_sizes=grid_sizes,
                )
            if self.condition_cross_attn is not None and condition_tokens is not None:
                num_frames = int(condition_tokens.shape[1]) if condition_tokens.ndim == 4 else 1
                frame_seqlen = x.shape[1] // num_frames
                x = x + self.condition_cross_attn(
                    self.norm3(x),
                    condition_tokens,
                    condition_mask,
                    condition_positions,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    grid_sizes=grid_sizes,
                )
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class GanAttentionBlock(nn.Module):

    def __init__(self,
                 dim=1536,
                 ffn_dim=8192,
                 num_heads=12,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        # self.norm1 = WanLayerNorm(dim, eps)
        # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        #   eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.cross_attn = WanGanCrossAttention(dim, num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        context,
        # seq_lens,
        # grid_sizes,
        # freqs,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        # e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # # self-attention
        # y = self.self_attn(
        #     self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        #     freqs)
        # # with amp.autocast(dtype=torch.float32):
        # x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context):
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token  # * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            # x = x + y * e[5]
            return y

        x = cross_attn_ffn(x, context)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            WanLayerNorm(in_dim, eps=1e-5, elementwise_affine=True), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            WanLayerNorm(out_dim, eps=1e-5, elementwise_affine=True))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(
        self,
        module=None,
        value=False,
        enable=None,
        gradient_checkpointing_func=None,
    ):
        self.gradient_checkpointing = bool(value if enable is None else enable)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        condition_tokens=None,
        condition_mask=None,
        condition_positions=None,
        action_tokens=None,
        action_mask=None,
        action_positions=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            condition_positions=condition_positions,
            action_tokens=action_tokens,
            action_mask=action_mask,
            action_positions=action_positions)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
            # x = torch.cat([registers, x], dim=1)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x

        return torch.stack(x)

    def add_condition_cross_attention(
        self,
        condition_dim=1024,
        dropout=0.0,
        position_encoding="learned_projection",
    ):
        for block in self.blocks:
            block.condition_cross_attn = WanCrossAttention(
                self.dim,
                self.num_heads,
                condition_dim=condition_dim,
                qk_norm=self.qk_norm,
                eps=self.eps,
                dropout=dropout,
                position_encoding=position_encoding,
            )

    def add_action_condition_cross_attention(
        self,
        condition_dim=1024,
        dropout=0.0,
        position_encoding="learned_projection",
    ):
        for block in self.blocks:
            block.action_condition_cross_attn = WanCrossAttention(
                self.dim,
                self.num_heads,
                condition_dim=condition_dim,
                qk_norm=self.qk_norm,
                eps=self.eps,
                dropout=dropout,
                position_encoding=position_encoding,
            )

    def set_condition_cross_attention_requires_grad(self, requires_grad=True):
        for block in self.blocks:
            if block.condition_cross_attn is not None:
                block.condition_cross_attn.requires_grad_(requires_grad)

    def set_action_condition_cross_attention_requires_grad(self, requires_grad=True):
        for block in self.blocks:
            if block.action_condition_cross_attn is not None:
                block.action_condition_cross_attn.requires_grad_(requires_grad)

    def _forward_classify(
        self,
        x,
        t,
        context,
        seq_len,
        register_tokens,
        cls_pred_branch,
        clip_fea=None,
        y=None,
    ):
        r"""
        Feature extraction through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of video features with original input shapes [C_block, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, c=self.dim // 4)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes, c=None):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim if c is None else c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
