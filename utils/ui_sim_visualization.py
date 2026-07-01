from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data._utils.collate import default_collate
from torchvision.io import write_png, write_video

from pipeline import CausalDiffusionInferencePipeline
from utils.ui_sim_conditioning import attach_ui_batch_conditioning
from utils.ui_sim_dataset import UISimLatentDataset


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class UISimTrainingVisualizer:
    """Deterministic denoising and causal-rollout previews for UI training."""

    def __init__(
        self,
        *,
        config: Any,
        model: Any,
        training_dataset: Any,
        output_path: str,
        device: int,
        dtype: torch.dtype,
        is_main_process: bool,
        disable_wandb: bool,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.dtype = dtype
        self.is_main_process = is_main_process
        self.disable_wandb = disable_wandb
        self.enabled = not bool(config.no_visualize)
        self.denoising_interval = int(getattr(config, "visualize_interval", 0))
        self.rollout_interval = int(getattr(config, "eval_interval", 0))
        self.num_frames = int(getattr(config, "eval_num_output_frames", 21))
        self.sample_index = int(getattr(config, "visualize_sample_index", 0))
        self.seed = int(getattr(config, "visualize_seed", 0))
        self.timestep_index = int(getattr(config, "visualize_timestep", 500))
        self.fps = int(getattr(config, "visualize_fps", 4))
        self.cross_attn_maps = _as_bool(getattr(config, "cross_attn_maps", False))
        self.cross_attn_map_max_blocks = int(
            getattr(config, "cross_attn_map_max_blocks", 4)
        )
        self.cross_attn_map_topk = int(getattr(config, "cross_attn_map_topk", 4))
        self.cross_attn_map_frame_limit = int(
            getattr(config, "cross_attn_map_frame_limit", 3)
        )
        self.output_dir = Path(output_path) / "visualizations"

        eval_data_path = str(
            getattr(config, "eval_data_path", "")
            or os.environ.get("UI_SIM_CF_VALIDATION_CACHE", "")
        ).strip()
        self.dataset = (
            UISimLatentDataset(eval_data_path, config)
            if eval_data_path
            else training_dataset
        )
        if self.enabled and self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Visualization samples: {eval_data_path or config.data_path}",
                flush=True,
            )

    def should_log_denoising(self, step: int) -> bool:
        return (
            self.enabled
            and self.denoising_interval > 0
            and step % self.denoising_interval == 0
        )

    def should_log_rollout(self, step: int) -> bool:
        return (
            self.enabled
            and self.rollout_interval > 0
            and step % self.rollout_interval == 0
        )

    def _fixed_batch(self) -> dict[str, Any]:
        index = self.sample_index % len(self.dataset)
        random_state = random.getstate()
        random.seed(self.seed)
        try:
            sample = self.dataset[index]
        finally:
            random.setstate(random_state)
        return default_collate([sample])

    def _clean_latent_and_condition(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        clean_latent = clean_latent[:, :min(self.num_frames, clean_latent.shape[1])]
        ui_batch = {
            key: batch[key]
            for key in ("actions", "node_tokens", "node_mask", "node_positions")
            if key in batch
        }
        return clean_latent, ui_batch

    def _generator(self) -> torch.Generator:
        generator = torch.Generator(device=f"cuda:{self.device}")
        generator.manual_seed(self.seed)
        return generator

    def _new_pipeline(self) -> CausalDiffusionInferencePipeline:
        return CausalDiffusionInferencePipeline(
            self.config,
            device=self.device,
            generator=self.model.generator,
            text_encoder=self.model.text_encoder,
            vae=self.model.vae,
        )

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "vae_decode_mode", "video") == "single_frame":
            video = self.model.vae.decode_framewise_to_pixel(latents)
        else:
            video = self.model.vae.decode_to_pixel(latents)
        return (video * 0.5 + 0.5).clamp(0, 1)

    def _decode_pair(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.vae.to(device=self.device, dtype=self.dtype)
        try:
            target_video = self._decode(target).cpu()
            self.model.vae.model.clear_cache()
            prediction_video = self._decode(prediction).cpu()
            self.model.vae.model.clear_cache()
            return target_video, prediction_video
        finally:
            self.model.vae.to(device="cpu")
            torch.cuda.empty_cache()

    def _write_comparison(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        name: str,
    ) -> Path:
        num_frames = min(target.shape[1], prediction.shape[1])
        comparison = torch.cat(
            [target[0, :num_frames], prediction[0, :num_frames]],
            dim=-1,
        )
        pixels = (
            comparison.permute(0, 2, 3, 1).float().cpu() * 255.0
        ).round().clamp(0, 255).to(torch.uint8)
        output_path = self.output_dir / name
        write_video(str(output_path), pixels, fps=self.fps)
        return output_path

    def _graph_cross_attn_modules(self) -> list[tuple[str, Any]]:
        modules: list[tuple[str, Any]] = []
        for name, module in self.model.generator.named_modules():
            if "condition_cross_attn" not in name:
                continue
            if "action_condition_cross_attn" in name:
                continue
            if module.__class__.__name__ != "WanCrossAttention":
                continue
            modules.append((name, module))
        return modules

    @staticmethod
    def _clear_cross_attn_module(module: Any) -> None:
        module.store_attn_weights = False
        for attr in (
            "first_attn_slot_map",
            "first_attn_patch_map",
            "last_attn_slot_map",
            "last_attn_patch_map",
            "last_attn_frame_has_condition",
        ):
            if hasattr(module, attr):
                setattr(module, attr, None)

    def _set_graph_cross_attn_capture(self, enabled: bool) -> list[tuple[str, Any]]:
        selected: list[tuple[str, Any]] = []
        max_blocks = self.cross_attn_map_max_blocks
        for name, module in self._graph_cross_attn_modules():
            self._clear_cross_attn_module(module)
            capture = enabled and (max_blocks <= 0 or len(selected) < max_blocks)
            module.store_attn_weights = capture
            if capture:
                selected.append((name, module))
        return selected

    def _collect_graph_cross_attn_maps(
        self,
        modules: list[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        captured: list[dict[str, Any]] = []
        for block_index, (name, module) in enumerate(modules):
            slot_map = getattr(module, "last_attn_slot_map", None)
            patch_map = getattr(module, "last_attn_patch_map", None)
            if slot_map is None or patch_map is None:
                continue
            captured.append(
                {
                    "block_index": block_index,
                    "name": name,
                    "slot_map": slot_map,
                    "patch_map": patch_map,
                    "num_frames": int(getattr(module, "last_attn_num_frames", 0)),
                    "tokens_per_frame": int(
                        getattr(module, "last_attn_tokens_per_frame", 0)
                    ),
                    "frame_seqlen": int(getattr(module, "last_attn_frame_seqlen", 0)),
                    "grid_hw": getattr(module, "last_attn_grid_hw", None),
                }
            )
        return captured

    @staticmethod
    def _heat_to_uint8(heat: torch.Tensor) -> torch.Tensor:
        heat = torch.nan_to_num(heat.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        if heat.numel() == 0:
            return torch.zeros_like(heat, dtype=torch.uint8)
        max_value = heat.max()
        max_scalar = float(max_value.item())
        if not torch.isfinite(max_value).item() or max_scalar <= 0.0:
            return torch.zeros_like(heat, dtype=torch.uint8)
        return (heat / max_value * 255.0).round().clamp(0, 255).to(torch.uint8)

    def _write_heat_png(self, heat: torch.Tensor, name: str) -> Path:
        image = self._heat_to_uint8(heat).cpu()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[-1] in {1, 3, 4}:
            image = image.permute(2, 0, 1)
        output_path = self.output_dir / name
        write_png(image.contiguous(), str(output_path))
        return output_path

    @staticmethod
    def _row_normalized_heat(heat: torch.Tensor) -> torch.Tensor:
        heat = torch.nan_to_num(heat.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        if heat.ndim != 2 or heat.numel() == 0:
            return heat
        row_max = heat.max(dim=-1, keepdim=True).values
        row_max = torch.where(row_max > 0.0, row_max, torch.ones_like(row_max))
        return heat / row_max

    def _selected_cross_attn_frames(self, slot_map: torch.Tensor) -> list[int]:
        limit = max(0, self.cross_attn_map_frame_limit)
        if limit <= 0:
            return []
        valid_frames = [
            frame_idx
            for frame_idx in range(slot_map.shape[0])
            if float(slot_map[frame_idx].sum()) > 0.0
        ]
        if len(valid_frames) <= limit:
            return valid_frames
        if limit == 1:
            return [valid_frames[-1]]
        return [
            valid_frames[round(i * (len(valid_frames) - 1) / (limit - 1))]
            for i in range(limit)
        ]

    def _top_cross_attn_slots(self, slot_row: torch.Tensor) -> list[int]:
        topk = min(max(0, self.cross_attn_map_topk), int(slot_row.numel()))
        if topk <= 0 or float(slot_row.sum()) <= 0.0:
            return []
        return torch.topk(slot_row.float(), k=topk).indices.tolist()

    def _cross_attn_patch_montage(
        self,
        patch_frame: torch.Tensor,
        slots: list[int],
        grid_hw: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if grid_hw is None or not slots:
            return None
        grid_h, grid_w = grid_hw
        if patch_frame.shape[0] != grid_h * grid_w:
            return None
        heatmaps = [
            patch_frame[:, slot].reshape(grid_h, grid_w).detach().float()
            for slot in slots
        ]
        return torch.cat(heatmaps, dim=1)

    def _graph_cross_attn_top_token_table(
        self,
        slot_map: torch.Tensor,
    ) -> wandb.Table:
        table = wandb.Table(columns=["frame", "rank", "slot", "weight"])
        topk = min(max(0, self.cross_attn_map_topk), int(slot_map.shape[-1]))
        if topk <= 0:
            return table
        for frame_idx in range(slot_map.shape[0]):
            row = slot_map[frame_idx].float()
            if float(row.sum()) <= 0.0:
                continue
            values, slots = torch.topk(row, k=topk)
            for rank, (slot, value) in enumerate(zip(slots, values), start=1):
                table.add_data(frame_idx, rank, int(slot), float(value))
        return table

    def _log_graph_cross_attn_maps(
        self,
        captured: list[dict[str, Any]],
        step: int,
    ) -> None:
        if not captured:
            return
        wandb_payload: dict[str, Any] = {}
        for item in captured:
            block_index = int(item["block_index"])
            num_frames = max(0, int(item["num_frames"]))
            slot_map = item["slot_map"]
            patch_map = item["patch_map"]
            if num_frames > 0:
                slot_map = slot_map[:num_frames]
                patch_map = patch_map[:num_frames]

            slot_path = self._write_heat_png(
                slot_map,
                f"cross_attn_block_{block_index:02d}_slots_step_{step:06d}.png",
            )
            slot_row_norm_path = self._write_heat_png(
                self._row_normalized_heat(slot_map),
                f"cross_attn_block_{block_index:02d}_slots_rowmax_step_{step:06d}.png",
            )
            slot_row_sum_path = self._write_heat_png(
                slot_map.sum(dim=-1, keepdim=True),
                f"cross_attn_block_{block_index:02d}_slots_rowsum_step_{step:06d}.png",
            )
            print(
                f"[visualization] wrote graph cross-attention slot map to {slot_path}",
                flush=True,
            )
            if not self.disable_wandb:
                wandb_payload[f"preview/graph_cross_attn_block{block_index}_slots"] = (
                    wandb.Image(
                        str(slot_path),
                        caption=(
                            f"Graph CA block {block_index}: rows=frames, "
                            "cols=graph-token slots."
                        ),
                    )
                )
                wandb_payload[
                    f"preview/graph_cross_attn_block{block_index}_slots_rowmax"
                ] = wandb.Image(
                    str(slot_row_norm_path),
                    caption=(
                        f"Graph CA block {block_index}: rows=frames, "
                        "cols=graph-token slots, each row normalized by its own max."
                    ),
                )
                wandb_payload[
                    f"preview/graph_cross_attn_block{block_index}_slots_rowsum"
                ] = wandb.Image(
                    str(slot_row_sum_path),
                    caption=(
                        f"Graph CA block {block_index}: per-frame slot attention sum. "
                        "Valid graph-conditioned rows should be bright."
                    ),
                )
                wandb_payload[
                    f"preview/graph_cross_attn_block{block_index}_top_tokens"
                ] = self._graph_cross_attn_top_token_table(slot_map)

            grid_hw = item["grid_hw"]
            for frame_idx in self._selected_cross_attn_frames(slot_map):
                slots = self._top_cross_attn_slots(slot_map[frame_idx])
                montage = self._cross_attn_patch_montage(
                    patch_map[frame_idx],
                    slots,
                    grid_hw,
                )
                if montage is None:
                    continue
                patch_path = self._write_heat_png(
                    montage,
                    (
                        f"cross_attn_block_{block_index:02d}_frame_{frame_idx:02d}"
                        f"_patches_step_{step:06d}.png"
                    ),
                )
                print(
                    f"[visualization] wrote graph cross-attention patch map to {patch_path} "
                    f"(frame {frame_idx}, slots {slots})",
                    flush=True,
                )
                if not self.disable_wandb:
                    wandb_payload[
                        (
                            f"preview/graph_cross_attn_block{block_index}"
                            f"_frame{frame_idx}_patches"
                        )
                    ] = wandb.Image(
                        str(patch_path),
                        caption=(
                            f"Graph CA block {block_index}, frame {frame_idx}: "
                            f"patch heatmaps for token slots {slots}."
                        ),
                    )

        if wandb_payload and not self.disable_wandb:
            wandb.log(wandb_payload, step=step)

    def log_denoising(self, step: int) -> None:
        batch = self._fixed_batch()
        clean_latent, ui_batch = self._clean_latent_and_condition(batch)
        noise = torch.randn(
            clean_latent.shape,
            generator=self._generator(),
            device=self.device,
            dtype=self.dtype,
        )
        timestep_index = max(
            0,
            min(self.timestep_index, len(self.model.scheduler.timesteps) - 1),
        )
        timestep_value = self.model.scheduler.timesteps[timestep_index].to(
            device=self.device,
            dtype=self.dtype,
        )
        timestep = timestep_value.expand(clean_latent.shape[:2]).clone()
        noisy_latent = self.model.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, clean_latent.shape[:2])
        noisy_latent[:, :1] = clean_latent[:, :1]
        timestep[:, :1] = 0

        was_training = self.model.generator.training
        self.model.generator.eval()
        capture_attn = self.cross_attn_maps and "node_tokens" in ui_batch
        captured_attn: list[dict[str, Any]] = []
        captured_modules: list[tuple[str, Any]] = []
        try:
            if capture_attn:
                captured_modules = self._set_graph_cross_attn_capture(True)
            with torch.inference_mode():
                conditional_dict = self.model.text_encoder(
                    text_prompts=list(batch["prompts"])
                )
                conditional_dict, _ = attach_ui_batch_conditioning(
                    ui_batch,
                    conditional_dict,
                    conditional_dict,
                    device=self.device,
                    dtype=self.dtype,
                    num_latent_frames=clean_latent.shape[1],
                    i2v=True,
                )
                _, x0_prediction = self.model.generator(
                    noisy_image_or_video=noisy_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    clean_x=clean_latent if self.model.teacher_forcing else None,
                    aug_t=None,
                )
                if capture_attn:
                    captured_attn = self._collect_graph_cross_attn_maps(
                        captured_modules,
                    )
        finally:
            if capture_attn:
                self._set_graph_cross_attn_capture(False)
            self.model.generator.train(was_training)

        if not self.is_main_process:
            return
        target, prediction = self._decode_pair(clean_latent, x0_prediction)
        output_path = self._write_comparison(
            target,
            prediction,
            f"denoising_step_{step:06d}.mp4",
        )
        print(
            f"[visualization] wrote fixed one-step denoising preview to {output_path} "
            f"(timestep index {timestep_index})",
            flush=True,
        )
        if not self.disable_wandb:
            wandb.log(
                {
                    "preview/denoising": wandb.Video(
                        str(output_path),
                        caption=(
                            "Left: ground truth. Right: one-step x0 prediction. "
                            f"Fixed timestep index: {timestep_index}."
                        ),
                        fps=self.fps,
                        format="mp4",
                    ),
                },
                step=step,
            )
        self._log_graph_cross_attn_maps(captured_attn, step)

    def log_rollout(self, step: int) -> None:
        if not self.enabled or self.rollout_interval <= 0:
            return
        batch = self._fixed_batch()
        clean_latent, ui_batch = self._clean_latent_and_condition(batch)
        noise = torch.randn(
            (
                clean_latent.shape[0],
                clean_latent.shape[1] - 1,
                clean_latent.shape[2],
                clean_latent.shape[3],
                clean_latent.shape[4],
            ),
            generator=self._generator(),
            device=self.device,
            dtype=self.dtype,
        )

        generator_was_training = self.model.generator.training
        text_encoder_was_training = self.model.text_encoder.training
        pipeline = self._new_pipeline()
        pipeline.eval()
        try:
            with torch.inference_mode():
                generated_latent = pipeline.inference(
                    noise=noise,
                    text_prompts=list(batch["prompts"]),
                    initial_latent=clean_latent[:, :1],
                    ui_batch=ui_batch,
                    return_video=False,
                )
        finally:
            self.model.generator.train(generator_was_training)
            self.model.text_encoder.train(text_encoder_was_training)
            del pipeline
            torch.cuda.empty_cache()

        if not self.is_main_process:
            return
        target, prediction = self._decode_pair(clean_latent, generated_latent)
        output_path = self._write_comparison(
            target,
            prediction,
            f"rollout_step_{step:06d}.mp4",
        )
        latent_mse = torch.mean(
            (generated_latent.float() - clean_latent.float()) ** 2
        ).item()
        print(
            f"[visualization] wrote fixed-seed causal rollout to {output_path} "
            f"(latent MSE {latent_mse:.6f})",
            flush=True,
        )
        if not self.disable_wandb:
            wandb.log(
                {
                    "preview/rollout": wandb.Video(
                        str(output_path),
                        caption=(
                            "Left: ground truth. Right: causal rollout from the "
                            "same initial screen and action sequence."
                        ),
                        fps=self.fps,
                        format="mp4",
                    ),
                    "preview/rollout_latent_mse": latent_mse,
                },
                step=step,
            )
