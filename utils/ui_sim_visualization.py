from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data._utils.collate import default_collate
from torchvision.io import write_video

from pipeline import CausalDiffusionInferencePipeline
from utils.ui_sim_conditioning import attach_ui_batch_conditioning
from utils.ui_sim_dataset import UISimLatentDataset


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

        self.pipeline = None
        if self.enabled and self.rollout_interval > 0:
            self.pipeline = CausalDiffusionInferencePipeline(
                config,
                device=device,
                generator=model.generator,
                text_encoder=model.text_encoder,
                vae=model.vae,
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
            for key in ("actions", "node_tokens", "node_mask")
            if key in batch
        }
        return clean_latent, ui_batch

    def _generator(self) -> torch.Generator:
        generator = torch.Generator(device=f"cuda:{self.device}")
        generator.manual_seed(self.seed)
        return generator

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "vae_decode_mode", "video") == "single_frame":
            video = self.model.vae.decode_framewise_to_pixel(latents)
        else:
            video = self.model.vae.decode_to_pixel(latents)
        return (video * 0.5 + 0.5).clamp(0, 1)

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
        try:
            with torch.no_grad():
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
        finally:
            self.model.generator.train(was_training)

        if not self.is_main_process:
            return
        target = self._decode(clean_latent)
        prediction = self._decode(x0_prediction)
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
        self.model.vae.model.clear_cache()

    def log_rollout(self, step: int) -> None:
        if self.pipeline is None:
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

        was_training = self.model.generator.training
        self.pipeline.eval()
        try:
            with torch.no_grad():
                generated_latent = self.pipeline.inference(
                    noise=noise,
                    text_prompts=list(batch["prompts"]),
                    initial_latent=clean_latent[:, :1],
                    ui_batch=ui_batch,
                    return_video=False,
                )
        finally:
            self.model.generator.train(was_training)

        if not self.is_main_process:
            return
        target = self._decode(clean_latent)
        prediction = self._decode(generated_latent)
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
        self.model.vae.model.clear_cache()
