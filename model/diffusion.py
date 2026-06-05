from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def _cfg_get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        if getattr(self.generator, "train_ui_conditioner_only", False):
            self.generator.enable_trainable_ui_conditioning_only()
        else:
            self.generator.model.requires_grad_(True)

        model_kwargs = getattr(args, "model_kwargs", {})
        self.text_encoder = WanTextEncoder(
            model_name=_cfg_get(model_kwargs, "model_name", "Wan2.1-T2V-1.3B"),
            model_root=_cfg_get(model_kwargs, "model_root", None),
        )
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(
            model_name=_cfg_get(model_kwargs, "model_name", "Wan2.1-T2V-1.3B"),
            model_root=_cfg_get(model_kwargs, "model_root", None),
        )
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        loss_weight: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)
        loss_frame_mask = torch.ones(
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.float32,
        )
        if getattr(self.args, "i2v", False) and initial_latent is not None:
            noisy_latents[:, :initial_latent.shape[1]] = initial_latent.to(
                device=self.device,
                dtype=self.dtype,
            )
            timestep[:, :initial_latent.shape[1]] = 0
            training_target[:, :initial_latent.shape[1]] = 0
            loss_frame_mask[:, :initial_latent.shape[1]] = 0

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                self.noise_augmentation_max_timestep,
                1000,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None
        # Compute loss
        
        
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None
        )
        loss_map = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        )
        if loss_weight is not None:
            loss_map = loss_map * loss_weight.to(
                device=loss_map.device,
                dtype=loss_map.dtype,
            )
        per_frame_loss = loss_map.mean(dim=(2, 3, 4))
        loss = per_frame_loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss * loss_frame_mask
        loss = loss.sum() / loss_frame_mask.sum().clamp_min(1.0)

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        if loss_weight is not None:
            log_dict["element_weight_mean"] = loss_weight.detach().float().mean()
        return loss, log_dict
