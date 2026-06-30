import gc
import logging

from model import CausalDiffusion
from utils.dataset import cycle
from utils.misc import set_seed
from utils.ui_sim_conditioning import attach_ui_batch_conditioning, ui_conditioning_dropout_kwargs
from utils.ui_sim_dataset import build_training_dataset
from utils.ui_sim_element_loss import (
    build_element_loss_weighter,
    build_element_loss_weight_map,
)
from utils.ui_sim_visualization import UISimTrainingVisualizer
import torch.distributed as dist
import torch
import wandb
import time
from utils.distributed import (
    EMA_FSDP,
    barrier,
    fsdp_optim_state_dict,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
    load_fsdp_optim_state_dict,
)
from utils.checkpoint_sync import CheckpointSyncManager
from utils.training_checkpoint import (
    atomic_torch_save,
    checkpoint_dir_for_step,
    checkpoint_metadata,
    DEFAULT_CHECKPOINT_DIR_NAME,
    extract_generator_state,
    load_checkpoint,
    load_trainer_payload,
)
from utils.training_utils import (
    maybe_cache_text_encoder,
    should_run_interval,
    training_dataloader_kwargs,
)
from utils.wandb_logging import init_wandb


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.memory_diagnostics = bool(
            getattr(config, "memory_diagnostics", False)
        )
        self.memory_log_interval = int(
            getattr(config, "memory_log_interval", 100)
        )
        self.log_interval = int(getattr(config, "log_iters", 1))
        self.max_train_steps = int(getattr(config, "max_train_steps", 0) or 0)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            init_wandb(config)

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.element_loss_weighter = build_element_loss_weighter(
            config,
            is_main_process=self.is_main_process,
        )
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self._log_cuda_memory("startup/after_generator_fsdp")
        torch.cuda.empty_cache()

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )
        self._log_cuda_memory("startup/after_text_encoder_fsdp")
        original_text_encoder = self.model.text_encoder
        self.model.text_encoder = maybe_cache_text_encoder(
            original_text_encoder,
            config,
        )
        if self.model.text_encoder is not original_text_encoder:
            del original_text_encoder
            gc.collect()
            torch.cuda.empty_cache()
            if self.is_main_process:
                print("Cached fixed UI text embeddings and released UMT5.")
        torch.cuda.empty_cache()

        if config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self._log_trainable_generator_parameters()

        # Step 3: Initialize the dataloader
        dataset = build_training_dataset(config)
       
        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            **training_dataloader_kwargs(config),
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ##############################################################################################################
        # 7. Load an initializer or resume the current stage.
        self.training_stage = str(getattr(config, "training_stage", ""))
        self.initialization_checkpoint = None
        checkpoint = None
        trainer_payload = None
        generator_state = None
        strict = not bool(getattr(config, "allow_partial_generator_load", False))
        if getattr(config, "generator_ckpt", False):
            checkpoint = load_checkpoint(
                config.generator_ckpt,
                current_stage=self.training_stage,
                checkpoint_mode=str(getattr(config, "checkpoint_mode", "auto")),
            )
            print(
                f"Loading generator from {checkpoint.model_path} "
                f"with checkpoint mode {checkpoint.mode}"
            )
            generator_state = extract_generator_state(
                checkpoint.payload,
                for_resume=checkpoint.is_resume,
            )
            load_result = self.model.generator.load_state_dict(
                generator_state,
                strict=strict,
            )
            if self.is_main_process and not strict:
                print(f"Generator load result: {load_result}")
            if checkpoint.is_resume:
                self.step = checkpoint.step
                self.initialization_checkpoint = checkpoint.initialization_checkpoint
                trainer_payload = load_trainer_payload(checkpoint)
            else:
                self.initialization_checkpoint = str(checkpoint.model_path)

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (
            (ema_weight is not None)
            and (ema_weight > 0.0)
            and self.step >= config.ema_start_step
        ):
            ema_state = (
                extract_generator_state(checkpoint.payload, for_resume=False)
                if checkpoint is not None
                and checkpoint.is_resume
                and "generator_ema" in checkpoint.payload
                else None
            )
            if ema_state is not None and generator_state is not None:
                self.model.generator.load_state_dict(ema_state, strict=strict)
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
            if ema_state is not None and generator_state is not None:
                self.model.generator.load_state_dict(generator_state, strict=strict)

        if checkpoint is not None and checkpoint.is_resume:
            if trainer_payload is not None and "generator_optimizer" in trainer_payload:
                load_fsdp_optim_state_dict(
                    self.model.generator,
                    self.generator_optimizer,
                    trainer_payload["generator_optimizer"],
                )
                if self.is_main_process:
                    print(f"Resumed optimizer and global step {self.step}")
            elif self.is_main_process:
                print(
                    "WARNING: legacy checkpoint has no trainer.pt; resumed model "
                    f"weights and global step {self.step}, but reinitialized AdamW state."
                )

        self._log_cuda_memory("startup/after_checkpoint_load")
        torch.cuda.empty_cache()
        self._log_cuda_memory("startup/ready")
        torch.cuda.reset_peak_memory_stats(self.device)
        self.run_start_step = self.step

        ##############################################################################################################

        self.max_grad_norm = 10.0
        self.previous_log_time = time.perf_counter()
        self.visualizer = UISimTrainingVisualizer(
            config=config,
            model=self.model,
            training_dataset=self.dataset,
            output_path=self.output_path,
            device=self.device,
            dtype=self.dtype,
            is_main_process=self.is_main_process,
            disable_wandb=self.disable_wandb,
        )
        self.checkpoint_sync = CheckpointSyncManager(
            config,
            output_path=self.output_path,
            is_main_process=self.is_main_process,
        )

    def _log_cuda_memory(
        self,
        label: str,
        *,
        wandb_step: int | None = None,
        reset_peak: bool = False,
    ) -> None:
        if not self.memory_diagnostics:
            return
        gib = 1024 ** 3
        allocated = torch.cuda.memory_allocated(self.device) / gib
        reserved = torch.cuda.memory_reserved(self.device) / gib
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / gib
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / gib
        if self.is_main_process:
            print(
                f"[cuda-memory] {label}: allocated={allocated:.2f} GiB, "
                f"reserved={reserved:.2f} GiB, "
                f"peak_allocated={peak_allocated:.2f} GiB, "
                f"peak_reserved={peak_reserved:.2f} GiB",
                flush=True,
            )
            if wandb_step is not None and not self.disable_wandb:
                wandb.log(
                    {
                        "memory/allocated_gib": allocated,
                        "memory/reserved_gib": reserved,
                        "memory/peak_allocated_gib": peak_allocated,
                        "memory/peak_reserved_gib": peak_reserved,
                    },
                    step=wandb_step,
                )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)

    @staticmethod
    def _normalize_parameter_name(name: str) -> str:
        return (
            name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )

    @staticmethod
    def _trainable_parameter_category(name: str) -> str:
        if "action_condition_cross_attn" in name:
            return "action_condition_cross_attn"
        if "condition_cross_attn" in name:
            return "graph_condition_cross_attn"
        if "ui_conditioner" in name:
            return "ui_conditioner"
        return "other"

    def _log_trainable_generator_parameters(self) -> None:
        if not self.is_main_process:
            return

        total_params = 0
        trainable_params = 0
        trainable_names = []
        trainable_by_category = {
            "graph_condition_cross_attn": 0,
            "action_condition_cross_attn": 0,
            "ui_conditioner": 0,
            "other": 0,
        }
        for name, param in self.model.generator.named_parameters():
            normalized_name = self._normalize_parameter_name(name)
            param_count = param.numel()
            total_params += param_count
            if not param.requires_grad:
                continue
            trainable_params += param_count
            category = self._trainable_parameter_category(normalized_name)
            trainable_by_category[category] += param_count
            trainable_names.append(normalized_name)

        print(
            "[trainable] generator parameters: "
            f"{trainable_params:,} / {total_params:,} trainable",
            flush=True,
        )
        for category, count in trainable_by_category.items():
            print(f"[trainable]   {category}: {count:,}", flush=True)
        print("[trainable] trainable parameter names:", flush=True)
        for name in trainable_names[:80]:
            print(f"[trainable]   {name}", flush=True)
        if len(trainable_names) > 80:
            print(
                f"[trainable]   ... {len(trainable_names) - 80} more",
                flush=True,
            )
        if trainable_by_category["other"] > 0:
            print(
                "[trainable] WARNING: parameters outside UI/condition CA are trainable.",
                flush=True,
            )
            
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        generator_optimizer_state_dict = fsdp_optim_state_dict(
            self.model.generator,
            self.generator_optimizer,
        )

        state_dict = checkpoint_metadata(
            training_stage=self.training_stage,
            step=self.step,
            initialization_checkpoint=self.initialization_checkpoint,
        )
        state_dict["generator"] = generator_state_dict
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.full_state_dict(
                self.model.generator
            )

        trainer_state_dict = checkpoint_metadata(
            training_stage=self.training_stage,
            step=self.step,
            initialization_checkpoint=self.initialization_checkpoint,
        )
        trainer_state_dict["generator_optimizer"] = generator_optimizer_state_dict

        if self.is_main_process:
            checkpoint_dir = checkpoint_dir_for_step(
                self.output_path,
                self.step,
                checkpoint_dir_name=getattr(
                    self.config,
                    "checkpoint_dir_name",
                    DEFAULT_CHECKPOINT_DIR_NAME,
                ),
            )
            model_path = checkpoint_dir / "model.pt"
            trainer_path = checkpoint_dir / "trainer.pt"
            atomic_torch_save(state_dict, model_path)
            atomic_torch_save(trainer_state_dict, trainer_path)
            print("Model saved to", model_path)
            self.checkpoint_sync.sync(
                checkpoint_dir,
                step=self.step,
                stage=self.training_stage,
            )

    def train_one_step(self, batch):
        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
           
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts) 
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict
        conditional_dict, unconditional_dict = attach_ui_batch_conditioning(
            batch,
            conditional_dict,
            unconditional_dict,
            device=self.device,
            dtype=self.dtype,
            num_latent_frames=clean_latent.shape[1],
            i2v=bool(getattr(self.config, "i2v", False)),
            **ui_conditioning_dropout_kwargs(self.config),
        )
        loss_weight = build_element_loss_weight_map(
            self.element_loss_weighter,
            batch,
            clean_latent,
            device=self.device,
        )

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            loss_weight=loss_weight,
        )
        self.generator_optimizer.zero_grad(set_to_none=True)
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        # Step 4: Logging
        if self.is_main_process and should_run_interval(
            self.step,
            self.log_interval,
        ):
            current_time = time.perf_counter()
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                "per iteration time": (
                    current_time - self.previous_log_time
                ) / self.log_interval,
            }
            self.previous_log_time = current_time
            if "element_weight_mean" in log_dict:
                wandb_loss_dict["element_weight_mean"] = (
                    log_dict["element_weight_mean"].item()
                )
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if should_run_interval(
            self.step,
            int(getattr(self.config, "gc_interval", 0)),
        ):
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
        if (
            self.step == self.run_start_step + 1
            or (
                self.memory_log_interval > 0
                and self.step % self.memory_log_interval == 0
            )
        ):
            self._log_cuda_memory(
                "training",
                wandb_step=self.step,
                reset_peak=True,
            )

    def _run_visualization(self, name, callback):
        try:
            callback(self.step)
        except Exception:
            logging.exception(
                "Skipping failed %s visualization at step %d on rank %d.",
                name,
                self.step,
                dist.get_rank(),
            )
            gc.collect()
            torch.cuda.empty_cache()
            if bool(getattr(self.config, "visualization_fail_fast", False)):
                raise

    def train(self):

        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            save_iters = getattr(self.config, "save_iters", self.config.log_iters)
            if (not self.config.no_save) and self.step % save_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.visualizer.should_log_denoising(self.step):
                barrier()
                self._run_visualization(
                    "denoising",
                    self.visualizer.log_denoising,
                )
                barrier()
            if self.visualizer.should_log_rollout(self.step):
                barrier()
                self._run_visualization(
                    "rollout",
                    self.visualizer.log_rollout,
                )
                barrier()
            if (
                self.max_train_steps > 0
                and self.step - self.run_start_step >= self.max_train_steps
            ):
                if dist.get_rank() == 0:
                    logging.info(
                        "Reached max_train_steps=%d after %d new steps; stopping.",
                        self.max_train_steps,
                        self.step - self.run_start_step,
                    )
                break
