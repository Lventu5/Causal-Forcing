import gc
import logging
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.ui_sim_conditioning import attach_ui_batch_conditioning, ui_conditioning_dropout_kwargs
from utils.ui_sim_dataset import build_training_dataset
from utils.ui_sim_element_loss import (
    build_element_loss_weighter,
    build_element_loss_weight_map,
)
from utils.distributed import (
    EMA_FSDP,
    fsdp_optim_state_dict,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
    load_fsdp_optim_state_dict,
)
from utils.misc import set_seed
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
import torch.distributed as dist
from model import DMD
import torch
import wandb
import time
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
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.log_interval = int(getattr(config, "log_iters", 1))

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
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")
        self.element_loss_weighter = build_element_loss_weighter(
            config,
            is_main_process=self.is_main_process,
        )

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
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

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if str(getattr(config, "dataset_type", "text")) == "ui_sim_latent":
            dataset = build_training_dataset(config)
        else:
            dataset = TextDataset(config.data_path)
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
        # 7. Load the previous-stage initializer or resume this DMD stage.
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

        if (
            checkpoint is not None
            and checkpoint.is_resume
            and trainer_payload is not None
            and "critic" in trainer_payload
        ):
            self.model.fake_score.load_state_dict(
                trainer_payload["critic"],
                strict=True,
            )

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
                else generator_state
            )
            if ema_state is not None:
                self.model.generator.load_state_dict(ema_state, strict=strict)
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(
                self.model.generator,
                decay=ema_weight,
                device=getattr(config, "ema_device", "cpu"),
            )
            if generator_state is not None:
                self.model.generator.load_state_dict(generator_state, strict=strict)

        if checkpoint is not None and checkpoint.is_resume:
            if trainer_payload is not None and {
                "generator_optimizer",
                "critic_optimizer",
                "critic",
            }.issubset(trainer_payload):
                load_fsdp_optim_state_dict(
                    self.model.generator,
                    self.generator_optimizer,
                    trainer_payload["generator_optimizer"],
                )
                load_fsdp_optim_state_dict(
                    self.model.fake_score,
                    self.critic_optimizer,
                    trainer_payload["critic_optimizer"],
                )
                if self.is_main_process:
                    print(f"Resumed optimizers, critic, and global step {self.step}")
            elif self.is_main_process:
                print(
                    "WARNING: legacy checkpoint has no complete trainer.pt; resumed "
                    f"generator weights and global step {self.step}, but reinitialized "
                    "DMD critic/optimizer state."
                )

        ##############################################################################################################

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_log_time = time.perf_counter()
        self.checkpoint_sync = CheckpointSyncManager(
            config,
            output_path=self.output_path,
            is_main_process=self.is_main_process,
        )

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)
        generator_optimizer_state_dict = fsdp_optim_state_dict(
            self.model.generator,
            self.generator_optimizer,
        )
        critic_optimizer_state_dict = fsdp_optim_state_dict(
            self.model.fake_score,
            self.critic_optimizer,
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
        trainer_state_dict.update(
            {
                "generator_optimizer": generator_optimizer_state_dict,
                "critic": critic_state_dict,
                "critic_optimizer": critic_optimizer_state_dict,
            }
        )

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

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

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
            atomic_torch_save(state_dict, model_path)
            print("Model saved to", model_path)
            self.checkpoint_sync.sync(
                checkpoint_dir,
                step=self.step,
                stage=self.training_stage,
            )
            
    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if clean_latent is None and "clean_latent" in batch:
            clean_latent = batch["clean_latent"].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
        if self.config.i2v:
            if "ode_latent" in batch:
                image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                    device=self.device, dtype=self.dtype)
            elif clean_latent is not None:
                image_latent = clean_latent[:, 0:1, ].to(
                    device=self.device, dtype=self.dtype)
            else:
                raise KeyError("I2V distillation requires ode_latent or clean_latent in the batch.")
        else:
            # clean_latent = None #original code here
            image_latent = None

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
        if clean_latent is not None:
            num_latent_frames = clean_latent.shape[1]
        elif self.config.i2v and image_latent is not None:
            num_latent_frames = int(self.config.image_or_video_shape[1])
        else:
            num_latent_frames = int(self.config.image_or_video_shape[1])
        conditional_dict, unconditional_dict = attach_ui_batch_conditioning(
            batch,
            conditional_dict,
            unconditional_dict,
            device=self.device,
            dtype=self.dtype,
            num_latent_frames=num_latent_frames,
            i2v=bool(getattr(self.config, "i2v", False)),
            **ui_conditioning_dropout_kwargs(self.config),
        )
        loss_weight = (
            build_element_loss_weight_map(
                self.element_loss_weighter,
                batch,
                clean_latent,
                device=self.device,
            )
            if clean_latent is not None
            else None
        )

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                loss_weight=loss_weight,
            )
           
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step
       
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                
                
                

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)
                
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(
                    self.model.generator,
                    decay=self.config.ema_weight,
                    device=getattr(self.config, "ema_device", "cpu"),
                )

            # Save the model
            save_iters = getattr(self.config, "save_iters", self.config.log_iters)
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % save_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process and should_run_interval(
                self.step,
                self.log_interval,
            ):
                current_time = time.perf_counter()
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    if "element_weight_mean" in generator_log_dict:
                        wandb_loss_dict["element_weight_mean"] = (
                            generator_log_dict["element_weight_mean"].mean().item()
                        )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item(),
                        "per iteration time": (
                            current_time - self.previous_log_time
                        ) / self.log_interval,
                    }
                )
                self.previous_log_time = current_time

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if should_run_interval(
                self.step,
                int(getattr(self.config, "gc_interval", 0)),
            ):
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
