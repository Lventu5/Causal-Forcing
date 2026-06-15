import gc
import logging
from utils.dataset import cycle
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
from utils.training_checkpoint import (
    atomic_torch_save,
    checkpoint_metadata,
    extract_generator_state,
    load_checkpoint,
    load_trainer_payload,
)
import torch.distributed as dist
import torch
import wandb
import time
from pathlib import Path
from model import NaiveConsistency
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
        self.model = NaiveConsistency(config, device=self.device)
        self.element_loss_weighter = build_element_loss_weighter(
            config,
            is_main_process=self.is_main_process,
        )

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )
        
        self.model.generator_ema = fsdp_wrap(
            self.model.generator_ema,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.teacher = fsdp_wrap(
            self.model.teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True
        )

        
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
 
        dataset = build_training_dataset(config)
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=int(getattr(config, "num_workers", 8)))

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
        # 7. Load the previous-stage initializer or resume this causal-CD stage.
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
                self.initialization_checkpoint = (
                    checkpoint.initialization_checkpoint
                    or getattr(config, "stage_initialization_ckpt", None)
                )
                trainer_payload = load_trainer_payload(checkpoint)
            else:
                self.initialization_checkpoint = str(checkpoint.model_path)

        teacher_state = generator_state
        if checkpoint is not None and checkpoint.is_resume:
            if self.initialization_checkpoint:
                initializer = load_checkpoint(
                    self.initialization_checkpoint,
                    current_stage="",
                    checkpoint_mode="initialize",
                )
                teacher_state = extract_generator_state(
                    initializer.payload,
                    for_resume=False,
                )
            elif self.is_main_process:
                print(
                    "WARNING: resumed causal-CD checkpoint has no recorded "
                    "previous-stage initializer; using its current generator as teacher. "
                    "Set stage_initialization_ckpt for legacy checkpoints."
                )
        if teacher_state is not None:
            teacher_result = self.model.teacher.load_state_dict(
                teacher_state,
                strict=strict,
            )
            if self.is_main_process and not strict:
                print(f"Teacher load result: {teacher_result}")

        ema_state = (
            extract_generator_state(checkpoint.payload, for_resume=False)
            if checkpoint is not None
            and checkpoint.is_resume
            and "generator_ema" in checkpoint.payload
            else generator_state
        )
        if ema_state is not None:
            self.model.generator_ema.load_state_dict(ema_state, strict=strict)
            self.model.generator.load_state_dict(ema_state, strict=strict)

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
        if generator_state is not None:
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

        #############################################################################################################
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        
        

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
            checkpoint_dir = (
                Path(self.output_path) / f"checkpoint_model_{self.step:06d}"
            )
            model_path = checkpoint_dir / "model.pt"
            trainer_path = checkpoint_dir / "trainer.pt"
            atomic_torch_save(state_dict, model_path)
            atomic_torch_save(trainer_state_dict, trainer_path)
            print("Model saved to", model_path)

            
    def fwdbwd_one_step(self, batch, clean_latent=None):
        self.model.eval()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
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

        # Step 3: Store gradients for the generator (if training the generator)
        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model=self.generator_ema,
            loss_weight=loss_weight,
        )
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator)

        generator_log_dict.update({"generator_loss": generator_loss,
                                    "generator_grad_norm": generator_grad_norm})

        return generator_log_dict
        

   

    def train(self):
        start_step = self.step

        while True:

            self.generator_optimizer.zero_grad(set_to_none=True)

            batch = next(self.dataloader)
            generator_log_dict = self.fwdbwd_one_step(batch, clean_latent=batch["clean_latent"])
            

            self.generator_optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
            
              

            # Increment the step since we finished gradient update
            self.step += 1

           
            # Save the model
            save_iters = getattr(self.config, "save_iters", self.config.log_iters)
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % save_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item()
                        }
                    )
                if "element_weight_mean" in generator_log_dict:
                    wandb_loss_dict["element_weight_mean"] = (
                        generator_log_dict["element_weight_mean"].mean().item()
                    )

              

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
