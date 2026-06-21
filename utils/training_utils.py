from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


class CachedTextEncoder(torch.nn.Module):
    """Serve precomputed embeddings for a fixed set of training prompts."""

    def __init__(self, embeddings: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        if not embeddings:
            raise ValueError("CachedTextEncoder requires at least one prompt.")

        self._prompt_to_buffer: dict[str, str] = {}
        for index, (prompt, embedding) in enumerate(embeddings.items()):
            if embedding.ndim < 2 or embedding.shape[0] != 1:
                raise ValueError(
                    "Each cached prompt embedding must have batch size 1, got "
                    f"{tuple(embedding.shape)} for {prompt!r}."
                )
            buffer_name = f"_prompt_embedding_{index}"
            self.register_buffer(
                buffer_name,
                embedding.detach().clone(),
                persistent=False,
            )
            self._prompt_to_buffer[prompt] = buffer_name

    @classmethod
    @torch.no_grad()
    def from_encoder(
        cls,
        text_encoder: torch.nn.Module,
        prompts: Iterable[str],
    ) -> "CachedTextEncoder":
        unique_prompts = list(dict.fromkeys(str(prompt) for prompt in prompts))
        output = text_encoder(text_prompts=unique_prompts)
        prompt_embeds = output["prompt_embeds"].detach()
        if prompt_embeds.shape[0] != len(unique_prompts):
            raise ValueError(
                "Text encoder returned the wrong batch size while caching: "
                f"{prompt_embeds.shape[0]} != {len(unique_prompts)}."
            )
        return cls(
            {
                prompt: prompt_embeds[index:index + 1]
                for index, prompt in enumerate(unique_prompts)
            }
        )

    def forward(self, text_prompts: Iterable[str]) -> dict[str, torch.Tensor]:
        prompts = [str(prompt) for prompt in text_prompts]
        if not prompts:
            raise ValueError("text_prompts must not be empty.")

        missing = sorted(set(prompts) - self._prompt_to_buffer.keys())
        if missing:
            raise ValueError(
                "Encountered prompts outside the fixed UI prompt cache: "
                f"{missing}. Disable cache_text_embeddings for variable prompts."
            )

        if len(set(prompts)) == 1:
            embedding = getattr(
                self,
                self._prompt_to_buffer[prompts[0]],
            )
            prompt_embeds = embedding.expand(
                len(prompts),
                *embedding.shape[1:],
            )
        else:
            prompt_embeds = torch.cat(
                [
                    getattr(self, self._prompt_to_buffer[prompt])
                    for prompt in prompts
                ],
                dim=0,
            )
        return {"prompt_embeds": prompt_embeds}


def maybe_cache_text_encoder(
    text_encoder: torch.nn.Module,
    config: Any,
) -> torch.nn.Module:
    if not bool(getattr(config, "cache_text_embeddings", False)):
        return text_encoder

    prompt_mode = str(getattr(config, "ui_prompt_mode", "fixed")).strip().lower()
    if prompt_mode != "fixed":
        raise ValueError(
            "cache_text_embeddings=true only supports ui_prompt_mode='fixed'. "
            "Use cache_text_embeddings=false for graph-derived block prompts."
        )

    positive_prompt = str(getattr(config, "ui_prompt"))
    negative_prompt = str(getattr(config, "negative_prompt"))
    return CachedTextEncoder.from_encoder(
        text_encoder,
        [positive_prompt, negative_prompt],
    )


def training_dataloader_kwargs(config: Any) -> dict[str, Any]:
    num_workers = int(getattr(config, "num_workers", 8))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(config, "pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            getattr(config, "persistent_workers", True)
        )
        prefetch_factor = int(getattr(config, "prefetch_factor", 2))
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def should_run_interval(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _normalize_parameter_name(name: str) -> str:
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module,
    source_model: torch.nn.Module,
    *,
    decay: float,
) -> None:
    """Update matching local parameter shards without materializing full weights."""

    ema_parameters = list(ema_model.named_parameters())
    source_parameters = list(source_model.named_parameters())
    if len(ema_parameters) != len(source_parameters):
        raise ValueError(
            "EMA/source parameter count mismatch: "
            f"{len(ema_parameters)} != {len(source_parameters)}."
        )

    for (ema_name, ema_param), (source_name, source_param) in zip(
        ema_parameters,
        source_parameters,
    ):
        if _normalize_parameter_name(ema_name) != _normalize_parameter_name(
            source_name
        ):
            raise ValueError(
                f"EMA/source parameter mismatch: {ema_name!r} != {source_name!r}."
            )
        if ema_param.shape != source_param.shape:
            raise ValueError(
                f"EMA/source shape mismatch for {ema_name}: "
                f"{tuple(ema_param.shape)} != {tuple(source_param.shape)}."
            )
        ema_param.mul_(decay).add_(
            source_param.detach().to(
                device=ema_param.device,
                dtype=ema_param.dtype,
            ),
            alpha=1.0 - decay,
        )
