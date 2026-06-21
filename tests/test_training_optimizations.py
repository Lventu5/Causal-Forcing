from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils.distributed import EMA_FSDP  # noqa: E402
from utils.training_utils import (  # noqa: E402
    CachedTextEncoder,
    maybe_cache_text_encoder,
    training_dataloader_kwargs,
    update_ema_model,
)


class CountingTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, text_prompts):
        self.calls += 1
        values = torch.tensor(
            [
                float(sum(ord(character) for character in prompt))
                for prompt in text_prompts
            ],
            dtype=torch.float32,
        )
        return {
            "prompt_embeds": values[:, None, None].expand(-1, 2, 3).clone()
        }


def test_cached_text_encoder_computes_once_and_reuses_batches() -> None:
    encoder = CountingTextEncoder()
    cached = CachedTextEncoder.from_encoder(
        encoder,
        ["positive prompt", "negative prompt"],
    )

    positive = cached(["positive prompt", "positive prompt"])
    negative = cached(["negative prompt"])

    assert encoder.calls == 1
    assert positive["prompt_embeds"].shape == (2, 2, 3)
    assert torch.equal(
        positive["prompt_embeds"][0],
        positive["prompt_embeds"][1],
    )
    assert not torch.equal(
        positive["prompt_embeds"][0],
        negative["prompt_embeds"][0],
    )


def test_cached_text_encoder_rejects_variable_prompts() -> None:
    cached = CachedTextEncoder({"fixed prompt": torch.ones(1, 2, 3)})

    with pytest.raises(ValueError, match="outside the fixed UI prompt cache"):
        cached(["different prompt"])


def test_fixed_embedding_cache_rejects_graph_prompt_mode() -> None:
    with pytest.raises(ValueError, match="only supports ui_prompt_mode='fixed'"):
        maybe_cache_text_encoder(
            CountingTextEncoder(),
            SimpleNamespace(
                cache_text_embeddings=True,
                ui_prompt_mode="graph_paths",
                ui_prompt="fixed",
                negative_prompt="negative",
            ),
        )


def test_training_dataloader_kwargs_enable_async_loading() -> None:
    config = SimpleNamespace(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )

    assert training_dataloader_kwargs(config) == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 3,
    }


def test_training_dataloader_kwargs_omit_worker_only_options() -> None:
    config = SimpleNamespace(num_workers=0, pin_memory=False)

    assert training_dataloader_kwargs(config) == {
        "num_workers": 0,
        "pin_memory": False,
    }


def test_update_ema_model_updates_matching_parameter_shards() -> None:
    source = torch.nn.Linear(2, 2, bias=True)
    ema = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(2.0)
        ema.weight.zero_()
        ema.bias.zero_()

    update_ema_model(ema, source, decay=0.75)

    assert torch.equal(ema.weight, torch.ones_like(ema.weight))
    assert torch.equal(ema.bias, torch.full_like(ema.bias, 0.5))


def test_ema_fsdp_shadow_updates_on_configured_device() -> None:
    module = torch.nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
    wrapper = SimpleNamespace(module=module)
    with torch.no_grad():
        module.weight.zero_()
    ema = EMA_FSDP(wrapper, decay=0.5, device="cpu")

    with torch.no_grad():
        module.weight.fill_(2.0)
    ema.update(wrapper)

    assert ema.shadow["weight"].device.type == "cpu"
    assert ema.shadow["weight"].dtype == torch.float32
    assert torch.equal(
        ema.shadow["weight"],
        torch.ones_like(ema.shadow["weight"]),
    )
