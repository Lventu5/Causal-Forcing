from pathlib import Path
import sys
import pytest
import torch


CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils import distributed  # noqa: E402


def test_launch_distributed_job_binds_cuda_before_process_group(monkeypatch) -> None:
    calls = []
    init_kwargs = {}

    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("MASTER_ADDR", "master-node")
    monkeypatch.setenv("MASTER_PORT", "29517")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: 4)

    def fake_set_device(local_rank):
        calls.append(("set_device", local_rank))

    def fake_init_process_group(
        *,
        rank,
        world_size,
        backend,
        init_method,
        timeout,
        device_id=None,
    ):
        calls.append(("init_process_group", device_id))
        init_kwargs.update(
            rank=rank,
            world_size=world_size,
            backend=backend,
            init_method=init_method,
            timeout=timeout,
            device_id=device_id,
        )

    monkeypatch.setattr(distributed.torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(
        distributed.dist,
        "init_process_group",
        fake_init_process_group,
    )

    distributed.launch_distributed_job()

    assert calls == [
        ("set_device", 1),
        ("init_process_group", torch.device("cuda", 1)),
    ]
    assert init_kwargs["rank"] == 5
    assert init_kwargs["world_size"] == 16
    assert init_kwargs["backend"] == "nccl"
    assert init_kwargs["init_method"] == "tcp://master-node:29517"


def test_launch_distributed_job_rejects_invisible_local_rank(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "9")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("MASTER_ADDR", "master-node")
    monkeypatch.setenv("MASTER_PORT", "29517")
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: 4)

    with pytest.raises(RuntimeError) as exc_info:
        distributed.launch_distributed_job()

    assert "LOCAL_RANK=9" in str(exc_info.value)
    assert "4 CUDA devices" in str(exc_info.value)
