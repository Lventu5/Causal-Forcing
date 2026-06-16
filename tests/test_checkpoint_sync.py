from pathlib import Path
from types import SimpleNamespace
import sys


CF_ROOT = Path(__file__).resolve().parents[1]
if str(CF_ROOT) not in sys.path:
    sys.path.insert(0, str(CF_ROOT))

from utils import checkpoint_sync  # noqa: E402
from utils.checkpoint_sync import (  # noqa: E402
    CheckpointSyncManager,
    detect_current_partition,
    parse_partition_list,
)


class DummyProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _config(**overrides):
    values = {
        "checkpoint_sync_partitions": "",
        "checkpoint_sync_rsync_args": "-a --delete --partial",
        "checkpoint_sync_log_dir": "",
        "checkpoint_sync_mkpath": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _checkpoint_dir(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "action_stage1" / "last"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(b"model")
    (checkpoint_dir / "trainer.pt").write_bytes(b"trainer")
    return checkpoint_dir


def test_parse_partition_list_accepts_commas_and_spaces() -> None:
    assert parse_partition_list("sof1, msp3 gcp-us,,sof1") == [
        "sof1",
        "msp3",
        "gcp-us",
    ]


def test_detect_current_partition_from_slurm_nodelist(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_NODELIST", "msp3-h200-[0-7]")

    assert detect_current_partition(["sof1", "msp3", "gcp-us"]) == "msp3"


def test_sync_launches_one_background_rsync_per_remote_partition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched = []

    def fake_popen(command, **kwargs):
        launched.append((command, kwargs))
        return DummyProcess()

    monkeypatch.setattr(checkpoint_sync.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "msp3-h200-[0-7]")
    checkpoint_dir = _checkpoint_dir(tmp_path)
    manager = CheckpointSyncManager(
        _config(
            checkpoint_sync_partitions="sof1,msp3,gcp-us",
            checkpoint_sync_log_dir=str(tmp_path / "logs"),
        ),
        output_path=tmp_path / "action_stage1",
        is_main_process=True,
    )

    manager.sync(checkpoint_dir, step=2000, stage="action_stage1")

    destinations = [command[-1] for command, _ in launched]
    assert destinations == [
        f"sof1:{checkpoint_dir.resolve()}/",
        f"gcp-us:{checkpoint_dir.resolve()}/",
    ]
    assert all(command[0] == "rsync" for command, _ in launched)
    assert all(kwargs["start_new_session"] for _, kwargs in launched)
    assert (checkpoint_dir / "checkpoint_ready.json").exists()


def test_sync_skips_when_previous_partition_sync_is_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched = []

    def fake_popen(command, **kwargs):
        launched.append(command)
        return DummyProcess(returncode=None)

    monkeypatch.setattr(checkpoint_sync.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "msp3-h200-[0-7]")
    checkpoint_dir = _checkpoint_dir(tmp_path)
    manager = CheckpointSyncManager(
        _config(
            checkpoint_sync_partitions="sof1,msp3",
            checkpoint_sync_log_dir=str(tmp_path / "logs"),
        ),
        output_path=tmp_path / "action_stage1",
        is_main_process=True,
    )

    manager.sync(checkpoint_dir, step=1, stage="action_stage1")
    manager.sync(checkpoint_dir, step=2, stage="action_stage1")

    assert len(launched) == 1


def test_sync_launch_failure_is_nonfatal(tmp_path: Path, monkeypatch) -> None:
    def failing_popen(command, **kwargs):
        raise OSError("rsync unavailable")

    monkeypatch.setattr(checkpoint_sync.subprocess, "Popen", failing_popen)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "msp3-h200-[0-7]")
    checkpoint_dir = _checkpoint_dir(tmp_path)
    manager = CheckpointSyncManager(
        _config(
            checkpoint_sync_partitions="sof1,msp3",
            checkpoint_sync_log_dir=str(tmp_path / "logs"),
        ),
        output_path=tmp_path / "action_stage1",
        is_main_process=True,
    )

    manager.sync(checkpoint_dir, step=1, stage="action_stage1")


def test_sync_skips_all_targets_when_current_partition_is_unknown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched = []

    def fake_popen(command, **kwargs):
        launched.append(command)
        return DummyProcess()

    monkeypatch.setattr(checkpoint_sync.subprocess, "Popen", fake_popen)
    checkpoint_dir = _checkpoint_dir(tmp_path)
    manager = CheckpointSyncManager(
        _config(
            checkpoint_sync_partitions="definitely-remote-a,definitely-remote-b",
            checkpoint_sync_log_dir=str(tmp_path / "logs"),
        ),
        output_path=tmp_path / "action_stage1",
        is_main_process=True,
    )

    manager.sync(checkpoint_dir, step=1, stage="action_stage1")

    assert launched == []
