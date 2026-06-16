from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_RSYNC_ARGS = ("-a", "--delete", "--partial")
CURRENT_PARTITION_ENV_KEYS = (
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_CONSTRAINTS",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "NODE",
    "HOSTNAME",
    "CONSTRAINT",
)


def parse_partition_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[\s,]+", value.strip())
    else:
        raw_items = [str(item).strip() for item in value]

    partitions = []
    seen = set()
    for item in raw_items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        partitions.append(item)
    return partitions


def partition_matches(candidate: str, current: str | None) -> bool:
    if not current:
        return False
    candidate_key = candidate.strip().lower()
    current_key = current.strip().lower()
    return (
        candidate_key == current_key
        or candidate_key in current_key
        or current_key in candidate_key
    )


def detect_current_partition(candidates: list[str]) -> str | None:
    env_values = [
        os.environ[key].strip()
        for key in CURRENT_PARTITION_ENV_KEYS
        if os.environ.get(key, "").strip()
    ]
    env_values.extend(
        value
        for value in {
            socket.gethostname(),
            socket.getfqdn(),
            getattr(os.uname(), "nodename", ""),
        }
        if value
    )
    for candidate in candidates:
        if any(partition_matches(candidate, value) for value in env_values):
            return candidate
    return None


def parse_rsync_args(value: Any) -> list[str]:
    if value is None or value == "":
        return list(DEFAULT_RSYNC_ARGS)
    if isinstance(value, str):
        return shlex.split(value)
    return [str(item) for item in value]


def as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class CheckpointSyncManager:
    def __init__(
        self,
        config: Any,
        *,
        output_path: str | os.PathLike[str],
        is_main_process: bool,
    ) -> None:
        self.is_main_process = is_main_process
        self.partitions = parse_partition_list(
            getattr(config, "checkpoint_sync_partitions", "")
        )
        self.current_partition = detect_current_partition(self.partitions)
        self.rsync_args = parse_rsync_args(
            getattr(config, "checkpoint_sync_rsync_args", None)
        )
        self.log_dir = Path(
            getattr(config, "checkpoint_sync_log_dir", "")
            or (Path(output_path).expanduser() / "checkpoint_sync_logs")
        )
        self.mkpath = as_bool(
            getattr(config, "checkpoint_sync_mkpath", True),
            default=True,
        )
        self._processes: dict[str, tuple[subprocess.Popen[Any], Path]] = {}
        self._warned_unknown_current = False

    @property
    def enabled(self) -> bool:
        return self.is_main_process and bool(self.partitions)

    def _reap_finished(self) -> None:
        for partition, (process, log_path) in list(self._processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            status = "completed" if returncode == 0 else f"failed ({returncode})"
            print(
                f"Checkpoint rsync to {partition} {status}; log: {log_path}",
                flush=True,
            )
            del self._processes[partition]

    def _write_ready_marker(
        self,
        checkpoint_dir: Path,
        *,
        step: int,
        stage: str,
    ) -> None:
        marker_path = checkpoint_dir / "checkpoint_ready.json"
        temporary_path = marker_path.with_suffix(".json.tmp")
        payload = {
            "stage": stage,
            "step": int(step),
            "checkpoint_dir": str(checkpoint_dir),
            "created_at": time.time(),
        }
        temporary_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(temporary_path, marker_path)

    def _rsync_command(self, checkpoint_dir: Path, partition: str) -> list[str]:
        command = ["rsync", *self.rsync_args]
        if self.mkpath:
            remote_dir = shlex.quote(str(checkpoint_dir))
            command.append(f"--rsync-path=mkdir -p {remote_dir} && rsync")
        command.extend(
            [
                f"{checkpoint_dir}/",
                f"{partition}:{checkpoint_dir}/",
            ]
        )
        return command

    def sync(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        step: int,
        stage: str,
    ) -> None:
        if not self.enabled:
            return

        self._reap_finished()
        if self.current_partition is None:
            if not self._warned_unknown_current:
                print(
                    "WARNING: checkpoint rsync current partition could not be inferred; "
                    "skipping cross-partition checkpoint sync.",
                    flush=True,
                )
                self._warned_unknown_current = True
            return
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if not checkpoint_dir.exists():
            print(
                f"Skipping checkpoint rsync; missing checkpoint dir: {checkpoint_dir}",
                flush=True,
            )
            return

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._write_ready_marker(checkpoint_dir, step=step, stage=stage)
        except Exception as exc:
            print(
                f"Skipping checkpoint rsync; could not write ready marker: {exc}",
                flush=True,
            )
            return

        for partition in self.partitions:
            if partition_matches(partition, self.current_partition):
                print(
                    f"Skipping checkpoint rsync to current partition {partition}.",
                    flush=True,
                )
                continue

            active = self._processes.get(partition)
            if active is not None and active[0].poll() is None:
                print(
                    f"Skipping checkpoint rsync to {partition}; previous sync still running.",
                    flush=True,
                )
                continue

            log_path = self.log_dir / (
                f"rsync_{re.sub(r'[^A-Za-z0-9_.-]+', '_', partition)}_"
                f"step_{int(step):06d}.log"
            )
            command = self._rsync_command(checkpoint_dir, partition)
            try:
                with log_path.open("ab", buffering=0) as log_file:
                    log_file.write(
                        (
                            "\n"
                            f"[checkpoint-sync] step={int(step)} "
                            f"stage={stage} partition={partition}\n"
                            f"command: {shlex.join(command)}\n"
                        ).encode("utf-8")
                    )
                    process = subprocess.Popen(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                        start_new_session=True,
                    )
                self._processes[partition] = (process, log_path)
                print(
                    f"Launched background checkpoint rsync to {partition}; log: {log_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"WARNING: failed to launch checkpoint rsync to {partition}: {exc}",
                    flush=True,
                )
