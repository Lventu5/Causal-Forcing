from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence


FIXED_PROMPT_MODE = "fixed"
GRAPH_PATH_PROMPT_MODE = "graph_paths"
UI_PROMPT_MODES = (FIXED_PROMPT_MODE, GRAPH_PATH_PROMPT_MODE)

_BBOX_SUFFIX_RE = re.compile(r"\s+@\d+,\d+,\d+,\d+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")


def load_graph_frame_states(token_sidecar_path: str | Path) -> list[str]:
    """Load the active-state summary stored in each graph-token frame."""
    path = Path(token_sidecar_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Graph-derived UI prompts require the graph-token sidecar: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    explicit_states = payload.get("frame_states")
    if isinstance(explicit_states, list):
        return [_normalise_raw_state(value) for value in explicit_states]

    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Graph-token sidecar must contain a frames list: {path}")
    states: list[str] = []
    for tokens in frames:
        first = tokens[0] if isinstance(tokens, list) and tokens else ""
        states.append(_normalise_raw_state(first))
    return states


def build_ui_block_prompt(
    frame_states: Sequence[str],
    *,
    fixed_prompt: str,
    max_states: int = 16,
) -> str:
    """Build one native WAN prompt describing a complete temporal block."""
    states = _deduplicate_consecutive(
        _describe_state(value) for value in frame_states if str(value).strip()
    )
    if not states:
        return str(fixed_prompt)
    states = _truncate_states(states, max_states=max_states)
    sequence = " then ".join(states)
    return (
        "A sharp Linux desktop screen recording. "
        f"The visible screen sequence shows {sequence}. "
        "The interface follows the supplied actions. UI text, folder names, "
        "and window titles are crisp and legible."
    )


def prompt_for_block(
    *,
    mode: str,
    fixed_prompt: str,
    frame_states: Sequence[str] | None,
    start: int,
    num_frames: int,
) -> str:
    """Return the fixed or graph-derived prompt for one frame interval."""
    mode = str(mode).strip().lower()
    if mode not in UI_PROMPT_MODES:
        raise ValueError(f"Unsupported UI prompt mode {mode!r}; choose from {UI_PROMPT_MODES}.")
    if mode == FIXED_PROMPT_MODE:
        return str(fixed_prompt)
    if frame_states is None:
        raise ValueError("graph_paths prompt mode requires graph frame states.")
    end = int(start) + int(num_frames)
    if start < 0 or num_frames < 1 or len(frame_states) < end:
        raise ValueError(
            "Graph frame states are too short for prompt block "
            f"start={start}, frames={num_frames}: got {len(frame_states)}, need {end}."
        )
    return build_ui_block_prompt(
        frame_states[int(start):end],
        fixed_prompt=fixed_prompt,
    )


def _normalise_raw_state(value: object) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
    text = _BBOX_SUFFIX_RE.sub("", text).strip()
    if " | " in text:
        _, text = text.split(" | ", 1)
    return text.strip()


def _describe_state(value: str) -> str:
    state = _normalise_raw_state(value)
    lowered = state.lower()
    if not state or lowered in {"<missing>", "missing", "unknown"}:
        return "an unknown desktop screen"
    if lowered.startswith("desktop:") or lowered.startswith("desktop://"):
        return "the desktop"
    if state.startswith("/"):
        name = Path(state.rstrip("/")).name
        if state == "/":
            return 'the root folder at "/"'
        parts = Path(state.rstrip("/")).parts
        if state.rstrip("/") == "/home" or (
            len(parts) == 3 and parts[:2] == ("/", "home")
        ):
            return f'the Home folder at "{state}"'
        return f'the folder "{name}" at "{state}"'
    if "://" in state:
        scheme, remainder = state.split("://", 1)
        label = remainder.strip("/") or scheme
        return f'the {scheme} screen "{label}"'
    return f'the application screen "{state}"'


def _deduplicate_consecutive(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not output or output[-1] != value:
            output.append(value)
    return output


def _truncate_states(states: Sequence[str], *, max_states: int) -> list[str]:
    if max_states < 2:
        raise ValueError("max_states must be at least 2.")
    if len(states) <= max_states:
        return list(states)
    return [*states[: max_states - 1], f"additional transitions ending at {states[-1]}"]
