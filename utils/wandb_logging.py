import os
from typing import Any

import wandb
from omegaconf import OmegaConf


_PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "{your key}",
    "{your entity}",
    "{your project}",
}


def _clean_config_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_VALUES:
        return None
    return value


def init_wandb(config: Any) -> None:
    """Initialize wandb without storing API keys in the logged config."""
    api_key = _clean_config_value(os.environ.get("WANDB_API_KEY"))
    if api_key is None:
        api_key = _clean_config_value(getattr(config, "wandb_key", None))

    host = _clean_config_value(os.environ.get("WANDB_HOST"))
    if host is None:
        host = _clean_config_value(getattr(config, "wandb_host", None))

    entity = _clean_config_value(os.environ.get("WANDB_ENTITY"))
    if entity is None:
        entity = _clean_config_value(getattr(config, "wandb_entity", None))

    project = _clean_config_value(os.environ.get("WANDB_PROJECT"))
    if project is None:
        project = _clean_config_value(getattr(config, "wandb_project", None))

    mode = _clean_config_value(os.environ.get("WANDB_MODE")) or "online"
    name = _clean_config_value(os.environ.get("WANDB_NAME")) or getattr(config, "config_name", None)
    group = _clean_config_value(os.environ.get("WANDB_GROUP"))
    tags_value = _clean_config_value(os.environ.get("WANDB_TAGS"))
    tags = [tag.strip() for tag in tags_value.split(",") if tag.strip()] if tags_value else None

    if mode != "offline":
        login_kwargs = {}
        if host is not None:
            login_kwargs["host"] = host
        if api_key is not None:
            login_kwargs["key"] = api_key
        wandb.login(**login_kwargs)

    logged_config = OmegaConf.to_container(config, resolve=True)
    if isinstance(logged_config, dict):
        logged_config.pop("wandb_key", None)

    wandb.init(
        config=logged_config,
        name=name,
        mode=mode,
        entity=entity,
        project=project,
        dir=getattr(config, "wandb_save_dir", ""),
        group=group,
        tags=tags,
    )
