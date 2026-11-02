"""Configuration loading: default.yaml + profile overlay + user files + ``--set`` overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from mrep.utils.hashing import short_hash

PROFILES = ("default", "light", "standard", "max_quality")


def find_config_dir() -> Path:
    """Locate the config directory: $MREP_CONFIG_DIR > repo checkout > packaged resources."""
    env = os.environ.get("MREP_CONFIG_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    here = Path(__file__).resolve()
    checkout = here.parents[2] / "config"
    if (checkout / "default.yaml").is_file():
        return checkout
    packaged = here.parent / "resources" / "config"
    if (packaged / "default.yaml").is_file():
        return packaged
    raise FileNotFoundError("MREP config directory not found (set MREP_CONFIG_DIR)")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_resource_yaml(name: str) -> dict[str, Any]:
    """Load an auxiliary config such as ``drum_map.yaml`` from the config directory."""
    path = Path(name)
    if not path.is_absolute():
        path = find_config_dir() / name
    return load_yaml(path)


def parse_set(expr: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value`` (value parsed as YAML scalar)."""
    if "=" not in expr:
        raise ValueError(f"--set expects key=value, got {expr!r}")
    key, value = expr.split("=", 1)
    return key.strip().split("."), yaml.safe_load(value)


def apply_set(cfg: dict[str, Any], keys: list[str], value: Any) -> None:
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load_config(
    profile: str = "default",
    overrides: list[Path] | None = None,
    sets: list[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    cfg_dir = find_config_dir()
    cfg = load_yaml(cfg_dir / "default.yaml")
    if profile and profile != "default":
        p = Path(profile)
        if not p.is_file():
            p = cfg_dir / f"{profile}.yaml"
        if not p.is_file():
            raise FileNotFoundError(f"unknown profile {profile!r} (expected {p})")
        cfg = deep_merge(cfg, load_yaml(p))
        cfg.setdefault("profile", profile)
    for path in overrides or []:
        cfg = deep_merge(cfg, load_yaml(Path(path)))
    for expr in sets or []:
        keys, value = parse_set(expr)
        apply_set(cfg, keys, value)
    if device:
        cfg["device"] = device
    return cfg


def stage_config(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    return dict(cfg.get("stages", {}).get(stage, {}) or {})


def stage_enabled(cfg: dict[str, Any], stage: str) -> bool:
    return bool(stage_config(cfg, stage).get("enabled", True))


def config_hash(cfg: dict[str, Any]) -> str:
    return short_hash(cfg, 16)


def stage_params_hash(cfg: dict[str, Any], stage: str) -> str:
    """Hash of everything that influences a stage's output (its section + global audio settings)."""
    payload = {"stage": stage_config(cfg, stage), "audio": cfg.get("audio", {}), "seed": cfg.get("seed", 0)}
    return short_hash(payload, 16)
