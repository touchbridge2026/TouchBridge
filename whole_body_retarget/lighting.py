"""Validated, reproducible lighting configuration for rgb_o3d rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


LIGHTING_SCHEMA_VERSION = "1.0"
LIGHTING_MODEL = "base_point_lights_lambert_blinn_phong"
DEFAULT_LIGHTING_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "rgb_o3d_lighting.yaml"
)


def _finite_vector(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly {length} finite values")
    return tuple(float(item) for item in array)


def _nonnegative(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _unit_interval(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class PointLight:
    name: str
    position_base_m: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float
    falloff_radius_m: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, expected_name: str) -> "PointLight":
        name = str(payload.get("name", expected_name))
        if name != expected_name:
            raise ValueError(f"Expected light name {expected_name!r}, got {name!r}")
        color = _finite_vector(payload.get("color"), name=f"{name}.color", length=3)
        if any(item < 0.0 or item > 1.0 for item in color):
            raise ValueError(f"{name}.color values must be in [0, 1]")
        falloff = _nonnegative(
            payload.get("falloff_radius_m"), name=f"{name}.falloff_radius_m"
        )
        if falloff <= 0.0:
            raise ValueError(f"{name}.falloff_radius_m must be positive")
        return cls(
            name=name,
            position_base_m=_finite_vector(
                payload.get("position_base_m"),
                name=f"{name}.position_base_m",
                length=3,
            ),
            color=color,
            intensity=_nonnegative(payload.get("intensity"), name=f"{name}.intensity"),
            falloff_radius_m=falloff,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position_base_m": list(self.position_base_m),
            "color": list(self.color),
            "intensity": self.intensity,
            "falloff_radius_m": self.falloff_radius_m,
        }


@dataclass(frozen=True)
class LightingConfig:
    ambient_intensity: float
    key_light: PointLight
    fill_light: PointLight
    specular_intensity: float
    shininess: float
    arm_lighting_strength: float
    hand_lighting_strength: float
    other_lighting_strength: float
    gamma_correct: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LightingConfig":
        if str(payload.get("schema_version")) != LIGHTING_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported lighting schema_version: "
                f"{payload.get('schema_version')!r}; expected {LIGHTING_SCHEMA_VERSION!r}"
            )
        if payload.get("coordinate_frame") != "robot_base":
            raise ValueError("Lighting coordinate_frame must be 'robot_base'")
        if payload.get("model") != LIGHTING_MODEL:
            raise ValueError(
                f"Lighting model must be {LIGHTING_MODEL!r}, got {payload.get('model')!r}"
            )
        material = payload.get("material")
        if not isinstance(material, dict):
            raise ValueError("Lighting config must contain a material mapping")
        lights = payload.get("lights")
        if not isinstance(lights, list) or len(lights) != 2:
            raise ValueError("Lighting config must contain exactly key and fill lights")
        by_name = {
            str(item.get("name")): item for item in lights if isinstance(item, dict)
        }
        if set(by_name) != {"key", "fill"}:
            raise ValueError("Lighting config lights must be named 'key' and 'fill'")
        shininess = _nonnegative(material.get("shininess"), name="material.shininess")
        if shininess < 1.0:
            raise ValueError("material.shininess must be at least 1")
        return cls(
            ambient_intensity=_nonnegative(
                payload.get("ambient_intensity"), name="ambient_intensity"
            ),
            key_light=PointLight.from_dict(by_name["key"], expected_name="key"),
            fill_light=PointLight.from_dict(by_name["fill"], expected_name="fill"),
            specular_intensity=_nonnegative(
                material.get("specular_intensity"), name="material.specular_intensity"
            ),
            shininess=shininess,
            arm_lighting_strength=_unit_interval(
                material.get("arm_lighting_strength"),
                name="material.arm_lighting_strength",
            ),
            hand_lighting_strength=_unit_interval(
                material.get("hand_lighting_strength"),
                name="material.hand_lighting_strength",
            ),
            other_lighting_strength=_unit_interval(
                material.get("other_lighting_strength"),
                name="material.other_lighting_strength",
            ),
            gamma_correct=bool(material.get("gamma_correct", True)),
        )

    @property
    def lights(self) -> tuple[PointLight, PointLight]:
        return (self.key_light, self.fill_light)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIGHTING_SCHEMA_VERSION,
            "coordinate_frame": "robot_base",
            "model": LIGHTING_MODEL,
            "ambient_intensity": self.ambient_intensity,
            "lights": [light.to_dict() for light in self.lights],
            "material": {
                "specular_intensity": self.specular_intensity,
                "shininess": self.shininess,
                "arm_lighting_strength": self.arm_lighting_strength,
                "hand_lighting_strength": self.hand_lighting_strength,
                "other_lighting_strength": self.other_lighting_strength,
                "gamma_correct": self.gamma_correct,
            },
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_lighting_config(path: str | Path) -> LightingConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Lighting config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lighting config root must be a mapping: {source}")
    return LightingConfig.from_dict(payload)


def save_lighting_config(path: str | Path, config: LightingConfig) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        config.to_dict(),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination
