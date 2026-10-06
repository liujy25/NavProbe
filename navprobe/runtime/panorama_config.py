from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanoramaConfig:
    observation_count: int
    turns_per_observation: int
    turn_direction: str
    turn_angle_degrees: float

    def __post_init__(self) -> None:
        if int(self.observation_count) <= 0:
            raise ValueError(f"observation_count must be positive: {self.observation_count}")
        if int(self.turns_per_observation) <= 0:
            raise ValueError(f"turns_per_observation must be positive: {self.turns_per_observation}")
        if str(self.turn_direction) not in {"left", "right"}:
            raise ValueError(f"turn_direction must be left or right: {self.turn_direction}")
        if float(self.turn_angle_degrees) <= 0.0:
            raise ValueError(f"turn_angle_degrees must be positive: {self.turn_angle_degrees}")

    @property
    def observation_spacing_degrees(self) -> float:
        return float(self.turns_per_observation) * float(self.turn_angle_degrees)


def localmap_panorama_config() -> PanoramaConfig:
    return objnav_panorama_config()


def objnav_panorama_config() -> PanoramaConfig:
    return PanoramaConfig(
        observation_count=4,
        turns_per_observation=3,
        turn_direction="left",
        turn_angle_degrees=30.0,
    )


def hm3dv2_panorama_config() -> PanoramaConfig:
    return vlnce_panorama_config()


def vlnce_panorama_config() -> PanoramaConfig:
    return PanoramaConfig(
        observation_count=4,
        turns_per_observation=3,
        turn_direction="left",
        turn_angle_degrees=30.0,
    )
