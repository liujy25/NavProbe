"""Read the active local Habitat agent without importing Habitat itself.

Prefer the live HabitatSim configuration, then native simulator configuration,
then Env's internal and public configurations. Roots may be a full config, a
HabitatConfig, or a SimulatorConfig. Explicit default-agent selection follows
Habitat's agents_order/default_agent_id. Without an explicit selection, use
main_agent first, then mapping/list order. Fields missing from one source fall
back to the next.
"""
from __future__ import annotations

from typing import Any, Iterator


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    try:
        return getattr(value, key)
    except AttributeError:
        pass
    if hasattr(value, "get"):
        return value.get(key, default)
    return default


def _iter_items(value: Any) -> list[tuple[Any, Any]]:
    if hasattr(value, "items"):
        return list(value.items())
    return []


def _simulator_configs(env: Any) -> Iterator[Any]:
    sim = _get_attr_or_key(env, "sim")
    seen: set[int] = set()
    for root in (
        _get_attr_or_key(sim, "habitat_config"),
        _get_attr_or_key(sim, "config"),
        _get_attr_or_key(env, "_config"),
        _get_attr_or_key(env, "config"),
    ):
        habitat = _get_attr_or_key(root, "habitat", root)
        simulator = _get_attr_or_key(habitat, "simulator", habitat)
        if simulator is not None and id(simulator) not in seen:
            seen.add(id(simulator))
            yield simulator


def _iter_agent_configs(simulator: Any) -> list[Any]:
    agents = _get_attr_or_key(simulator, "agents")
    order = _get_attr_or_key(simulator, "agents_order")
    agent_id = _get_attr_or_key(simulator, "default_agent_id", 0)
    if order is not None:
        # Do not read a different agent's camera or radius if this one lacks it.
        name = order[int(agent_id)]
        agent = _get_attr_or_key(agents, str(name))
        return [] if agent is None else [agent]
    if isinstance(agents, (list, tuple)):
        # Native Habitat-Sim config stores AgentConfiguration objects in a list.
        if _get_attr_or_key(simulator, "default_agent_id") is not None:
            return [agents[int(agent_id)]] if agents else []
        return list(agents)
    configs: list[Any] = []
    seen: set[int] = set()
    for agent in [_get_attr_or_key(agents, "main_agent"), *[value for _, value in _iter_items(agents)]]:
        if agent is not None and id(agent) not in seen:
            seen.add(id(agent))
            configs.append(agent)
    return configs


def habitat_agent_configs(env: Any) -> Iterator[Any]:
    for simulator in _simulator_configs(env):
        yield from _iter_agent_configs(simulator)


def habitat_agent_radius_m(habitat_env: Any) -> float | None:
    for candidate in habitat_agent_configs(habitat_env):
        radius = _get_attr_or_key(candidate, "radius")
        if radius is not None:
            return float(radius)
    sim = _get_attr_or_key(habitat_env, "sim")
    agents = _get_attr_or_key(sim, "agents") or []
    if agents:
        agent_id = next((value for config in _simulator_configs(habitat_env)
                         if (value := _get_attr_or_key(config, "default_agent_id")) is not None), None)
        candidates = agents if agent_id is None else [agents[int(agent_id)]]
        for sim_agent in candidates:
            radius = _get_attr_or_key(_get_attr_or_key(sim_agent, "agent_config"), "radius")
            if radius is not None:
                return float(radius)
    return None


def _sensor_config(env: Any, sensor_name: str | None = None) -> Any:
    for agent in habitat_agent_configs(env):
        sensors = _iter_items(_get_attr_or_key(agent, "sim_sensors"))
        if sensor_name is not None:
            for name, sensor in sensors:
                if str(name) == sensor_name:
                    return sensor
        else:
            for kind in ("depth", "rgb"):
                for name, sensor in sensors:
                    if kind in str(name).lower():
                        return sensor
    return None
