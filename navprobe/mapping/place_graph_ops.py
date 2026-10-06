from __future__ import annotations

from navprobe.memory.graph.graph import Graph
from navprobe.runtime.cache import RuntimeCache


def _create_place_node(
    graph: Graph,
    cache: RuntimeCache,
    anchor_obs_id: str,
    obs_ids: list[str],
    floor_id: str,
) -> str:
    anchor_observation = cache.get_observation(str(anchor_obs_id)).observation
    pose = anchor_observation.pose
    new_node = graph.add_node(
        position=(float(pose.x), float(pose.y), float(pose.z)),
        yaw=float(pose.yaw),
        floor_id=str(floor_id),
        obs_id=str(anchor_obs_id),
        node_kind="place",
        category="place",
    )
    new_node.obs_ids = [str(obs_id) for obs_id in obs_ids]
    return str(new_node.id)
