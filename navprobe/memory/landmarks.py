from __future__ import annotations

from dataclasses import dataclass, field
import math

from navprobe.memory.entity_knowledge import EntityKnowledge
from navprobe.memory.entity_knowledge import merge_entity_knowledge




def landmark_position_distance_3d(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between landmark observations in world coordinates."""
    return math.hypot(
        float(a["x"]) - float(b["x"]),
        float(a["y"]) - float(b["y"]),
        float(a["z"]) - float(b["z"]),
    )


@dataclass
class NavProbeLandmarkRecord:
    landmark_id: str
    class_name: str
    first_step_index: int
    last_step_index: int
    position: dict[str, float]
    detections: list[dict[str, object]] = field(default_factory=list)
    knowledge: list[EntityKnowledge] = field(default_factory=list)

    def observation_count(self) -> int:
        return len(
            {
                str(item.get("obs_id", "") or "")
                for item in self.detections
                if str(item.get("obs_id", "") or "").strip() != ""
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "landmark_id": str(self.landmark_id),
            "class_name": str(self.class_name),
            "status": "confirmed",
            "first_step_index": int(self.first_step_index),
            "last_step_index": int(self.last_step_index),
            "observation_count": int(self.observation_count()),
            "position": dict(self.position),
            "detections": [dict(item) for item in self.detections],
            "knowledge": [item.to_dict() for item in self.knowledge],
        }


class NavProbeLandmarkMemory:
    def __init__(
        self,
        *,
        merge_distance_m: float,
        merge_distance_by_class: dict[str, float] | None = None,
        max_detections_per_entity: int,
    ) -> None:
        self.merge_distance_m = float(merge_distance_m)
        self.merge_distance_by_class = self._normalize_merge_distance_by_class(
            merge_distance_by_class
        )
        self.max_detections_per_entity = int(max_detections_per_entity)
        self._next_landmark_index = 0
        self._records: dict[str, NavProbeLandmarkRecord] = {}
        self._merged_into: dict[str, str] = {}
        # Lightweight observation records; representative RGB evidence stays bounded.
        self._detections_by_obs_id: dict[str, dict[str, dict[str, object]]] = {}

    def reset(self) -> None:
        self._next_landmark_index = 0
        self._records = {}
        self._merged_into = {}
        self._detections_by_obs_id = {}

    def _next_landmark_id(self) -> str:
        landmark_id = f"landmark_{self._next_landmark_index}"
        self._next_landmark_index += 1
        return landmark_id

    @staticmethod
    def _normalize_class_name(value: object) -> str:
        return str(value).strip().replace("_", " ")

    @classmethod
    def _normalize_merge_distance_by_class(
        cls,
        merge_distance_by_class: dict[str, float] | None,
    ) -> dict[str, float]:
        if merge_distance_by_class is None:
            return {}
        normalized: dict[str, float] = {}
        for class_name, distance in merge_distance_by_class.items():
            normalized_class_name = cls._normalize_class_name(class_name)
            normalized[normalized_class_name] = float(distance)
        return normalized

    @staticmethod
    def _normalize_position(value: object) -> dict[str, float]:
        return {
            "x": float(value["x"]),
            "y": float(value["y"]),
            "z": float(value.get("z", 0.0)),
        }

    def merge_distance_for_class(self, class_name: str) -> float:
        normalized_class_name = self._normalize_class_name(class_name)
        return float(self.merge_distance_by_class.get(normalized_class_name, self.merge_distance_m))


    def _find_match(
        self,
        *,
        class_name: str,
        position: dict[str, float],
    ) -> tuple[NavProbeLandmarkRecord, float] | None:
        best: NavProbeLandmarkRecord | None = None
        best_distance = float("inf")
        class_text = self._normalize_class_name(class_name)
        merge_distance_m = self.merge_distance_for_class(class_text)
        for record in self._records.values():
            if self._normalize_class_name(record.class_name) != class_text:
                continue
            distance = self._record_min_position_distance_3d(
                position=position,
                record=record,
            )
            if distance > merge_distance_m:
                continue
            if distance >= best_distance:
                continue
            best = record
            best_distance = distance
        if best is None:
            return None
        return best, float(best_distance)

    def _record_min_position_distance_3d(
        self,
        *,
        position: dict[str, float],
        record: NavProbeLandmarkRecord,
    ) -> float:
        positions = self._record_evidence_positions(record)
        return min(landmark_position_distance_3d(position, existing) for existing in positions)

    def _record_pair_min_position_distance_3d(
        self,
        a: NavProbeLandmarkRecord,
        b: NavProbeLandmarkRecord,
    ) -> float:
        a_positions = self._record_evidence_positions(a)
        b_positions = self._record_evidence_positions(b)
        return min(
            landmark_position_distance_3d(a_position, b_position)
            for a_position in a_positions
            for b_position in b_positions
        )

    def _record_evidence_positions(
        self,
        record: NavProbeLandmarkRecord,
    ) -> list[dict[str, float]]:
        positions = [
            self._normalize_position(item["position"])
            for item in record.detections
            if isinstance(item.get("position"), dict)
        ]
        if positions == []:
            positions.append(self._normalize_position(record.position))
        return positions

    @staticmethod
    def _evidence_sort_key(evidence: dict[str, object]) -> tuple[float, int, str]:
        return (
            -float(evidence.get("score", 0.0)),
            -int(evidence.get("step_index", 0)),
            str(evidence.get("det_id", "")),
        )

    def _refresh_record(self, record: NavProbeLandmarkRecord) -> None:
        record.detections.sort(key=self._evidence_sort_key)
        record.detections = record.detections[: self.max_detections_per_entity]
        positions = self._record_evidence_positions(record)
        record.position = {
            "x": sum(float(item["x"]) for item in positions) / len(positions),
            "y": sum(float(item["y"]) for item in positions) / len(positions),
            "z": sum(float(item["z"]) for item in positions) / len(positions),
        }

    def _add_or_replace_evidence(
        self,
        *,
        record: NavProbeLandmarkRecord,
        evidence: dict[str, object],
    ) -> None:
        obs_id = str(evidence.get("obs_id", "") or "")
        existing_index = None
        for index, item in enumerate(record.detections):
            if str(item.get("obs_id", "") or "") == obs_id:
                existing_index = index
                break
        if existing_index is None:
            record.detections.append(evidence)
        elif float(evidence["score"]) > float(record.detections[existing_index].get("score", 0.0)):
            record.detections[existing_index] = evidence

    def _consolidate_record(
        self,
        record: NavProbeLandmarkRecord,
    ) -> NavProbeLandmarkRecord:
        while True:
            merge_target = self._find_record_consolidation_target(record)
            if merge_target is None:
                return record
            survivor, absorbed = self._ordered_record_pair(record, merge_target)
            survivor.first_step_index = min(survivor.first_step_index, absorbed.first_step_index)
            survivor.last_step_index = max(survivor.last_step_index, absorbed.last_step_index)
            merge_entity_knowledge(survivor.knowledge, absorbed.knowledge)
            for evidence in absorbed.detections:
                self._add_or_replace_evidence(record=survivor, evidence=dict(evidence))
            self._refresh_record(survivor)
            self._merged_into[absorbed.landmark_id] = survivor.landmark_id
            self._records.pop(str(absorbed.landmark_id), None)
            record = survivor

    def _find_record_consolidation_target(
        self,
        record: NavProbeLandmarkRecord,
    ) -> NavProbeLandmarkRecord | None:
        best: NavProbeLandmarkRecord | None = None
        best_distance = float("inf")
        class_name = self._normalize_class_name(record.class_name)
        merge_distance_m = self.merge_distance_for_class(class_name)
        for other in self._records.values():
            if other is record:
                continue
            if self._normalize_class_name(other.class_name) != class_name:
                continue
            distance = self._record_pair_min_position_distance_3d(record, other)
            if distance > merge_distance_m:
                continue
            if distance >= best_distance:
                continue
            best = other
            best_distance = distance
        return best

    @staticmethod
    def _landmark_id_index(landmark_id: str) -> int:
        return int(str(landmark_id).rsplit("_", 1)[-1])

    def _ordered_record_pair(
        self,
        a: NavProbeLandmarkRecord,
        b: NavProbeLandmarkRecord,
    ) -> tuple[NavProbeLandmarkRecord, NavProbeLandmarkRecord]:
        ordered = sorted(
            [a, b],
            key=lambda item: (
                int(item.first_step_index),
                self._landmark_id_index(str(item.landmark_id)),
                str(item.landmark_id),
            ),
        )
        return ordered[0], ordered[1]

    def add_detection(
        self,
        *,
        detection: dict[str, object],
        source: str,
        step_index: int,
    ) -> NavProbeLandmarkRecord:
        class_name = self._normalize_class_name(detection.get("class_name", "") or "")
        det_id = str(detection.get("det_id", "") or "")
        obs_id = str(detection.get("obs_id", "") or "")
        position = self._normalize_position(detection.get("position"))
        evidence = {
            "det_id": det_id,
            "obs_id": obs_id,
            "class_name": class_name,
            "detector_class_name": str(detection.get("detector_class_name", class_name)),
            "score": float(detection.get("score", 0.0)),
            "bbox": [float(value) for value in list(detection.get("bbox", []))],
            "source": str(source),
            "step_index": int(step_index),
            "position": dict(position),
            "geometry": str(detection.get("geometry", "")),
        }
        match = self._find_match(class_name=class_name, position=position)
        if match is None:
            record = NavProbeLandmarkRecord(
                landmark_id=self._next_landmark_id(),
                class_name=class_name,
                first_step_index=int(step_index),
                last_step_index=int(step_index),
                position=dict(position),
                detections=[evidence],
            )
            self._records[str(record.landmark_id)] = record
            self._refresh_record(record)
        else:
            record, _distance = match
            self._add_or_replace_evidence(record=record, evidence=evidence)
            self._refresh_record(record)
        # Entity lifetime covers every associated detection, including evidence
        # that did not enter the bounded representative image set.
        record.first_step_index = min(record.first_step_index, int(step_index))
        record.last_step_index = max(record.last_step_index, int(step_index))
        record = self._consolidate_record(record)
        self._detections_by_obs_id.setdefault(obs_id, {})[det_id] = {
            "landmark_id": record.landmark_id,
            "obs_id": obs_id,
            "det_id": det_id,
            "bbox": list(evidence["bbox"]),
            "score": evidence["score"],
            "position": dict(position),
        }
        return record

    def iter_records(self) -> list[NavProbeLandmarkRecord]:
        """Return current entities without serializing their evidence or knowledge."""
        return list(self._records.values())

    def get_record(self, landmark_id: str) -> NavProbeLandmarkRecord:
        record = self.find_record(landmark_id)
        if record is None:
            raise ValueError(f"unknown landmark_id: {landmark_id!r}")
        return record

    def find_record(self, landmark_id: str) -> NavProbeLandmarkRecord | None:
        current = str(landmark_id)
        path = []
        while current in self._merged_into:
            path.append(current)
            current = self._merged_into[current]
        for alias in path:
            self._merged_into[alias] = current
        return self._records.get(current)

    def detections_for_obs_ids(self, obs_ids: list[str]) -> dict[str, list[dict[str, object]]]:
        result = {}
        for obs_id in obs_ids:
            best_by_id = {}
            for item in self._detections_by_obs_id.get(str(obs_id), {}).values():
                record = self.find_record(str(item["landmark_id"]))
                if record is None:
                    continue
                current = {
                    **item,
                    "bbox": list(item["bbox"]),
                    "position": dict(item["position"]),
                    "landmark_id": record.landmark_id,
                    "class_name": record.class_name,
                    "status": "confirmed",
                }
                previous = best_by_id.get(record.landmark_id)
                if previous is None or (float(current["score"]), str(current["det_id"])) > (
                    float(previous["score"]), str(previous["det_id"])
                ):
                    best_by_id[record.landmark_id] = current
            result[str(obs_id)] = list(best_by_id.values())
        return result


    def to_context(self, *, include_observations: bool = False) -> dict[str, object]:
        records = [record.to_dict() for record in self._records.values()]
        records.sort(
            key=lambda item: (
                -int(item["observation_count"]),
                str(item["landmark_id"]),
            )
        )
        return {
            **({
                "merged_into": dict(self._merged_into),
                "next_landmark_index": self._next_landmark_index,
                "observation_detections": {
                    obs_id: [dict(item) for item in detections.values()]
                    for obs_id, detections in self._detections_by_obs_id.items()
                },
            } if include_observations else {}),
            "merge_distance_m": float(self.merge_distance_m),
            "merge_distance_by_class": dict(self.merge_distance_by_class),
            "max_detections_per_entity": int(self.max_detections_per_entity),
            "confirmed_landmarks": records,
        }
