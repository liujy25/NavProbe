from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionCall:
    action: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "args": self.args}


@dataclass
class ActionResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "message": self.message,
        }
