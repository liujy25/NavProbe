from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import os
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image

from navprobe.llm.log_format import format_llm_logs_text


def _decode_image_data_url(url: str) -> tuple[str, str, bytes] | None:
    if not url.startswith("data:image/"):
        return None
    header, separator, payload = url.partition(",")
    if separator == "":
        raise ValueError("image data URL is missing a comma separator")
    mime_type = header.removeprefix("data:").split(";", 1)[0]
    extension = {
        "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png", "image/webp": "webp",
    }.get(mime_type, "img")
    return mime_type, extension, base64.b64decode(payload)


def _safe_filename_component(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_") or "unknown"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_llm_request_artifacts(
    *,
    step_dir: Path,
    logs: list[dict[str, object]],
    save_images: bool = True,
) -> dict[str, Any]:
    """Store request text and optionally exact images; never modify live logs."""
    llm_dir = step_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    images = {}
    for step_index, source_log in enumerate(logs, start=1):
        log = deepcopy(source_log)
        request_index = int(log.get("request_index", step_index))
        call_name = str(log.get("call_name", "unknown"))
        module, separator, operation = call_name.partition(".")
        if not separator:
            module, operation = "other", call_name
        module, operation = map(_safe_filename_component, (module, operation))
        context = dict(log.get("context") or {})
        round_number = context.get("retrieval_round")
        folder = llm_dir / module
        if round_number is not None:
            folder = folder / "retrieve" / f"round_{int(round_number):02d}"
        else:
            folder = folder / operation
        folder = folder / f"request_{request_index:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        image_count = 0
        omitted_image_count = 0
        prompt = log.get("prompt") or {}
        user = prompt.get("user")
        for part in user if isinstance(user, list) else []:
            if part.get("type") != "image_url":
                continue
            if not save_images:
                part["image_url"] = {"saved": False, "reason": "output.visualize=false"}
                omitted_image_count += 1
                continue
            image_url = part.get("image_url") or {}
            decoded = _decode_image_data_url(str(image_url.get("url", "")))
            if decoded is None:
                continue
            mime_type, extension, image_bytes = decoded
            image_dir = llm_dir / "images"
            image_dir.mkdir(exist_ok=True)
            filename = f"{hashlib.sha256(image_bytes).hexdigest()}.{extension}"
            image_path = image_dir / filename
            if not image_path.exists():
                image_path.write_bytes(image_bytes)
            relative_path = image_path.relative_to(step_dir).as_posix()
            if relative_path not in images:
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                images[relative_path] = {
                    "path": relative_path, "width": width, "height": height,
                    "mime_type": mime_type,
                }
            image_url["url"] = Path(os.path.relpath(image_path, folder)).as_posix()
            image_count += 1
        log.update(module=module, operation=operation, context=context)
        _write_json(folder / "request.json", log)
        (folder / "request.txt").write_text(format_llm_logs_text([log]), encoding="utf-8")
        requests.append({
            "request_index": request_index, "step_request_index": step_index,
            "call_name": call_name, "module": module, "operation": operation,
            "context": context, "elapsed_seconds": log.get("elapsed_seconds", 0.0),
            "status": log.get("status", "completed"),
            "request_attempt": log.get("request_attempt", 1),
            "error": log.get("error"),
            "request_ref": (folder / "request.json").relative_to(step_dir).as_posix(),
            "text_ref": (folder / "request.txt").relative_to(step_dir).as_posix(),
            "image_count": image_count,
            "omitted_image_count": omitted_image_count,
        })
    summary = {
        "schema_version": 3, "request_count": len(requests), "requests": requests,
        "image_count": len(images), "dir": "llm", "images": list(images.values()),
        "omitted_image_count": sum(request["omitted_image_count"] for request in requests),
    }
    _write_json(llm_dir / "index.json", summary)
    overview = [
        "# LLM requests", "",
        "Requests are listed in execution order; paths are grouped by module and retrieval round.", "",
        "| Request | Module / operation | Status | Attempt | Retrieve round | Seconds | Files |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for request in requests:
        text_ref = Path(request["text_ref"]).relative_to("llm").as_posix()
        json_ref = Path(request["request_ref"]).relative_to("llm").as_posix()
        overview.append(
            f"| {request['request_index']} | {request['module']} / {request['operation']} | "
            f"{request['status']} | {request['request_attempt']} | "
            f"{request['context'].get('retrieval_round') or '—'} | {float(request['elapsed_seconds']):.3f} | "
            f"[text]({text_ref}) · [JSON]({json_ref}) · {request['image_count']} saved images"
            f" · {request['omitted_image_count']} omitted |"
        )
    (llm_dir / "README.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    return summary
