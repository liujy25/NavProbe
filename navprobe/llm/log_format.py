from __future__ import annotations

import json


def _format_string(text: str) -> str:
    stripped = text.strip()
    if stripped == "":
        return ""
    if stripped[0] in "{[":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return text
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return text


def _format_image_url(image_url: object) -> str:
    if not isinstance(image_url, dict):
        return str(image_url)
    if image_url.get("saved") is False:
        return f"[image not saved: {image_url['reason']}]"
    url = str(image_url.get("url", ""))
    if url.startswith("data:image/"):
        header, _, payload = url.partition(",")
        formatted_url = f"{header},<base64 omitted: {len(payload)} chars>"
    else:
        formatted_url = url
    return formatted_url


def _format_message_part(part: object, index: int) -> str:
    if not isinstance(part, dict):
        return f"[part {index}]\n{format_llm_log_value(part)}"
    part_type = str(part.get("type", ""))
    if part_type == "text":
        return f"[part {index}: text]\n{format_llm_log_value(part.get('text', ''))}"
    if part_type == "image_url":
        return f"[part {index}: image_url]\n{_format_image_url(part.get('image_url'))}"
    return f"[part {index}: {part_type or 'unknown'}]\n{json.dumps(part, ensure_ascii=False, indent=2)}"


def format_llm_log_value(value: object) -> str:
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, list):
        return "\n\n".join(
            _format_message_part(part, index)
            for index, part in enumerate(value, start=1)
        )
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2)


def format_llm_logs_text(logs: list[dict[str, object]]) -> str:
    if logs == []:
        return "No LLM requests in this step.\n"
    sections: list[str] = []
    for step_request_index, log in enumerate(logs, start=1):
        prompt = log.get("prompt", {})
        if not isinstance(prompt, dict):
            prompt = {}
        global_request_index = int(log.get("request_index", step_request_index))
        call_name = str(log.get("call_name", ""))
        model = str(log.get("model", ""))
        sections.append(
            "\n".join(
                [
                    f"LLM Request {step_request_index}",
                    "",
                    "Global request index:",
                    str(global_request_index),
                    "",
                    "Call name:",
                    call_name,
                    "",
                    "Model:",
                    model,
                    "",
                    "Elapsed seconds:",
                    f"{float(log.get('elapsed_seconds', 0.0)):.3f}",
                    "",
                    "Execution context:",
                    format_llm_log_value(log.get("context", {})),
                    "",
                    "Request status:",
                    str(log.get("status", "completed")),
                    "",
                    "Request attempt:",
                    str(log.get("request_attempt", 1)),
                    "",
                    "Request error:",
                    format_llm_log_value(log.get("error")),
                    "",
                    "Prompt:",
                    "[system]",
                    format_llm_log_value(prompt.get("system", "")),
                    "",
                    "[user]",
                    format_llm_log_value(prompt.get("user", "")),
                    "",
                    "Response:",
                    format_llm_log_value(log.get("response", "")),
                ]
            )
        )
    return "\n\n".join(sections) + "\n"
