from __future__ import annotations


import json
import re
import threading
import time
from dataclasses import dataclass


from navprobe.llm.request_config import _model_uses_temperature_parameter
from navprobe.llm.request_config import REASONING_EFFORT_CHOICES, model_request_budget
from navprobe.llm.request_config import NAVPROBE_REASONING_MODULES, validate_module_reasoning_efforts
from navprobe.llm.request_context import current_request_context
try:
    from openai import APIConnectionError
    from openai import APITimeoutError
    from openai import APIStatusError
    from openai import OpenAI
    from openai import omit
except ModuleNotFoundError as exc:  # The dependency is only needed for model-backed runs.
    if exc.name != "openai":
        raise

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    OpenAI = None  # type: ignore[assignment,misc]
    omit = None


class _ModelResponseError(ValueError):
    """A returned response violates the JSON output contract."""


def _retryable_request_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        status = error.status_code
        return status in {408, 409, 429} or status >= 500
    return False


def _normalize_prompt_text(text: str) -> str:
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", str(text).strip())


def _normalize_prompt_content(
    content: str | list[dict[str, object]],
) -> str | list[dict[str, object]]:
    if isinstance(content, str):
        return _normalize_prompt_text(content)
    normalized: list[dict[str, object]] = []
    for part in content:
        normalized_part = dict(part)
        if str(normalized_part.get("type", "")) == "text":
            normalized_part["text"] = _normalize_prompt_text(
                str(normalized_part.get("text", ""))
            )
        normalized.append(normalized_part)
    return normalized


@dataclass
class LLMUsageEntry:
    request_index: int
    call_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    temperature: float | None
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    max_completion_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_index": int(self.request_index),
            "call_name": str(self.call_name),
            "model": str(self.model),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_tokens is not None:
            payload["reasoning_tokens"] = self.reasoning_tokens
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        return payload


class LLMClient:
    def __init__(
        self,
        *,
        reasoning_effort: str | None,
        module_reasoning_efforts: list[int] | None = None,
        reasoning_by_module: dict[str, str],
        request_timeout_seconds: float,
        min_completion_tokens: int,
        request_limits: dict[str, dict[str, int]],
        model: str,
        api_key: str,
        base_url: str,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_CHOICES:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort!r}")
        self.reasoning_effort = None if reasoning_effort in {None, "none"} else reasoning_effort
        self.module_reasoning_efforts = validate_module_reasoning_efforts(module_reasoning_efforts)
        self.reasoning_by_module = dict(reasoning_by_module or {})
        self.request_timeout_seconds = request_timeout_seconds
        if any(k not in NAVPROBE_REASONING_MODULES or v not in REASONING_EFFORT_CHOICES for k, v in self.reasoning_by_module.items()):
            raise ValueError("invalid named module reasoning effort")
        self.model = str(model)
        self.base_url = base_url
        self.api_key = api_key
        if OpenAI is None:
            raise RuntimeError(
                "LLMClient requires the optional 'llm' dependency; install "
                "navprobe[llm] (or openai) before starting model-backed navigation"
            )
        self.client = OpenAI(
            # The SDK requires an initialization key. Unauthenticated requests
            # explicitly omit the Authorization header below.
            api_key=self.api_key or "unused",
            base_url=self.base_url,
            timeout=self.request_timeout_seconds,
            max_retries=0,  # The shared request loop owns the retry budget.
        )
        self._usage_totals = dict(request_count=0, prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self._llm_logs: list[dict[str, object]] = []
        self._llm_log_offset = 0
        self._request_index = 0
        self._call_lock = threading.Lock()
        self.min_completion_tokens = int(min_completion_tokens)
        self.request_limits = {name: dict(limits) for name, limits in request_limits.items()}

    def _reasoning_request_config(self, call_name: str) -> dict[str, str]:
        effort = self.reasoning_effort
        module = str(call_name).split(".", 1)[0]
        levels = self.module_reasoning_efforts
        if levels is not None and module in NAVPROBE_REASONING_MODULES:
            level = levels[NAVPROBE_REASONING_MODULES.index(module)]
            effort = REASONING_EFFORT_CHOICES[level] if level else None
        named = self.reasoning_by_module
        if module in named:
            effort = None if named[module] == "none" else named[module]
        return {} if effort is None else {"reasoning_effort": effort}


    @staticmethod
    def _base_url_is_openai_official(base_url: str | None) -> bool:
        normalized = str(base_url or "").strip().lower().rstrip("/")
        return normalized in {"", "https://api.openai.com/v1", "https://api.openai.com"}

    def reset_usage(self) -> None:
        self._usage_totals = dict(request_count=0, prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self._llm_logs = []
        self._llm_log_offset = 0
        self._request_index = 0

    def get_llm_log_count(self) -> int:
        return self._llm_log_offset + len(self._llm_logs)

    def get_llm_logs_since(self, start_index: int) -> list[dict[str, object]]:
        offset = self._llm_log_offset
        if int(start_index) < offset:
            raise ValueError("requested LLM logs have already been archived")
        return list(self._llm_logs[int(start_index) - offset:])

    def discard_llm_logs_before(self, end_index: int) -> None:
        """Release archived prompt/image payloads while keeping stable cursors."""
        with self._call_lock:
            offset = self._llm_log_offset
            if not offset <= int(end_index) <= self.get_llm_log_count():
                raise ValueError("LLM log archive boundary is outside retained logs")
            del self._llm_logs[:int(end_index) - offset]
            self._llm_log_offset = int(end_index)


    def get_usage_summary(self) -> dict[str, object]:
        return {
            **self._usage_totals,
            "attempt_count": int(self._request_index),
            "request_error_count": int(self._request_index) - self._usage_totals["request_count"],
        }

    def _record_usage(
        self,
        *,
        call_name: str,
        completion: object,
        request_config: dict[str, object],
    ) -> LLMUsageEntry:
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens_raw = getattr(usage, "total_tokens", None)
        total_tokens = int(total_tokens_raw) if total_tokens_raw is not None else int(prompt_tokens + completion_tokens)
        request_index = int(self._request_index) + 1
        entry = LLMUsageEntry(
            request_index=request_index,
            call_name=str(call_name),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            temperature=None if "temperature" not in request_config else float(request_config["temperature"]),
            reasoning_effort=request_config.get("reasoning_effort"),
            reasoning_tokens=getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
            finish_reason=(getattr(completion.choices[0], "finish_reason", None) if getattr(completion, "choices", []) else None),
            max_completion_tokens=request_config.get("max_completion_tokens", request_config.get("max_tokens")),
        )
        self._usage_totals["request_count"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._usage_totals[key] += getattr(entry, key)
        self._request_index = request_index
        return entry

    def _record_llm_log(
        self,
        *,
        usage_entry: LLMUsageEntry,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
        completion: object,
        elapsed_seconds: float,
    ) -> None:
        response_items = [
            getattr(getattr(choice, "message", None), "content", None)
            for choice in getattr(completion, "choices", [])
        ]
        response: object
        if len(response_items) == 1:
            response = response_items[0]
        else:
            response = response_items
        self._llm_logs.append(
            {
                **usage_entry.to_dict(),
                "prompt": {
                    "system": system_prompt,
                    "user": user_prompt,
                },
                "response": response,
                "elapsed_seconds": float(elapsed_seconds),
                "context": current_request_context(),
            }
        )


    def _create_visual_json_completion(
        self,
        *,
        call_name: str,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
        max_new_tokens: int,
        token_field: str,
        response_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> dict[str, object]:
        system_prompt = _normalize_prompt_text(system_prompt)
        user_prompt = _normalize_prompt_content(user_prompt)
        if token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"unsupported visual token field: {token_field!r}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        official_request_config: dict[str, object] = self._reasoning_request_config(call_name)
        fallback_request_config = dict(official_request_config)
        if temperature is not None and _model_uses_temperature_parameter(
            self.model
        ):
            fallback_request_config["temperature"] = float(temperature)
        official_openai = self._base_url_is_openai_official(self.base_url)
        effective_token_field = (
            "max_completion_tokens"
            if official_openai and token_field == "max_tokens"
            else token_field
        )
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            effective_token_field: max(int(max_new_tokens), int(self.min_completion_tokens)),
            "timeout": self.request_timeout_seconds,
        }
        if not self.api_key:
            request_kwargs["extra_headers"] = {"Authorization": omit}
        if official_openai and response_schema is not None:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        if official_openai:
            effective_request_config = dict(official_request_config)
            if "temperature" in effective_request_config:
                request_kwargs["temperature"] = float(effective_request_config["temperature"])
        else:
            effective_request_config = dict(fallback_request_config)
            if "temperature" in effective_request_config:
                request_kwargs["temperature"] = float(
                    effective_request_config["temperature"]
                )
            gpt_family_model = str(self.model).strip().lower().startswith(
                ("gpt-", "chatgpt-")
            )
            if not gpt_family_model:
                request_kwargs["extra_body"] = {"enable_thinking": False}
        request_kwargs.update(effective_request_config)
        with model_request_budget() as budget:
            while budget.remaining:
                attempt = budget.consume()
                with self._call_lock:
                    request_started_at = time.perf_counter()
                    try:
                        completion = self.client.chat.completions.create(**request_kwargs)
                    except Exception as exc:
                        elapsed_seconds = time.perf_counter() - request_started_at
                        self._request_index += 1
                        self._llm_logs.append({
                            "request_index": self._request_index, "call_name": call_name,
                            "model": self.model,
                            **effective_request_config,
                            "max_completion_tokens": request_kwargs[effective_token_field],
                            "prompt": {"system": system_prompt, "user": user_prompt},
                            "response": None, "context": current_request_context(),
                            "elapsed_seconds": elapsed_seconds, "request_attempt": attempt,
                            "status": "request_error", "usage_available": False,
                            "error": {"type": type(exc).__name__, "message": str(exc),
                                      "status_code": getattr(exc, "status_code", None)},
                        })
                        if not _retryable_request_error(exc) or not budget.remaining:
                            raise
                    else:
                        elapsed_seconds = time.perf_counter() - request_started_at
                        usage_entry = self._record_usage(
                            call_name=call_name, completion=completion,
                            request_config={**effective_request_config, effective_token_field: request_kwargs[effective_token_field]},
                        )
                        self._record_llm_log(
                            usage_entry=usage_entry, system_prompt=system_prompt,
                            user_prompt=user_prompt, completion=completion, elapsed_seconds=elapsed_seconds,
                        )
                        log = self._llm_logs[-1]
                        budget.last_response_log = log
                        log.update(status="completed", request_attempt=attempt)
                        try:
                            if len(completion.choices) == 0:
                                raise _ModelResponseError(f"{call_name} returned no choices")
                            choice = completion.choices[0]
                            if getattr(choice, "finish_reason", None) == "length":
                                raise _ModelResponseError(
                                    f"{call_name} returned truncated content; max_completion_tokens={usage_entry.max_completion_tokens}, "
                                    f"completion_tokens={usage_entry.completion_tokens}, reasoning_tokens={usage_entry.reasoning_tokens}"
                                )
                            parsed = self._parse_visual_json_content(call_name=call_name, content=choice.message.content)
                        except _ModelResponseError as exc:
                            log.update(status="response_error", error={"type": type(exc).__name__, "message": str(exc)})
                            if not budget.remaining:
                                raise
                        else:
                            return parsed
                time.sleep(min(2.0, 0.5 * float(attempt)))

    def _parse_visual_json_content(
        self,
        *,
        call_name: str,
        content: object,
    ) -> dict[str, object]:
        if not isinstance(content, str):
            raise _ModelResponseError(f"{call_name} returned non-string content: {content!r}")
        stripped = content.strip()
        if stripped == "":
            raise _ModelResponseError(f"{call_name} returned empty content")
        if stripped.startswith("<think>"):
            closing_tag = "</think>"
            closing_index = stripped.find(closing_tag, len("<think>"))
            if closing_index < 0:
                raise _ModelResponseError(
                    f"{call_name} returned an unclosed <think> prefix: {content!r}"
                )
            stripped = stripped[closing_index + len(closing_tag):].strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(stripped.lstrip())
        except json.JSONDecodeError as exc:
            raise _ModelResponseError(f"{call_name} returned invalid JSON: {content!r}") from exc
        if not isinstance(parsed, dict):
            raise _ModelResponseError(f"{call_name} must return a JSON object: {parsed!r}")
        return parsed

    def initialize_task_state(
        self,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
    ) -> dict[str, object]:
        return self._create_visual_json_completion(
            call_name="task_executive.initialize",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.request_limits["task_executive_initialize"]["max_tokens"],
            token_field="max_completion_tokens",
        )

    def assess_task_state(
        self,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
    ) -> dict[str, object]:
        return self._create_visual_json_completion(
            call_name="task_executive.assess",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.request_limits["task_executive_assess"]["max_tokens"],
            token_field="max_completion_tokens",
        )

    def select_skill(
        self,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
    ) -> dict[str, object]:
        return self._create_visual_json_completion(
            call_name="skill_selector.select",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.request_limits["skill_selector"]["max_tokens"],
            token_field="max_completion_tokens",
        )

    def manage_knowledge(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
        return self._create_visual_json_completion(
            call_name="entity_knowledge_manager.update",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.request_limits["entity_knowledge_manager"]["max_tokens"],
            token_field="max_completion_tokens",
        )

    def summarize_node(
        self,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
    ) -> dict[str, object]:
        return self._create_visual_json_completion(
            call_name="place_memory.summarize",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.request_limits["place_memory"]["max_tokens"],
            token_field="max_completion_tokens",
        )
