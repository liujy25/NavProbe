"""Validate shared runtime resources before starting a Habitat episode."""
from pathlib import Path

from navprobe.config.settings import configuration_for_args
from navprobe.llm.client import APIStatusError


def validate_runtime_resources(args) -> None:
    config = configuration_for_args(args)
    config.model.client_kwargs()
    resources = {"environment.habitat_config": config.values["environment"]["habitat_config"]}
    perception = config.perception
    detector = perception["detector"]
    resources.update({
        f"landmark_perception.{detector}.{key}": value
        for key, value in perception[detector].items() if key.endswith("path")
    })
    for name, value in resources.items():
        if not Path(value).is_file():
            raise FileNotFoundError(f"Missing {name}: {value}")
    scenes_dir = config.values["dataset"]["scenes_dir"]
    if not Path(scenes_dir).is_dir():
        raise FileNotFoundError(f"Missing dataset.scenes_dir: {scenes_dir}")


def is_global_model_error(error: Exception) -> bool:
    """Identify authentication, endpoint and explicit model/parameter failures."""
    if not isinstance(error, APIStatusError):
        return False
    if error.status_code in {401, 403, 404}:
        return True
    if error.status_code != 400:
        return False
    body = error.body
    if not isinstance(body, dict):
        return False
    details = body.get("error", body)
    if not isinstance(details, dict):
        return False
    return details.get("code") in {
        "model_not_found", "invalid_model", "unsupported_parameter",
        "unsupported_value", "invalid_api_key",
    } or details.get("param") in {
        "model", "reasoning_effort", "temperature", "max_tokens",
        "max_completion_tokens", "response_format", "enable_thinking",
    }
