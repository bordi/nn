from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def generate_structured_answer(
    prompt: str,
    *,
    api_key: str | None = None,
    client: Any | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> dict[str, Any]:
    """Generate a structured grounded answer from an OpenAI chat completion.

    The model is expected to return JSON with ``answer`` and ``used_chunk_ids``.
    """

    resolved_api_key = _resolve_api_key(api_key)
    openai_client = client or _create_openai_client(resolved_api_key)

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = _extract_response_content(response)
    payload = json.loads(content)

    answer = payload.get("answer")
    used_chunk_ids = payload.get("used_chunk_ids")
    if not isinstance(answer, str):
        raise ValueError("OpenAI response must include a string 'answer'.")
    if not isinstance(used_chunk_ids, list):
        raise ValueError("OpenAI response must include a list 'used_chunk_ids'.")

    return {
        "answer": answer,
        "used_chunk_ids": [str(chunk_id) for chunk_id in used_chunk_ids],
    }


def _resolve_api_key(api_key: str | None) -> str:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in the environment before generating an answer."
        )
    return resolved_api_key


def _create_openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "The openai package is not installed. Add it to requirements before generating answers."
        ) from exc

    return OpenAI(api_key=api_key)


def _extract_response_content(response: Any) -> str:
    if isinstance(response, Mapping):
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("OpenAI response did not contain any choices.")
        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            message = first_choice.get("message", {})
            if isinstance(message, Mapping):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
        else:
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
    else:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("OpenAI response did not contain any choices.")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None

    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI response did not include text content.")
    return content
