"""모델 호출 래퍼. 응답이 준 토큰 수를 그대로 Usage 로 돌려준다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic
from dotenv import load_dotenv

from rnd_rag.agents.state import Usage
from rnd_rag.paths import ROOT

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"

TIMEOUT = 60.0
MAX_RETRIES = 3  # 529 가 측정 실패로 남지 않게

load_dotenv(ROOT / ".env")

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=TIMEOUT, max_retries=MAX_RETRIES)
    return _client


@dataclass(frozen=True)
class Reply:
    text: str
    usage: Usage


def _usage(model: str, response: anthropic.types.Message) -> Usage:
    return Usage.of(model, response.usage.input_tokens, response.usage.output_tokens)


def _strict(node: Any) -> Any:
    """object 마다 additionalProperties 를 닫는다. 구조화 출력이 이를 요구한다."""
    if isinstance(node, list):
        return [_strict(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {key: _strict(value) for key, value in node.items()}
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    return out


def _create(model: str, system: str, prompt: str, max_tokens: int, **extra: Any):
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    return client().messages.create(**kwargs, **extra)


def call_text(prompt: str, model: str = HAIKU, system: str = "",
              max_tokens: int = 1024) -> Reply:
    response = _create(model, system, prompt, max_tokens)
    text = "".join(b.text for b in response.content if b.type == "text")
    return Reply(text.strip(), _usage(model, response))


def call_json(prompt: str, schema: dict[str, Any], model: str = HAIKU,
              system: str = "", max_tokens: int = 1024) -> tuple[dict[str, Any], Usage]:
    """자유 텍스트를 파싱하면 파싱 실패가 측정 잡음이 된다."""
    response = _create(
        model, system, prompt, max_tokens,
        output_config={"format": {"type": "json_schema", "schema": _strict(schema)}},
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} 응답이 max_tokens={max_tokens} 에서 잘렸다")
    text = "".join(b.text for b in response.content if b.type == "text")
    return json.loads(text), _usage(model, response)


def call_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]],
               schema: dict[str, Any], model: str = SONNET, max_tokens: int = 1024,
               allow_tools: bool = True) -> tuple[anthropic.types.Message, Usage]:
    """도구 호출 왕복 한 번. 도구를 안 부르고 끝내면 schema 형식으로 답한다."""
    response = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        tool_choice={"type": "auto" if allow_tools else "none"},
        output_config={"format": {"type": "json_schema", "schema": _strict(schema)}},
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} 응답이 max_tokens={max_tokens} 에서 잘렸다")
    return response, _usage(model, response)
