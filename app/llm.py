"""LLM providers for the chat UI.

Two adapters cover essentially every endpoint:

* ``anthropic``         Claude via the official Anthropic SDK. Credentials resolve the SDK's
                        usual way (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
                        ``ant auth login`` profile) unless a key is supplied per request.
* ``openai_compatible`` any ``POST {base_url}/chat/completions`` endpoint: OpenAI, Azure
                        OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp server, ...

Keys are never written to disk by the server: they come from the request (the UI keeps
them in the browser session) or from the environment variable named in config.yaml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    pass


@dataclass
class ProviderConfig:
    provider: str                       # "anthropic" | "openai_compatible"
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    system: str | None = None
    max_tokens: int = 4096
    extra: dict = field(default_factory=dict)

    def key(self) -> str | None:
        return self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)


def chat(cfg: ProviderConfig, messages: list[dict]) -> str:
    """messages: [{"role": "user"|"assistant", "content": str}, ...] -> assistant text."""
    if cfg.provider == "anthropic":
        return _anthropic(cfg, messages)
    if cfg.provider == "openai_compatible":
        return _openai_compatible(cfg, messages)
    raise LLMError(f"unknown provider {cfg.provider!r}")


def _anthropic(cfg: ProviderConfig, messages: list[dict]) -> str:
    import anthropic

    kwargs = {}
    if cfg.key():
        kwargs["api_key"] = cfg.key()
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    client = anthropic.Anthropic(**kwargs)
    req = dict(model=cfg.model, max_tokens=cfg.max_tokens, messages=messages)
    if cfg.system:
        req["system"] = cfg.system
    try:
        # Server-side refusal fallback ("default" routing) - only on the first-party API.
        if not cfg.base_url:
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                               fallbacks="default", **req)
        else:
            resp = client.messages.create(**req)
    except TypeError as e:
        # The SDK raises TypeError (not an API error) when no credential source resolves.
        if "authentication method" not in str(e):
            raise
        raise LLMError("No Anthropic credentials: enter an API key in the sidebar, or start the "
                       "server with ANTHROPIC_API_KEY set (or `ant auth login`)") from e
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic rejected the API key - check it and try again") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Anthropic model or endpoint not found: {cfg.model}") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Anthropic rate limit hit - retry shortly") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"cannot reach Anthropic API: {e}") from e
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        return f"[The model declined to answer{f' ({cat})' if cat else ''}.]"
    return "".join(b.text for b in resp.content if b.type == "text")


def _openai_compatible(cfg: ProviderConfig, messages: list[dict]) -> str:
    if not cfg.base_url:
        raise LLMError("openai_compatible provider needs a base_url (e.g. http://localhost:11434/v1)")
    msgs = ([{"role": "system", "content": cfg.system}] if cfg.system else []) + messages
    headers = {"Content-Type": "application/json"}
    if cfg.key():
        headers["Authorization"] = f"Bearer {cfg.key()}"
    body = {"model": cfg.model, "messages": msgs, "max_tokens": cfg.max_tokens, **cfg.extra}
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=300)
    except httpx.HTTPError as e:
        raise LLMError(f"cannot reach {url}: {e}") from e
    if r.status_code != 200:
        raise LLMError(f"{url} returned {r.status_code}: {r.text[:300]}")
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError(f"unexpected response shape from {url}: {r.text[:300]}") from e
