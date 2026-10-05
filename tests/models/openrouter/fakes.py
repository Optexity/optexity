import json

import httpx

from optexity.inference.models.openrouter import LLMConfig, OpenRouterClient


def completion(content="ok", model="google/gemini-3.5-flash-lite", **extra):
    body = {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "provider": "Google",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": extra.pop("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 1},
        },
    }
    body.update(extra)
    return httpx.Response(200, json=body)


def error(status, message="boom", headers=None):
    return httpx.Response(
        status, json={"error": {"code": status, "message": message}}, headers=headers
    )


class FakeOpenRouter:
    """Records every request; replies from a queue (callable or Response)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0) if self.replies else completion()
        if callable(reply) and not isinstance(reply, httpx.Response):
            return reply(request, body)
        return reply

    @property
    def models(self) -> list[str]:
        return [r["model"] for r in self.requests]


def make_client(fake: FakeOpenRouter, **config) -> OpenRouterClient:
    config.setdefault("api_key", "test-key")
    return OpenRouterClient(
        LLMConfig(**config),
        http_client=httpx.Client(transport=httpx.MockTransport(fake)),
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
    )
