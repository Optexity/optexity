import asyncio

import httpx
import pytest
from pydantic import BaseModel

from optexity.inference.models.openrouter import (
    LLMAllModelsFailed,
    LLMBadRequest,
    LLMConfig,
    LLMConfigError,
    LLMOutOfCredits,
    OpenRouterClient,
    Tier,
    user_message,
)

from .fakes import FakeOpenRouter, completion, error, make_client

MSG = [user_message("hello")]
LITE0, LITE1, LITE2 = (
    "google/gemini-3.5-flash-lite",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.6-luna",
)


class Answer(BaseModel):
    answer: str


def test_tags_reasoning_and_usage_are_sent(no_sleep):
    fake = FakeOpenRouter(completion())
    client = make_client(fake, service="optexity-worker", environment="prod")
    client.complete(MSG, feature="llm_extraction", session_id="task-1")
    body = fake.requests[0]
    assert body["model"] == LITE0
    assert body["user"] == "optexity-worker"
    assert body["session_id"] == "task-1"
    assert body["trace"] == {
        "service": "optexity-worker",
        "feature": "llm_extraction",
        "environment": "prod",
    }
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"effort": "medium", "exclude": True}
    assert "provider" not in body and "plugins" not in body


def test_pro_tier_sends_no_reasoning():
    fake = FakeOpenRouter(completion(model="google/gemini-2.5-pro"))
    make_client(fake).complete(MSG, model=Tier.PRO)
    assert "reasoning" not in fake.requests[0]


def test_untagged_when_service_not_configured():
    fake = FakeOpenRouter(completion())
    make_client(fake).complete(MSG)
    assert "user" not in fake.requests[0] and "trace" not in fake.requests[0]


def test_schema_sends_strict_json_schema_and_require_parameters():
    fake = FakeOpenRouter(completion('{"answer": "42"}'))
    result = make_client(fake).complete(MSG, schema=Answer)
    body = fake.requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}
    assert result.parsed == Answer(answer="42")


def test_pdf_pins_native_file_parser():
    fake = FakeOpenRouter(completion())
    make_client(fake).complete([user_message("read", pdfs=[b"%PDF-1.4"])])
    assert fake.requests[0]["plugins"] == [
        {"id": "file-parser", "pdf": {"engine": "native"}}
    ]


def test_result_has_usage_cost_and_model():
    fake = FakeOpenRouter(completion(model=LITE0))
    r = make_client(fake).complete(MSG)
    assert (r.text, r.model_used, r.provider, r.generation_id) == (
        "ok",
        LITE0,
        "Google",
        "gen-test",
    )
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cost) == (
        10,
        5,
        0.001,
    )
    assert (r.usage.reasoning_tokens, r.usage.cached_tokens) == (2, 1)
    assert r.fallback_count == 0


def test_transient_error_retries_once_then_next_model(no_sleep):
    fake = FakeOpenRouter(error(503), error(503), completion(model=LITE1))
    r = make_client(fake).complete(MSG)
    assert fake.models == [LITE0, LITE0, LITE1]
    assert r.model_used == LITE1 and r.fallback_count == 1
    assert [a.outcome for a in r.attempts] == ["LLMUnavailable", "LLMUnavailable", "ok"]
    assert len(no_sleep) == 1


def test_retry_after_header_is_honoured_and_capped(no_sleep):
    fake = FakeOpenRouter(error(429, headers={"retry-after": "3"}), completion())
    make_client(fake).complete(MSG)
    assert no_sleep == [3.0]
    fake = FakeOpenRouter(error(429, headers={"retry-after": "120"}), completion())
    make_client(fake).complete(MSG)
    assert no_sleep[-1] == 10.0


def test_timeout_counts_as_transient(no_sleep):
    def timeout(request, body):
        raise httpx.ReadTimeout("slow", request=request)

    fake = FakeOpenRouter(timeout, timeout, completion(model=LITE1))
    r = make_client(fake).complete(MSG)
    assert fake.models == [LITE0, LITE0, LITE1]
    assert r.attempts[0].outcome == "LLMTimeout"


def test_out_of_credits_stops_immediately():
    fake = FakeOpenRouter(error(402, "Insufficient credits"))
    with pytest.raises(LLMOutOfCredits):
        make_client(fake).complete(MSG)
    assert fake.models == [LITE0]


def test_bad_request_stops_immediately():
    fake = FakeOpenRouter(error(400, "invalid parameter"))
    with pytest.raises(LLMBadRequest):
        make_client(fake).complete(MSG)
    assert len(fake.requests) == 1


def test_context_length_error_moves_to_next_model_without_retry():
    fake = FakeOpenRouter(error(400, "maximum context length exceeded"), completion())
    make_client(fake).complete(MSG)
    assert fake.models == [LITE0, LITE1]


def test_invalid_output_retries_twice_then_next_model():
    fake = FakeOpenRouter(
        completion("nope"),
        completion("still nope"),
        completion("no json"),
        completion('{"answer": "ok"}', model=LITE1),
    )
    r = make_client(fake).complete(MSG, schema=Answer)
    assert fake.models == [LITE0, LITE0, LITE0, LITE1]
    assert r.parsed.answer == "ok"
    # every billed attempt is counted
    assert r.usage.cost == pytest.approx(0.004)


def test_http_200_with_error_body_is_treated_as_failure(no_sleep):
    bad = httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": LITE0,
            "choices": [],
            "error": {"code": 502, "message": "upstream died"},
        },
    )
    fake = FakeOpenRouter(bad, bad, completion(model=LITE1))
    r = make_client(fake).complete(MSG)
    assert r.model_used == LITE1


def test_all_models_failing_raises_with_every_error(no_sleep):
    fake = FakeOpenRouter(*[error(503)] * 6)
    with pytest.raises(LLMAllModelsFailed) as exc:
        make_client(fake).complete(MSG)
    assert fake.models == [LITE0, LITE0, LITE1, LITE1, LITE2, LITE2]
    assert len(exc.value.errors) == 6 and exc.value.retryable


def test_tier_call_never_uses_env_fallback(no_sleep):
    fake = FakeOpenRouter(*[error(503)] * 6)
    client = make_client(fake, env_fallback="openai/gpt-5-mini")
    with pytest.raises(LLMAllModelsFailed):
        client.complete(MSG, model=Tier.PRO)
    assert "openai/gpt-5-mini" not in fake.models


def test_missing_api_key_is_a_config_error():
    with pytest.raises(LLMConfigError):
        OpenRouterClient(LLMConfig(api_key=None))


def test_async_path_falls_back_too(monkeypatch):
    async def no_wait(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    fake = FakeOpenRouter(
        error(503), error(503), completion('{"answer": "a"}', model=LITE1)
    )
    r = asyncio.run(make_client(fake).acomplete(MSG, schema=Answer))
    assert fake.models == [LITE0, LITE0, LITE1]
    assert r.parsed.answer == "a"


def test_provider_rejection_moves_to_next_model_without_retry():
    rejected = httpx.Response(
        400,
        json={
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {"raw": "The compiled grammar is too large"},
            }
        },
    )
    fake = FakeOpenRouter(rejected, completion(model=LITE1))
    r = make_client(fake).complete(MSG)
    assert fake.models == [LITE0, LITE1] and r.model_used == LITE1


def test_invalid_model_id_is_still_fatal():
    fake = FakeOpenRouter(error(400, "gemini/x is not a valid model ID"))
    with pytest.raises(LLMBadRequest):
        make_client(fake).complete(MSG, model="gemini/x")
    assert len(fake.requests) == 1


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_doc_file",
            "description": "Read a doc file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": ["string", "null"]}},
            },
        },
    }
]


def tool_call_reply(arguments='{"path": "a.md"}', model=LITE0, **message_extra):
    return completion(
        content="",
        model=model,
        finish_reason="tool_calls",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_doc_file",
                                "arguments": arguments,
                            },
                        }
                    ],
                    **message_extra,
                },
            }
        ],
    )


def test_tools_are_sent_with_strict_routing():
    fake = FakeOpenRouter(completion())
    make_client(fake).complete(MSG, tools=TOOLS, tool_choice="auto")
    body = fake.requests[0]
    assert body["tools"] == TOOLS and body["tool_choice"] == "auto"
    assert body["provider"] == {"require_parameters": True}


def test_tool_calls_are_parsed_and_message_is_ready_for_history():
    fake = FakeOpenRouter(
        tool_call_reply(
            reasoning_details=[{"type": "reasoning.encrypted", "data": "x"}]
        )
    )
    r = make_client(fake).complete(MSG, tools=TOOLS)
    (call,) = r.tool_calls
    assert (call.id, call.name, call.arguments) == (
        "call_1",
        "read_doc_file",
        {"path": "a.md"},
    )
    assert r.finish_reason == "tool_calls"
    assert r.message["role"] == "assistant"
    assert r.message["tool_calls"][0]["function"]["name"] == "read_doc_file"
    assert r.message["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "x"}
    ]


def test_invalid_tool_arguments_retry_then_next_model():
    fake = FakeOpenRouter(
        tool_call_reply("{not json"),
        tool_call_reply("{not json"),
        tool_call_reply("{not json"),
        tool_call_reply(model=LITE1),
    )
    r = make_client(fake).complete(MSG, tools=TOOLS)
    assert fake.models == [LITE0, LITE0, LITE0, LITE1]
    assert r.tool_calls[0].arguments == {"path": "a.md"}


def test_text_reply_has_no_tool_calls_and_plain_message():
    fake = FakeOpenRouter(completion("done"))
    r = make_client(fake).complete(MSG, tools=TOOLS)
    assert r.tool_calls == []
    assert r.message == {"role": "assistant", "content": "done"}


def test_stream_collects_text_usage_and_finish_reason():
    from .fakes import chunk, sse

    fake = FakeOpenRouter(
        sse(
            [
                chunk({"role": "assistant", "content": "Hel"}),
                chunk({"content": "lo"}),
                chunk({}, finish_reason="stop"),
                chunk(
                    None,
                    usage={
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                        "cost": 0.0005,
                    },
                ),
            ]
        )
    )
    r = make_client(fake).complete(MSG, model=Tier.PRO, stream=True)
    assert fake.requests[0]["stream"] is True
    assert (r.text, r.finish_reason, r.model_used) == (
        "Hello",
        "stop",
        "google/gemini-2.5-pro",
    )
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cost) == (
        7,
        3,
        0.0005,
    )


def test_stream_collects_tool_call_pieces():
    from .fakes import chunk, sse

    fake = FakeOpenRouter(
        sse(
            [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_doc_file",
                                    "arguments": '{"pa',
                                },
                            }
                        ]
                    }
                ),
                chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'th": "a.md"}'}}
                        ]
                    }
                ),
                chunk({}, finish_reason="tool_calls"),
            ]
        )
    )
    r = make_client(fake).complete(MSG, tools=TOOLS, stream=True)
    (call,) = r.tool_calls
    assert (call.id, call.name, call.arguments) == (
        "call_1",
        "read_doc_file",
        {"path": "a.md"},
    )


def test_error_in_the_middle_of_a_stream_moves_on(no_sleep):
    from .fakes import chunk, sse

    broken = sse(
        [
            chunk({"content": "partial"}),
            chunk(
                {},
                finish_reason="error",
                error={"code": 502, "message": "upstream died"},
            ),
        ]
    )
    recovered = sse(
        [
            chunk({"content": "ok"}, model="anthropic/claude-opus-4.7"),
            chunk({}, finish_reason="stop"),
        ]
    )
    fake = FakeOpenRouter(broken, broken, recovered)
    r = make_client(fake).complete(MSG, model=Tier.PRO, stream=True)
    assert fake.models == [
        "google/gemini-2.5-pro",
        "google/gemini-2.5-pro",
        "anthropic/claude-opus-4.7",
    ]
    assert r.text == "ok"


def test_async_stream():
    from .fakes import chunk, sse

    fake = FakeOpenRouter(
        sse([chunk({"content": "async ok"}), chunk({}, finish_reason="stop")])
    )
    r = asyncio.run(make_client(fake).acomplete(MSG, stream=True))
    assert r.text == "async ok"


def test_web_search_plugin_and_citations():
    from optexity.inference.models.openrouter import Citation, WebSearch

    fake = FakeOpenRouter(
        completion(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Python 3.14 is out.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://python.org",
                                    "title": "Python",
                                    "content": "Python 3.14 released",
                                    "start_index": 0,
                                    "end_index": 10,
                                },
                            }
                        ],
                    },
                }
            ]
        )
    )
    r = make_client(fake).complete(
        [user_message("latest python?", pdfs=[b"%PDF-1.4"])],
        web_search=WebSearch(max_results=3, include_domains=["python.org"]),
    )
    plugins = fake.requests[0]["plugins"]
    assert {"id": "web", "max_results": 3, "include_domains": ["python.org"]} in plugins
    assert any(p["id"] == "file-parser" for p in plugins)  # PDF plugin kept too
    assert r.citations == [
        Citation(
            url="https://python.org", title="Python", content="Python 3.14 released"
        )
    ]


def test_stream_rebuilds_reasoning_details_for_the_next_turn():
    from .fakes import chunk, sse

    fake = FakeOpenRouter(
        sse(
            [
                chunk(
                    {
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "index": 0,
                                "text": "Need the ",
                                "format": "x",
                            }
                        ]
                    }
                ),
                chunk(
                    {
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "index": 0,
                                "text": "docs.",
                                "signature": "sig",
                            }
                        ]
                    }
                ),
                chunk(
                    {
                        "reasoning_details": [
                            {"type": "reasoning.encrypted", "index": 1, "data": "enc"}
                        ]
                    }
                ),
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_doc_file",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                ),
                chunk({}, finish_reason="tool_calls"),
            ]
        )
    )
    r = make_client(fake).complete(MSG, tools=TOOLS, stream=True)
    assert r.message["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "index": 0,
            "text": "Need the docs.",
            "format": "x",
            "signature": "sig",
        },
        {"type": "reasoning.encrypted", "index": 1, "data": "enc"},
    ]
    assert r.tool_calls[0].name == "read_doc_file"
