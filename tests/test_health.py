import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

import app.api.health as health_module
from app.api.health import router
from app.llm.client import HAIKU, SONNET
from tests import anthropic_fake as fake

_app = FastAPI()
_app.include_router(router)



def _mock_collection(count: int):
    col = MagicMock()
    col.count.return_value = count
    return col


def _mock_client(raise_exc=None):
    client = MagicMock()
    if raise_exc:
        client.messages.create = AsyncMock(side_effect=raise_exc)
    else:
        client.messages.create = AsyncMock(return_value=MagicMock())
    return client


@pytest.mark.asyncio
async def test_kb_ok():
    with patch("app.api.health.get_collection", return_value=_mock_collection(42)), \
         patch("app.api.health.get_client", return_value=_mock_client()):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["knowledge_base"]["status"] == "ok"
    assert data["knowledge_base"]["detail"] == "42"


@pytest.mark.asyncio
async def test_kb_zero():
    with patch("app.api.health.get_collection", return_value=_mock_collection(0)), \
         patch("app.api.health.get_client", return_value=_mock_client()):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    assert r.json()["knowledge_base"]["status"] == "error"


@pytest.mark.asyncio
async def test_kb_error():
    col = MagicMock()
    col.count.side_effect = RuntimeError("db down")
    with patch("app.api.health.get_collection", return_value=col), \
         patch("app.api.health.get_client", return_value=_mock_client()):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    assert r.json()["knowledge_base"]["status"] == "error"


@pytest.mark.asyncio
async def test_model_ok():
    with patch("app.api.health.get_collection", return_value=_mock_collection(1)), \
         patch("app.api.health.get_client", return_value=_mock_client()):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    data = r.json()["model"]
    assert data["status"] == "ok"
    assert data["detail"] == "haiku"


@pytest.mark.asyncio
async def test_model_sonnet_fallback():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=[Exception("haiku down"), MagicMock()])
    with patch("app.api.health.get_collection", return_value=_mock_collection(1)), \
         patch("app.api.health.get_client", return_value=client):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    data = r.json()["model"]
    assert data["status"] == "ok"
    assert data["detail"] == "sonnet"


@pytest.mark.asyncio
async def test_model_error():
    with patch("app.api.health.get_collection", return_value=_mock_collection(1)), \
         patch("app.api.health.get_client", return_value=_mock_client(raise_exc=Exception("api down"))):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    data = r.json()["model"]
    assert data["status"] == "error"
    assert data["detail"] == "both down"


@pytest.mark.asyncio
async def test_overall_degraded():
    with patch("app.api.health.get_collection", return_value=_mock_collection(10)), \
         patch("app.api.health.get_client", return_value=_mock_client(raise_exc=Exception("api down"))):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    assert r.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_cache():
    col = _mock_collection(5)
    client = _mock_client()
    with patch("app.api.health.get_collection", return_value=col), \
         patch("app.api.health.get_client", return_value=client):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            await ac.get("/api/health")
            await ac.get("/api/health")
    assert col.count.call_count == 2
    assert client.messages.create.call_count == 2


# --- probe request on the real SDK over a fake API (ADR 009) ---


async def _probe(monkeypatch, respond):
    api = fake.install(monkeypatch, respond)
    with patch("app.api.health.get_collection", return_value=_mock_collection(21)):
        async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
            r = await ac.get("/api/health")
    return api, r.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, stop_reason",
    [
        ([fake.thinking()], "max_tokens"),
        ([{"type": "text", "text": "ok"}], "end_turn"),
        ([fake.thinking(), {"type": "text", "text": "o"}], "max_tokens"),
    ],
    ids=["cut_off_while_thinking", "plain_ok", "cut_off_mid_text"],
)
async def test_probe_counts_any_reply_as_ok(monkeypatch, content, stop_reason):
    api, data = await _probe(
        monkeypatch, lambda body: fake.message_response(content, stop_reason)
    )

    assert data["model"] == {"status": "ok", "detail": "haiku"}
    assert data["status"] == "ok"
    assert api.models() == [HAIKU]


@pytest.mark.asyncio
async def test_probe_request_identical_for_both_models(monkeypatch):
    def respond(body):
        if body["model"] == HAIKU:
            return fake.error_response(529)
        return fake.message_response([fake.thinking()], "max_tokens", model=SONNET)

    api, data = await _probe(monkeypatch, respond)

    assert data["model"] == {"status": "ok", "detail": "sonnet"}
    haiku_body, sonnet_body = api.bodies
    for body in (haiku_body, sonnet_body):
        assert body["max_tokens"] == 16
        assert body["output_config"] == {"effort": "high"}
        assert "thinking" not in body and "temperature" not in body
    assert haiku_body.pop("model") == HAIKU
    assert sonnet_body.pop("model") == SONNET
    assert haiku_body == sonnet_body


@pytest.mark.asyncio
async def test_probe_both_down(monkeypatch):
    api, data = await _probe(monkeypatch, lambda body: fake.error_response(500))

    assert data["model"] == {"status": "error", "detail": "both down"}
    assert data["status"] == "degraded"
    assert api.models() == [HAIKU, SONNET]
