import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

import app.api.health as health_module
from app.api.health import router

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
