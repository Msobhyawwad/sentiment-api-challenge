from httpx import ASGITransport, AsyncClient
from app import app, get_cached_result, set_cached_result
import redis
import json
import pytest
# Mock Redis client for testing
redis_client = redis.Redis(host="localhost", port=6379, db=0)

def test_get_cached_result():
    # Clear any existing cache
    redis_client.flushdb()
    
    # Test when cache is empty
    assert get_cached_result("test_text") is None
    
    # Test when cache is not empty
    set_cached_result("test_text", [{"label": "POSITIVE", "score": 0.99}])
    cached_result = get_cached_result("test_text")
    assert cached_result == [{"label": "POSITIVE", "score": 0.99}]

def test_set_cached_result():
    # Clear any existing cache
    redis_client.flushdb()
    
    # Test setting cache
    set_cached_result("test_text", [{"label": "POSITIVE", "score": 0.99}])
    cached_result = redis_client.get("test_text")
    assert json.loads(cached_result) == [{"label": "POSITIVE", "score": 0.99}]

@pytest.mark.anyio
async def test_predict_endpoint():
    # Clear any existing cache
    redis_client.flushdb()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with a short text
        response = await ac.post("/predict", json={"text": "I love FastAPI!"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)

@pytest.mark.anyio
async def test_predict_endpoint_cached():
    # Clear any existing cache
    redis_client.flushdb()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with cached result
        response = await ac.post("/predict", json={"text": "I love FastAPI!"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.anyio
async def test_predict_quantized_endpoint():
    # Clear any existing cache
    redis_client.flushdb()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with a short text
        response = await ac.post("/predict_quantized", json={"text": "I love FastAPI!"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)

@pytest.mark.anyio
async def test_predict_quantized_endpoint_cached():
    # Clear any existing cache
    redis_client.flushdb()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:    
         # Test with cached result
        response = await ac.post("/predict_quantized", json={"text": "I love FastAPI!"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.anyio
async def test_predict_endpoint_with_invalid_input():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with invalid input (missing text field)
        response = await ac.post("/predict", json={})
        assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.anyio
async def test_predict_quantized_endpoint_with_invalid_input():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Test with invalid input (missing text field)
        response = await ac.post("/predict_quantized", json={})
        assert response.status_code == 422  # Unprocessable Entity