from fastapi.testclient import TestClient

from vllm.entrypoints.openai import api_server


def test_openai_server_requires_initialized_engine_for_chat_requests() -> None:
    """TestClient imports the app only; real GGUF chat still needs engine wiring.

    DeepSeek V4 Flash currently exposes only the Task 6 limited model.forward smoke:
    one input token through embedding, output_norm, and Q8_0 output projection.
    The OpenAI chat route goes through AsyncLLM/LiteEngine, which allocates the
    normal KV cache and calls the model via prefill/decode executors with full
    autoregressive metadata. That path is not the limited one-token smoke path.
    """
    previous_engine = api_server.engine
    api_server.engine = None
    try:
        response = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0,
                "max_tokens": 1,
            },
        )
    finally:
        api_server.engine = previous_engine

    assert response.status_code == 503
    assert response.json() == {"detail": "engine not initialized"}


def test_openai_server_still_exposes_chat_route_for_deepseek_support() -> None:
    paths = {route.path for route in api_server.app.routes}
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths
