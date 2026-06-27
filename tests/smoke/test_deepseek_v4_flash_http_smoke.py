from fastapi.testclient import TestClient

from vllm.entrypoints.openai import api_server


class _FakeOutput:
    def __init__(self, text: str, finished: bool) -> None:
        self.outputs = [type("_FakeToken", (), {"text": text})()]
        self.finished = finished


class _FakeEngine:
    def __init__(self, payload_text: str = "ok") -> None:
        self.payload_text = payload_text

    def generate_greedy_reference_chat(self, prompt: str, *, max_tokens: int) -> str:
        raise AssertionError(
            "production chat route must not call direct-reference helper"
        )

    async def generate(
        self,
        prompt: str,
        sampling_params,
        request_id: str,
        lora_request=None,
        multi_modal_data=None,
        **kwargs,
    ):
        assert prompt == "x"
        assert getattr(sampling_params, "temperature", None) == 0
        assert request_id.startswith("chat-")
        assert lora_request is None
        assert multi_modal_data is None
        yield _FakeOutput(self.payload_text, True)

    async def get_model_config(self):
        return type("_FakeModelConfig", (), {"model": "deepseek-v4-flash"})()

    def stats(self):
        return {}

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        return None


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


def test_openai_server_uses_normal_generate_path_for_chat_requests() -> None:
    previous_engine = api_server.engine
    api_server.engine = _FakeEngine()  # type: ignore[assignment]
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

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "ok"


def test_openai_server_streams_normal_generate_path_for_chat_requests() -> None:
    previous_engine = api_server.engine
    api_server.engine = _FakeEngine("streamed")  # type: ignore[assignment]
    try:
        chunks: list[str] = []
        with TestClient(api_server.app).stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0,
                "max_tokens": 1,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            for line in response.iter_lines():
                if line:
                    chunks.append(line.decode() if isinstance(line, bytes) else line)
    finally:
        api_server.engine = previous_engine

    assert any(line.startswith("data: {") for line in chunks)
    assert any("streamed" in line for line in chunks)
    assert "data: [DONE]" in chunks
