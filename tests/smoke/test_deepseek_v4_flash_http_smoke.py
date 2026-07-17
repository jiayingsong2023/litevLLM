from fastapi.testclient import TestClient

from vllm.engine.runtime_observer import InMemoryRuntimeObserver
from vllm.entrypoints.openai import api_server


class _FakeOutput:
    def __init__(self, text: str, finished: bool) -> None:
        self.outputs = [type("_FakeToken", (), {"text": text, "token_ids": [1]})()]
        self.prompt_token_ids = [0]
        self.finished = finished


class _FakeEngine:
    def __init__(self, payload_text: str = "ok") -> None:
        self.payload_text = payload_text
        self.submitted: dict[str, tuple[str, object, object]] = {}
        self.aborted: list[str] = []

    def generate_greedy_reference_chat(self, prompt: str, *, max_tokens: int) -> str:
        raise AssertionError(
            "production chat route must not call direct-reference helper"
        )

    async def submit(
        self,
        prompt: str,
        sampling_params,
        request_id: str,
        multi_modal_data=None,
    ) -> None:
        assert prompt == "x"
        assert getattr(sampling_params, "temperature", None) == 0
        assert request_id.startswith("chat-")
        assert multi_modal_data is None
        self.submitted[request_id] = (prompt, sampling_params, multi_modal_data)

    async def stream(self, request_id: str):
        assert request_id in self.submitted
        yield _FakeOutput(self.payload_text, True)

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

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


def test_openai_server_enables_bounded_runtime_observer(monkeypatch) -> None:
    config = type("_Config", (), {})()
    captured = {}
    previous_engine = api_server.engine
    previous_debug = api_server.debug_endpoints_enabled

    monkeypatch.setattr(api_server, "build_vllm_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        api_server,
        "AsyncLLM",
        lambda received_config: captured.setdefault("config", received_config),
    )
    uvicorn_kwargs = {}
    monkeypatch.setattr(
        api_server.uvicorn,
        "run",
        lambda *_args, **kwargs: uvicorn_kwargs.update(kwargs),
    )
    monkeypatch.setattr("sys.argv", ["api_server", "--model", "models/mock"])
    try:
        api_server.main()
    finally:
        api_server.engine = previous_engine
        api_server.debug_endpoints_enabled = previous_debug

    assert isinstance(captured["config"].runtime_observer, InMemoryRuntimeObserver)
    assert uvicorn_kwargs["host"] == "127.0.0.1"
