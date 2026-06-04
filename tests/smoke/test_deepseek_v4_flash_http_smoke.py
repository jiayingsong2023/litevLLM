from vllm.entrypoints.openai.api_server import app


def test_openai_server_still_exposes_chat_route_for_deepseek_support() -> None:
    paths = {route.path for route in app.routes}
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths
