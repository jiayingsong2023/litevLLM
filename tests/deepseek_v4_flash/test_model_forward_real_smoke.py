import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

import pytest
import torch

from vllm.model_executor.model_loader import get_model
from vllm.serving.config_builder import build_vllm_config

MODEL = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)
REAL_SMOKE_TIMEOUT_SECONDS = 1800


@contextmanager
def _real_smoke_timeout_guard(seconds: int) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        pytest.skip("SIGALRM/setitimer unavailable for real-model smoke timeout")

    def _raise_timeout(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        raise TimeoutError(
            f"DeepSeek V4 Flash real smoke timed out after {seconds} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(
                signal.ITIMER_REAL,
                previous_timer[0],
                previous_timer[1],
            )


def test_real_smoke_timeout_guard_restores_previous_handler_and_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    previous_handler = object()

    def fake_signal(signum: object, handler: object) -> object:
        calls.append(("signal", signum, handler))
        return previous_handler

    def fake_setitimer(
        which: object,
        seconds: object,
        interval: object = 0.0,
    ) -> tuple[float, float]:
        calls.append(("setitimer", which, seconds, interval))
        return (12.0, 3.0)

    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr(signal, "setitimer", fake_setitimer)

    with _real_smoke_timeout_guard(REAL_SMOKE_TIMEOUT_SECONDS):
        pass

    assert calls[0] == ("signal", signal.SIGALRM, calls[0][2])
    assert calls[1] == ("setitimer", signal.ITIMER_REAL, 1800.0, 0.0)
    assert calls[2] == ("setitimer", signal.ITIMER_REAL, 0.0, 0.0)
    assert calls[3] == ("signal", signal.SIGALRM, previous_handler)
    assert calls[4] == ("setitimer", signal.ITIMER_REAL, 12.0, 3.0)


def test_real_smoke_timeout_guard_raises_timeout() -> None:
    with _real_smoke_timeout_guard(REAL_SMOKE_TIMEOUT_SECONDS):
        handler = signal.getsignal(signal.SIGALRM)
        assert callable(handler)
        with pytest.raises(
            TimeoutError,
            match="DeepSeek V4 Flash real smoke timed out",
        ):
            handler(signal.SIGALRM, None)


@pytest.mark.skipif(not MODEL.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_embedding_output_projection_smoke_returns_finite_logits() -> None:
    with _real_smoke_timeout_guard(REAL_SMOKE_TIMEOUT_SECONDS):
        model = get_model(build_vllm_config(str(MODEL), max_model_len=4096))
        try:
            logits = model.forward(torch.tensor([1], dtype=torch.long))

            assert model.limited_forward_smoke_only is True
            assert logits.shape == (1, model.shape.vocab_size)
            assert logits.dtype == torch.float32
            assert torch.isfinite(logits).all()
        finally:
            model.close()
