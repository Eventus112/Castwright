"""Synthesis poison responses retain a safe GPU OOM hint for Node's taxonomy."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

import main  # noqa: E402


REQUESTS = (
    ("/synthesize", {"engine": "qwen", "model": "1.7b", "voice": "narrator", "text": "Hello."}),
    ("/synthesize-batch", {
        "engine": "qwen", "model": "1.7b",
        "items": [{"voice": "narrator", "text": "Hello."}],
    }),
)
PRIVATE_DETAIL = " /home/private/book.txt secret manuscript text"
FENCE_DETAIL = (
    "Voice engine is in a poisoned CUDA state from a prior CUDA "
    "error and must be restarted. A fresh process is being "
    "respawned automatically; retry once /health responds again."
)


class _FailingEngine:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    def synthesize(self, model, voice, text, language=None):
        self.calls += 1
        raise RuntimeError(self.message)

    def synthesize_batch(self, model, items, live_instruct=False):
        self.calls += 1
        raise RuntimeError(self.message)


@pytest.fixture(autouse=True)
def _reset_poison(monkeypatch):
    main._reset_poison_for_test()
    monkeypatch.setattr(main, "_schedule_poison_exit", lambda: None)
    monkeypatch.setattr(main, "_restart_pending", False)
    monkeypatch.setattr(main, "_capacity_admission_enabled", lambda: False)
    yield
    main._reset_poison_for_test()


@pytest.mark.parametrize("endpoint,body", REQUESTS)
@pytest.mark.parametrize("message", (
    "CUDA out of memory. Tried to allocate 66.00 MiB (80.56 MiB free).",
    "CUDA error: out of memory",
    "out of memory while allocating on CUDA device 0",
    "HIP out of memory",
    "hip error: out of memory",
))
def test_synthesis_oom_returns_only_safe_cause(monkeypatch, endpoint, body, message):
    engine = _FailingEngine(message + PRIVATE_DETAIL)
    monkeypatch.setitem(main.ENGINES, "qwen", engine)
    response = TestClient(main.app).post(endpoint, json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": "CUDA out of memory.", "poisoned": True}
    assert engine.calls == 1
    assert main._process_poison_reason == message + PRIVATE_DETAIL
    assert main._poison_exit_scheduled is True


@pytest.mark.parametrize("endpoint,body", REQUESTS)
def test_synthesis_oom_fence_preserves_first_trigger(monkeypatch, endpoint, body):
    engine = _FailingEngine("must never run")
    monkeypatch.setitem(main.ENGINES, "qwen", engine)
    first_trigger = "CUDA out of memory." + PRIVATE_DETAIL
    main._mark_cuda_poisoned(first_trigger)
    main._mark_cuda_poisoned("CUDA error: device-side assert triggered")

    response = TestClient(main.app).post(endpoint, json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": "CUDA out of memory.", "poisoned": True}
    assert engine.calls == 0
    assert main._process_poison_reason == first_trigger


@pytest.mark.parametrize("endpoint,body", REQUESTS)
@pytest.mark.parametrize("message", (
    "CUDA error: device-side assert triggered",
    "HIP error: invalid device function",
))
def test_synthesis_non_oom_poison_keeps_generic_detail(monkeypatch, endpoint, body, message):
    engine = _FailingEngine(message + PRIVATE_DETAIL)
    monkeypatch.setitem(main.ENGINES, "qwen", engine)
    client = TestClient(main.app)

    response = client.post(endpoint, json=body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Internal error.", "poisoned": True}

    for fence_endpoint, fence_body in REQUESTS:
        response = client.post(fence_endpoint, json=fence_body)
        assert response.status_code == 503
        assert response.json() == {"detail": FENCE_DETAIL, "poisoned": True}
    assert engine.calls == 1


@pytest.mark.parametrize("endpoint,body", REQUESTS)
@pytest.mark.parametrize("message", ("CPU out of memory", "model load failed"))
def test_synthesis_unrelated_error_stays_generic(monkeypatch, endpoint, body, message):
    engine = _FailingEngine(message + PRIVATE_DETAIL)
    monkeypatch.setitem(main.ENGINES, "qwen", engine)
    response = TestClient(main.app).post(endpoint, json=body)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error."}
    assert main._process_poisoned is False
    assert main._poison_exit_scheduled is False
