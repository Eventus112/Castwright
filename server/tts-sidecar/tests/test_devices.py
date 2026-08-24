import asyncio, importlib, json, os, sys, threading, time, types
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main = importlib.import_module("main")

import pytest
from fastapi.testclient import TestClient

def _fake_torch():
    def props(i):
        return types.SimpleNamespace(name=["RTX 4070","RTX 5070 Ti"][i],
                                     total_memory=[8*10**9,16*10**9][i], uuid=f"GPU-{i}")
    cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2,
        get_device_properties=props,
        mem_get_info=lambda i: ([6*10**9,14*10**9][i], [8*10**9,16*10**9][i]))
    return types.SimpleNamespace(cuda=cuda)

def test_enumerate_cards():
    out = main._enumerate_cuda_devices(_fake_torch())
    assert [d["idx"] for d in out] == [0, 1]
    assert out[1] == {"uuid": "GPU-1", "idx": 1, "name": "RTX 5070 Ti", "total_mb": 16000, "free_mb": 14000}

def test_enumerate_empty_without_cuda():
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    assert main._enumerate_cuda_devices(fake) == []

def test_engine_actual_card_detects_cpu_fallback():
    # a fake engine that REQUESTED cuda:1 but actually resolved to cpu
    eng = types.SimpleNamespace(_requested_device="cuda:1", device="cpu", _model=object())
    card = main._engine_actual_card(eng)
    assert card["family"] == "cpu"
    assert card["fell_back"] is True

def test_engine_actual_card_none_when_unloaded():
    eng = types.SimpleNamespace(_requested_device="cuda:1", device="cpu", _model=None)
    assert main._engine_actual_card(eng) is None

def test_engine_actual_card_unknown_family_when_all_probes_fail():
    # model is a plain object(): no parameters(), no device/kokoro attrs on engine
    eng = types.SimpleNamespace(_requested_device="cuda:1", _model=object())
    card = main._engine_actual_card(eng)
    assert card is not None
    assert card["family"] == "unknown"
    assert card["index"] is None
    assert card["fell_back"] is False


# --- Kokoro ORT provider reconcile ---

def _fake_kokoro_cpu_session():
    """Kokoro engine: requested cuda:1 but ORT resolved to CPU-only providers."""
    sess = types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    kok = types.SimpleNamespace(sess=sess)
    # No device/_device/_model attrs — mirrors a real KokoroEngine
    return types.SimpleNamespace(_requested_device="cuda:1", _kokoro=kok)


def _fake_kokoro_cuda_session():
    """Kokoro engine: requested cuda:0, ORT kept the CUDA EP."""
    sess = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    kok = types.SimpleNamespace(sess=sess)
    return types.SimpleNamespace(_requested_device="cuda:0", _kokoro=kok)


def test_engine_actual_card_kokoro_cuda_to_cpu_provider_drop_flags_fell_back():
    # Regression: tier-2 _parse_device(None) returns "auto", not None, so the
    # tier-3 guard `if family is None:` was False and the Kokoro reconcile was
    # unreachable. Fix: guard must be `if family in (None, "auto"):`.
    card = main._engine_actual_card(_fake_kokoro_cpu_session())
    assert card["family"] == "cpu"
    assert card["index"] is None
    assert card["fell_back"] is True


def test_engine_actual_card_kokoro_cuda_resident_no_fallback():
    card = main._engine_actual_card(_fake_kokoro_cuda_session())
    assert card["family"] == "cuda"
    assert card["index"] is None
    assert card["fell_back"] is False


# --- _resident_engines_by_card + _build_gpus_payload (Task 9) ---

def test_resident_buckets_engines_by_card(monkeypatch):
    # ENGINES["qwen"] loaded on card 1; ASR fell back to cpu
    monkeypatch.setattr(main, "_engine_actual_card",
        lambda e: {"family": "cuda", "index": 1, "fell_back": False} if e is main.ENGINES["qwen"]
        else ({"family": "cpu", "index": None, "fell_back": True} if e is main.ASR else None))
    by_card = main._resident_engines_by_card([{"idx": 0}, {"idx": 1}])
    assert {"engine": "qwen", "actual_card": 1} in by_card[1]
    # a fell_back engine is recorded with stale_reason (card key is the cpu bucket convention)
    flat = [r for v in by_card.values() for r in v]
    assert any(r.get("stale_reason") == "cpu_fallback" and r["engine"] == "asr" for r in flat)


def test_build_gpus_payload_merges(monkeypatch):
    monkeypatch.setattr(main, "_enumerate_cuda_devices", lambda tm=None: [{"uuid":"GPU-1","idx":1,"name":"x","total_mb":16000,"free_mb":14000}])
    monkeypatch.setattr(main, "_resident_engines_by_card", lambda cards: {1: [{"engine":"qwen","actual_card":1}]})
    out = main._build_gpus_payload(_fake_torch())
    assert out[0]["resident"] == [{"engine": "qwen", "actual_card": 1}]
    assert "torch_reserved_mb" in out[0]


def test_health_gpus_field_additive(monkeypatch):
    """gpus key appears in /health and pre-existing keys are byte-for-byte unchanged."""
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main, "_build_gpus_payload", lambda torch_module=None: [])
    client = TestClient(main.app)
    body = client.get("/health").json()
    assert "gpus" in body
    # Additive contract: none of the pre-existing keys were removed or renamed
    assert "devices" in body
    assert "asr_device" in body
    assert "spk_device" in body


# ── final-review fixes: loaded Coqui visibility + unindexed cpu_fallback ──

def test_engine_actual_card_detects_loaded_coqui_via_tts():
    """A loaded Coqui keeps its model in `_tts` (not _model/_kokoro/_base) and its
    ACTUAL device in `_resolved_device` (the pref lives in `_device`). A Coqui that
    requested cuda but resolved cpu must be visible (not None) and flagged fell_back."""
    coqui = types.SimpleNamespace(
        _requested_device="cuda:1", _tts=object(), _resolved_device="cpu", _device="cuda:1"
    )
    card = main._engine_actual_card(coqui)
    assert card is not None  # was None before the _tts lookup fix
    assert card["family"] == "cpu"  # resolved device, not the cuda pref
    assert card["fell_back"] is True


def test_engine_actual_card_loaded_coqui_on_cuda_no_fallback():
    coqui = types.SimpleNamespace(
        _requested_device="cuda:1", _tts=object(), _resolved_device="cuda:1", _device="cuda:1"
    )
    card = main._engine_actual_card(coqui)
    assert card is not None
    assert card["family"] == "cuda"
    assert card["fell_back"] is False


def test_build_gpus_payload_surfaces_unindexed_cpu_fallback(monkeypatch):
    """An ORT/CT2 (or cpu-fallen) engine has index=None → the -1 bucket, which no
    real card claims. _build_gpus_payload must surface it as a synthetic entry so the
    cpu_fallback is visible in gpus[] rather than silently dropped."""
    monkeypatch.setattr(main, "_enumerate_cuda_devices",
        lambda tm=None: [{"uuid": "GPU-1", "idx": 1, "name": "x", "total_mb": 16000, "free_mb": 14000}])
    monkeypatch.setattr(main, "_resident_engines_by_card", lambda cards: {
        1: [{"engine": "qwen", "actual_card": 1}],
        -1: [{"engine": "kokoro", "actual_card": None, "stale_reason": "cpu_fallback"}],
    })
    out = main._build_gpus_payload(_fake_torch())
    unindexed = [c for c in out if c["idx"] == -1]
    assert len(unindexed) == 1
    assert {"engine": "kokoro", "actual_card": None, "stale_reason": "cpu_fallback"} in unindexed[0]["resident"]


def test_build_gpus_payload_no_unindexed_entry_when_bucket_empty(monkeypatch):
    """A fully-indexed box (no -1 bucket) sees no synthetic entry — additive only."""
    monkeypatch.setattr(main, "_enumerate_cuda_devices",
        lambda tm=None: [{"uuid": "GPU-1", "idx": 1, "name": "x", "total_mb": 16000, "free_mb": 14000}])
    monkeypatch.setattr(main, "_resident_engines_by_card", lambda cards: {1: [{"engine": "qwen", "actual_card": 1}]})
    out = main._build_gpus_payload(_fake_torch())
    assert [c for c in out if c["idx"] == -1] == []


def test_build_gpus_payload_includes_per_card_ceilings(monkeypatch):
    monkeypatch.setenv("SIDECAR_VRAM_FREE_FLOOR_MB", "1024")
    monkeypatch.setattr(main, "_enumerate_cuda_devices",
        lambda tm=None: [{"uuid": "GPU-1", "idx": 1, "name": "x", "total_mb": 16000, "free_mb": 14000}])
    monkeypatch.setattr(main, "_resident_engines_by_card", lambda cards: {})
    out = main._build_gpus_payload(_fake_torch())
    assert out[0]["free_floor_mb"] == 1024.0
    assert out[0]["reserved_ceiling_mb"] == main._VRAM_HARD_FRACTION * 16000


def test_build_gpus_payload_unindexed_row_also_carries_free_floor(monkeypatch):
    """The synthetic -1 (unindexed cpu/ORT/CT2) row must carry free_floor_mb
    too — it's a global absolute, not per-card, and this bucket IS the
    Kokoro/Whisper CPU-fallback engines the floor exists to protect. A
    consumer that expects every gpus[] row to carry the field (e.g. a
    staleness display) must not see it silently missing here."""
    monkeypatch.setenv("SIDECAR_VRAM_FREE_FLOOR_MB", "2048")
    monkeypatch.setattr(main, "_enumerate_cuda_devices",
        lambda tm=None: [{"uuid": "GPU-1", "idx": 1, "name": "x", "total_mb": 16000, "free_mb": 14000}])
    monkeypatch.setattr(main, "_resident_engines_by_card", lambda cards: {
        -1: [{"engine": "kokoro", "actual_card": None, "stale_reason": "cpu_fallback"}],
    })
    out = main._build_gpus_payload(_fake_torch())
    unindexed = [c for c in out if c["idx"] == -1]
    assert len(unindexed) == 1
    assert unindexed[0]["free_floor_mb"] == 2048.0
    assert unindexed[0]["reserved_ceiling_mb"] == 0.0


# --- /synthesize capacity admission (SEG_CAPACITY_ADMISSION, task 4) ---


class _FakeSynthEngine(main.CoquiEngine):
    """Minimal Coqui stand-in so /synthesize never touches the real
    multi-gigabyte XTTS model — same shape as test_smoke.py's _FakeEngine,
    duplicated locally to keep this file's admission tests self-contained."""

    name = "coqui"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, model, voice, text, language=None):
        self.calls.append((model, voice, text))
        return main.SynthResult(pcm=b"\x00\x00", sample_rate=24000, substituted_from=None)


@pytest.fixture
def synth_client(monkeypatch):
    fake = _FakeSynthEngine()
    monkeypatch.setitem(main.ENGINES, "coqui", fake)
    # Drop the real Kokoro engine so TestClient's startup event doesn't try
    # to eager-preload it (mirrors test_smoke.py's `client` fixture).
    monkeypatch.delitem(main.ENGINES, "kokoro", raising=False)
    main._reset_poison_for_test()
    with TestClient(main.app) as c:
        yield c
    main._reset_poison_for_test()


def _synth_body():
    return {"engine": "coqui", "model": "xtts_v2", "voice": "v", "text": "hi"}


def test_synthesize_flag_off_ignores_full_device(monkeypatch, synth_client):
    """Explicit opt-out (SEG_CAPACITY_ADMISSION=0): /synthesize behaves exactly
    as the pre-admission build — no admission check runs at all — even when the
    probe reports a device with no free room. This is the rollback path."""
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "0")
    monkeypatch.setattr(
        main._placement,
        "probe",
        lambda: [{"kind": "cuda", "index": 0, "label": "g", "totalMb": 8192, "freeMb": 50}],
    )
    r = synth_client.post("/synthesize", json=_synth_body())
    assert r.status_code == 200
    assert r.content == b"\x00\x00"


def test_capacity_admission_default_on(monkeypatch):
    """Regression for the #1720 flag flip: capacity admission is ON by default
    (unset), stays ON for any value that isn't the explicit "0" opt-out, and
    only "0" turns it off (the rollback path)."""
    monkeypatch.delenv("SEG_CAPACITY_ADMISSION", raising=False)
    assert main._capacity_admission_enabled() is True
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    assert main._capacity_admission_enabled() is True
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "0")
    assert main._capacity_admission_enabled() is False


def test_synthesize_flag_on_no_capacity_returns_503(monkeypatch, synth_client):
    """Flag ON + a probe that can't fit the peak -> 503 noCapacity, engine
    never called."""
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setattr(
        main._placement,
        "probe",
        lambda: [{"kind": "cuda", "index": 0, "label": "g", "totalMb": 8192, "freeMb": 50}],
    )
    r = synth_client.post("/synthesize", json=_synth_body())
    assert r.status_code == 503
    body = r.json()
    assert body["noCapacity"] is True
    assert body["neededMb"] > 0
    assert body["deviceKey"] == "cuda:0"


def test_synthesize_flag_on_fits_proceeds(monkeypatch, synth_client):
    """Flag ON + a roomy probe -> the reservation admits and /synthesize
    proceeds as normal."""
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setattr(
        main._placement,
        "probe",
        lambda: [{"kind": "cuda", "index": 0, "label": "g", "totalMb": 16384, "freeMb": 16000}],
    )
    r = synth_client.post("/synthesize", json=_synth_body())
    assert r.status_code == 200
    assert r.content == b"\x00\x00"


def test_loaded_qwen_17b_synthesize_admits_without_self_eviction(monkeypatch):
    """The factual /load -> /synthesize repro keeps the loaded model warm.

    The cold 6144 MB footprint does not fit this simulated card while the
    resident model is present.  Before the regression fix, admission therefore
    evicted ``qwen.base17``, admitted against the newly freed VRAM, and made the
    same request reload the model.  The request still returned 200, masking the
    churn unless unload/reload are asserted explicitly.
    """

    class FakeResidentQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._device = "cuda:0"
            self._base17 = None
            self.base17_loads = 0
            self.base17_unloads = 0

        def _ensure_base17_loaded(self, device=None):
            if self._base17 is None:
                self._base17 = object()
                self.base17_loads += 1

        def maybe_free_idle_base17(self, ttl_seconds):
            if self._base17 is None or self._base17_in_flight.busy:
                return False
            self._base17 = None
            self.base17_unloads += 1
            return True

        def synthesize(self, model, voice, text, language=None):
            assert model == "1.7b"
            assert self._base17_residency_leases == 1
            self._ensure_base17_loaded()
            return main.SynthResult(pcm=b"\x00\x00", sample_rate=24000, substituted_from=None)

    qwen = FakeResidentQwen()
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.delitem(main.ENGINES, "kokoro", raising=False)
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setenv("PRELOAD_QWEN", "0")
    monkeypatch.setenv("PRELOAD_QWEN_BASE17", "0")
    monkeypatch.setattr(main._placement, "footprints", main.FootprintTable())
    monkeypatch.setattr(main._placement, "ledger", main.ReservationLedger())
    monkeypatch.setattr(
        main._placement,
        "probe",
        lambda: [{
            "kind": "cuda",
            "index": 0,
            "label": "g",
            "totalMb": 12288,
            "freeMb": 2000 if qwen._base17 is not None else 9000,
        }],
    )
    main._reset_poison_for_test()

    with TestClient(main.app) as client:
        loaded = client.post("/load", json={"engine": "qwen", "model": "1.7b"})
        response = client.post(
            "/synthesize",
            json={"engine": "qwen", "model": "1.7b", "voice": "v", "text": "hi"},
        )

    assert loaded.status_code == 200
    assert loaded.json() == {"status": "ready"}
    assert response.status_code == 200
    assert response.content == b"\x00\x00"
    assert qwen.base17_loads == 1
    assert qwen.base17_unloads == 0
    main._reset_poison_for_test()


def test_plain_qwen_17b_lease_closes_admission_to_activity_unload_race(monkeypatch):
    class FakeQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._device = "cuda:0"
            self._base17 = object()

        def synthesize(self, model, voice, text, language=None):
            assert self._base17 is not None
            assert self._base17_residency_leases == 1
            with self._base17_activity():
                return main.SynthResult(
                    pcm=b"\x00\x00", sample_rate=24000, substituted_from=None
                )

    qwen = FakeQwen()
    unload_started = threading.Event()
    unload_threads = []

    def unload_during_admission():
        unload_started.set()
        qwen.unload_base17()

    @asynccontextmanager
    async def racing_reservation(*args, **kwargs):
        assert qwen._base17_residency_leases == 1
        assert qwen.maybe_free_idle_base17(0.0) is False
        thread = threading.Thread(target=unload_during_admission)
        unload_threads.append(thread)
        thread.start()
        assert unload_started.wait(1.0)
        thread.join(0.05)
        assert thread.is_alive()
        assert qwen._base17 is not None
        yield {"device": "cuda:0"}

    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.delitem(main.ENGINES, "kokoro", raising=False)
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setenv("PRELOAD_QWEN", "0")
    monkeypatch.setenv("PRELOAD_QWEN_BASE17", "0")
    monkeypatch.setattr(main._placement, "reservation", racing_reservation)
    monkeypatch.setattr(main, "_reclaim_host_and_vram", lambda: None)
    main._reset_poison_for_test()

    with TestClient(main.app) as client:
        response = client.post(
            "/synthesize",
            json={"engine": "qwen", "model": "1.7b", "voice": "v", "text": "hi"},
        )

    assert response.status_code == 200
    assert response.content == b"\x00\x00"
    assert len(unload_threads) == 1
    unload_threads[0].join(1.0)
    assert not unload_threads[0].is_alive()
    assert qwen._base17 is None
    main._reset_poison_for_test()


def test_base17_lease_wait_keeps_health_and_event_loop_responsive(monkeypatch):
    qwen = main.QwenEngine()
    qwen._device = "cuda:0"
    qwen._base17 = object()
    events = []

    with qwen._base17_lease_cv:
        qwen._base17_unloading = True

    def finish_unload():
        time.sleep(0.15)
        events.append("unload-finished")
        with qwen._base17_lease_cv:
            qwen._base17_unloading = False
            qwen._base17_lease_cv.notify_all()

    thread = threading.Thread(target=finish_unload)
    thread.start()
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.delitem(main.ENGINES, "kokoro", raising=False)

    async def body():
        async def wait_for_lease():
            async with qwen._base17_residency_lease():
                events.append("lease-entered")

        lease_task = asyncio.create_task(wait_for_lease())
        await asyncio.sleep(0.02)
        health = main.health()
        events.append("health-returned")
        assert health["ok"] is True
        await lease_task

    asyncio.run(body())
    thread.join(1.0)
    assert events.index("health-returned") < events.index("unload-finished")
    assert events[-1] == "lease-entered"


def test_base17_unload_writer_excludes_new_leases_while_waiting_and_reclaiming(
    monkeypatch,
):
    qwen = main.QwenEngine()
    qwen._base17 = object()
    assert qwen._try_acquire_base17_residency_lease() == "leased"

    reclaim_started = threading.Event()
    reclaim_release = threading.Event()

    def blocking_reclaim():
        reclaim_started.set()
        assert reclaim_release.wait(1.0)

    monkeypatch.setattr(main, "_reclaim_host_and_vram", blocking_reclaim)
    unload_thread = threading.Thread(target=qwen.unload_base17)
    unload_thread.start()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with qwen._base17_lease_cv:
            if qwen._base17_unloading:
                break
        time.sleep(0.001)
    else:
        pytest.fail("unloader did not close new Base17 leases")

    with qwen._base17_unload_guard(wait=False) as allowed:
        assert allowed is False

    def forbidden_to_thread(*args, **kwargs):
        raise AssertionError("Base17 lease wait must not consume an executor thread")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)

    async def exercise_burst_waiters():
        async def acquire_after_unload():
            async with qwen._base17_residency_lease() as leased:
                return leased

        waiters = [asyncio.create_task(acquire_after_unload()) for _ in range(64)]
        await asyncio.sleep(0.03)
        assert not any(task.done() for task in waiters)

        # This direct release is a bounded lock/counter operation. The writer
        # proceeds, but all burst waiters must keep yielding while its unload
        # body is active instead of occupying the default executor.
        qwen._release_base17_residency_lease()
        deadline = asyncio.get_running_loop().time() + 1.0
        while not reclaim_started.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)
        assert not any(task.done() for task in waiters)

        reclaim_release.set()
        assert await asyncio.gather(*waiters) == [False] * len(waiters)

    asyncio.run(exercise_burst_waiters())
    unload_thread.join(1.0)
    assert not unload_thread.is_alive()
    with qwen._base17_lease_cv:
        assert qwen._base17_unloading is False
        assert qwen._base17_unload_waiters == 0


def test_base17_concurrent_unloaders_serialize_the_unload_body(monkeypatch):
    qwen = main.QwenEngine()
    qwen._base17 = object()
    first_reclaim_started = threading.Event()
    first_reclaim_release = threading.Event()
    state_lock = threading.Lock()
    active_reclaims = 0
    max_active_reclaims = 0
    reclaim_calls = 0

    def recording_reclaim():
        nonlocal active_reclaims, max_active_reclaims, reclaim_calls
        with state_lock:
            reclaim_calls += 1
            call = reclaim_calls
            active_reclaims += 1
            max_active_reclaims = max(max_active_reclaims, active_reclaims)
        try:
            if call == 1:
                first_reclaim_started.set()
                assert first_reclaim_release.wait(1.0)
        finally:
            with state_lock:
                active_reclaims -= 1

    monkeypatch.setattr(main, "_reclaim_host_and_vram", recording_reclaim)
    first = threading.Thread(target=qwen.unload_base17)
    second = threading.Thread(target=qwen.unload_base17)
    first.start()
    assert first_reclaim_started.wait(1.0)
    second.start()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with qwen._base17_lease_cv:
            if qwen._base17_unload_waiters == 1:
                break
        time.sleep(0.001)
    else:
        pytest.fail("second unloader did not wait for the active unloader")
    with state_lock:
        assert reclaim_calls == 1
        assert max_active_reclaims == 1

    first_reclaim_release.set()
    first.join(1.0)
    second.join(1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert reclaim_calls == 2
    assert max_active_reclaims == 1
    with qwen._base17_lease_cv:
        assert qwen._base17_unloading is False
        assert qwen._base17_unload_waiters == 0


def test_wait_task_ignoring_cancellation_terminates_for_cancelled_child():
    async def body():
        child = asyncio.create_task(asyncio.sleep(10.0))
        child.cancel()
        await asyncio.sleep(0)
        assert child.cancelled()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                main._wait_task_ignoring_cancellation(child), timeout=0.2
            )

    asyncio.run(body())


def test_cancelled_base17_lease_releases_synchronously_without_executor(monkeypatch):
    qwen = main.QwenEngine()
    qwen._base17 = object()

    def forbidden_to_thread(*args, **kwargs):
        raise AssertionError("Base17 lease cleanup must not use the executor")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)

    async def body():
        entered = asyncio.Event()
        stay_open = asyncio.Event()

        async def holder():
            async with qwen._base17_residency_lease() as leased:
                assert leased is True
                entered.set()
                await stay_open.wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        assert qwen._base17_residency_leases == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert qwen._base17_residency_leases == 0

    asyncio.run(body())
    with qwen._base17_unload_guard(wait=False) as allowed:
        assert allowed is True


def test_cancelled_synth_holds_reservation_and_lease_until_worker_finishes(monkeypatch):
    worker_started = threading.Event()
    worker_release = threading.Event()
    worker_finished = threading.Event()
    reservation_active = False

    class FakeRequest:
        async def body(self):
            return json.dumps({
                "engine": "qwen",
                "model": "1.7b",
                "voice": "v",
                "text": "hi",
            }).encode()

    class FakeQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._device = "cuda:0"
            self._base17 = object()

        def synthesize(self, model, voice, text, language=None):
            worker_started.set()
            worker_release.wait()
            worker_finished.set()
            return main.SynthResult(
                pcm=b"\x00\x00", sample_rate=24000, substituted_from=None
            )

    qwen = FakeQwen()

    @asynccontextmanager
    async def recording_reservation(*args, **kwargs):
        nonlocal reservation_active
        reservation_active = True
        try:
            yield {"device": "cuda:0"}
        finally:
            reservation_active = False

    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setattr(main._placement, "reservation", recording_reservation)
    main._reset_poison_for_test()
    main._reset_restart_state_for_test()

    async def body():
        task = asyncio.create_task(main.synthesize(FakeRequest()))
        assert await asyncio.to_thread(worker_started.wait, 1.0)
        assert reservation_active is True
        assert qwen._base17_residency_leases == 1

        task.cancel()
        await asyncio.sleep(0.02)
        assert task.done() is False
        assert reservation_active is True
        assert qwen._base17_residency_leases == 1

        worker_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker_finished.is_set()
        assert reservation_active is False
        assert qwen._base17_residency_leases == 0

    asyncio.run(body())
    main._reset_poison_for_test()


def test_qwen_base17_unload_route_burst_uses_one_executor_writer(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    synth_started = threading.Event()
    synth_release = threading.Event()
    unload_started = threading.Event()
    unload_calls = 0
    count_lock = threading.Lock()

    class FakeQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._device = "cuda:0"
            self._base17 = object()

        def synthesize(self, model, voice, text, language=None):
            synth_started.set()
            assert synth_release.wait(1.0)
            return main.SynthResult(
                pcm=b"\x00\x00", sample_rate=24000, substituted_from=None
            )

        def unload_base17(self):
            nonlocal unload_calls
            with count_lock:
                unload_calls += 1
            unload_started.set()
            return super().unload_base17()

    class SynthRequest:
        async def body(self):
            return json.dumps({
                "engine": "qwen",
                "model": "1.7b",
                "voice": "v",
                "text": "hi",
            }).encode()

    class UnloadRequest:
        async def body(self):
            return b'{"engine":"qwen","model":"1.7b"}'

    qwen = FakeQwen()

    @asynccontextmanager
    async def reservation(*args, **kwargs):
        yield {"device": "cuda:0"}

    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setattr(main._placement, "reservation", reservation)
    monkeypatch.setattr(main, "_reclaim_host_and_vram", lambda: None)
    main._reset_poison_for_test()
    main._reset_restart_state_for_test()

    async def wait_until_set(event):
        deadline = asyncio.get_running_loop().time() + 1.0
        while not event.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)

    executor = ThreadPoolExecutor(max_workers=2)

    async def burst():
        asyncio.get_running_loop().set_default_executor(executor)
        synth_task = asyncio.create_task(main.synthesize(SynthRequest()))
        await wait_until_set(synth_started)
        assert qwen._base17_residency_leases == 1

        unload_tasks = [
            asyncio.create_task(main.unload_model(UnloadRequest()))
            for _ in range(32)
        ]
        await wait_until_set(unload_started)
        await asyncio.sleep(0.03)
        with count_lock:
            assert unload_calls == 1
        assert not any(task.done() for task in unload_tasks)

        # One executor worker is waiting in unload_base17; the other synth
        # worker can still finish, after which the route releases its lease
        # synchronously and lets that single unload writer proceed.
        synth_release.set()
        synth_response = await synth_task
        assert synth_response.status_code == 200
        responses = await asyncio.gather(*unload_tasks)
        assert all(response.status_code == 200 for response in responses)

    asyncio.run(burst())
    executor.shutdown(wait=True)
    assert qwen._base17 is None
    assert qwen._base17_residency_leases == 0
    with count_lock:
        assert unload_calls == 1

    # The lazy route lock must be recreated rather than reused across the
    # separate loops common in these tests.
    qwen._base17 = object()
    response = asyncio.run(main.unload_model(UnloadRequest()))
    assert response.status_code == 200
    with count_lock:
        assert unload_calls == 2
    main._reset_poison_for_test()


def test_cancelled_qwen_base17_unload_leader_holds_mutex_until_worker_finishes(
    monkeypatch,
):
    unload_started = threading.Event()
    unload_calls = 0
    count_lock = threading.Lock()

    class FakeQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._base17 = object()

        def unload_base17(self):
            nonlocal unload_calls
            with count_lock:
                unload_calls += 1
            unload_started.set()
            return super().unload_base17()

    class UnloadRequest:
        async def body(self):
            return b'{"engine":"qwen","model":"1.7b"}'

    qwen = FakeQwen()
    assert qwen._try_acquire_base17_residency_lease() == "leased"
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.setattr(main, "_reclaim_host_and_vram", lambda: None)

    async def wait_until_started():
        deadline = asyncio.get_running_loop().time() + 1.0
        while not unload_started.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)

    async def body():
        leader = asyncio.create_task(main.unload_model(UnloadRequest()))
        await wait_until_started()
        follower = asyncio.create_task(main.unload_model(UnloadRequest()))
        await asyncio.sleep(0)

        leader.cancel()
        await asyncio.sleep(0.03)
        assert leader.done() is False
        assert follower.done() is False
        assert qwen._base17_route_unload_lock is not None
        assert qwen._base17_route_unload_lock.locked()
        with count_lock:
            assert unload_calls == 1

        qwen._release_base17_residency_lease()
        with pytest.raises(asyncio.CancelledError):
            await leader
        response = await follower
        assert response.status_code == 200

    asyncio.run(body())
    assert qwen._base17 is None
    assert qwen._base17_residency_leases == 0
    with count_lock:
        assert unload_calls == 1


def test_resident_qwen_17b_still_refuses_when_warm_footprint_cannot_fit(monkeypatch):
    """Warm admission remains capacity-safe instead of self-evicting to fit."""

    class FakeResidentQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._device = "cuda:0"
            self._base17 = object()
            self.base17_unloads = 0
            self.synth_calls = 0

        def maybe_free_idle_base17(self, ttl_seconds):
            self._base17 = None
            self.base17_unloads += 1
            return True

        def synthesize(self, model, voice, text, language=None):
            self.synth_calls += 1
            return main.SynthResult(pcm=b"\x00\x00", sample_rate=24000, substituted_from=None)

    qwen = FakeResidentQwen()
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.delitem(main.ENGINES, "kokoro", raising=False)
    monkeypatch.setenv("SEG_CAPACITY_ADMISSION", "1")
    monkeypatch.setenv("PRELOAD_QWEN", "0")
    monkeypatch.setenv("PRELOAD_QWEN_BASE17", "0")
    monkeypatch.setattr(main._placement, "footprints", main.FootprintTable())
    monkeypatch.setattr(main._placement, "ledger", main.ReservationLedger())
    monkeypatch.setattr(
        main._placement,
        "probe",
        lambda: [{"kind": "cuda", "index": 0, "label": "g", "totalMb": 12288, "freeMb": 600}],
    )
    main._reset_poison_for_test()

    with TestClient(main.app) as client:
        response = client.post(
            "/synthesize",
            json={"engine": "qwen", "model": "1.7b", "voice": "v", "text": "hi"},
        )

    assert response.status_code == 503
    assert response.json()["neededMb"] == main.SEED_FOOTPRINTS_MB["qwen.1.7b.warm"]
    assert qwen.base17_unloads == 0
    assert qwen.synth_calls == 0
    main._reset_poison_for_test()


def test_qwen_design_keeps_engine_residency_and_can_evict_base17(monkeypatch):
    """Design keeps pre-patch engine residency and may evict Base17."""

    class FakeQwen(main.QwenEngine):
        def __init__(self):
            super().__init__()
            self._device = "cuda:0"
            self._base17 = object()
            self._design = None
            self.base17_unloads = 0

        def maybe_free_idle_base17(self, ttl_seconds):
            self._base17 = None
            self.base17_unloads += 1
            return True

    qwen = FakeQwen()
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.delitem(main.ENGINES, "coqui", raising=False)

    assert main._is_model_resident(
        "qwen", "1.7b", {"op": "design"}
    ) == main._is_resident("qwen")
    assert main._is_model_resident(
        "qwen", "1.7b", {"op": "mint"}
    ) == main._is_resident("qwen")

    pc = main.PlacementController(
        probe=lambda: [{
            "kind": "cuda",
            "index": 0,
            "label": "g",
            "totalMb": 12288,
            "freeMb": 2000 if qwen._base17 is not None else 9000,
        }],
        footprints=main.FootprintTable(),
        ledger=main.ReservationLedger(),
        reserve_mb=lambda: 500,
        idle_evict_steps=main._idle_evict_steps,
        is_resident=main._is_resident,
        is_model_resident=main._is_model_resident,
        reclaim=lambda device_key: None,
    )

    async def reserve_design():
        async with pc.reservation(
            "qwen", "1.7b", {"op": "design"}, cpu_capable=False, heavy=True
        ) as admission:
            assert admission == {"device": "cuda:0"}

    asyncio.run(reserve_design())
    assert qwen.base17_unloads == 1


def test_qwen_model_residency_uses_the_selected_wrappers_card(monkeypatch):
    class FakeDevice:
        def __init__(self, index):
            self.type = "cuda"
            self.index = index

    class FakeModule:
        def __init__(self, index):
            self._parameter = types.SimpleNamespace(device=FakeDevice(index))

        def parameters(self):
            return iter([self._parameter])

    qwen = main.QwenEngine()
    qwen._device = "cuda:0"
    qwen._base = FakeModule(0)
    qwen._base17 = FakeModule(1)
    qwen._design = FakeModule(2)
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)

    assert main._is_model_resident("qwen", "0.6b", {}) == "cuda:0"
    assert main._is_model_resident("qwen", "1.7b", {}) == "cuda:1"
    assert main._is_model_resident("qwen", "1.7b", {"op": "mint"}) == "cuda:0"
    assert main._is_model_resident("qwen", "1.7b", {"op": "design"}) == "cuda:0"


# --- device-targeted idle_evict (#1721) ---


def test_same_card_normalises_and_rejects_off_card():
    """`_same_card` compares two concrete device strings by CUDA index only.
    Unindexed "cuda" is card 0 (torch default); cpu/mps/auto/None never match a
    GPU target — an engine off the target card holds no VRAM there to free."""
    assert main._same_card("cuda:1", "cuda:1") is True
    assert main._same_card("cuda", "cuda:0") is True  # unindexed -> card 0
    assert main._same_card("cuda:0", "cuda:1") is False
    assert main._same_card("cpu", "cuda:0") is False
    assert main._same_card("auto", "cuda:0") is False
    assert main._same_card(None, "cuda:0") is False


class _FakeEvictQwen(main.QwenEngine):
    """QwenEngine stand-in whose idle-evict methods just record they were
    called (real QwenEngine() construction is cheap — no model I/O). Both the
    VoiceDesign and 1.7B-Base models live on the shared `_device`, so one card
    string gates both."""

    name = "qwen"

    def __init__(self, device):
        super().__init__()
        self._device = device
        self._design = object()
        self._base17 = object()
        self.design_freed = 0
        self.base17_freed = 0

    def maybe_free_idle_design(self, ttl_seconds):
        self.design_freed += 1
        self._design = None
        return True

    def maybe_free_idle_base17(self, ttl_seconds):
        self.base17_freed += 1
        self._base17 = None
        return True


class _FakeEvictSingleton:
    """ASR / SPK stand-in: carries a resolved device attr (name differs — ASR
    uses `_device`, SPK uses `device`) and records `maybe_free_idle` calls."""

    def __init__(self, device_attr, device_val):
        setattr(self, device_attr, device_val)
        self.freed = 0

    def maybe_free_idle(self, ttl_seconds):
        self.freed += 1
        return True


class _FakeEvictCoqui(main.CoquiEngine):
    """CoquiEngine stand-in whose `maybe_free_idle` just records it was
    called (real CoquiEngine() construction is cheap — no model I/O). Must
    subclass CoquiEngine, not just duck-type its surface: `_idle_evict_steps`'s
    Coqui branch is `isinstance`-guarded (#1894 review) the same way its Qwen
    sibling already was."""

    def __init__(self, device):
        super().__init__()
        self._device = device
        self.freed = 0
        self.ttls = []

    def maybe_free_idle(self, ttl_seconds):
        self.freed += 1
        self.ttls.append(ttl_seconds)
        return True


def _wire_evict_engines(monkeypatch, qwen_dev, asr_dev, spk_dev, coqui_dev="cpu"):
    qwen = _FakeEvictQwen(qwen_dev)
    asr = _FakeEvictSingleton("_device", asr_dev)
    spk = _FakeEvictSingleton("device", spk_dev)
    coqui = _FakeEvictCoqui(coqui_dev)
    monkeypatch.setitem(main.ENGINES, "qwen", qwen)
    monkeypatch.setitem(main.ENGINES, "coqui", coqui)
    monkeypatch.setattr(main, "ASR", asr)
    monkeypatch.setattr(main, "SPK", spk)
    return qwen, asr, spk, coqui


def _run(device_key, engine):
    """Build the steps, assert on their names, then RUN them — the builder
    never invokes the lambdas, so the existing per-engine call counters and
    the TTL assertion only mean anything if we drive them here."""
    steps = main._idle_evict_steps(device_key, engine)
    names = [s.name for s in steps]
    freed = any([s.run() for s in steps])  # list, not generator — no short-circuit
    return names, freed


def test_idle_evict_only_frees_engines_on_target_card(monkeypatch):
    """Qwen on cuda:0, ASR on cuda:1, SPK on cuda:0. Admitting onto cuda:0 must
    free Qwen (design + base17) and SPK, but NOT ASR (different card) — the
    multi-GPU over-eviction #1721 fixes."""
    qwen, asr, spk, coqui = _wire_evict_engines(monkeypatch, "cuda:0", "cuda:1", "cuda:0")
    names, freed = _run("cuda:0", "qwen")
    assert "qwen.design" in names and "qwen.base17" in names
    assert "spk" in names
    assert "asr" not in names  # resident on the OTHER card — no step built
    assert freed is True
    assert (qwen.design_freed, qwen.base17_freed) == (1, 1)
    assert spk.freed == 1
    assert asr.freed == 0  # resident on the OTHER card — left alone


def test_idle_evict_targets_the_other_card(monkeypatch):
    """Same layout, admitting onto cuda:1 instead: only ASR (the cuda:1
    resident) is freed; Qwen + SPK on cuda:0 are untouched."""
    qwen, asr, spk, coqui = _wire_evict_engines(monkeypatch, "cuda:0", "cuda:1", "cuda:0")
    names, freed = _run("cuda:1", "qwen")
    assert names == ["asr"]
    assert freed is True
    assert asr.freed == 1
    assert (qwen.design_freed, qwen.base17_freed) == (0, 0)
    assert spk.freed == 0


def test_idle_evict_skips_cpu_resident_engines(monkeypatch):
    """An engine resident on cpu holds no VRAM on any GPU, so a GPU admission
    never evicts it (evicting would just cost a needless reload). ASR defaults
    to cpu — admitting onto cuda:0 must leave it alone."""
    qwen, asr, spk, coqui = _wire_evict_engines(monkeypatch, "cuda:0", "cpu", "cpu")
    names, freed = _run("cuda:0", "qwen")
    assert "qwen.design" in names and "qwen.base17" in names  # qwen still freed
    assert "asr" not in names
    assert "spk" not in names
    assert freed is True
    assert asr.freed == 0
    assert spk.freed == 0


def test_idle_evict_unindexed_cuda_is_card_zero(monkeypatch):
    """A resident engine reporting bare "cuda" (no index) normalises to card 0,
    so it's freed for a cuda:0 admission and spared for cuda:1."""
    qwen, asr, spk, coqui = _wire_evict_engines(monkeypatch, "cuda", "cuda", "cuda")
    names, freed = _run("cuda:1", "qwen")
    assert names == []  # nothing on card 1
    assert freed is False
    assert (qwen.design_freed, spk.freed, asr.freed) == (0, 0, 0)
    names, freed = _run("cuda:0", "qwen")
    assert freed is True
    assert (qwen.design_freed, qwen.base17_freed, asr.freed, spk.freed) == (1, 1, 1, 1)


def test_idle_evict_frees_an_idle_coqui_on_the_target_card(monkeypatch):
    """#1894 — a starved non-Coqui op reclaims a resident, idle XTTS."""
    _q, _a, _s, coqui = _wire_evict_engines(monkeypatch, "cpu", "cpu", "cpu", coqui_dev="cuda:0")
    names, freed = _run("cuda:0", "qwen")
    assert "coqui" in names
    assert freed is True
    assert coqui.freed == 1
    assert coqui.ttls == [main._coqui_idle_ttl()]


def test_idle_evict_skips_coqui_on_another_card(monkeypatch):
    _q, _a, _s, coqui = _wire_evict_engines(monkeypatch, "cpu", "cpu", "cpu", coqui_dev="cuda:1")
    names, freed = _run("cuda:0", "qwen")
    assert names == []
    assert freed is False
    assert coqui.freed == 0


def test_idle_evict_never_evicts_coqui_for_a_coqui_op(monkeypatch):
    """Coqui is a PRIMARY engine, unlike the transient models beside it here:
    evicting it for a starved Coqui op would unload the very model that op is
    about to reload. Admission gives a resident engine no free pass, so this
    is reachable."""
    _q, _a, _s, coqui = _wire_evict_engines(monkeypatch, "cpu", "cpu", "cpu", coqui_dev="cuda:0")
    names, freed = _run("cuda:0", "coqui")
    assert "coqui" not in names  # self-eviction guard — no step built at all
    assert freed is False
    assert coqui.freed == 0


def test_coqui_idle_ttl_defaults_and_floors(monkeypatch):
    monkeypatch.delenv("COQUI_IDLE_TTL", raising=False)
    assert main._coqui_idle_ttl() == 30.0
    monkeypatch.setenv("COQUI_IDLE_TTL", "90")
    assert main._coqui_idle_ttl() == 90.0
    monkeypatch.setenv("COQUI_IDLE_TTL", "1")  # below the 5s floor -> default
    assert main._coqui_idle_ttl() == 30.0
    monkeypatch.setenv("COQUI_IDLE_TTL", "not-a-number")
    assert main._coqui_idle_ttl() == 30.0
