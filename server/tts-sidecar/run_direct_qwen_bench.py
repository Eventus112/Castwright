#!/usr/bin/env python3
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGINT, signal.SIG_IGN)

SIDECAR_DIR = Path(__file__).resolve().parent
if str(SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(SIDECAR_DIR))

import torch
from qwen_tts import Qwen3TTSModel

def main():
    print("=== Qwen TTS CUDA Benchmark ===", flush=True)
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}", flush=True)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)

    voices_dir = SIDECAR_DIR / "voices" / "qwen"
    prompt_file = voices_dir / "test-ru.pt"
    if not prompt_file.exists():
        prompt_file = list(voices_dir.glob("*.pt"))[0]
    print(f"Voice prompt: {prompt_file.name}", flush=True)
    prompt_items = torch.load(str(prompt_file), weights_only=False)
    if not isinstance(prompt_items, list):
        prompt_items = [prompt_items]

    # Load Model
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "to"):
        inner.to("cuda")
    else:
        model.to("cuda")
    try:
        model.device = torch.device("cuda")
    except Exception:
        pass
    load_time = time.perf_counter() - t0
    vram_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    vram_res = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"Model Load Time: {load_time:.2f}s | VRAM: {vram_alloc:.1f} MB alloc / {vram_res:.1f} MB res", flush=True)

    # Benchmark Sentences
    sentences = [
        ("RU Короткое", "Russian", "Привет! Это проверка работы синтеза речи на видеокарте."),
        ("RU Среднее", "Russian", "Модель генерации речи Qwen обеспечивает естественное звучание и интонации текста."),
        ("RU Длинное", "Russian", "В тишине осеннего вечера за окном медленно падали золотые листья, укрывая дорожки старого сада, и в воздухе разливался аромат приближающихся заморозков."),
        ("EN Short", "English", "Hello! This is a fast speech synthesis test on CUDA GPU."),
        ("EN Medium", "English", "Neural voice synthesis with autoregressive transformers delivers remarkable naturalness and consistent prosody.")
    ]

    results = []
    print("\n--- Running Synthesis Benchmark ---", flush=True)
    for label, lang, text in sentences:
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        with torch.no_grad():
            wavs, sr = model.generate_voice_clone(
                text=[text],
                language=[lang],
                voice_clone_prompt=prompt_items,
            )
        t_gen = time.perf_counter() - t_start
        audio_dur = len(wavs[0]) / sr
        rtf = t_gen / audio_dur if audio_dur > 0 else 0.0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        res = {
            "label": label,
            "lang": lang,
            "char_len": len(text),
            "gen_time_s": round(t_gen, 2),
            "audio_dur_s": round(audio_dur, 2),
            "rtf": round(rtf, 3),
            "speed_x": round(audio_dur / t_gen, 2) if t_gen > 0 else 0.0,
            "peak_vram_mb": round(peak_vram, 1),
        }
        results.append(res)
        print(f"[{label:12s}] {len(text):3d} симв. | Синтез: {t_gen:5.2f}с | Аудио: {audio_dur:5.2f}с | RTF: {rtf:.3f} ({res['speed_x']:.2f}x) | Пик VRAM: {peak_vram:.1f} MB", flush=True)

    with open(SIDECAR_DIR / "direct_qwen_bench_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nBenchmark completed successfully!", flush=True)

if __name__ == "__main__":
    main()
