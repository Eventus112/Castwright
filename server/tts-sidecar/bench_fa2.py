#!/usr/bin/env python3
import gc
import json
import os
import sys
import time
from pathlib import Path

SIDECAR_DIR = Path(__file__).resolve().parent
if str(SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(SIDECAR_DIR))

import torch
from qwen_tts import Qwen3TTSModel
import flash_attn

def run():
    print("==================================================", flush=True)
    print("      Qwen TTS + FlashAttention-2 Benchmark       ", flush=True)
    print("==================================================", flush=True)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    print(f"FlashAttention: {flash_attn.__version__}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("\n--- 1. Loading 0.6B Base with FlashAttention-2 ---", flush=True)
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        attn_implementation="flash_attention_2",
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
    print(f"Model loaded in: {load_time:.2f}s | VRAM: {vram_alloc:.1f} MB alloc / {vram_res:.1f} MB reserved", flush=True)

    prompt_file = SIDECAR_DIR / "voices" / "qwen" / "test-ru.pt"
    prompt_items = torch.load(str(prompt_file), weights_only=False)
    if not isinstance(prompt_items, list):
        prompt_items = [prompt_items]

    # Test sentences
    test_cases = [
        ("RU Короткое", "Russian", "Привет! Это быстрый тест синтеза речи на видеокарте."),
        ("RU Среднее", "Russian", "Генеративный синтез речи на базе моделей Qwen демонстрирует высокую выразительность и естественность интонаций."),
        ("RU Длинное", "Russian", "В тишине осеннего вечера за окном медленно падали золотые листья, укрывая мокрые дорожки старого сада, и в воздухе разливался тонкий аромат прелой листвы и приближающихся заморозков."),
        ("EN Short", "English", "Hello! This is a fast speech synthesis test on CUDA GPU."),
        ("EN Medium", "English", "Neural voice synthesis with autoregressive transformers delivers remarkable naturalness and consistent prosody.")
    ]

    print("\n--- 2. Single Sentence Synthesis on CUDA ---", flush=True)
    results = []
    for label, lang, text in test_cases:
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
        speed_x = audio_dur / t_gen if t_gen > 0 else 0.0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        res = {
            "label": label,
            "lang": lang,
            "char_len": len(text),
            "gen_time_s": round(t_gen, 2),
            "audio_dur_s": round(audio_dur, 2),
            "rtf": round(rtf, 3),
            "speed_x": round(speed_x, 2),
            "peak_vram_mb": round(peak_vram, 1),
        }
        results.append(res)
        print(f"[{label:12s}] {len(text):3d} симв. | Синтез: {t_gen:5.2f}с | Аудио: {audio_dur:5.2f}с | RTF: {rtf:.3f} ({speed_x:.2f}x realtime) | Пик VRAM: {peak_vram:.1f} MB", flush=True)

    out_file = SIDECAR_DIR / "fa2_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved benchmark results to {out_file}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    run()
