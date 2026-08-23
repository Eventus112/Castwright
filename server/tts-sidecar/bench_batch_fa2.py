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
import flash_attn

def main():
    print("==========================================================", flush=True)
    print("       Qwen TTS Batched Throughput & Scaling Benchmark     ", flush=True)
    print("==========================================================", flush=True)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    print(f"FlashAttention: {flash_attn.__version__}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} (12GB VRAM)", flush=True)

    print("\n--- 1. Loading 0.6B Base Model on CUDA (BF16 + FlashAttention) ---", flush=True)
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
    print(f"Model loaded in: {load_time:.2f}s | Baseline VRAM: {vram_alloc:.1f} MB", flush=True)

    prompt_file = SIDECAR_DIR / "voices" / "qwen" / "test-ru.pt"
    prompt_items = torch.load(str(prompt_file), weights_only=False)
    if not isinstance(prompt_items, list):
        prompt_items = [prompt_items]

    test_sentences = [
        "Первое предложение тестового пакета синтеза речи.",
        "Вторая фраза описывает развитие событий в главе аудиокниги.",
        "Третья реплика персонажа прозвучала неожиданно громко в тихой комнате.",
        "Четвёртая строка завершает первую смысловую часть диалога между героями.",
        "Пятый фрагмент текста используется для проверки масштабирования при параллельной обработке.",
        "Шестой отрезок проверяет устойчивость видеопамяти под возрастающей нагрузкой.",
        "Седьмое предложение приближает размер пакета к восьми параллельным элементам.",
        "Восьмая строка полностью заполняет стандартный батч генератора аудиокниги.",
        "Девятое предложение начинает вторую половину расширенного пакета.",
        "Десятый фрагмент текста тестирует работу тензорных ядер на максимальной плотности.",
        "Одиннадцатая строка проверяет скорость слитных ядер FlashAttention-2.",
        "Двенадцатая реплика диалога подтверждает стабильность генерации речи.",
        "Тринадцатое предложение проверяет отсутствие деградации памяти при высоком батче.",
        "Четырнадцатая фраза демонстрирует масштабирование параллельного синтеза.",
        "Пятнадцатый отрезок аудиокниги приближает размер к шестнадцати потокам.",
        "Шестнадцатая строка завершает полный батч из шестнадцати параллельных предложений."
    ]

    batch_sizes = [1, 2, 4, 8, 16]
    results = []

    print("\n--- 2. Running Batched Synthesis Benchmark ---", flush=True)
    print(f"{'Batch':<8} | {'Текстов':<8} | {'Время (с)':<10} | {'Аудио (с)':<10} | {'RTF':<8} | {'Скорость':<12} | {'На 1 фразу':<12} | {'VRAM (MB)'}", flush=True)
    print("-" * 95, flush=True)

    for bs in batch_sizes:
        sub_texts = test_sentences[:bs]
        langs = ["Russian"] * bs
        prompts = prompt_items * bs

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t_start = time.perf_counter()

        with torch.no_grad():
            wavs, sr = model.generate_voice_clone(
                text=sub_texts,
                language=langs,
                voice_clone_prompt=prompts,
                max_new_tokens=150,
            )

        t_wall = time.perf_counter() - t_start
        total_audio_s = sum(len(w) for w in wavs) / sr
        rtf = t_wall / total_audio_s if total_audio_s > 0 else 0.0
        speed_x = total_audio_s / t_wall if t_wall > 0 else 0.0
        time_per_sentence = t_wall / bs
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        res = {
            "batch_size": bs,
            "wall_time_s": round(t_wall, 2),
            "total_audio_s": round(total_audio_s, 2),
            "rtf": round(rtf, 3),
            "speed_x": round(speed_x, 2),
            "time_per_sentence_s": round(time_per_sentence, 2),
            "peak_vram_mb": round(peak_vram, 1),
        }
        results.append(res)
        print(f"BS={bs:<5} | {bs:<8} | {t_wall:<10.2f} | {total_audio_s:<10.2f} | {rtf:<8.3f} | {speed_x:<9.2f}x | {time_per_sentence:<9.2f}с   | {peak_vram:.1f} MB", flush=True)

    out_file = SIDECAR_DIR / "fa2_batch_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("-" * 95, flush=True)
    print(f"Результаты сохранены в {out_file}", flush=True)
    print("==========================================================", flush=True)

if __name__ == "__main__":
    main()
