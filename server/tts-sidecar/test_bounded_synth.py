#!/usr/bin/env python3
import time
import sys
from pathlib import Path

SIDECAR_DIR = Path(__file__).resolve().parent
if str(SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(SIDECAR_DIR))

import torch
from qwen_tts import Qwen3TTSModel

def run():
    print("--- 1. Loading 0.6B Base Model on CUDA ---", flush=True)
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
    t_load = time.perf_counter() - t0
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Loaded in {t_load:.2f}s | VRAM: {vram_mb:.1f} MB", flush=True)

    prompt_file = SIDECAR_DIR / "voices" / "qwen" / "test-ru.pt"
    prompt_items = torch.load(str(prompt_file), weights_only=False)
    if not isinstance(prompt_items, list):
        prompt_items = [prompt_items]

    print("--- 2. Synthesizing with max_new_tokens=256 ---", flush=True)
    text = "Привет! Это проверка работы синтеза речи на видеокарте."
    torch.cuda.reset_peak_memory_stats()
    t_synth0 = time.perf_counter()
    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=[text],
            language=["Russian"],
            voice_clone_prompt=prompt_items,
            max_new_tokens=256,
        )
    t_synth = time.perf_counter() - t_synth0
    audio_dur = len(wavs[0]) / sr
    rtf = t_synth / audio_dur if audio_dur > 0 else 0.0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"Synthesis time : {t_synth:.2f} s", flush=True)
    print(f"Audio duration : {audio_dur:.2f} s ({sr} Hz)", flush=True)
    print(f"RTF            : {rtf:.3f} ({audio_dur/t_synth:.2f}x realtime)", flush=True)
    print(f"Peak VRAM      : {peak_vram:.1f} MB", flush=True)

if __name__ == "__main__":
    run()
