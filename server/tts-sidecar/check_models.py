#!/usr/bin/env python3
import os
from pathlib import Path

hub_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))

models = [
    ("0.6B-Base (Synthesis)", "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"),
    ("1.7B-VoiceDesign (Design)", "models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ("1.7B-Base (Quality tier)", "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"),
]

out_lines = ["=== Qwen3-TTS Model Cache Check ==="]
for name, folder in models:
    p = hub_dir / folder
    if p.exists():
        snapshots = p / "snapshots"
        files_count = 0
        total_size = 0
        if snapshots.exists():
            for f in snapshots.rglob("*"):
                if f.is_file():
                    files_count += 1
                    total_size += f.stat().st_size
        size_gb = total_size / (1024 ** 3)
        out_lines.append(f"[FOUND] {name}: {size_gb:.2f} GB ({files_count} files)")
    else:
        out_lines.append(f"[MISSING] {name}: NOT FOUND in {p}")

report = "\n".join(out_lines)
print(report)
with open(Path(__file__).parent / "models_status.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")
