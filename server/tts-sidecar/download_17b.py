#!/usr/bin/env python3
import sys
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

print("=== Starting Qwen 1.7B Models Download ===")
for repo_id in models:
    print(f"\n--> Fetching {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model"],
        resume_download=True,
    )
    print(f"--> [DONE] {repo_id} downloaded successfully.")

print("\n=== All 1.7B weights are ready! ===")
