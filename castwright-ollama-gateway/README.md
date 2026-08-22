# Castwright Ollama gateway

Local adapter from the Ollama endpoints used by Castwright to the existing
OpenAI-compatible `llama-swap` / NInfer Qwen3.8-27B service.

It uses the dedicated llama-swap alias `castwright`, which disables thinking
and applies a low-temperature sampling profile without changing the shared
`qwen` alias used by other clients.

llama-swap reports a running alias under its canonical model key (`qwen`), so
the gateway uses `RUNNING_MODEL=qwen` when translating `/running` to `/api/ps`.

The gateway is deliberately bound to `127.0.0.1:11435`. It does not download
or duplicate model weights. An Ollama `keep_alive: 0` request is translated to
llama-swap's `/unload`, so Castwright can release the GPU before TTS starts.
