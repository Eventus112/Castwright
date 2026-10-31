# Qwen long-input acceptance — chapters 96 and 101

Status: **owed**, 2026-10-31. Register: **A48**. Provider regression plan:
[plan 136](../features/archive/136-qwen-token-budget-batching.md#long-input-correction-2026-10-31).
GitHub Issues is disabled (HTTP 410); no issue number is claimed.

Prerequisites: the authorized live book with pending chapters 96 and 101,
a real sidecar and the same Qwen 1.7B model, cast, voice, language and delivery
settings as the failed run. No manuscript/cast edits or output-token caps.

1. Record the tested source SHA and verify the runtime contains the provider
   fragmentation patch. Capture hashes of manuscript/cast inputs before the run.
   Result: **partial** — isolated image built from `32cb643435f9a67ed3d841589bd9757ad3768c25` plus deployment changes at `78e3fcf20`; manuscript/cast hashes captured before chapter generation.
2. Generate chapter 96. Its first raw input was 2,703 characters (normalized
   2,680), with many paragraphs; prior retries returned exactly 655.28 seconds
   of audio at the 8,192-token ceiling. Confirm provider pieces are <=600 Unicode
   code points, preserve the full text, finish in order and produce one result
   per original group. Capture sidecar logs and the chapter completion event.
   A pathological blank-only piece stays with a spoken neighbor; only that
   whitespace may exceed the bound. Normalized generation text has no such run.
   Result: **partial** — actual first block synthesized through the new provider in an isolated container: lossless pieces 563/566/528/569/454; 184.48 seconds of audio in 253.474 seconds. Signal QA `ok` (longest silence 0.848s), Russian ASR `ok` (WER 0.2114). Full chapter still pending. Local evidence: `.run/runtime-generation-reconnect/qwen-probe-20261031/report.json` and WAV.
3. Generate chapter 101, including its 1,096- and 833-character blocks; check the
   same constraints and completion. Confirm original sentence/group identities
   and ordering remain intact, and manuscript/cast hashes match the baseline.
   Result: **not run**.
4. Verify both chapter audio files exist and decode successfully with ffprobe;
   check segments reference the intended voice/model and include the final
   original group. Listen across fragment joins and chapter endings for missing,
   repeated or truncated speech and unacceptable delivery/voice discontinuities.
   Result: **not run** — automated HTTP/PCM tests do not establish audio quality.
5. Confirm the requested export block recognizes both completed chapters.
   Record file paths, durations and observed result, not just a successful API
   response. Update the register and its styled live view together.
   Result: **not run**.

Publication: the canonical Claude artifact cannot be published by this tool
surface; the tracked register/live-view pair is the reviewable source.
