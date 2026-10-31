# Qwen long-input acceptance — chapters 96 and 101

Status: **owed**, 2026-10-31. Register: **A48**. Provider regression plan:
[plan 136](../features/archive/136-qwen-token-budget-batching.md#long-input-correction-2026-10-31).
GitHub Issues is disabled (HTTP 410); no issue number is claimed.

Prerequisites: the authorized live book with pending chapters 96 and 101,
a real sidecar and the same Qwen 1.7B model, cast, voice, language and delivery
settings as the failed run. No manuscript/cast edits or output-token caps.

1. Record the tested source SHA and verify the runtime contains the provider
   fragmentation patch. Capture hashes of manuscript/cast inputs before the run.
   Result: **not run**.
2. Generate chapter 96. Its first raw input was 2,703 characters (normalized
   2,680), with many paragraphs; prior retries returned exactly 655.28 seconds
   of audio at the 8,182-token ceiling. Confirm provider pieces are <=600 Unicode
   code points, preserve the full text, finish in order and produce one result
   per original group. Capture sidecar logs and the chapter completion event.
   A pathological blank-only piece stays with a spoken neighbor; only that
   whitespace may exceed the bound. Normalized generation text has no such run.
   Result: **not run**.
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
