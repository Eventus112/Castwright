---
status: active
shipped: null
owner: null
---

# Queue robustness: failure circuit-breaker + orphaned-entry recovery

> Status: active
> Key files: `src/store/queue-dispatcher-middleware.ts`, `server/src/workspace/queue-io.ts`, `server/src/routes/generation.ts`
> URL surface: indirect — the queue modal (`#/...` any view, `View queue`) reflects the paused state + toast
> OpenAPI ops: none new (reuses `POST /api/queue/pause`, the generation SSE close)

Two independent robustness fixes for the persisted generation queue (plan 102 / plan 111):

- **srv-11 — consecutive-identical-failure circuit breaker (frontend dispatcher).**
- **srv-12 — orphaned `in_progress` recovery on SSE last-subscriber disconnect (server).**

## Benefit / Rationale

- **User (srv-11):** a wedged sidecar / bad config no longer silently burns through every queued chapter producing the same failure. After 3 identical-in-a-row failures for a book, the queue pauses and a toast names the book + the repeated error, so the user fixes the root cause once instead of retrying N failed entries.
- **User (srv-12):** a short connection drop leaves the chapter running for a 30-second reconnect grace period. If no subscriber returns, the unwatched synthesis is aborted without changing the queue row. A returning observer persists Failed with Retry; without a client, reload observation or boot recovery handles the remaining `in_progress` row. A reconnect only attaches to the same live attempt; an unavailable attempt asks for explicit Retry.
- **Technical:** srv-11 is in-memory (per-book streak) — a reload naturally resets it, and failed entries already persist as `failed`+`errorReason` (PR #254), so nothing extra needs persisting. srv-12 aborts expired orphan jobs without queue persistence, so it cannot overwrite a concurrent frontend failure/removal or automatically replay a chapter.
- **Architectural:** srv-12 establishes `res.on('close')` (not `req.on('close')`) as the reliable client-gone signal on the generation SSE — once `express.json()` consumes the request body the `req` stream is already ended, so `req.on('close')` can miss a mid-stream disconnect.

## Architectural impact

- **srv-11 (`queue-dispatcher-middleware.ts`):** new in-memory `failureStreakByBook: Map<string, { reason, count }>` alongside the existing `inFlight` / `completed` local state. The reconcile-loop failure branch increments the streak on an *identical* repeated reason (resets to 1 on a differing reason); crossing `CONSECUTIVE_FAILURE_THRESHOLD` (3) dispatches `setQueuePaused(true)` + a `pushToast`. The success branch deletes the book's streak. Pause granularity is **global** (`setQueuePaused`) — the queue has no per-book/per-entry pause shape (`queue-io.setPaused` flips one global flag; `QueueFile.paused` is a single boolean), so the toast names the book + reason instead.
- **srv-12 (`generation.ts`):** every subscriber uses `res.close`. The last disconnect schedules a single 30-second timer; a successful reattach cancels it. Expiry synchronously rechecks registration, zero subscribers and the abort signal, then only aborts the controller. It performs no queue read/reset/write. Completion, displacement and explicit pause clear the timer; explicit pause stays immediate.
- **Preflight disconnect (2026-10-27):** response close is observed from route entry. A request that closes during awaited preflight cannot subscribe, displace an existing job, or start synthesis, and leaves the queue unchanged. A reconnect observer receives unavailable-attempt failure plus `idle`, so the normal frontend lifecycle persists Failed with Retry; without a client, reload observation or boot recovery handles the row. Preflight performs no queue cleanup write that could overwrite a concurrent frontend failure/removal or automatically replay the chapter.
- **Reconnect contract (2026-10-26):** the client retains chapter/model/queue ids and sends `resume: true`, omitting `force`, on reconnect and on the first observation request after a browser reload. The server checks this before provider selection, cache reads, voice validation, preloads or displacement. It subscribes only to the matching live request and queue entry, never starts synthesis, and never treats an old forced-regeneration MP3 as completion. Missing/aborted/mismatched attempts emit scoped `chapter_failed` then `idle` and require Retry. Non-resume explicit regeneration still displaces intentionally.
- **Reload ownership (2026-10-27):** before new fills and before paused/recycling gates, the dispatcher adopts every unowned persisted `in_progress` entry into its local `inFlight` map and opens an observation-only stream. Adoption does not POST `/start`, reset chapter progress, or force regeneration. All adopted streams occupy worker slots, even when their count exceeds a lowered worker setting. Repeated snapshots do not duplicate subscriptions; closed failed/parked streams remain guarded while the local snapshot still says `in_progress`, then become eligible for explicit Retry/confirmation once that status changes.
- **Invariants preserved:** the frontend owns the success lifecycle (it POSTs `/complete` BEFORE the SSE closes), so a clean disconnect-after-completion hits a *deregistered* job and the registration guard skips it — orphan recovery only ever touches genuine orphans. RTK Immer drafts untouched (the streak map is plain middleware-local state, not slice state).
- **Reversibility:** both are additive. Removing srv-11 = drop the streak map + its two branches. Removing srv-12 = drop the orphan timer + the close-handler block (revert the `res.on('close')` rename back to `req.on('close')`).

## Invariants to preserve

- `CONSECUTIVE_FAILURE_THRESHOLD` in `src/store/queue-dispatcher-middleware.ts` is `3`. Only an *identical* repeated `errorReason` increments the streak; a differing reason resets to 1; a success deletes the book's entry.
- `resetEntryToQueued` in `server/src/workspace/queue-io.ts` is a no-op (returns the input `file` by reference) for a missing id or any non-`in_progress` status; it `renumber()`s like the sibling `resetInProgressToQueued`.
- Every generation subscriber uses response close; orphan recovery requires a full 30-second disconnect plus zero subscribers, a queue entry, live registration and an un-aborted controller. Expiry leaves the queue unchanged, preserving concurrent frontend failure/removal; an unavailable-attempt observer emits `chapter_failed` then `idle` so the frontend persists Failed with Retry. Non-queue legacy jobs retain their existing disconnect behavior.
- A target `chapter_failed` or `chapter_complete` followed by EOF produces `idle` without reconnecting. Failure code, reason and remediation survive unchanged; an unscoped setup failure is scoped to the sole requested chapter. Sibling catch-up completions do not finish the target. Empty/setup/network/exhausted exits emit scoped failure then idle; cancellation emits neither.
- For explicit chapter targets, `idle` before a matching terminal tick emits scoped `chapter_failed` first so interrupted work stays Failed with Retry instead of being done-pruned. Progress reaching 100%, sibling completions and another queue entry's completions cannot finish the target. A matching `chapter_awaiting_fallback_confirm` preserves the confirmation park; legacy streams without explicit targets still accept bare idle.
- A cold-boot `in_progress` row receives a subscribe-only first request, including while the queue is paused or recycling. Observe all existing attempts before filling any queued sibling; missing attempts fail and release their slots without re-adopting a stale snapshot.

## Test plan

### Automated coverage

- Vitest unit (`src/store/queue-dispatcher-middleware.test.ts`, `describe('consecutive-failure circuit breaker (srv-11)')`) — asserts: no trip on a single transient failure; no trip on three *differing*-reason failures; trip after 3 *identical* failures → queue paused + an `error` toast (`dedupeKey` `queue-failure-breaker:<book>`) naming the book + reason; a success resets the streak so prior failures don't accumulate.
- Vitest server (`server/src/workspace/queue-io.test.ts`, `describe('queue-io.resetEntryToQueued ...')`) — asserts: flips only the targeted `in_progress` entry (siblings untouched); no-op (returns input by reference) for a missing id; no-op for a non-`in_progress` (queued/failed) entry.
- Vitest server (`server/src/routes/generation-orphan-recovery.test.ts`) — response-close events, controlled preflight/synthesis promises and fake timers prove one synth through forced start → drop → resume, progress replay, sibling/queue/model isolation, rejection of absent/aborted attempts despite old audio, write-free expiry, observer failure persistence through the actual queue completion route, expiry racing with frontend failure/removal, timer cleanup after completion/displacement/pause, legacy subscriptions, and ordinary disk completion. A preflight close starts no job, leaves the queue unchanged, preserves failures/removals persisted through the real queue completion route while the disconnected request exits, and leaves existing target/sibling jobs and their grace timers intact. No real server, sidecar or GPU is required.
- Vitest client (`src/lib/api-stream-reconnect.test.ts`) — fake timers pin reconnect and first-request reload observation bodies, terminal EOF handling, unfinished-target idle failures, sibling/queue identity isolation, confirmation parks, legacy bare idle, scoped setup/empty/network failures, bounded exhaustion and quiet cancellation. `generation-stream-runner.test.ts` pins resume pass-through.
- Vitest dispatcher (`src/store/queue-dispatcher-middleware.test.ts`, `reload adoption`) — persisted live rows preserve progress and model/queue identity without `/start` or force, repeated snapshots are idempotent, paused/recycling queues still observe, all live rows count against a lowered worker limit, and missing attempts release slots while stale closed snapshots stay guarded. Existing Retry/confirmation tests preserve explicit regeneration behavior.
- Playwright (`e2e/queue-modal.spec.ts`, hard reload) — the Layout mount restores visible generation activity before the queue modal is opened, including after `page.reload()`, while a paused queue keeps siblings queued. Uses the existing browser mock-queue seed; server attempt identity remains covered by HTTP lifecycle tests.

### Manual acceptance walkthrough

1. Queue 3+ chapters for one book with a broken engine (e.g. stop the sidecar) → each chapter fails with the same reason. After the 3rd identical failure the queue flips to **Paused** and a red toast reads *Paused the queue — book "…" failed 3 times in a row: …*.
2. Interrupt only the SSE connection mid-chapter, then reconnect within 30 seconds → the same attempt and progress continue without another synth. Leave all subscribers disconnected beyond 30 seconds → the unwatched synth aborts and the entry stays `in_progress` until a returning observer persists Failed with Retry (or boot recovery handles it). No automatic replay occurs on expiry. If the server restarted or the attempt ended before reattach, the client shows an interrupted failure with Retry and releases the runner. Existing audio from before a forced regeneration must not mark that attempt successful.
3. Hard-refresh within the grace period with an `in_progress` entry and queued siblings → the first request is `resume: true` with the original ids and no `force`; no `/start` or progress reset occurs. Existing streams continue to occupy slots before queued siblings can start. Repeat with a paused/recycling queue and a worker setting below the number of live attempts → every existing attempt is observed but new fills stay blocked. An unavailable attempt becomes Failed with Retry and releases its slot.

The automated lifecycle checks are GPU-free. Actual live generation and a physical network drop were not exercised for the 2026-10-26 reconnect fix.

## Out of scope

- Per-book / per-entry queue pause (the queue has only a global pause flag). srv-11 uses the global pause + a book-naming toast.
- Persisting the srv-11 streak across reloads (intentionally in-memory; failed entries already persist with their reason).

## Ship notes

(Filled in when status flips to `stable`.)
