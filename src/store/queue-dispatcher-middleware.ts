/* Plan 102 + plan 111 — queue dispatcher middleware (sole concurrency
 * authority).
 *
 * Drains the workspace queue (queue-slice mirror of <workspace>/.queue.json)
 * with up to N concurrent workers (N = the `generationWorkers` user setting,
 * default 2). The dispatcher is the SOLE stream-opener — the old
 * generation-stream-middleware `hasWork` "generating override" is gone, so
 * every run flows through the persisted queue. One queue worker = one
 * chapter: the within-book server pool was removed, so N workers map to N
 * concurrent chapters uniformly across ALL books, including sibling chapters
 * of the SAME book.
 *
 * Each tick:
 *   1. RECONCILE COMPLETIONS — for every entry we claimed whose CHAPTER no
 *      longer has an open stream, that chapter finished: POST /complete to drop
 *      it from the server queue (status-agnostic — the entry is in_progress by
 *      now, so a user-cancel DELETE would 409). Chapter-level, so an entry
 *      leaves the queue the instant its own chapter completes, independent of
 *      siblings.
 *   Before fills, ADOPT persisted in_progress entries with no local owner
 *   through resume-only observers. These count against N even while paused /
 *   recycling or when a lowered worker limit is below the live job count.
 *   2. FILL TO N — claim up to (N − inFlight) queued entries from the flat
 *      queue across books, opening ONE stream per claimed entry (one chapter
 *      each, force:true). On claim, POST /start to flip the entry to
 *      in_progress so the modal renders it "In flight" (multiple can be in
 *      flight at once) and the persisted .queue.json survives a mid-run reload.
 *      Skip chapters already streaming. For the VIEWED book, flip its rows
 *      (regenerateChapterIds — every entry is a whole-chapter regen) so the
 *      Generate view reflects the run; cross-book opens are direct.
 *
 * Correctness:
 *   - No double-claim: middleware + queueMicrotask run single-threaded; we add
 *     to `inFlight` synchronously before any await, and the runner's
 *     per-chapter singleton is the second guard.
 *   - No regen loop: an entry is DELETEd only AFTER its chapter's stream
 *     closes (the chapter is `done` by then); a done row is never re-claimed
 *     (entry gone) nor re-enqueued (the enqueue-on-work observer only enqueues
 *     queued/in_progress rows).
 *   - Same-book no-abort: each new queued chapter starts its own forced
 *     request keyed `${bookId}::${chapterId}` server-side; reload observers
 *     attach with resume:true and never displace a live attempt.
 *
 * srv-11 — consecutive-failure circuit breaker: a per-book in-memory streak
 *   counts back-to-back IDENTICAL chapter-failure reasons; crossing the
 *   threshold (3) pauses the queue + toasts the book + reason so a wedged
 *   sidecar can't burn the whole queue. A differing reason or a success resets
 *   the streak. */

import type { Middleware, MiddlewareAPI } from '@reduxjs/toolkit';
import { chaptersActions, type ChaptersState } from './chapters-slice';
import { queueActions, type QueueState } from './queue-slice';
import { completeQueueEntry, setQueuePaused, startQueueEntry } from './queue-thunks';
import { notificationsActions } from './notifications-slice';
import type { UiState } from './ui-slice';
import type { AppDispatch } from './index';
import type { StreamRunner } from './generation-stream-runner';

interface DispatcherRootState {
  ui: UiState;
  chapters: ChaptersState;
  queue: QueueState;
  /* Optional so lean test stores without the account slice still work; the
     worker count falls back to the default (2). */
  account?: { generationWorkers?: number };
}

const DEFAULT_WORKERS = 2;

/* srv-11 — consecutive-IDENTICAL-failure circuit breaker. When a single book's
   chapters fail repeatedly with the SAME error reason this many times in a row,
   the dispatcher pauses the queue and surfaces a toast naming the book + the
   repeated error — so a wedged sidecar / bad config can't burn through every
   queued chapter producing the identical failure. A differing reason or a
   success resets the streak. */
const CONSECUTIVE_FAILURE_THRESHOLD = 3;

export function queueDispatcherMiddleware(getRunner: () => StreamRunner): Middleware {
  return (storeApi: MiddlewareAPI) => {
    /* Entries the dispatcher has opened a stream for, mapped to their
       (bookId, chapterId). An entry is removed when its CHAPTER's stream
       closes (that chapter finished). The mapping survives the entry leaving
       the queue snapshot, so we can still DELETE it + reconcile per chapter. */
    const inFlight = new Map<string, { bookId: string; chapterId: number }>();

    /* Closed streams awaiting a queue snapshot. Successful entries stay
       blocked until removed, preventing a stale snapshot's regen loop.
       Failed/parked entries stay blocked only while still in_progress, so
       observing a stale row cannot reopen it and Retry/confirm can reclaim. */
    const completed = new Map<string, 'done' | 'closed'>();

    /* srv-11 — per-book consecutive-IDENTICAL-failure streak. `count` only
       grows while the SAME error reason repeats back-to-back for a book; a
       differing reason resets it to 1 (and records the new reason), and a
       success clears the book's entry entirely. Crossing
       CONSECUTIVE_FAILURE_THRESHOLD pauses the queue + toasts once. In-memory
       by design: a reload naturally resets the streak (and failed entries
       already persist as `failed`+`errorReason`), so there's nothing worth
       surviving a restart. */
    const failureStreakByBook = new Map<string, { reason: string; count: number }>();

    const dispatch = storeApi.dispatch as AppDispatch;

    const tick = (): void => {
      const state = storeApi.getState() as DispatcherRootState;
      const { queue, chapters, ui } = state;
      const runner = getRunner();

      /* Cold-boot gate — wait for the initial GET /api/queue. */
      if (!queue.loaded) return;

      /* Forget closed entries once the authoritative snapshot catches up. */
      if (completed.size > 0) {
        const live = new Map(queue.entries.map((e) => [e.id, e.status]));
        for (const [id, outcome] of completed) {
          if (!live.has(id) || (outcome === 'closed' && live.get(id) !== 'in_progress'))
            completed.delete(id);
        }
      }

      /* STEP 1 — reconcile completions. An entry whose CHAPTER has no open
         stream finished (or was torn down). DELETE it and remember it as
         pending so STEP 2 can't re-claim it before the snapshot catches up.
         Chapter-level: completing ch3's stream DELETEs ch3's entry without
         waiting on a sibling ch4 of the same book. */
      for (const [entryId, { bookId, chapterId }] of [...inFlight.entries()]) {
        if (!runner.hasOpenStreamForChapter(bookId, chapterId)) {
          inFlight.delete(entryId);
          completed.set(entryId, 'closed');
          /* Loud-fallback gate: the worker PARKED this chapter (entry flipped to
             `awaiting_confirm`) and closed its stream without completing. Free
             the slot without /complete; it must stay re-claimable once the
             user confirms it back to `queued`. Keep the stale in_progress
             snapshot blocked only until that status changes, so confirmation
             can re-dispatch it.

             Asked of the RUNNER, not the queue slice's `status` field (#1284):
             nothing dispatches a queue-snapshot update off the park tick, so
             `queue.entries` here can still show the entry's pre-park status
             (`in_progress`) — reading that stale value routed a park down the
             "real completion" branch below, /completed it (a server no-op, but
             `completed.add(entryId)` isn't undone by that no-op), and
             permanently blacklisted the entry from STEP 2's re-claim once the
             user confirmed it back to `queued`. */
          if (runner.takeChapterAwaitingConfirm(bookId, chapterId)) {
            continue;
          }
          /* Did this chapter FAIL? If so, complete it as `failed` so the entry
             LINGERS in the queue (rendered "Failed · <reason>" with a Retry
             control) instead of being done-pruned. Its temporary closed guard
             clears when the snapshot changes status, allowing explicit Retry. */
          const failure = runner.takeChapterFailure(bookId, chapterId);
          if (failure != null) {
            void dispatch(
              completeQueueEntry(entryId, { outcome: 'failed', errorReason: failure.reason }),
            ).catch(() => {
              /* Best-effort — the next snapshot reconciles. */
            });

            /* srv-11 — track the consecutive-IDENTICAL-failure streak for this
               book. Only a repeated identical reason trips the breaker; a
               differing reason resets the count to 1 + records the new reason.
               voice-not-designed is exempt: it's a deterministic per-book
               cast-config issue (the #1263 fail-fast that replaced the
               awaiting_confirm park, which was ALSO exempt from this streak),
               not the "something's systemically wrong" signal the breaker
               exists to catch — it shouldn't pause unrelated books' queues. */
            if (failure.errorCode !== 'voice-not-designed') {
              const streak = failureStreakByBook.get(bookId);
              if (streak && streak.reason === failure.reason) {
                streak.count += 1;
              } else {
                failureStreakByBook.set(bookId, { reason: failure.reason, count: 1 });
              }
              const current = failureStreakByBook.get(bookId)!;
              if (current.count >= CONSECUTIVE_FAILURE_THRESHOLD) {
                /* Trip: pause the queue (no per-book pause exists — the queue
                   pause flag is global) and toast naming the book + the repeated
                   error so the user knows which run wedged and why. Reset the
                   streak so a Resume doesn't immediately re-trip on the same
                   already-counted failures. */
                failureStreakByBook.delete(bookId);
                void dispatch(setQueuePaused(true)).catch(() => {
                  /* Best-effort — the toast still surfaces the problem. */
                });
                dispatch(
                  notificationsActions.pushToast({
                    kind: 'error',
                    message: `Paused the queue — book "${bookId}" failed ${CONSECUTIVE_FAILURE_THRESHOLD} times in a row: ${failure.reason}`,
                    dedupeKey: `queue-failure-breaker:${bookId}`,
                  }),
                );
              }
            }
          } else {
            /* Completion removal, NOT a user cancel: the entry is in_progress
               by now (we POSTed /start on claim), so DELETE would 409.
               /complete done-prunes it. Remember it as pending so STEP 2 can't
               re-claim it before the snapshot catches up. Idempotent for an
               already-gone id; the next /api/queue snapshot reconciles. */
            completed.set(entryId, 'done');
            /* srv-11 — a successful chapter resets the book's failure streak. */
            failureStreakByBook.delete(bookId);
            void dispatch(completeQueueEntry(entryId)).catch(() => {
              /* Best-effort — the next snapshot reconciles. */
            });
          }
        }
      }

      /* A reload loses local ownership while the server keeps jobs alive
         through its reconnect grace. Observe every persisted live entry
         before gates/fills, even if the worker limit has since been lowered.
         This must never /start or reset rows: resume only attaches to the
         original attempt, and a missing attempt fails through normal cleanup. */
      for (const e of queue.entries) {
        if (e.status !== 'in_progress') continue;
        if (inFlight.has(e.id) || completed.has(e.id)) continue;
        if (runner.hasOpenStreamForChapter(e.bookId, e.chapterId)) continue;
        inFlight.set(e.id, { bookId: e.bookId, chapterId: e.chapterId });
        runner.open(
          e.bookId,
          e.modelKey ?? ui.ttsModelKey,
          { chapterIds: [e.chapterId] },
          { queueEntryId: e.id, chapterId: e.chapterId, resume: true },
        );
      }

      /* Queue-global pause or sidecar-recycling stops NEW fills at the next
         boundary (in-flight streams finish via their own idle teardown /
         server readiness gate). */
      if (queue.paused || queue.recycling) return;

      /* STEP 2 — fill up to N workers (chapters). Flat per-entry claim: one
         stream per claimed entry, one chapter each. Skip chapters already
         streaming and entries already in flight / pending-delete. */
      const workers = state.account?.generationWorkers ?? DEFAULT_WORKERS;
      let slots = workers - inFlight.size;
      if (slots <= 0) return;

      for (const e of queue.entries) {
        if (slots <= 0) break;
        if (e.status !== 'queued') continue;
        if (inFlight.has(e.id)) continue;
        if (completed.has(e.id)) continue;
        if (runner.hasOpenStreamForChapter(e.bookId, e.chapterId)) continue;

        /* Claim synchronously before any await so a back-to-back snapshot
           tick can't double-claim. */
        inFlight.set(e.id, { bookId: e.bookId, chapterId: e.chapterId });
        slots--;

        /* Flip the entry to in_progress at the moment of claim — one entry =
           one chapter actively starting, so claim == in_progress (no need to
           wait for the first progress tick). The modal then renders this row
           as "In flight" instead of "Queued", and the persisted .queue.json
           shows in_progress mid-run so a reload reflects accurate state.
           Best-effort: a failed mark only affects the label, not the run. */
        void dispatch(startQueueEntry(e.id)).catch(() => {
          /* Status label only — the stream still runs. */
        });

        /* VIEWED book — flip rows so the Generate view shows the run. This
           dispatch no longer triggers an open (the override is gone); it only
           updates slice rows. Every entry is a whole-chapter regen now (the
           per-character scope path was removed — see plan
           docs/features/archive/114-profile-regen-preview.md), so a tolerated-but-
           unused `scope:'character'` on an old .queue.json row maps to the
           same chapter regen. */
        if (chapters.currentBookId === e.bookId) {
          dispatch(chaptersActions.regenerateChapterIds({ chapterIds: [e.chapterId] }));
        }

        /* Open ONE stream for this chapter, keyed `${bookId}::${chapterId}`
           server-side (force:true). queueEntryId + chapterId correlate ticks
           and key the runner handle. */
        runner.open(
          e.bookId,
          /* Per-entry model override (a regenerate requested at a chosen quality
             tier, e.g. Qwen 1.7B); absent → the session default. */
          e.modelKey ?? ui.ttsModelKey,
          { chapterIds: [e.chapterId], force: true },
          {
            queueEntryId: e.id,
            chapterId: e.chapterId,
            /* Loud-fallback gate: a confirmed entry renders straight through —
               tell the worker so it doesn't re-park on the same undesigned
               voices. */
            ...(e.fallbackConfirmed ? { fallbackConfirmed: true } : {}),
          },
        );
      }
    };

    return (next) => (action) => {
      const result = next(action);
      const a = action as { type?: string };
      const type = a?.type;
      /* React to anything that could change the dispatcher's decision:
         - queue/setSnapshot — entries or paused flag changed.
         - chapters/setActiveStream + clearActiveStream — a stream opened/closed.
         - chapters/setCurrentBookId + hydrateFromBookState — the viewed book
           changed (affects which claims flip rows). */
      if (
        type === queueActions.setSnapshot.type ||
        type === 'chapters/setActiveStream' ||
        type === 'chapters/clearActiveStream' ||
        type === 'chapters/setCurrentBookId' ||
        type === 'chapters/hydrateFromBookState'
      ) {
        /* Defer to a microtask so the runner's setActiveStream / clear lands
           before we re-read the open-stream set. */
        queueMicrotask(() => tick());
      }
      return result;
    };
  };
}
