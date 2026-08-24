/* Per-book export job tracker. Drives the Listen view's Export modal and
   the `ExportQueue` rail beneath the Listen view.

   Shape:
   - `byBookId[bookId]` — newest-first list of jobs for that book. The
     modal pushes a job on `createBookExport` (`exportStarted`); the
     store-level `exportPollMiddleware` then polls `getBookExport` and
     dispatches `exportUpdated` as the server reports back, until the
     job reaches a terminal state.
   - `lanUrls` — non-loopback IPv4 URLs the server is reachable on.
     Hydrated once when the modal opens so the user can scan the QR to
     download the audiobook to their Android phone over Wi-Fi. */

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { BookExportJob } from '../lib/types';

export interface ExportsState {
  byBookId: Record<string, BookExportJob[]>;
  lanUrls: string[];
  lanPort: number | null;
  /** fs-54 — per-book terminal-completion snapshot for the global Export
      status pill's brief "done"/"failed" summary. Keyed by bookId; a later
      completion for the same book overwrites an earlier one (consistent
      with how the other three status pills aggregate per-book, not
      per-job). Populated + cleared by export-pill-middleware.ts. */
  linger: Record<string, { state: 'done' | 'failed' }>;
}

const initialState: ExportsState = {
  byBookId: {},
  lanUrls: [],
  lanPort: null,
  linger: {},
};

function captionKeyPart(value: string | null | undefined): string {
  return value === undefined ? '<undefined>' : value === null ? '<null>' : value;
}

function revocationKey(job: BookExportJob): string {
  if (job.format !== 'captions') return job.format;
  return [
    job.format,
    captionKeyPart(job.captionFileFormat),
    captionKeyPart(job.captionGranularity),
    captionKeyPart(job.captionScope),
  ].join('|');
}

function completedTimestamp(job: BookExportJob): number {
  const parsed = Date.parse(job.completedAt ?? job.createdAt);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

/** Mirror server revokeStaleSameFormat for completed artifacts while keeping
    concurrent/rerunnable non-done rows visible in the queue. */
function withoutRevokedDoneJobs(jobs: BookExportJob[]): BookExportJob[] {
  const winners = new Map<string, BookExportJob>();
  for (const job of jobs) {
    if (job.status !== 'done') continue;
    const key = revocationKey(job);
    const current = winners.get(key);
    if (!current || completedTimestamp(job) > completedTimestamp(current)) {
      winners.set(key, job);
    }
  }
  return jobs.filter(
    (job) => job.status !== 'done' || winners.get(revocationKey(job))?.id === job.id,
  );
}

export const exportsSlice = createSlice({
  name: 'exports',
  initialState,
  reducers: {
    /* Newest-first prepend so the modal's "current export" view shows the
       latest job at the top of the queue without filtering. */
    exportStarted: (s, a: PayloadAction<BookExportJob>) => {
      const job = a.payload;
      const list = s.byBookId[job.bookId] ?? (s.byBookId[job.bookId] = []);
      list.unshift(job);
    },

    /* Patch in-place on poll updates. We accept the full BookExportJob
       payload because the server is the source of truth for every
       transient field — partial merges would just risk drift. */
    exportUpdated: (s, a: PayloadAction<BookExportJob>) => {
      const job = a.payload;
      const list = s.byBookId[job.bookId];
      if (!list) {
        s.byBookId[job.bookId] = [job];
        return;
      }
      const idx = list.findIndex((j) => j.id === job.id);
      if (idx < 0) list.unshift(job);
      else list[idx] = job;
      if (job.status === 'done') {
        s.byBookId[job.bookId] = withoutRevokedDoneJobs(list);
      }
    },

    /* Remove a job from the rail. The modal exposes this via a "Dismiss"
       button on done/failed rows. */
    exportDismissed: (s, a: PayloadAction<{ bookId: string; exportId: string }>) => {
      const list = s.byBookId[a.payload.bookId];
      if (!list) return;
      s.byBookId[a.payload.bookId] = list.filter((j) => j.id !== a.payload.exportId);
    },

    /* Replace a book's job list from a server snapshot (Listen-mount
       rehydrate). On a fresh mount the slice is empty so this seeds it;
       on a revisit it reconciles to the server's authoritative set. The
       poll middleware picks up any non-terminal jobs from here. */
    exportsHydrated: (s, a: PayloadAction<{ bookId: string; jobs: BookExportJob[] }>) => {
      s.byBookId[a.payload.bookId] = withoutRevokedDoneJobs(a.payload.jobs);
    },

    lanUrlsHydrated: (s, a: PayloadAction<{ urls: string[]; port: number }>) => {
      s.lanUrls = a.payload.urls;
      s.lanPort = a.payload.port;
    },

    /* fs-54 — Export pill completion linger (see ExportsState.linger doc).
       Dispatched by export-pill-middleware.ts. */
    exportLingerSet: (s, a: PayloadAction<{ bookId: string; state: 'done' | 'failed' }>) => {
      s.linger[a.payload.bookId] = { state: a.payload.state };
    },
    exportLingerCleared: (s, a: PayloadAction<{ bookId: string }>) => {
      delete s.linger[a.payload.bookId];
    },
  },
});

export const exportsActions = exportsSlice.actions;
