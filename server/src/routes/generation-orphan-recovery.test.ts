/* Response-close events and fake timers pin reconnect/orphan behavior without
   sockets, polling budgets, real synthesis, or encoding. */
import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import { EventEmitter } from 'node:events';
import { mkdirSync, writeFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Request, Response } from '../http.js';

type SynthArgs = { signal: AbortSignal; onGroupComplete: (tick: object) => void };
let synthesiseImpl: (args: SynthArgs) => Promise<unknown>;
const synth = vi.fn((args: SynthArgs) => synthesiseImpl(args));
const selectProvider = vi.fn(() => ({ synthesize: vi.fn() }));
const hydrateCast = vi.fn(
  async (characters: import('../tts/synthesise-chapter.js').CastCharacter[]) => characters,
);
let beforeQueueWrite: (() => Promise<void>) | undefined;
let onQueueWrite: (() => void) | undefined;
vi.mock('../tts/synthesise-chapter.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../tts/synthesise-chapter.js')>()),
  synthesiseChapter: (args: SynthArgs) => synth(args),
}));
vi.mock('../tts/index.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../tts/index.js')>()),
  selectTtsProvider: () => selectProvider(),
}));
vi.mock('../tts/hydrate-reused-voice-workspace.js', () => ({
  hydrateCastReusedVoices: (characters: import('../tts/synthesise-chapter.js').CastCharacter[]) =>
    hydrateCast(characters),
}));
vi.mock('../tts/ensure-sidecar-loaded.js', () => ({
  ensureSidecarEngineReady: vi.fn(async () => undefined),
  reconcileResidentQwenTiers: vi.fn(async () => undefined),
  SIDECAR_ENGINES: new Set(),
}));
vi.mock('../system/prevent-sleep.js', () => ({ preventSleep: vi.fn(), allowSleep: vi.fn() }));
vi.mock('../audio/finalize-chapter-write.js', () => ({
  finalizeChapterAudioWrite: async () => ({
    audioQa: { status: 'pass', reasons: [] },
    audioModelKey: 'gemini-2.5-flash',
    audioEngines: ['gemini'],
  }),
}));
vi.mock('../workspace/queue-migrate.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../workspace/queue-migrate.js')>();
  return {
    ...actual,
    writeQueueFile: async (...args: Parameters<typeof actual.writeQueueFile>) => {
      await beforeQueueWrite?.();
      await actual.writeQueueFile(...args);
      onQueueWrite?.();
    },
  };
});
function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
class SseResponse extends EventEmitter {
  ticks: Record<string, unknown>[] = [];
  destroyed = false;
  writableEnded = false;
  setHeader() {}
  flushHeaders() {}
  write(frame: string) {
    this.ticks.push(JSON.parse(frame.slice(6)));
    return true;
  }
  end() {
    this.writableEnded = true;
    this.emit('close');
    return this;
  }
  status() {
    return this;
  }
  json() {
    return this.end();
  }
}
let workspaceRoot: string;
let bookDir: string;
let bookId: string;
let queuePath: string;
let generationRouter: import('express').Router;
let isGenerationActive: typeof import('./generation.js').isGenerationActive;
let readQueueFile: typeof import('../workspace/queue-migrate.js').readQueueFile;
let writeQueueFile: typeof import('../workspace/queue-migrate.js').writeQueueFile;
let pending: Promise<unknown>[] = [];
let nextSynth = deferred<SynthArgs>();
const ENTRY_ID = 'orphan-entry-1';
const modelKey = 'gemini-2.5-flash';
function dispatch(body: object, path = '/:bookId/generation', res = new SseResponse()) {
  const layer = generationRouter.stack.find((entry) => entry.route?.path === path)!;
  const req = Object.assign(new EventEmitter(), { params: { bookId }, body });
  const done = Promise.resolve(
    layer.route!.stack[0].handle(req as unknown as Request, res as unknown as Response, vi.fn()),
  );
  pending.push(done);
  return { res, done };
}
async function open(body: object = {}) {
  nextSynth = deferred<SynthArgs>();
  const stream = dispatch({
    modelKey,
    chapterIds: [1],
    force: true,
    queueEntryId: ENTRY_ID,
    ...body,
  });
  await Promise.race([stream.done, nextSynth.promise]);
  return stream;
}
function blockSynth() {
  synthesiseImpl = (args) => {
    nextSynth.resolve(args);
    args.onGroupComplete({ group: { characterId: 'narrator' }, completed: 1, totalGroups: 4 });
    return new Promise((_resolve, reject) => {
      const abort = () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }));
      if (args.signal.aborted) abort();
      else args.signal.addEventListener('abort', abort, { once: true });
    });
  };
}
async function entryStatus() {
  return (await readQueueFile(queuePath)).entries.find((e) => e.id === ENTRY_ID)?.status;
}
beforeAll(async () => {
  workspaceRoot = await mkdtemp(join(tmpdir(), 'generation-reconnect-'));
  process.env.WORKSPACE_DIR = workspaceRoot;
  process.env.GEN_WORKERS = '1';
  ({ generationRouter, isGenerationActive } = await import('./generation.js'));
  const paths = await import('../workspace/paths.js');
  ({ readQueueFile, writeQueueFile } = await import('../workspace/queue-migrate.js'));
  const cache = await import('../store/analysis-cache.js');
  bookId = paths.makeBookId('Test Author', 'Standalones', 'Reconnect Test');
  queuePath = paths.queueJsonPath();
  bookDir = join(workspaceRoot, 'books', 'Test Author', 'Standalones', 'Reconnect Test');
  mkdirSync(join(bookDir, '.audiobook'), { recursive: true });
  mkdirSync(join(bookDir, 'audio'), { recursive: true });
  writeFileSync(
    join(bookDir, '.audiobook', 'state.json'),
    JSON.stringify({
      bookId,
      manuscriptId: 'm_reconnect_test',
      author: 'Test Author',
      title: 'Reconnect Test',
      series: 'Standalones',
      updatedAt: '2026-05-23T00:00:00.000Z',
      schema: 1,
      chapters: [1, 2].map((id) => ({ id, title: `Chapter ${id}`, slug: `chapter-${id}` })),
    }),
  );
  writeFileSync(
    join(bookDir, '.audiobook', 'cast.json'),
    JSON.stringify({ characters: [{ id: 'narrator', name: 'Narrator', voiceId: 'af_alloy' }] }),
  );
  await cache.saveAnalysisCache('m_reconnect_test', {
    chapters: {
      1: [{ id: 1, chapterId: 1, characterId: 'narrator', text: 'Hello.' }],
      2: [{ id: 2, chapterId: 2, characterId: 'narrator', text: 'Goodbye.' }],
    },
  });
});
beforeEach(async () => {
  synth.mockClear();
  selectProvider.mockClear();
  hydrateCast.mockReset();
  hydrateCast.mockImplementation(async (characters) => characters);
  pending = [];
  beforeQueueWrite = undefined;
  onQueueWrite = undefined;
  await rm(join(bookDir, 'audio', 'chapter-1.mp3'), { force: true });
  await writeQueueFile(queuePath, {
    entries: [
      {
        id: ENTRY_ID,
        bookId,
        chapterId: 1,
        scope: 'this',
        addedAt: '2026-05-23T00:00:00.000Z',
        status: 'in_progress',
        order: 0,
      },
    ],
    paused: false,
  });
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  blockSynth();
});
afterEach(async () => {
  beforeQueueWrite = undefined;
  onQueueWrite = undefined;
  dispatch({}, '/:bookId/generation/pause');
  await Promise.all(pending);
  vi.useRealTimers();
});
afterAll(async () => {
  await rm(workspaceRoot, { recursive: true, force: true });
});

describe('generation reconnect and orphan grace', () => {
  it.each(['destroyed', 'writableEnded'] as const)(
    'does not subscribe an already closed resume response or clear its orphan grace: %s',
    async (closedFlag) => {
      const starter = await open();
      const signal = synth.mock.calls[0][0].signal;
      starter.res.emit('close');
      expect(vi.getTimerCount()).toBe(1);
      const res = new SseResponse();
      res[closedFlag] = true;
      res.emit('close');
      const resumed = dispatch(
        { modelKey, chapterIds: [1], resume: true, queueEntryId: ENTRY_ID },
        undefined,
        res,
      );
      await resumed.done;
      expect(vi.getTimerCount()).toBe(1);
      expect(synth).toHaveBeenCalledTimes(1);
      expect(signal.aborted).toBe(false);
      expect(res.ticks).toEqual([]);
    },
  );
  it.each(['destroyed', 'writableEnded'] as const)(
    'never starts for a response already closed before route entry: %s',
    async (closedFlag) => {
      const res = new SseResponse();
      res[closedFlag] = true;
      res.emit('close');
      nextSynth = deferred<SynthArgs>();
      const queueBefore = await readQueueFile(queuePath);
      const stream = dispatch(
        { modelKey, chapterIds: [1], force: true, queueEntryId: ENTRY_ID },
        undefined,
        res,
      );
      await Promise.race([stream.done, nextSynth.promise]);
      expect(synth).not.toHaveBeenCalled();
      await stream.done;
      expect(isGenerationActive(bookId)).toBe(false);
      expect(await readQueueFile(queuePath)).toEqual(queueBefore);
      expect(res.ticks).toEqual([]);
      expect(vi.getTimerCount()).toBe(0);
    },
  );
  it.each(['in_progress', 'failed', 'removed'] as const)(
    'never starts after a preflight disconnect and preserves a %s queue entry lifecycle',
    async (status) => {
      const entered = deferred();
      const release = deferred();
      hydrateCast.mockImplementationOnce(async (characters) => {
        entered.resolve();
        await release.promise;
        return characters;
      });
      const stream = dispatch({ modelKey, chapterIds: [1], force: true, queueEntryId: ENTRY_ID });
      await entered.promise;
      stream.res.emit('close');
      const resumed = await open({ resume: true, force: undefined });
      expect(resumed.res.ticks.map((t) => t.type)).toEqual(['chapter_failed', 'idle']);
      if (status !== 'in_progress') {
        const before = await readQueueFile(queuePath);
        await writeQueueFile(queuePath, {
          ...before,
          entries:
            status === 'removed'
              ? []
              : before.entries.map((entry) => ({
                  ...entry,
                  status: 'failed',
                  errorReason: 'Live attempt unavailable. Click Retry.',
                })),
        });
      }
      const queueBefore = await readQueueFile(queuePath);
      release.resolve();
      await Promise.race([stream.done, nextSynth.promise]);
      expect(synth).not.toHaveBeenCalled();
      await stream.done;
      expect(isGenerationActive(bookId)).toBe(false);
      expect(await readQueueFile(queuePath)).toEqual(queueBefore);
      if (status === 'failed')
        expect((await readQueueFile(queuePath)).entries[0].errorReason).toBe(
          'Live attempt unavailable. Click Retry.',
        );
      expect(vi.getTimerCount()).toBe(0);
    },
  );
  it.each(['failed', 'done'] as const)(
    'preserves frontend %s completion racing with disconnected preflight',
    async (outcome) => {
      const entered = deferred();
      const releasePreflight = deferred();
      const writeEntered = deferred();
      const releaseWrite = deferred();
      hydrateCast.mockImplementationOnce(async (characters) => {
        entered.resolve();
        await releasePreflight.promise;
        return characters;
      });
      const stream = dispatch({ modelKey, chapterIds: [1], force: true, queueEntryId: ENTRY_ID });
      await entered.promise;
      stream.res.emit('close');
      const resumed = await open({ resume: true, force: undefined });
      expect(resumed.res.ticks.map((tick) => tick.type)).toEqual(['chapter_failed', 'idle']);
      /* Hold any stale cleanup snapshot until the real frontend completion
         route has persisted its failure/removal. A write-free preflight
         returns directly instead. */
      beforeQueueWrite = async () => {
        beforeQueueWrite = undefined;
        writeEntered.resolve();
        await releaseWrite.promise;
      };
      releasePreflight.resolve();
      await Promise.race([stream.done, writeEntered.promise]);
      beforeQueueWrite = undefined;
      const { queueRouter } = await import('./queue.js');
      const route = queueRouter.stack.find((entry) => entry.route?.path === '/:entryId/complete')!;
      const errorReason = 'Live attempt unavailable. Click Retry.';
      const req = { params: { entryId: ENTRY_ID }, body: { outcome, errorReason } };
      try {
        await route.route!.stack[0].handle(
          req as unknown as Request,
          new SseResponse() as unknown as Response,
          vi.fn(),
        );
        expect(await entryStatus()).toBe(outcome === 'failed' ? 'failed' : undefined);
      } finally {
        releaseWrite.resolve();
      }
      await stream.done;
      expect(await entryStatus()).toBe(outcome === 'failed' ? 'failed' : undefined);
      if (outcome === 'failed')
        expect((await readQueueFile(queuePath)).entries[0].errorReason).toBe(errorReason);
      expect(synth).not.toHaveBeenCalled();
      expect(isGenerationActive(bookId)).toBe(false);
      expect(vi.getTimerCount()).toBe(0);
    },
  );
  it.each([{ force: true }, { chapterIds: undefined, force: undefined }])(
    'does not attach or displace target or sibling jobs after a preflight disconnect: %j',
    async (body) => {
      const before = await readQueueFile(queuePath);
      await writeQueueFile(queuePath, {
        ...before,
        entries: [
          ...before.entries,
          { ...before.entries[0], id: 'entry-two', chapterId: 2, order: 1 },
        ],
      });
      const target = await open();
      const sibling = await open({ chapterIds: [2], queueEntryId: 'entry-two' });
      const targetSignal = synth.mock.calls[0][0].signal;
      const siblingSignal = synth.mock.calls[1][0].signal;
      const entered = deferred();
      const release = deferred();
      hydrateCast.mockImplementationOnce(async (characters) => {
        entered.resolve();
        await release.promise;
        return characters;
      });
      nextSynth = deferred<SynthArgs>();
      const stream = dispatch({ modelKey, chapterIds: [1], queueEntryId: ENTRY_ID, ...body });
      await entered.promise;
      stream.res.emit('close');
      const runningQueue = await readQueueFile(queuePath);
      release.resolve();
      await Promise.race([stream.done, nextSynth.promise]);
      expect(synth).toHaveBeenCalledTimes(2);
      await stream.done;
      expect(targetSignal.aborted).toBe(false);
      expect(siblingSignal.aborted).toBe(false);
      expect(await readQueueFile(queuePath)).toEqual(runningQueue);
      const queued = deferred();
      onQueueWrite = () => {
        void readQueueFile(queuePath).then((queue) => {
          if (queue.entries.every((entry) => entry.status === 'queued')) queued.resolve();
        });
      };
      target.res.emit('close');
      sibling.res.emit('close');
      expect(vi.getTimerCount()).toBe(2);
      await vi.advanceTimersByTimeAsync(30_000);
      await queued.promise;
      expect(targetSignal.aborted).toBe(true);
      expect(siblingSignal.aborted).toBe(true);
    },
  );
  it('keeps a forced run alive through a drop and scoped resume, preserving progress with one synth', async () => {
    const starter = await open();
    const signal = synth.mock.calls[0][0].signal;
    starter.res.emit('close');
    expect(signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(500);
    const resumed = await open({ resume: true, force: undefined });
    expect(synth).toHaveBeenCalledTimes(1);
    expect(selectProvider).toHaveBeenCalledTimes(1);
    expect(resumed.res.ticks[0]).toMatchObject({ type: 'resume_from', queueEntryId: ENTRY_ID });
    expect(resumed.res.ticks).toContainEqual(
      expect.objectContaining({
        type: 'progress',
        chapterId: 1,
        progress: 0.25,
        queueEntryId: ENTRY_ID,
      }),
    );
    await vi.advanceTimersByTimeAsync(30_000);
    expect(signal.aborted).toBe(false);
    expect(await entryStatus()).toBe('in_progress');
    const queued = deferred();
    onQueueWrite = () => queued.resolve();
    resumed.res.emit('close');
    await vi.advanceTimersByTimeAsync(30_000);
    await queued.promise;
    expect(signal.aborted).toBe(true);
    expect(await entryStatus()).toBe('queued');
  });
  it('only aborts and requeues an orphan after the full 30-second grace', async () => {
    const stream = await open();
    const signal = synth.mock.calls[0][0].signal;
    stream.res.emit('close');
    await vi.advanceTimersByTimeAsync(29_999);
    expect(signal.aborted).toBe(false);
    expect(await entryStatus()).toBe('in_progress');
    const queued = deferred();
    onQueueWrite = () => queued.resolve();
    await vi.advanceTimersByTimeAsync(1);
    await queued.promise;
    expect(signal.aborted).toBe(true);
    expect(await entryStatus()).toBe('queued');
  });
  it('attaches only to its own chapter and queue while a sibling keeps running', async () => {
    const own = await open();
    const sibling = await open({ chapterIds: [2], queueEntryId: 'entry-two' });
    const ownSignal = synth.mock.calls[0][0].signal;
    const siblingSignal = synth.mock.calls[1][0].signal;
    own.res.emit('close');
    /* resume wins over force even if a caller mistakenly sends both. */
    const resumed = await open({ resume: true });
    expect(synth).toHaveBeenCalledTimes(2);
    expect(resumed.res.ticks.filter((t) => t.type === 'progress').map((t) => t.chapterId)).toEqual([
      1,
    ]);
    expect(sibling.res.ticks.some((t) => t.type === 'progress' && t.chapterId === 1)).toBe(false);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(ownSignal.aborted).toBe(false);
    expect(siblingSignal.aborted).toBe(false);
  });
  it('legacy non-queue subscriptions keep running after every observer disconnects', async () => {
    const starter = await open({ queueEntryId: undefined });
    const resumed = await open({
      chapterIds: undefined,
      force: undefined,
      queueEntryId: undefined,
    });
    starter.res.emit('close');
    resumed.res.emit('close');
    await vi.advanceTimersByTimeAsync(30_000);
    expect(synth).toHaveBeenCalledTimes(1);
    expect(synth.mock.calls[0][0].signal.aborted).toBe(false);
  });
  it.each([{ chapterIds: [2] }, { queueEntryId: 'another-entry' }, { modelKey: 'kokoro-v1' }])(
    'rejects a mismatched resume without touching another job: %j',
    async (mismatch) => {
      await open();
      const signal = synth.mock.calls[0][0].signal;
      const resumed = await open({ resume: true, ...mismatch });
      expect(resumed.res.ticks).toEqual([
        expect.objectContaining({
          type: 'chapter_failed',
          chapterId: mismatch.chapterIds?.[0] ?? 1,
          queueEntryId: mismatch.queueEntryId ?? ENTRY_ID,
          errorReason: expect.stringMatching(/Retry/),
        }),
        { type: 'idle' },
      ]);
      expect(synth).toHaveBeenCalledTimes(1);
      expect(signal.aborted).toBe(false);
    },
  );
  it('never starts an absent attempt or claims an old forced-regeneration MP3 completed it', async () => {
    writeFileSync(join(bookDir, 'audio', 'chapter-1.mp3'), Buffer.from([0, 0]));
    const resumed = await open({ resume: true, force: undefined });
    expect(synth).not.toHaveBeenCalled();
    expect(resumed.res.ticks.map((t) => t.type)).toEqual(['chapter_failed', 'idle']);
    expect(await entryStatus()).toBe('in_progress');
  });
  it('does not resume an aborted job and explicit pause still aborts immediately', async () => {
    const stream = await open();
    const signal = synth.mock.calls[0][0].signal;
    stream.res.emit('close');
    expect(vi.getTimerCount()).toBe(1);
    dispatch({}, '/:bookId/generation/pause');
    expect(signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    const resumed = await open({ resume: true });
    expect(resumed.res.ticks.map((t) => t.type)).toEqual(['chapter_failed', 'idle']);
    expect(synth).toHaveBeenCalledTimes(1);
  });
  it('ordinary explicit regeneration still displaces the same chapter', async () => {
    const stream = await open();
    const oldSignal = synth.mock.calls[0][0].signal;
    stream.res.emit('close');
    expect(vi.getTimerCount()).toBe(1);
    await open();
    expect(oldSignal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    expect(synth).toHaveBeenCalledTimes(2);
    expect(synth.mock.calls[1][0].signal.aborted).toBe(false);
  });
  it('does not reset a completed job after its disconnected grace timer would expire', async () => {
    const completed = deferred<unknown>();
    synthesiseImpl = (args) => {
      nextSynth.resolve(args);
      return completed.promise;
    };
    const stream = await open();
    stream.res.emit('close');
    const doneWrite = deferred();
    onQueueWrite = () => doneWrite.resolve();
    completed.resolve({ pcm: Buffer.alloc(2), sampleRate: 24_000, durationSec: 1, segments: [] });
    await stream.done;
    await doneWrite.promise;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(await entryStatus()).toBeUndefined();
    expect(vi.getTimerCount()).toBe(0);
  });
  it('still done-prunes an ordinary non-force queue request for existing audio', async () => {
    writeFileSync(join(bookDir, 'audio', 'chapter-1.mp3'), Buffer.from([0, 0]));
    const doneWrite = deferred();
    onQueueWrite = () => doneWrite.resolve();
    const stream = await open({ force: undefined });
    await stream.done;
    await doneWrite.promise;
    expect(await entryStatus()).toBeUndefined();
    expect(synth).not.toHaveBeenCalled();
  });
});
