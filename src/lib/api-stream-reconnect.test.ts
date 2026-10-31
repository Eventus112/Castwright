/* Reconnects attach to the same attempt and always release the runner. */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { GenerationTick } from './types';
import type { StreamArgs } from './api';

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  vi.useFakeTimers();
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
function sseResponse(ticks: object[]): Response {
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const tick of ticks)
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(tick)}\n\n`));
        controller.close();
      },
    }),
    text: async () => '',
  } as Response;
}
async function start(overrides: Partial<StreamArgs> = {}) {
  const { api } = await import('./api');
  const ticks: GenerationTick[] = [];
  const cancel = api.streamGeneration({
    bookId: 'book-A',
    modelKey: 'kokoro-v1',
    chapterIds: [1],
    force: true,
    queueEntryId: 'queue-entry-xyz',
    ...overrides,
    onTick: (tick) => ticks.push(tick),
  });
  await vi.advanceTimersByTimeAsync(0);
  return { ticks, cancel };
}
const scoped = { chapterId: 1, queueEntryId: 'queue-entry-xyz' };

describe('realStreamGeneration auto-reconnect', () => {
  it('sends resume on the first request for a reloaded live attempt, omitting force', async () => {
    const { api } = await import('./api');
    fetchMock.mockResolvedValueOnce(sseResponse([{ type: 'idle' }]));
    api.streamGeneration({
      bookId: 'book-A',
      modelKey: 'kokoro-v1',
      chapterIds: [1],
      queueEntryId: scoped.queueEntryId,
      resume: true,
      force: true,
      onTick: () => {},
    });
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      modelKey: 'kokoro-v1',
      chapterIds: [1],
      queueEntryId: scoped.queueEntryId,
      resume: true,
    });
  });

  it('uses resume only on reconnect, retaining the target and omitting force', async () => {
    fetchMock
      .mockResolvedValueOnce(sseResponse([{ type: 'progress', chapterId: 1, progress: 0.3 }]))
      .mockResolvedValueOnce(
        sseResponse([{ type: 'chapter_complete', chapterId: 1 }, { type: 'idle' }]),
      );
    const { ticks } = await start();
    await vi.advanceTimersByTimeAsync(500);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      modelKey: 'kokoro-v1',
      chapterIds: [1],
      force: true,
      queueEntryId: scoped.queueEntryId,
    });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
      modelKey: 'kokoro-v1',
      chapterIds: [1],
      resume: true,
      queueEntryId: scoped.queueEntryId,
    });
    expect(ticks.map((t) => t.type)).toEqual(['progress', 'chapter_complete', 'idle']);
  });
  it.each([
    ['no progress', []],
    ['full progress', [{ type: 'progress', chapterId: 1, progress: 1 }]],
    ['sibling completion', [{ type: 'chapter_complete', chapterId: 2 }]],
    ['another queue entry', [{ type: 'chapter_complete', chapterId: 1, queueEntryId: 'another-entry' }]],
  ])('fails an unfinished target before idle without reconnecting (%s)', async (_label, beforeIdle) => {
    fetchMock.mockResolvedValueOnce(sseResponse([...beforeIdle, { type: 'idle' }]));
    const { ticks } = await start();
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(ticks).toEqual([
      ...beforeIdle,
      {
        type: 'chapter_failed',
        ...scoped,
        errorReason: expect.stringMatching(/before this chapter finished.*Retry/),
      },
      { type: 'idle' },
    ]);
  });
  it.each(['chapter_complete', 'chapter_failed', 'chapter_awaiting_fallback_confirm'] as const)(
    'preserves a target %s followed by idle',
    async (type) => {
      const terminal = {
        type,
        ...scoped,
        errorReason: 'Original failure',
        remediation: 'Open cast',
      };
      fetchMock.mockResolvedValueOnce(sseResponse([terminal, { type: 'idle' }]));
      const { ticks } = await start();
      await vi.runAllTimersAsync();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(ticks).toEqual([terminal, { type: 'idle' }]);
    },
  );
  it('fails only unfinished targets when idle follows a partial completion', async () => {
    const complete = { type: 'chapter_complete', chapterId: 1 };
    fetchMock.mockResolvedValueOnce(sseResponse([complete, { type: 'idle' }]));
    const { ticks } = await start({ chapterIds: [1, 2] });
    await vi.runAllTimersAsync();
    expect(ticks).toEqual([
      complete,
      expect.objectContaining({
        type: 'chapter_failed',
        chapterId: 2,
        queueEntryId: scoped.queueEntryId,
      }),
      { type: 'idle' },
    ]);
  });
  it.each([undefined, []])('allows legacy idle without explicit targets (%j)', async (chapterIds) => {
    fetchMock.mockResolvedValueOnce(sseResponse([{ type: 'idle' }]));
    const { ticks } = await start({ chapterIds });
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(ticks).toEqual([{ type: 'idle' }]);
  });
  it.each(['chapter_complete', 'chapter_failed'] as const)(
    'ends with idle after target %s without retrying or replacing the payload',
    async (type) => {
      const terminal = {
        type,
        ...scoped,
        errorReason: 'Original CUDA error',
        errorCode: 'cuda-poisoned',
        remediation: 'Restart engine',
      };
      fetchMock.mockResolvedValueOnce(sseResponse([terminal]));
      const { ticks } = await start();
      await vi.runAllTimersAsync();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(ticks).toEqual([terminal, { type: 'idle' }]);
    },
  );
  it('ignores sibling catch-up completions when deciding whether the target ended', async () => {
    fetchMock
      .mockResolvedValueOnce(sseResponse([{ type: 'chapter_complete', chapterId: 2 }]))
      .mockResolvedValueOnce(sseResponse([{ type: 'chapter_complete', chapterId: 1 }]));
    const { ticks } = await start();
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(ticks.map((t) => t.type)).toEqual(['chapter_complete', 'chapter_complete', 'idle']);
  });
  it('scopes a setup failure and supplies idle without replacing its original reason', async () => {
    fetchMock.mockResolvedValueOnce(
      sseResponse([
        { type: 'chapter_failed', errorReason: 'Cast not confirmed', remediation: 'Open cast' },
      ]),
    );
    const { ticks } = await start();
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(ticks).toEqual([
      {
        type: 'chapter_failed',
        ...scoped,
        errorReason: 'Cast not confirmed',
        remediation: 'Open cast',
      },
      { type: 'idle' },
    ]);
  });
  it.each(['empty', 'http', 'network'])(
    'releases the target runner after a final %s setup error',
    async (kind) => {
      if (kind === 'empty') fetchMock.mockResolvedValueOnce(sseResponse([]));
      if (kind === 'http')
        fetchMock.mockResolvedValueOnce({
          ok: false,
          status: 503,
          text: async () => 'Unavailable',
        });
      if (kind === 'network') fetchMock.mockRejectedValueOnce(new Error('Network unavailable'));
      const { ticks } = await start();
      await vi.runAllTimersAsync();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(ticks).toEqual([
        expect.objectContaining({
          type: 'chapter_failed',
          ...scoped,
          errorReason: expect.any(String),
        }),
        { type: 'idle' },
      ]);
    },
  );
  it('exhausts reconnects with a scoped failure and idle', async () => {
    fetchMock.mockImplementation(async () => sseResponse([{ type: 'progress', chapterId: 1 }]));
    const { ticks } = await start();
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(ticks.slice(-2)).toEqual([
      expect.objectContaining({
        type: 'chapter_failed',
        ...scoped,
        errorReason: expect.stringMatching(/Retry/),
      }),
      { type: 'idle' },
    ]);
  });
  it('stays quiet when cancelled during reconnect backoff', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([{ type: 'progress', chapterId: 1 }]));
    const { ticks, cancel } = await start();
    cancel();
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(ticks.map((t) => t.type)).toEqual(['progress']);
  });
  it('stays quiet when a pending first fetch is cancelled', async () => {
    fetchMock.mockImplementation(
      (_url, opts) =>
        new Promise((_resolve, reject) => {
          opts.signal.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          );
        }),
    );
    const { ticks, cancel } = await start();
    cancel();
    await vi.runAllTimersAsync();
    expect(ticks).toEqual([]);
  });
});
