import { readFile } from 'node:fs/promises';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { errorPath } from '../handoff/protocol.js';
import { AnalysisAbortedError } from './ollama.js';
import {
  OpenAICompatibleAnalyzer,
  OpenAICompatibleHttpError,
  normaliseOpenAIBaseUrl,
  resolveOpenAICompatibleConfigFromEnv,
} from './openai-compatible.js';

const originalEnv = { ...process.env };

function completion(content: string, status = 200, statusText = ''): Response {
  return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content } }] }), {
    status,
    statusText,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  delete process.env.ANALYZER_PROVIDER;
  delete process.env.OPENAI_BASE_URL;
  delete process.env.OPENAI_MODEL;
  delete process.env.OPENAI_API_KEY;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  process.env = { ...originalEnv };
});

describe('normaliseOpenAIBaseUrl', () => {
  it('appends /chat/completions to an API root and preserves a full endpoint', () => {
    expect(normaliseOpenAIBaseUrl(' http://localhost:8000/v1/ ')).toBe(
      'http://localhost:8000/v1/chat/completions',
    );
    expect(normaliseOpenAIBaseUrl('https://example.test/api/chat/completions')).toBe(
      'https://example.test/api/chat/completions',
    );
  });

  it.each([
    'ftp://example.test/v1',
    'http://user:pass@example.test/v1',
    'http://example.test/v1?token=secret',
    'http://example.test/v1#fragment',
    'not a URL',
    '   ',
  ])('rejects unsafe or invalid URL %j', (url) => {
    expect(() => normaliseOpenAIBaseUrl(url)).toThrow(/OPENAI_BASE_URL/);
  });
});

describe('resolveOpenAICompatibleConfigFromEnv', () => {
  it('is disabled when ANALYZER_PROVIDER is unset', () => {
    expect(resolveOpenAICompatibleConfigFromEnv()).toEqual({ enabled: false });
  });

  it('returns one validated server-internal configuration', () => {
    process.env.ANALYZER_PROVIDER = 'openai';
    process.env.OPENAI_BASE_URL = 'http://model-host:8000/v1';
    process.env.OPENAI_MODEL = 'local-model';
    process.env.OPENAI_API_KEY = '  secret  ';
    expect(resolveOpenAICompatibleConfigFromEnv()).toEqual({
      enabled: true,
      baseUrl: 'http://model-host:8000/v1',
      endpoint: 'http://model-host:8000/v1/chat/completions',
      model: 'local-model',
      apiKey: 'secret',
    });
  });
});

describe('OpenAICompatibleAnalyzer', () => {
  it('sends the minimal non-streaming request, optional auth, and reports one chunk', async () => {
    const fetchMock = vi.fn().mockResolvedValue(completion('{"nonStory":false}'));
    vi.stubGlobal('fetch', fetchMock);
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'qwen-local',
      apiKey: 'test-key',
    });
    const onChunk = vi.fn();

    await expect(
      analyzer.runNonStoryClassification('openai_success', 1, '# classify this', { onChunk }),
    ).resolves.toEqual({ nonStory: false });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://localhost:8000/v1/chat/completions');
    expect(init.headers).toEqual({
      'Content-Type': 'application/json',
      Authorization: 'Bearer test-key',
    });
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(Object.keys(body)).toEqual(['model', 'messages', 'stream']);
    expect(body.model).toBe('qwen-local');
    expect(body.stream).toBe(false);
    expect(body.messages).toEqual([
      expect.objectContaining({ role: 'system' }),
      { role: 'user', content: '# classify this' },
    ]);
    expect(onChunk).toHaveBeenCalledTimes(1);
    expect(onChunk).toHaveBeenCalledWith(
      expect.objectContaining({ receivedText: '{"nonStory":false}' }),
    );
  });

  it('omits Authorization when OPENAI_API_KEY is blank', async () => {
    const fetchMock = vi.fn().mockResolvedValue(completion('{"nonStory":true}'));
    vi.stubGlobal('fetch', fetchMock);
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
      apiKey: '   ',
    });

    await analyzer.runNonStoryClassification('openai_no_auth', 1, '# prompt', {});
    const headers = (fetchMock.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it('performs one corrective validation retry with assistant history', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(completion('{"nonStory":"maybe"}'))
      .mockResolvedValueOnce(completion('{"nonStory":true}'));
    vi.stubGlobal('fetch', fetchMock);
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
    });
    const onChunk = vi.fn();

    await expect(
      analyzer.runNonStoryClassification('openai_retry', 2, '# prompt', { onChunk }),
    ).resolves.toEqual({ nonStory: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(onChunk).toHaveBeenCalledTimes(2);

    const retryBody = JSON.parse(fetchMock.mock.calls[1][1].body as string) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(retryBody.messages.map((message) => message.role)).toEqual([
      'system',
      'user',
      'assistant',
      'user',
    ]);
    expect(retryBody.messages[2].content).toBe('{"nonStory":"maybe"}');
    expect(retryBody.messages[3].content).toMatch(/schema validation/i);

    const forensic = JSON.parse(
      await readFile(errorPath('openai_retry', 'nonstory-ch2'), 'utf8'),
    ) as { attempt: number; kind: string };
    expect(forensic).toMatchObject({ attempt: 1, kind: 'schema-validation' });
  });

  it('rejects a malformed chat-completions envelope', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"choices":[]}')));
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
    });
    await expect(
      analyzer.runNonStoryClassification('openai_malformed', 1, '# prompt', {}),
    ).rejects.toThrow(/choices\[0\]\.message\.content/);
  });

  it('retains HTTP status, bounds error text, and redacts the API key', async () => {
    const secret = 'super-secret-key';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(`${secret}:${'x'.repeat(10_000)}`, {
          status: 503,
          statusText: 'Unavailable',
        }),
      ),
    );
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
      apiKey: secret,
    });

    const err = await analyzer
      .runNonStoryClassification('openai_http_error', 1, '# prompt', {})
      .then(
        () => null,
        (error: unknown) => error as OpenAICompatibleHttpError,
      );
    expect(err).toBeInstanceOf(OpenAICompatibleHttpError);
    expect(err?.status).toBe(503);
    expect(err?.message).not.toContain(secret);
    expect(err?.message.length).toBeLessThan(4_400);
  });

  it('maps an aborted request to AnalysisAbortedError without calling fetch', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
    });
    const controller = new AbortController();
    controller.abort();

    await expect(
      analyzer.runNonStoryClassification('openai_abort', 1, '# prompt', {
        signal: controller.signal,
      }),
    ).rejects.toBeInstanceOf(AnalysisAbortedError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('ticks onWaiting while a response is pending', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise<Response>((resolve) =>
            setTimeout(() => resolve(completion('{"nonStory":false}')), 550),
          ),
      ),
    );
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
    });
    const onWaiting = vi.fn();
    await analyzer.runNonStoryClassification('openai_waiting', 1, '# prompt', { onWaiting });
    expect(onWaiting).toHaveBeenCalled();
  });

  it('treats escalation failures as best-effort but preserves aborts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(completion('not json')));
    const analyzer = new OpenAICompatibleAnalyzer({
      baseUrl: 'http://localhost:8000/v1',
      model: 'local-model',
    });
    await expect(
      analyzer.runAttributionEscalation('openai_escalation', 1, 0, '# prompt', {}),
    ).resolves.toBeNull();
  });
});
