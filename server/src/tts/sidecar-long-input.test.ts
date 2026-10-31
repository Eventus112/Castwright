import { afterEach, describe, expect, it } from 'vitest';
import { createServer, type Server, type ServerResponse } from 'node:http';
import { SidecarTtsProvider } from './sidecar.js';
import type { SynthesizeBatchInput, SynthesizeInput, TtsEngine } from './index.js';

interface WireItem {
  text: string;
  voice: string;
  instruct?: string;
  emotion?: string;
  language?: string;
}
interface WireBody extends WireItem {
  engine: string;
  model: string;
  liveInstruct?: boolean;
  items?: WireItem[];
}
const LONG_TEXT = '«Это длинный абзац, который нужно прочитать целиком?»\n\n— Да, — ответила она.\n'
  .repeat(60)
  .slice(0, 2703);
const SINGLE: SynthesizeInput = {
  text: LONG_TEXT,
  voiceName: 'qwen-clone',
  modelKey: 'qwen3-tts-1.7b',
  language: 'ru',
  cloned: true,
};
const MIXED: SynthesizeBatchInput = {
  items: [
    { text: 'Before.', voiceName: 'qwen-narrator' },
    {
      text: LONG_TEXT,
      voiceName: 'qwen-clone',
      cloned: true,
      instruct: 'Quietly.',
      emotion: 'whisper',
    },
    { text: 'After.', voiceName: 'qwen-other', instruct: 'Firmly.', emotion: 'angry' },
  ],
  modelKey: 'qwen3-tts-1.7b',
  liveInstruct: true,
  language: 'ru',
};
let server: Server | undefined;
let requests: { path: string; body: WireBody; raw: string }[] = [];
type Reply = (res: ServerResponse, body: WireBody, index: number) => void;
function respond(
  res: ServerResponse,
  body: WireBody,
  index: number,
  sampleRate = 24000,
  metrics: { genMs?: number; audioMs?: number } = { genMs: 3, audioMs: 5 },
) {
  const pcm = Buffer.from([index, 0]);
  if (body.items) {
    const pcms = body.items.map(() => pcm);
    res.end(
      Buffer.concat([
        Buffer.from(
          JSON.stringify({ sampleRate, lengths: pcms.map((p) => p.length), ...metrics }) + '\n',
        ),
        ...pcms,
      ]),
    );
  } else {
    res.setHeader('content-type', `audio/L16;codec=pcm;rate=${sampleRate}`);
    res.setHeader('x-sample-rate', sampleRate);
    res.end(pcm);
  }
}
async function provider(reply: Reply = respond, engine: TtsEngine = 'qwen') {
  requests = [];
  server = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const raw = Buffer.concat(chunks).toString();
    const body = JSON.parse(raw) as WireBody;
    requests.push({ path: req.url!, body, raw });
    reply(res, body, requests.length);
  });
  await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Missing test server address');
  return new SidecarTtsProvider({
    url: `http://127.0.0.1:${address.port}`,
    engine,
    engineSynths: new Map(),
  });
}
afterEach(async () => {
  if (server) {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  }
});
function expectBoundedLossless(texts: string[], original: string) {
  expect(texts.length).toBeGreaterThan(1);
  expect(texts.join('')).toBe(original);
  expect(texts.every((text) => Array.from(text).length <= 600 && text.length > 0)).toBe(true);
}
describe('Qwen long provider input', () => {
  it('fragments single input preserving voice/model/language and joins PCM', async () => {
    const result = await (await provider()).synthesize(SINGLE);
    expectBoundedLossless(
      requests.map((r) => r.body.text),
      LONG_TEXT,
    );
    for (const { path, body } of requests) {
      expect(path).toBe('/synthesize');
      expect(body).toEqual({
        engine: 'qwen',
        model: '1.7b',
        voice: 'qwen-clone',
        text: body.text,
        language: 'Russian',
      });
    }
    expect(result).toEqual({
      pcm: Buffer.concat(requests.map((_, i) => Buffer.from([i + 1, 0]))),
      sampleRate: 24000,
      mimeType: 'audio/L16;codec=pcm;rate=24000',
    });
  });
  it('keeps mixed batch order, instruct, emotion and clone language with one PCM per original', async () => {
    const result = await (await provider()).synthesizeBatch(MIXED);
    const parts = requests.slice(1, -1);
    expectBoundedLossless(
      parts.map((r) => r.body.items![0].text),
      LONG_TEXT,
    );
    expect(requests[0].body.items).toEqual([{ text: 'Before.', voice: 'qwen-narrator' }]);
    expect(requests.at(-1)!.body.items).toEqual([
      { text: 'After.', voice: 'qwen-other', instruct: 'Firmly.', emotion: 'angry' },
    ]);
    for (const { path, body } of requests) {
      expect(path).toBe('/synthesize-batch');
      expect(body.engine).toBe('qwen');
      expect(body.model).toBe('1.7b');
      expect(body.liveInstruct).toBe(true);
      expect(body.items).toHaveLength(1);
    }
    for (const { body } of parts)
      expect(body.items![0]).toEqual({
        text: body.items![0].text,
        voice: 'qwen-clone',
        language: 'Russian',
        instruct: 'Quietly.',
        emotion: 'whisper',
      });
    expect(result).toEqual({
      pcms: [
        Buffer.from([1, 0]),
        Buffer.concat(parts.map((_, i) => Buffer.from([i + 2, 0]))),
        Buffer.from([requests.length, 0]),
      ],
      sampleRate: 24000,
      genMs: requests.length * 3,
      audioMs: requests.length * 5,
    });
  });
  it.each([833, 1096])(
    'fragments a %i-character passage on the 0.6B designed-voice path',
    async (length) => {
      const p = await provider();
      const text = LONG_TEXT.slice(0, length);
      await p.synthesize({ ...SINGLE, text, modelKey: 'qwen3-tts-0.6b', cloned: false });
      expectBoundedLossless(
        requests.map((r) => r.body.text),
        text,
      );
      for (const { body } of requests) {
        expect(body.model).toBe('0.6b');
        expect(body).not.toHaveProperty('language');
      }
    },
  );
  it('does not report incomplete totals when an older sidecar omits metrics', async () => {
    const p = await provider((res, body, i) =>
      respond(res, body, i, 24000, i === 2 ? {} : { genMs: 3, audioMs: 5 }),
    );
    const result = await p.synthesizeBatch(MIXED);
    expect(requests.length).toBeGreaterThan(2);
    expect(result.genMs).toBeUndefined();
    expect(result.audioMs).toBeUndefined();
  });
  for (const batch of [false, true]) {
    const run = (p: SidecarTtsProvider, signal?: AbortSignal) =>
      batch
        ? p.synthesizeBatch({ ...MIXED, items: [MIXED.items[1]], signal })
        : p.synthesize({ ...SINGLE, signal });
    it(`${batch ? 'batch' : 'single'} stops on fragment error without partial success`, async () => {
      const p = await provider((res, body, i) => {
        if (i === 2) {
          res.statusCode = 400;
          res.end('fragment failed');
        } else respond(res, body, i);
      });
      await expect(run(p)).rejects.toThrow('fragment failed');
      expect(requests).toHaveLength(2);
    });
    it(`${batch ? 'batch' : 'single'} rejects inconsistent fragment sample rates`, async () => {
      const p = await provider((res, body, i) => respond(res, body, i, i === 2 ? 22050 : 24000));
      await expect(run(p)).rejects.toThrow(/sample rate/i);
      expect(requests).toHaveLength(2);
    });
    it(`${batch ? 'batch' : 'single'} honors abort during a fragment and never starts the next`, async () => {
      const ac = new AbortController();
      const p = await provider((res, body, i) => {
        if (i === 2) ac.abort();
        else respond(res, body, i);
      });
      await expect(run(p, ac.signal)).rejects.toMatchObject({ name: 'AbortError' });
      expect(requests).toHaveLength(2);
    });
    it(`${batch ? 'batch' : 'single'} checks cancellation before the first fragment`, async () => {
      const p = await provider();
      const ac = new AbortController();
      ac.abort();
      await expect(run(p, ac.signal)).rejects.toMatchObject({ name: 'AbortError' });
      expect(requests).toHaveLength(0);
    });
  }
  it('rejects differing sample rates between original batch items', async () => {
    const p = await provider((res, body, i) => respond(res, body, i, i === 2 ? 22050 : 24000));
    await expect(p.synthesizeBatch(MIXED)).rejects.toThrow(/sample rate/i);
    expect(requests).toHaveLength(2);
  });
  it('leaves short single and batch request bytes unchanged', async () => {
    const p = await provider();
    const single = await p.synthesize({ ...SINGLE, text: 'Short.' });
    const batch = await p.synthesizeBatch({ ...MIXED, items: [MIXED.items[0], MIXED.items[2]] });
    expect(requests.map((r) => r.raw)).toEqual([
      '{"engine":"qwen","model":"1.7b","voice":"qwen-clone","text":"Short.","language":"Russian"}',
      '{"engine":"qwen","model":"1.7b","liveInstruct":true,"items":[{"voice":"qwen-narrator","text":"Before."},{"voice":"qwen-other","text":"After.","instruct":"Firmly.","emotion":"angry"}]}',
    ]);
    expect(single.pcm).toEqual(Buffer.from([1, 0]));
    expect(batch).toEqual({
      pcms: [Buffer.from([2, 0]), Buffer.from([2, 0])],
      sampleRate: 24000,
      genMs: 3,
      audioMs: 5,
    });
  });
  it('leaves a long non-Qwen request whole', async () => {
    const p = await provider(respond, 'coqui');
    await p.synthesize({ ...SINGLE, modelKey: 'coqui-xtts-v2' });
    expect(requests).toHaveLength(1);
    expect(requests[0].body.text).toBe(LONG_TEXT);
    expect(requests[0].body.language).toBe('ru');
  });
});
