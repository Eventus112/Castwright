/* OpenAI-compatible analyzer for self-hosted/local chat-completions servers.
   It deliberately uses the small, widely-supported non-streaming request
   shape instead of an SDK or provider-specific structured-output options. */

import { writeFile } from 'node:fs/promises';
import type { z } from 'zod';
import {
  stage1Schema,
  stage1ChapterSchema,
  stage2ChapterSchema,
  emotionAnnotationSchema,
  scriptReviewSchema,
  stage3ChapterSchema,
  escalationSchema,
  nonStoryClassificationSchema,
  type Stage1Output,
  type Stage1ChapterOutput,
  type Stage2ChapterOutput,
  type EmotionAnnotationOutput,
  type ScriptReviewOutput,
  type Stage3ChapterOutput,
  type EscalationOutput,
  type NonStoryClassificationOutput,
} from '../handoff/schemas.js';
import {
  writeInbox,
  errorPath,
  rawAttemptPath,
  stage2HandoffKey,
  type HandoffKey,
} from '../handoff/protocol.js';
import type { Analyzer, StageCall, StageChunkInfo } from './index.js';
import { AnalysisAbortedError } from './ollama.js';
import {
  buildRetryMessage,
  buildSystemInstruction,
  loadSkill,
  parseAndValidate,
  persistResponse,
  summariseDetail,
  type SkillName,
} from './gemini.js';

const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
const MAX_ERROR_BYTES = 4 * 1024;

type ChatMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

export interface OpenAICompatibleOptions {
  baseUrl: string;
  model: string;
  apiKey?: string;
}

export type OpenAICompatibleEnvConfig =
  | { enabled: false }
  | {
      enabled: true;
      baseUrl: string;
      endpoint: string;
      model: string;
      apiKey: string;
    };

export class OpenAICompatibleHttpError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'OpenAICompatibleHttpError';
  }
}

export function normaliseOpenAIBaseUrl(raw: string): string {
  const value = raw.trim();
  if (!value) throw new Error('OPENAI_BASE_URL is required when ANALYZER_PROVIDER=openai.');

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error('OPENAI_BASE_URL must be a valid http:// or https:// URL.');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('OPENAI_BASE_URL must use http:// or https://.');
  }
  if (url.username || url.password) {
    throw new Error('OPENAI_BASE_URL must not contain credentials. Use OPENAI_API_KEY instead.');
  }
  if (url.search || url.hash) {
    throw new Error('OPENAI_BASE_URL must not contain a query string or fragment.');
  }

  const path = url.pathname.replace(/\/+$/, '');
  url.pathname = path.endsWith('/chat/completions') ? path : `${path}/chat/completions`;
  return url.toString();
}

function requireModel(raw: string): string {
  const model = raw.trim();
  if (!model) throw new Error('OPENAI_MODEL is required when ANALYZER_PROVIDER=openai.');
  return model;
}

/** Single validation seam for analyzer selection and readiness/config layers.
    The key is returned for server-internal request construction only; callers
    must never serialize this object into an API response or log it. */
export function resolveOpenAICompatibleConfigFromEnv(): OpenAICompatibleEnvConfig {
  const provider = process.env.ANALYZER_PROVIDER?.trim();
  if (!provider) return { enabled: false };
  if (provider.toLowerCase() !== 'openai') {
    throw new Error(
      `Unknown ANALYZER_PROVIDER "${provider}". Supported values: openai (or leave it unset).`,
    );
  }

  const baseUrl = process.env.OPENAI_BASE_URL?.trim() ?? '';
  const model = requireModel(process.env.OPENAI_MODEL ?? '');
  const endpoint = normaliseOpenAIBaseUrl(baseUrl);
  return {
    enabled: true,
    baseUrl,
    endpoint,
    model,
    apiKey: process.env.OPENAI_API_KEY?.trim() ?? '',
  };
}

function redact(value: string, secret: string): string {
  return secret ? value.split(secret).join('[redacted]') : value;
}

async function readBoundedBody(
  response: Response,
  maxBytes: number,
): Promise<{ text: string; truncated: boolean }> {
  if (!response.body) {
    const text = await response.text();
    return { text: text.slice(0, maxBytes), truncated: text.length > maxBytes };
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = '';
  let received = 0;
  let truncated = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      const remaining = maxBytes - received;
      if (value.byteLength > remaining) {
        if (remaining > 0) text += decoder.decode(value.subarray(0, remaining), { stream: true });
        truncated = true;
        await reader.cancel().catch(() => undefined);
        break;
      }
      received += value.byteLength;
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    return { text, truncated };
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // The reader may already have been released by cancellation.
    }
  }
}

export class OpenAICompatibleAnalyzer implements Analyzer {
  private readonly endpoint: string;
  private readonly model: string;
  private readonly apiKey: string;

  constructor(opts: OpenAICompatibleOptions) {
    this.endpoint = normaliseOpenAIBaseUrl(opts.baseUrl);
    this.model = requireModel(opts.model);
    this.apiKey = opts.apiKey?.trim() ?? '';
  }

  runStage1(manuscriptId: string, promptMd: string, call: StageCall): Promise<Stage1Output> {
    return this.runStage(manuscriptId, '1', 'whole_book_stage1', promptMd, stage1Schema, call);
  }

  runStage1Chapter(
    manuscriptId: string,
    chapterId: number,
    promptMd: string,
    call: StageCall,
  ): Promise<Stage1ChapterOutput> {
    return this.runStage(
      manuscriptId,
      `1-ch${chapterId}`,
      'per_chapter_stage1',
      promptMd,
      stage1ChapterSchema,
      call,
    );
  }

  runStage2Chapter(
    manuscriptId: string,
    chapterId: number,
    promptMd: string,
    call: StageCall,
  ): Promise<Stage2ChapterOutput> {
    return this.runStage(
      manuscriptId,
      stage2HandoffKey(chapterId, call.stage2CallSeq),
      'per_chapter_stage2',
      promptMd,
      stage2ChapterSchema,
      call,
    );
  }

  runEmotionChapter(
    manuscriptId: string,
    chapterId: number,
    promptMd: string,
    call: StageCall,
  ): Promise<EmotionAnnotationOutput> {
    return this.runStage(
      manuscriptId,
      `emotion-ch${chapterId}`,
      'emotion_annotation',
      promptMd,
      emotionAnnotationSchema,
      call,
    );
  }

  runScriptReviewChapter(
    manuscriptId: string,
    chapterId: number,
    promptMd: string,
    call: StageCall,
  ): Promise<ScriptReviewOutput> {
    return this.runStage(
      manuscriptId,
      `review-ch${chapterId}`,
      'script_review',
      promptMd,
      scriptReviewSchema,
      call,
    );
  }

  runStage3Chapter(
    manuscriptId: string,
    chapterId: number,
    promptMd: string,
    call: StageCall,
  ): Promise<Stage3ChapterOutput> {
    return this.runStage(
      manuscriptId,
      `instruct-ch${chapterId}`,
      'instruct_annotation',
      promptMd,
      stage3ChapterSchema,
      call,
    );
  }

  runNonStoryClassification(
    manuscriptId: string,
    chapterId: number,
    promptMd: string,
    call: StageCall,
  ): Promise<NonStoryClassificationOutput> {
    return this.runStage(
      manuscriptId,
      `nonstory-ch${chapterId}`,
      'non_story_classification',
      promptMd,
      nonStoryClassificationSchema,
      call,
    );
  }

  async runAttributionEscalation(
    manuscriptId: string,
    chapterId: number,
    windowIndex: number,
    prompt: string,
    call: StageCall,
  ): Promise<EscalationOutput | null> {
    const key = `escalation-ch${chapterId}-w${windowIndex}` as const;
    await writeInbox(manuscriptId, key, prompt);
    return this.withWaiting(call, async () => {
      let text: string;
      try {
        text = await this.chat([{ role: 'user', content: prompt }], call);
      } catch (err) {
        if (err instanceof AnalysisAbortedError) throw err;
        console.warn(
          `[openai-compatible] ${this.model} ${key} produced no usable response: ${(err as Error)?.message ?? err}`,
        );
        return null;
      }
      const attempt = parseAndValidate(text, escalationSchema);
      if (!attempt.ok) {
        console.warn(`[openai-compatible] ${this.model} ${key} failed to parse: ${attempt.kind}`);
        return null;
      }
      await persistResponse(manuscriptId, key, text);
      return attempt.value;
    });
  }

  private async runStage<T>(
    manuscriptId: string,
    key: HandoffKey,
    skillName: SkillName,
    promptMd: string,
    validationSchema: z.ZodType<T>,
    call: StageCall,
  ): Promise<T> {
    await writeInbox(manuscriptId, key, promptMd);
    const skill = await loadSkill(skillName);
    const systemInstruction = buildSystemInstruction(skill, call.language, skillName);

    return this.withWaiting(call, async () => {
      const firstText = await this.chat(
        [
          { role: 'system', content: systemInstruction },
          { role: 'user', content: promptMd },
        ],
        call,
      );
      const firstAttempt = parseAndValidate(firstText, validationSchema);
      if (firstAttempt.ok) {
        await persistResponse(manuscriptId, key, firstText);
        return firstAttempt.value;
      }

      await writeFile(rawAttemptPath(manuscriptId, key, 1), firstText, 'utf8');
      await writeFile(
        errorPath(manuscriptId, key),
        JSON.stringify(
          { kind: firstAttempt.kind, detail: firstAttempt.detail, attempt: 1 },
          null,
          2,
        ),
        'utf8',
      );

      const secondText = await this.chat(
        [
          { role: 'system', content: systemInstruction },
          { role: 'user', content: promptMd },
          { role: 'assistant', content: firstText },
          { role: 'user', content: buildRetryMessage(firstAttempt) },
        ],
        call,
      );
      const secondAttempt = parseAndValidate(secondText, validationSchema);
      if (secondAttempt.ok) {
        await persistResponse(manuscriptId, key, secondText);
        return secondAttempt.value;
      }

      await writeFile(rawAttemptPath(manuscriptId, key, 2), secondText, 'utf8');
      await writeFile(
        errorPath(manuscriptId, key),
        JSON.stringify(
          {
            kind: secondAttempt.kind,
            detail: secondAttempt.detail,
            attempt: 2,
            firstError: { kind: firstAttempt.kind, detail: firstAttempt.detail },
          },
          null,
          2,
        ),
        'utf8',
      );
      throw new Error(
        `OpenAI-compatible ${this.model} ${key} failed validation after retry: ${secondAttempt.kind} — ${summariseDetail(secondAttempt.detail)}`,
      );
    });
  }

  private async withWaiting<T>(call: StageCall, fn: () => Promise<T>): Promise<T> {
    const start = Date.now();
    const tick = call.onWaiting
      ? setInterval(() => call.onWaiting!(Date.now() - start), 500)
      : null;
    try {
      return await fn();
    } finally {
      if (tick) clearInterval(tick);
    }
  }

  private async chat(messages: ChatMessage[], call: StageCall): Promise<string> {
    if (call.signal?.aborted) {
      throw new AnalysisAbortedError(
        `OpenAI-compatible ${this.model} call aborted before fetch (client disconnected).`,
      );
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const start = Date.now();
    let response: Response;
    try {
      response = await fetch(this.endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model: this.model, messages, stream: false }),
        signal: call.signal,
      });
    } catch (err) {
      if (call.signal?.aborted || (err as Error)?.name === 'AbortError') {
        throw new AnalysisAbortedError(
          `OpenAI-compatible ${this.model} fetch aborted (client disconnected).`,
        );
      }
      const detail = redact((err as Error)?.message ?? String(err), this.apiKey).slice(0, 500);
      throw new Error(`OpenAI-compatible request failed: ${detail}`);
    }

    let bounded: { text: string; truncated: boolean };
    try {
      bounded = await readBoundedBody(response, response.ok ? MAX_RESPONSE_BYTES : MAX_ERROR_BYTES);
    } catch (err) {
      if (call.signal?.aborted || (err as Error)?.name === 'AbortError') {
        throw new AnalysisAbortedError(
          `OpenAI-compatible ${this.model} response aborted (client disconnected).`,
        );
      }
      throw err;
    }

    if (!response.ok) {
      const suffix = bounded.truncated ? '…' : '';
      const detail = redact(bounded.text, this.apiKey);
      throw new OpenAICompatibleHttpError(
        redact(
          `OpenAI-compatible endpoint returned ${response.status} ${response.statusText}: ${detail}${suffix}`,
          this.apiKey,
        ).slice(0, MAX_ERROR_BYTES + 200),
        response.status,
      );
    }
    if (bounded.truncated) {
      throw new Error(
        `OpenAI-compatible ${this.model} response exceeded ${MAX_RESPONSE_BYTES} bytes.`,
      );
    }

    let envelope: unknown;
    try {
      envelope = JSON.parse(bounded.text);
    } catch {
      throw new Error('OpenAI-compatible endpoint returned malformed JSON.');
    }
    const content = (envelope as { choices?: Array<{ message?: { content?: unknown } }> })
      ?.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || content.length === 0) {
      throw new Error(
        'OpenAI-compatible endpoint returned a malformed response: choices[0].message.content is missing.',
      );
    }

    const elapsedMs = Date.now() - start;
    const info: StageChunkInfo = {
      receivedBytes: Buffer.byteLength(content, 'utf8'),
      receivedText: content,
      sinceLastChunkMs: elapsedMs,
      elapsedMs,
    };
    call.onChunk?.(info);
    return content;
  }
}
