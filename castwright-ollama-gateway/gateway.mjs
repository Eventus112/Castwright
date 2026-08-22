import http from 'node:http';

const host = process.env.GATEWAY_HOST ?? '127.0.0.1';
const port = Number(process.env.GATEWAY_PORT ?? 11435);
const swapUrl = (process.env.LLAMA_SWAP_URL ?? 'http://127.0.0.1:5001').replace(/\/+$/, '');
const publicModel = process.env.GATEWAY_MODEL ?? 'qwen3.8:27b';
const upstreamModel = process.env.UPSTREAM_MODEL ?? 'qwen';
const runningModel = process.env.RUNNING_MODEL ?? 'qwen';
const modelSize = Number(process.env.MODEL_SIZE_BYTES ?? 21492695040);
const modelVram = Number(process.env.MODEL_VRAM_BYTES ?? 27250393088);
const maxOutputTokens = Number(process.env.MAX_OUTPUT_TOKENS ?? 8192);
const requestTimeoutMs = Number(process.env.REQUEST_TIMEOUT_MS ?? 1800000);

const details = {
  parent_model: '',
  format: 'ninfer',
  family: 'qwen3.8',
  families: ['qwen3.8'],
  parameter_size: '27B',
  quantization_level: 'NVFP4',
};

function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 32 * 1024 * 1024) throw new Error('request body is too large');
    chunks.push(chunk);
  }
  if (size === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function asOpenAiMessages(body) {
  const messages = Array.isArray(body.messages)
    ? body.messages.map(({ role, content }) => ({ role, content }))
    : [{ role: 'user', content: String(body.prompt ?? '') }];

  if (body.format && typeof body.format === 'object') {
    messages.unshift({
      role: 'system',
      content: [
        'STRICT OUTPUT CONTRACT:',
        'Return only one valid JSON value matching the JSON Schema below.',
        'Use ordinary ASCII double quotes (\") for every JSON key and string value.',
        'Never use guillemets (« »), Markdown fences, comments, or text outside the JSON.',
        JSON.stringify(body.format),
      ].join('\n'),
    });
  } else if (body.format === 'json') {
    messages.unshift({
      role: 'system',
      content: 'Return only one valid JSON value. Use ASCII double quotes; no Markdown fences or surrounding text.',
    });
  }
  return messages;
}

function openAiRequest(body, stream) {
  const requested = Number(body?.options?.num_predict);
  const maxTokens = Number.isFinite(requested) && requested > 0
    ? Math.min(requested, maxOutputTokens)
    : maxOutputTokens;
  return {
    model: upstreamModel,
    messages: asOpenAiMessages(body),
    stream,
    max_tokens: maxTokens,
    ...(stream ? { stream_options: { include_usage: true } } : {}),
  };
}

async function upstreamFetch(path, init = {}) {
  return fetch(`${swapUrl}${path}`, {
    ...init,
    signal: init.signal ?? AbortSignal.timeout(requestTimeoutMs),
  });
}

async function chat(req, res, body) {
  const stream = body.stream !== false;
  const upstream = await upstreamFetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(openAiRequest(body, stream)),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    sendJson(res, upstream.status, { error: `llama-swap: ${text.slice(0, 2000)}` });
    return;
  }

  if (!stream) {
    const json = await upstream.json();
    const choice = json.choices?.[0] ?? {};
    sendJson(res, 200, {
      model: publicModel,
      created_at: new Date().toISOString(),
      message: { role: 'assistant', content: choice.message?.content ?? '' },
      done: true,
      done_reason: choice.finish_reason ?? 'stop',
      prompt_eval_count: json.usage?.prompt_tokens ?? 0,
      eval_count: json.usage?.completion_tokens ?? 0,
    });
    return;
  }

  if (!upstream.body) {
    sendJson(res, 502, { error: 'llama-swap returned an empty stream' });
    return;
  }

  res.writeHead(200, {
    'content-type': 'application/x-ndjson; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let doneReason = 'stop';
  let completionTokens = 0;
  let promptTokens = 0;

  const emit = (value) => res.write(`${JSON.stringify(value)}\n`);
  const consume = (line) => {
    line = line.trim();
    if (!line || line.startsWith(':') || !line.startsWith('data:')) return;
    const data = line.slice(5).trim();
    if (!data || data === '[DONE]') return;
    const json = JSON.parse(data);
    if (json.error) throw new Error(json.error.message ?? JSON.stringify(json.error));
    const choice = json.choices?.[0];
    const content = choice?.delta?.content;
    if (content) {
      emit({
        model: publicModel,
        created_at: new Date().toISOString(),
        message: { role: 'assistant', content },
        done: false,
      });
    }
    if (choice?.finish_reason) doneReason = choice.finish_reason;
    completionTokens = json.usage?.completion_tokens ?? completionTokens;
    promptTokens = json.usage?.prompt_tokens ?? promptTokens;
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        consume(line);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) consume(buffer);
    emit({
      model: publicModel,
      created_at: new Date().toISOString(),
      message: { role: 'assistant', content: '' },
      done: true,
      done_reason: doneReason,
      prompt_eval_count: promptTokens,
      eval_count: completionTokens,
    });
    res.end();
  } catch (error) {
    emit({ error: `gateway stream error: ${error.message}` });
    res.end();
  } finally {
    reader.releaseLock();
  }
}

async function runningModels() {
  const response = await upstreamFetch('/running', { signal: AbortSignal.timeout(5000) });
  if (!response.ok) return [];
  const json = await response.json();
  const running = Array.isArray(json.running) ? json.running : [];
  return running.some((entry) =>
    entry.model === upstreamModel ||
    entry.id === upstreamModel ||
    entry.model === runningModel ||
    entry.id === runningModel
  )
    ? [{
        name: publicModel,
        model: publicModel,
        size: modelSize,
        size_vram: modelVram,
        digest: 'ninfer-qwen3.8-27b-nvfp4',
        details,
        expires_at: '9999-12-31T23:59:59Z',
      }]
    : [];
}

async function generate(req, res, body) {
  if (Number(body.keep_alive) === 0 && !body.prompt) {
    const unloaded = await upstreamFetch('/unload', { signal: AbortSignal.timeout(120000) });
    if (!unloaded.ok) {
      sendJson(res, unloaded.status, { error: await unloaded.text() });
      return;
    }
    sendJson(res, 200, { model: publicModel, done: true, done_reason: 'unload' });
    return;
  }

  const warmBody = {
    ...body,
    messages: [{ role: 'user', content: body.prompt || 'Reply only with OK.' }],
    stream: false,
    options: { ...(body.options ?? {}), num_predict: body.prompt ? undefined : 1 },
  };
  await chat(req, res, warmBody);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
    if (req.method === 'GET' && url.pathname === '/api/version') {
      sendJson(res, 200, { version: 'castwright-llama-swap-gateway/1.0.0' });
    } else if (req.method === 'GET' && url.pathname === '/api/tags') {
      sendJson(res, 200, {
        models: [{
          name: publicModel,
          model: publicModel,
          modified_at: new Date(0).toISOString(),
          size: modelSize,
          digest: 'ninfer-qwen3.8-27b-nvfp4',
          details,
        }],
      });
    } else if (req.method === 'GET' && url.pathname === '/api/ps') {
      sendJson(res, 200, { models: await runningModels() });
    } else if (req.method === 'GET' && url.pathname === '/health') {
      const response = await upstreamFetch('/health', { signal: AbortSignal.timeout(5000) });
      sendJson(res, response.ok ? 200 : 503, { status: response.ok ? 'ok' : 'unavailable' });
    } else if (req.method === 'POST' && url.pathname === '/api/chat') {
      await chat(req, res, await readJson(req));
    } else if (req.method === 'POST' && url.pathname === '/api/generate') {
      await generate(req, res, await readJson(req));
    } else {
      sendJson(res, 404, { error: 'not found' });
    }
  } catch (error) {
    if (!res.headersSent) sendJson(res, 502, { error: error.message });
    else res.end();
  }
});

server.listen(port, host, () => {
  console.log(`Castwright Ollama gateway listening on http://${host}:${port}; upstream ${swapUrl}/${upstreamModel}`);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
