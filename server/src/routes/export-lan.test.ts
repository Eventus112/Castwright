/* Plan 81 wave 1 — export-lan protocol switching.
 *
 * Asserts that the existing GET /api/export/lan endpoint emits HTTPS URLs
 * on port LAN_HTTPS_PORT when LAN_HTTPS=1 is set, and HTTP URLs on port
 * PORT otherwise. The pre-plan-81 default (no env vars) is the regression
 * baseline. */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import express, { type Express } from 'express';
import request from 'supertest';
import { enumerateLanUrls, exportLanRouter, isLanHttpsEnabled } from './export-lan.js';
import { setLanRuntime } from '../lan-runtime.js';

function makeApp(): Express {
  const app = express();
  app.use('/api/export', exportLanRouter);
  return app;
}

describe('enumerateLanUrls', () => {
  it('defaults to http when no protocol is passed', () => {
    const out = enumerateLanUrls(8080);
    expect(out.protocol).toBe('http');
    expect(out.port).toBe(8080);
    // URLs may be [] in CI sandboxes without LAN interfaces — only assert
    // the prefix when at least one is present.
    if (out.urls.length > 0) {
      expect(out.urls[0]).toMatch(/^http:\/\//);
    }
  });

  it('emits https URLs when protocol is "https"', () => {
    const out = enumerateLanUrls(8443, 'https');
    expect(out.protocol).toBe('https');
    expect(out.port).toBe(8443);
    if (out.urls.length > 0) {
      expect(out.urls[0]).toMatch(/^https:\/\//);
      expect(out.urls[0]).toContain(':8443');
    }
  });
});

describe('isLanHttpsEnabled', () => {
  const savedNodeEnv = process.env.NODE_ENV;
  beforeEach(() => {
    delete process.env.LAN_HTTPS;
  });
  afterEach(() => {
    delete process.env.LAN_HTTPS;
    if (savedNodeEnv === undefined) delete process.env.NODE_ENV;
    else process.env.NODE_ENV = savedNodeEnv;
  });

  it('defaults OFF outside production when env unset (dev/test keep plain HTTP)', () => {
    process.env.NODE_ENV = 'development';
    expect(isLanHttpsEnabled()).toBe(false);
  });
  it('defaults ON in production when env unset (installers get LAN by default)', () => {
    process.env.NODE_ENV = 'production';
    expect(isLanHttpsEnabled()).toBe(true);
  });
  it('explicit LAN_HTTPS=1 wins even outside production', () => {
    process.env.NODE_ENV = 'development';
    process.env.LAN_HTTPS = '1';
    expect(isLanHttpsEnabled()).toBe(true);
  });
  it('explicit LAN_HTTPS=0 wins even in production', () => {
    process.env.NODE_ENV = 'production';
    process.env.LAN_HTTPS = '0';
    expect(isLanHttpsEnabled()).toBe(false);
  });
  it('any EXPLICIT non-1 value disables, even in production (LAN_HTTPS=false is not "on")', () => {
    process.env.NODE_ENV = 'production';
    for (const v of ['true', 'false', 'no', 'off', 'yes', '2']) {
      process.env.LAN_HTTPS = v;
      expect(isLanHttpsEnabled(), `LAN_HTTPS=${v}`).toBe(false);
    }
  });
});

describe('GET /api/export/lan — reflects the ACTUAL bound runtime, not the requested flag', () => {
  afterEach(() => {
    // reset to the module default (loopback HTTP)
    setLanRuntime({ httpsActive: false, port: 8080 });
  });

  it('returns http URLs on the bound port when the server degraded to loopback HTTP', async () => {
    setLanRuntime({ httpsActive: false, port: 8080 });
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    expect(res.body.protocol).toBe('http');
    expect(res.body.port).toBe(8080);
  });

  it('returns https URLs on the bound LAN port when HTTPS is actually active', async () => {
    setLanRuntime({ httpsActive: true, port: 8443 });
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    expect(res.body.protocol).toBe('https');
    expect(res.body.port).toBe(8443);
  });

  it('honours a non-default bound LAN port', async () => {
    setLanRuntime({ httpsActive: true, port: 9443 });
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    expect(res.body.protocol).toBe('https');
    expect(res.body.port).toBe(9443);
  });

  it('prioritises the reachable request Origin ahead of Docker interface URLs', async () => {
    setLanRuntime({ httpsActive: false, port: 8080 });
    const res = await request(makeApp())
      .get('/api/export/lan')
      .set('Origin', 'http://192.168.1.42:8080')
      .set('Host', '172.21.0.2:8080');
    expect(res.status).toBe(200);
    expect(res.body.urls[0]).toBe('http://192.168.1.42:8080');
    expect(res.body.protocol).toBe('http');
    expect(res.body.port).toBe(8080);
  });

  it('uses a valid forwarded host and protocol when Origin is absent', async () => {
    setLanRuntime({ httpsActive: true, port: 8443 });
    const res = await request(makeApp())
      .get('/api/export/lan')
      .set('X-Forwarded-Host', 'castwright.local')
      .set('X-Forwarded-Proto', 'https');
    expect(res.status).toBe(200);
    expect(res.body.urls[0]).toBe('https://castwright.local');
    expect(res.body.protocol).toBe('https');
    expect(res.body.port).toBe(8443);
  });

  it('rejects loopback and malformed request hosts from the advertised URLs', async () => {
    setLanRuntime({ httpsActive: false, port: 8080 });
    const loopback = await request(makeApp())
      .get('/api/export/lan')
      .set('Origin', 'http://localhost:8080')
      .set('X-Forwarded-Host', '127.0.0.1:8080')
      .set('Host', '0.0.0.0:8080');
    expect(loopback.status).toBe(200);
    expect(loopback.body.urls).not.toContain('http://localhost:8080');
    expect(loopback.body.urls).not.toContain('http://127.0.0.1:8080');
    expect(loopback.body.urls).not.toContain('http://0.0.0.0:8080');

    const ipv6Loopback = await request(makeApp())
      .get('/api/export/lan')
      .set('Origin', 'http://[::1]:8080')
      .set('Host', '127.0.0.1:8080');
    expect(ipv6Loopback.status).toBe(200);
    expect(ipv6Loopback.body.urls).not.toContain('http://[::1]:8080');

    for (const mapped of ['[::ffff:7f00:1]', '[::ffff:127.0.0.1]']) {
      const mappedLoopback = await request(makeApp())
        .get('/api/export/lan')
        .set('Origin', `http://${mapped}:8080`)
        .set('Host', '127.0.0.1:8080');
      expect(mappedLoopback.status).toBe(200);
      expect(mappedLoopback.body.urls).not.toContain(`http://[::ffff:7f00:1]:8080`);
    }

    const tailscale = await request(makeApp())
      .get('/api/export/lan')
      .set('Origin', 'http://100.100.10.20:8080');
    expect(tailscale.status).toBe(200);
    expect(tailscale.body.urls[0]).toBe('http://100.100.10.20:8080');

    const malformed = await request(makeApp())
      .get('/api/export/lan')
      .set('X-Forwarded-Host', 'user@example.com')
      .set('Host', '127.0.0.1:8080');
    expect(malformed.status).toBe(200);
    expect(malformed.body.urls).not.toContain('http://user@example.com');
  });

  it('the CRITICAL regression: LAN requested but certs missing → HTTP, never advertises https:8443', async () => {
    // requested flag on, but the server bound loopback HTTP because certs were absent
    process.env.LAN_HTTPS = '1';
    setLanRuntime({ httpsActive: false, port: 8080 });
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.body.protocol).toBe('http');
    expect(res.body.port).toBe(8080);
    delete process.env.LAN_HTTPS;
  });
});

describe('GET /api/export/lan — CORS-from-LAN-origin acceptance', () => {
  /* The endpoint is JSON, no auth, no body-handling — verify that an
     `Origin` header from a typical LAN address doesn't trip any existing
     middleware. Production runs same-origin (Node serves both bundle +
     API on the same port) so this is mostly a regression seatbelt:
     adding CORS lockdowns to other routes must not accidentally affect
     this one. */
  it('accepts requests from an https LAN origin', async () => {
    const res = await request(makeApp())
      .get('/api/export/lan')
      .set('Origin', 'https://192.168.1.50:8443');
    expect(res.status).toBe(200);
  });
});

describe('GET /api/export/lan — srv-20 pairing payload (token + caFingerprint)', () => {
  const origToken = process.env.LAN_AUTH_TOKEN;
  afterEach(() => {
    if (origToken === undefined) delete process.env.LAN_AUTH_TOKEN;
    else process.env.LAN_AUTH_TOKEN = origToken;
  });

  it('omits token when LAN_AUTH_TOKEN is unset', async () => {
    delete process.env.LAN_AUTH_TOKEN;
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    expect(res.body.token).toBeUndefined();
  });

  it('surfaces the token when LAN_AUTH_TOKEN is set', async () => {
    process.env.LAN_AUTH_TOKEN = 'pair-secret-123';
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    expect(res.body.token).toBe('pair-secret-123');
  });

  it('omits token for an empty LAN_AUTH_TOKEN', async () => {
    process.env.LAN_AUTH_TOKEN = '';
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    expect(res.body.token).toBeUndefined();
  });

  it('caFingerprint, when present, is a non-empty fingerprint string (best-effort)', async () => {
    const res = await request(makeApp()).get('/api/export/lan');
    expect(res.status).toBe(200);
    if (res.body.caFingerprint !== undefined) {
      expect(typeof res.body.caFingerprint).toBe('string');
      expect(res.body.caFingerprint.length).toBeGreaterThan(0);
    }
  });
});
