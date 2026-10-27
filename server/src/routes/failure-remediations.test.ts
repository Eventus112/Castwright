import { describe, it, expect } from 'vitest';
import { FAILURE_REMEDIATIONS } from './failure-remediations.js';

/* #1979 — the Help view's static 'cloned-voice-broken' remediation used to
   list re-upload / restore consent / switch engine / reassign, and never
   mentioned re-running the clone and checking the sidecar log — the remedy
   #1978 added for 'derive-failed'. The live render path short-circuits to
   UnresolvableClonedVoiceError's own message (generation.ts), so this static
   copy is the fallback the offline Help view shows; it must not omit the
   fix for the failure a user is most likely to open Help about. */
describe('FAILURE_REMEDIATIONS.cloned-voice-broken (#1979)', () => {
  it('names re-running the clone and checking the sidecar log', () => {
    const { remediation } = FAILURE_REMEDIATIONS['cloned-voice-broken'];
    expect(remediation).toMatch(/re-run the clone/i);
    expect(remediation).toMatch(/sidecar log/i);
  });

  it('still carries every pre-existing remedy — the addition does not replace them', () => {
    const { remediation } = FAILURE_REMEDIATIONS['cloned-voice-broken'];
    expect(remediation).toMatch(/re-upload/i);
    expect(remediation).toMatch(/restore consent/i);
    expect(remediation).toMatch(/switch the book/i);
    expect(remediation).toMatch(/reassign the character/i);
  });
});

describe('FAILURE_REMEDIATIONS.cuda-poisoned', () => {
  it('asks for a healthy restarted engine and the original log, without guessing at text or restart timing', () => {
    const { userMessage, remediation } = FAILURE_REMEDIATIONS['cuda-poisoned'];
    for (const copy of [userMessage, remediation]) {
      expect(copy).toMatch(/restart/i);
      expect(copy).toMatch(/pill.*green|green.*pill/i);
      expect(copy).toMatch(/log/i);
      expect(copy).not.toMatch(/10 seconds|auto-restart|usually|zero-width|control char/i);
    }
    expect(remediation).toMatch(/model.*input.*memory.*driver/i);
  });
});
