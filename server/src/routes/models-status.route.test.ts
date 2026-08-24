import { describe, it, expect, vi, beforeEach } from 'vitest';

/* sidecarVenvPresent, getActiveSupervisor, probeSidecarHealth, and
   findPython312 are vi.fn() spies (not plain returns) so call-count and
   per-test overrides are assertable — see the "single probe" and
   "venv-gated probe" invariants pinned below, mirroring
   setup-readiness.orchestration.test.ts's pattern for the same class of
   regression on the sibling route. */
const sidecarVenvPresent = vi.fn();
const getActiveSupervisor = vi.fn();
const probeSidecarHealth = vi.fn();
const findPython312 = vi.fn();
/* #1963 — a controllable spy (not the plain `() => false` the other two
   disk probes below use) so the coqui-import-honesty regression test can
   flip the disk probe to true for the one case that needs packageOnDisk
   true to observe packageBroken flip. */
const coquiPackageInstalledDisk = vi.fn(() => false);

// Mock the I/O seams so the route is deterministic.
vi.mock('../diagnostics/venv.js', () => ({ sidecarVenvPresent: () => sidecarVenvPresent() }));
vi.mock('../tts/sidecar-supervisor.js', () => ({ getActiveSupervisor: () => getActiveSupervisor() }));
vi.mock('../gpu/device-total.js', () => ({ getDeviceTotalVramMb: () => 8192 }));
vi.mock('./sidecar-health.js', () => ({ probeSidecarHealth: () => probeSidecarHealth() }));
/* Mocked so the venv-absent path (which falls through to
   probePython312Cached() -> findPython312()) doesn't spawn a real
   interpreter-hunt subprocess in a unit test. */
vi.mock('../tts/python-discovery.js', () => ({ findPython312: () => findPython312() }));
vi.mock('../tts/kokoro-install-detect.js', () => ({
  kokoroPackageInstalled: () => true, detectKokoroInstalledOnDisk: () => true,
}));
vi.mock('../tts/qwen-install-detect.js', () => ({
  qwenPackageInstalled: () => false, qwenWeightsPresent: () => false,
}));
vi.mock('../tts/coqui-install-detect.js', () => ({
  coquiPackageInstalled: () => coquiPackageInstalledDisk(), coquiWeightsPresent: () => false,
}));

import { computeModelsStatus } from './models-status.js';
import { _resetPythonProbeCacheForTests } from './setup-diagnosis.js';

describe('computeModelsStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    /* probePython312Cached() TTL-caches its result for 10s in module-level
       state (setup-diagnosis.ts) — reset so the venv-absent test's
       findPython312 override can't leak into (or be poisoned by) another
       test in the same wall-clock window. */
    _resetPythonProbeCacheForTests();
    sidecarVenvPresent.mockReturnValue(true);
    getActiveSupervisor.mockReturnValue({ tripEvent: () => null, exhaustedEvent: () => false });
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable', kokoroLoaded: false, kokoroPackageInstalled: true,
      qwenPackageInstalled: false, coquiPackageInstalled: false, modelLoaded: false,
    });
    findPython312.mockReturnValue(null);
    coquiPackageInstalledDisk.mockReturnValue(false);
  });

  it('reports kokoro ready, qwen not-installed, and runtime installed + reachable', async () => {
    const s = await computeModelsStatus('/repo');
    expect(s.engines.kokoro.state).toBe('ready');
    expect(s.engines.qwen.state).toBe('not-installed');
    expect(s.runtime.installedOnDisk).toBe(true);
    expect(s.runtime.process).toBe('ready');
    expect(s.info.vramTotalMb).toBe(8192);
  });

  it('calls probeSidecarHealth exactly once per computeModelsStatus call', async () => {
    await computeModelsStatus('/repo');

    expect(probeSidecarHealth).toHaveBeenCalledTimes(1);
  });

  it('trusts a reachable external sidecar when the web container has no local venv', async () => {
    sidecarVenvPresent.mockReturnValue(false);
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      qwenLoaded: false,
      qwenPackageInstalled: true,
      qwenWeightsPresent: true,
      coquiPackageInstalled: false,
      kokoroPackageInstalled: false,
      vramTotalMb: 12_288,
      device: 'cuda',
    });

    const s = await computeModelsStatus('/repo');

    expect(probeSidecarHealth).toHaveBeenCalledTimes(1);
    expect(s.runtime.installedOnDisk).toBe(true);
    expect(s.runtime.pythonFound).toBe(true);
    expect(findPython312).not.toHaveBeenCalled();
    expect(s.runtime.process).toBe('ready');
    expect(s.engines.qwen.state).toBe('ready');
    expect(s.engines.qwen.packageFault).toBe('ok');
    expect(s.info.vramTotalMb).toBe(12_288);
  });

  it('uses live Kokoro weights for a cold external sidecar', async () => {
    sidecarVenvPresent.mockReturnValue(false);
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroLoaded: false,
      kokoroPackageInstalled: true,
      kokoroWeightsPresent: true,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.state).toBe('ready');
  });

  it('keeps a reachable legacy external payload unknown without consulting local engine disks', async () => {
    sidecarVenvPresent.mockReturnValue(false);
    probeSidecarHealth.mockResolvedValue({ status: 'reachable', engines: ['kokoro', 'qwen'] });

    const s = await computeModelsStatus('/repo');

    /* The mocked web-container disk says Kokoro is fully installed. A legacy
       external payload has no capability evidence, so that local fact must
       not resurrect the remote engine or manufacture package-missing. */
    expect(s.engines.kokoro).toMatchObject({ state: 'not-installed', packageFault: 'ok' });
    expect(s.engines.kokoro.availabilityKnown).toBe(false);
    expect(s.engines.qwen).toMatchObject({
      state: 'not-installed',
      packageFault: 'ok',
      availabilityKnown: false,
    });
  });

  it('marks explicit modern external false capabilities as confirmed not-installed', async () => {
    sidecarVenvPresent.mockReturnValue(false);
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroPackageInstalled: false,
      kokoroWeightsPresent: false,
      qwenPackageInstalled: false,
      qwenWeightsPresent: false,
      coquiPackageInstalled: false,
      coquiWeightsPresent: false,
    });

    const s = await computeModelsStatus('/repo');

    for (const engine of Object.values(s.engines)) {
      expect(engine).toMatchObject({ state: 'not-installed', availabilityKnown: true });
    }
  });

  it('treats package=false as known without weights, but package=true without weights as unknown', async () => {
    sidecarVenvPresent.mockReturnValue(false);
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroPackageInstalled: true,
      qwenPackageInstalled: false,
      coquiPackageInstalled: false,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.availabilityKnown).toBe(false);
    expect(s.engines.qwen).toMatchObject({
      state: 'not-installed',
      availabilityKnown: true,
    });
    expect(s.engines.coqui.availabilityKnown).toBe(true);
  });

  it('still reports a missing runtime when neither a local venv nor the configured sidecar is available', async () => {
    sidecarVenvPresent.mockReturnValue(false);
    probeSidecarHealth.mockResolvedValue({ status: 'unreachable' });

    const s = await computeModelsStatus('/repo');

    expect(probeSidecarHealth).toHaveBeenCalledTimes(1);
    expect(findPython312).toHaveBeenCalledTimes(1);
    expect(s.runtime).toEqual({ installedOnDisk: false, pythonFound: false, process: 'starting' });
  });
});

describe('computeModelsStatus — coqui import honesty (#1963)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _resetPythonProbeCacheForTests();
    sidecarVenvPresent.mockReturnValue(true);
    getActiveSupervisor.mockReturnValue({ tripEvent: () => null, exhaustedEvent: () => false });
    findPython312.mockReturnValue(null);
    // Package present on disk so packageBroken's `p.packageOnDisk && …` half
    // can actually flip true — this is the producer half of the regression:
    // without it, packageBroken would read false regardless of importable.
    coquiPackageInstalledDisk.mockReturnValue(true);
  });

  it('coqui_import_ok: false flags coqui packageBroken even though find_spec (coquiPackageInstalled) says true', async () => {
    /* THE DEFECT (#1963): before the fix, coqui's livePackageImportable read
       straight off coquiPackageInstalled (find_spec), so a package that
       cannot actually import — the #1944 speechbrain lazy-proxy collision —
       reported packageBroken: false and the engine looked usable. */
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      modelLoaded: false,
      coquiPackageInstalled: true,
      coquiImportOk: false,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.coqui.packageBroken).toBe(true);
  });

  it('coqui_import_ok: null falls back to coquiPackageInstalled — packageBroken stays false, unchanged from today', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      modelLoaded: false,
      coquiPackageInstalled: true,
      coquiImportOk: null,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.coqui.packageBroken).toBe(false);
  });

  it('coqui_import_ok: true reports packageBroken false', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      modelLoaded: false,
      coquiPackageInstalled: true,
      coquiImportOk: true,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.coqui.packageBroken).toBe(false);
  });
});

describe('computeModelsStatus — kokoro/qwen import honesty (#1965)', () => {
  /* End-to-end through the route composition: a /health body carrying
     kokoro_import_ok: false must reach packageBroken. Before #1965 the
     registry's single accessor read kokoroPackageInstalled (find_spec) alone,
     so kokoro and qwen could not produce this signal AT ALL — a package present
     on disk that genuinely will not import (#1944) looked perfectly usable and
     Model Manager never offered Repair.

     Kokoro is the engine under test here because its disk probe is mocked true
     above, which is what lets packageBroken's `packageOnDisk && …` half flip.
     The qwen half of the same regression is pinned on the pure predicate in
     tts/models-status.test.ts and on the accessor in
     tts/voice-engine-registry.test.ts. */
  beforeEach(() => {
    vi.clearAllMocks();
    _resetPythonProbeCacheForTests();
    sidecarVenvPresent.mockReturnValue(true);
    getActiveSupervisor.mockReturnValue({ tripEvent: () => null, exhaustedEvent: () => false });
    findPython312.mockReturnValue(null);
    coquiPackageInstalledDisk.mockReturnValue(false);
  });

  it('kokoro_import_ok: false flags kokoro packageBroken even though find_spec says true', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroLoaded: false,
      kokoroPackageInstalled: true,
      kokoroImportOk: false,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.packageBroken).toBe(true);
  });

  it('kokoroImportOk: null (the common value — nothing has loaded kokoro yet) falls back to find_spec', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroLoaded: false,
      kokoroPackageInstalled: true,
      kokoroImportOk: null,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.packageBroken).toBe(false);
  });

  it('kokoroImportOk: null + find_spec false → packageBroken, exactly as before the split', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroLoaded: false,
      kokoroPackageInstalled: false,
      kokoroImportOk: null,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.packageBroken).toBe(true);
  });

  it('kokoroImportOk: true outranks a find_spec false — a real import beats the probe', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroLoaded: false,
      kokoroPackageInstalled: false,
      kokoroImportOk: true,
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.packageBroken).toBe(false);
  });

  it('sidecar unreachable → both signals unknown, never broken-confirmed', async () => {
    probeSidecarHealth.mockResolvedValue({ status: 'unreachable' });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.kokoro.packageBroken).toBe(false);
  });

  /* #2533 — qwenPackageInstalled forwards undefined (not a coerced false) when
     the wire omits qwen_package_installed and qwen isn't loaded (see the
     sidecar-health.ts fix and its own regression test in
     sidecar-health.test.ts's "qwenPackageInstalled (#2533)" describe block).
     This proves the fix reaches all the way through this route's composition
     into voice-engine-registry.ts's liveSpecPresent and classifyPackageFault's
     fail-open rule: an old sidecar that has never sent this field must not
     read as a confirmed-missing qwen package. */
  it('qwen_package_installed omitted from the wire, qwen not loaded → stays unfaulted (fail-open, not a false "missing")', async () => {
    probeSidecarHealth.mockResolvedValue({
      status: 'reachable',
      kokoroLoaded: false,
      kokoroPackageInstalled: true,
      qwenLoaded: false,
      // qwenPackageInstalled deliberately omitted — the pre-#1965 sidecar shape.
    });

    const s = await computeModelsStatus('/repo');

    expect(s.engines.qwen.packageFault).toBe('ok');
    expect(s.engines.qwen.packageBroken).toBe(false);
  });
});
