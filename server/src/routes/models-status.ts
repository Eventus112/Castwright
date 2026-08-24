/* GET /api/setup/models-status — the single client-facing voice-engine status
   payload. Does the I/O (disk probes, one /health probe, VRAM, runtime axes)
   then delegates the pure composition to buildModelsStatus. `computeModelsStatus`
   is exported so setup-readiness.ts derives its sidecar/tts badges from the SAME
   computation (no second source). */
import { Router } from 'express';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Request, Response } from '../http.js';
import {
  buildModelsStatus,
  type EngineProbeResult,
  type ModelsStatus,
  type RuntimeProcessState,
} from '../tts/models-status.js';
import { VOICE_ENGINES, type VoiceEngineId } from '../tts/voice-engine-registry.js';
import { sidecarVenvPresent } from '../diagnostics/venv.js';
import { probePython312Cached } from './setup-diagnosis.js';
import { probeSidecarHealth, type SidecarHealthResult } from './sidecar-health.js';
import { getActiveSupervisor } from '../tts/sidecar-supervisor.js';
import { getDeviceTotalVramMb } from '../gpu/device-total.js';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

/** Map the sidecar/supervisor liveness to the runtime.process axis. */
function deriveProcess(input: {
  reachable: boolean;
  supervisorActive: boolean;
  supervisorTripped: boolean;
  supervisorExhausted: boolean;
}): RuntimeProcessState {
  if (input.reachable) return 'ready';
  if (input.supervisorTripped || input.supervisorExhausted) return 'crashed';
  if (input.supervisorActive) return 'starting';
  return 'down';
}

/* Build the human GPU string from the SINGLE /health probe result (+ the
   boot-time nvidia-smi total for the sidecar-down case). This deliberately does
   NOT call buildDiagnostics — computeModelsStatus must make exactly ONE
   probeSidecarHealth() call (setup-readiness.ts:119-127 documents a "no second
   probe" invariant; a second one here would re-fire the setLastKnown* side
   effects and double the Models-step probe count). */
function gpuDetail(health: Partial<SidecarHealthResult>, bootTotalMb: number | null): string {
  const gb = (mb: number | null | undefined) => (mb != null ? (mb / 1024).toFixed(1) : '?');
  if (health.status === 'reachable') {
    const total = health.vramTotalMb ?? bootTotalMb;
    const reserved = health.vramReservedMb ?? null;
    const cuda = health.device === 'cuda' || total != null;
    if (!cuda) return health.device ? `device: ${health.device}` : 'CPU — no GPU detected';
    if (reserved != null && total != null) return `cuda · ${gb(reserved)} / ${gb(total)} GB reserved`;
    return total != null ? `cuda · ${gb(total)} GB` : `device: ${health.device}`;
  }
  // Sidecar down — fall back to the boot nvidia-smi total (sidecar-independent).
  return bootTotalMb != null ? `GPU · ~${gb(bootTotalMb)} GB (voice engine offline)` : 'CPU — no GPU detected';
}

export async function computeModelsStatus(repoRoot: string): Promise<ModelsStatus> {
  const localVenvPresent = sidecarVenvPresent(repoRoot);
  const supervisor = getActiveSupervisor();
  /* The configured sidecar may run in another container/host. Its health is
     independent of whether this web process can see a local Python venv. */
  const health: Partial<SidecarHealthResult> = await probeSidecarHealth();
  const reachable = health.status === 'reachable';
  /* `installedOnDisk` is the client-facing "runtime available" axis. A
     reachable configured sidecar owns its own Python environment, so the web
     container neither needs nor should offer to bootstrap one locally. */
  const installedOnDisk = localVenvPresent || reachable;
  const pythonFound = installedOnDisk ? true : probePython312Cached();

  const engines = {} as Record<VoiceEngineId, EngineProbeResult>;
  for (const e of VOICE_ENGINES) {
    const externalRuntime = reachable && !localVenvPresent;
    const livePackage = externalRuntime ? e.liveSpecPresent(health) : undefined;
    const liveWeights = externalRuntime
      ? e.id === 'kokoro'
        ? health.kokoroWeightsPresent
        : e.id === 'qwen'
          ? health.qwenWeightsPresent
          : e.id === 'coqui'
            ? health.coquiWeightsPresent
            : undefined
      : undefined;
    const loaded = reachable ? e.liveLoaded(health) : false;
    /* A confirmed-absent package is a complete availability answer by itself:
       weights cannot make that engine usable. A present package still needs a
       weights capability before availability is known. */
    const externalAvailabilityKnown =
      loaded || livePackage === false || (livePackage === true && liveWeights !== undefined);
    engines[e.id] = {
      packageOnDisk: loaded
        ? true
        : externalRuntime
          ? externalAvailabilityKnown && livePackage === true
          : e.packageInstalledOnDisk(repoRoot),
      weightsOnDisk: loaded
        ? true
        : externalRuntime
          ? externalAvailabilityKnown && liveWeights === true
          : e.weightsPresentOnDisk(repoRoot),
      loaded,
      availabilityKnown: externalRuntime ? externalAvailabilityKnown : true,
      /* #1965 — the real-import result and the find_spec probe are forwarded
         separately; buildModelsStatus decides which one governs. Sidecar down →
         both unknown (undefined), which can never read as broken. */
      importOk: reachable ? e.liveImportOk(health) : undefined,
      specPresent: reachable ? e.liveSpecPresent(health) : undefined,
    };
  }

  // GPU string + total from the SINGLE probe (no second buildDiagnostics probe).
  const bootTotalMb = getDeviceTotalVramMb();
  const totalMb = reachable ? health.vramTotalMb ?? bootTotalMb : bootTotalMb;

  return buildModelsStatus({
    runtime: {
      installedOnDisk,
      pythonFound,
      process: deriveProcess({
        reachable,
        supervisorActive: supervisor !== null,
        supervisorTripped: supervisor?.tripEvent() != null,
        supervisorExhausted: supervisor?.exhaustedEvent() ?? false,
      }),
    },
    engines,
    info: { gpu: gpuDetail(health, bootTotalMb), vramTotalMb: totalMb },
  });
}

export const modelsStatusRouter = Router();

modelsStatusRouter.get('/models-status', async (_req: Request, res: Response) => {
  res.json(await computeModelsStatus(REPO_ROOT));
});
