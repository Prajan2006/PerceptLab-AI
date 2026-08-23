/* ============================================================
   Composition root for frontend services.
   ============================================================ */

import type { CameraService } from '@/services/interfaces/CameraService';
import type { ExperimentService } from '@/services/interfaces/ExperimentService';
import type { ResearchService } from '@/services/research/types';
import { MockCameraService } from '@/services/mocks/MockCameraService';
import { MockExperimentService } from '@/services/mocks/MockExperimentService';
import { HttpCameraService } from '@/services/http/HttpCameraService';
import { MockResearchService } from '@/services/research/MockResearchService';

/**
 * VITE_USE_MOCKS=true (default): simulated services, no backend required.
 * VITE_USE_MOCKS=false: live transport — camera frames flow through the
 * isolated WebSocket adapter implementing the same CameraService contract.
 *
 * Optional: VITE_WS_URL (default ws://127.0.0.1:8000/api/camera/ws).
 */

const useMocks =
  ((import.meta.env.VITE_USE_MOCKS as string | undefined) ?? 'true') !== 'false';

export const isMockMode = useMocks;

const cameraServiceInstance: CameraService = useMocks
  ? new MockCameraService()
  : new HttpCameraService();

// Experiment lifecycle keeps its local simulation until its own bridge ships.
const experimentServiceInstance: ExperimentService = new MockExperimentService();

/**
 * Research catalog/runs use the simulator until the training backend
 * exists; an HTTP adapter will implement the identical ResearchService.
 */
export const isResearchSimulated = true;
const researchServiceInstance: ResearchService = new MockResearchService();

export const cameraService: CameraService = cameraServiceInstance;
export const experimentService: ExperimentService = experimentServiceInstance;
export const researchService: ResearchService = researchServiceInstance;
