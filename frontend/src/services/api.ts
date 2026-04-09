/**
 * GrizliV2 API client
 * Communicates with the FastAPI backend.
 */
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1'

const http = axios.create({ baseURL: BASE_URL, timeout: 30_000 })

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Bus {
  id: number
  name: string
  vn_kv: number
  vm_pu?: number
  va_degree?: number
}

export interface Line {
  id: number
  from_bus: number
  to_bus: number
  length_km?: number
  loading_pct?: number
  p_mw?: number
  in_service?: boolean
}

export interface GridMetrics {
  total_losses_mw: number
  max_line_loading_pct: number
  min_voltage_pu: number
  max_voltage_pu: number
  voltage_violations: number
  overloaded_lines: number
  congestion_index: number
  renewable_curtailment_mw: number
  is_feasible: boolean
}

export interface GridState {
  buses: Bus[]
  lines: Line[]
  n_buses: number
  n_lines: number
  converged: boolean
  metrics: GridMetrics | null
}

export interface AgentInfo {
  status: string
  policy?: string
  device?: string
  n_timesteps?: number
}

export interface TrainingStatus {
  active: boolean
  progress: {
    timestep?: number
    reward?: number
    status?: string
    error?: string
  }
}

// ---------------------------------------------------------------------------
// Grid API
// ---------------------------------------------------------------------------

export const gridApi = {
  getTopology: (): Promise<GridState> =>
    http.get('/grid/topology').then(r => r.data),

  runPowerFlow: (algorithm = 'nr', scenarioId?: number): Promise<GridState> =>
    http.post('/grid/power-flow', { algorithm, scenario_id: scenarioId }).then(r => r.data),

  applySwitch: (switchIdx: number, close: boolean) =>
    http.post('/grid/switch', { switch_idx: switchIdx, close }).then(r => r.data),

  getMetrics: (): Promise<{ converged: boolean; metrics: GridMetrics | null }> =>
    http.get('/grid/metrics').then(r => r.data),

  listScenarios: () =>
    http.get('/grid/scenarios').then(r => r.data),
}

// ---------------------------------------------------------------------------
// Agent API
// ---------------------------------------------------------------------------

export const agentApi = {
  getInfo: (): Promise<AgentInfo> =>
    http.get('/agent/info').then(r => r.data),

  step: (deterministic = true) =>
    http.post('/agent/step', { deterministic }).then(r => r.data),

  startTraining: (totalTimesteps = 100_000, checkpointFreq = 10_000) =>
    http.post('/agent/train', {
      total_timesteps: totalTimesteps,
      checkpoint_freq: checkpointFreq,
    }).then(r => r.data),

  getTrainingStatus: (): Promise<TrainingStatus> =>
    http.get('/agent/training-status').then(r => r.data),

  evaluate: (nEpisodes = 10) =>
    http.post(`/agent/evaluate?n_episodes=${nEpisodes}`).then(r => r.data),
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export const healthApi = {
  check: () => http.get('/health').then(r => r.data),
}

// ---------------------------------------------------------------------------
// WebSocket helper
// ---------------------------------------------------------------------------

export function createGridWebSocket(
  onMessage: (data: { type: string; data: GridState; training_active: boolean; training_progress: object }) => void,
  onError?: (e: Event) => void,
): WebSocket {
  const wsUrl = (import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/api/v1') + '/ws/grid'
  const ws = new WebSocket(wsUrl)
  ws.onmessage = e => {
    try {
      onMessage(JSON.parse(e.data))
    } catch (_) {}
  }
  if (onError) ws.onerror = onError
  return ws
}
