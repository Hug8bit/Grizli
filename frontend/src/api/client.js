/**
 * Typed API client for the GRIZLI FastAPI backend.
 * Base URL is read from the Vite proxy (/api → http://localhost:8000/api).
 */
import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL ?? ''

const http = axios.create({ baseURL: BASE, timeout: 30_000 })

// ── Simulation ────────────────────────────────────────────────────────────────

/**
 * @param {{ network_type, solar_factor, wind_factor, load_factor, optimize, num_buses }} params
 */
export async function runSimulation(params = {}) {
  const { data } = await http.post('/api/simulate', {
    network_type: 'ieee33',
    solar_factor: 1.0,
    wind_factor:  1.0,
    load_factor:  1.0,
    optimize:     false,
    num_buses:    30,
    ...params,
  })
  return data
}

export async function getNetwork() {
  const { data } = await http.get('/api/network')
  return data
}

// ── ML ────────────────────────────────────────────────────────────────────────

/**
 * @param {{ model_type, n_hours }} params
 */
export async function trainModel(params = {}) {
  const { data } = await http.post('/api/ml/train', {
    model_type: 'gradient_boosting',
    n_hours:    8760,
    ...params,
  })
  return data
}

/**
 * @param {{ hour, day_of_week, month, is_weekend, temperature_c, humidity_pct, sfmta_ridership }} params
 */
export async function predict(params = {}) {
  const { data } = await http.get('/api/ml/predict', { params })
  return data
}

/**
 * @param {number} nSamples
 */
export async function getMlStatus(nSamples = 96) {
  const { data } = await http.get('/api/ml/status', { params: { n_samples: nSamples } })
  return data
}

// ── SF Data ───────────────────────────────────────────────────────────────────

/**
 * @param {number} energySampleSize
 */
export async function getSfData(energySampleSize = 168) {
  const { data } = await http.get('/api/data/sf', {
    params: { energy_sample_size: energySampleSize },
  })
  return data
}
