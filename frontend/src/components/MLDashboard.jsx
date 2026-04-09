/**
 * [B] ML DASHBOARD — learning curves, predictions vs actual, KPIs.
 *
 * Uses Recharts for:
 *  - Learning curves (train_loss / val_loss over iterations)
 *  - Predictions vs actual scatter/line
 *  - KPI cards (MAE, RMSE, R², savings estimate, peak forecast)
 */
import React, { useEffect, useState, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine,
} from 'recharts'
import { Brain, RefreshCw, TrendingDown, Zap, Activity, DollarSign } from 'lucide-react'
import { getMlStatus, trainModel } from '../api/client'

// ── KPI card ──────────────────────────────────────────────────────────────────
function KpiCard({ icon: Icon, label, value, sub, color = 'text-brand-400' }) {
  return (
    <div className="bg-gray-800/60 rounded-xl px-4 py-3 flex items-center gap-3">
      <div className={`p-2 rounded-lg bg-gray-700/60 ${color}`}>
        <Icon size={18} />
      </div>
      <div>
        <p className="text-xs text-gray-500">{label}</p>
        <p className={`text-base font-bold ${color}`}>{value}</p>
        {sub && <p className="text-xs text-gray-600">{sub}</p>}
      </div>
    </div>
  )
}

// ── Custom tooltip ────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs">
      <p className="text-gray-400 mb-1">{label !== undefined ? `Iter ${label}` : `#${payload[0]?.payload?.index}`}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function MLDashboard() {
  const [status, setStatus]   = useState(null)
  const [training, setTraining] = useState(false)
  const [loading, setLoading] = useState(false)
  const [modelType, setModelType] = useState('gradient_boosting')
  const [nHours, setNHours]   = useState(8760)

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    try {
      const s = await getMlStatus(96)
      setStatus(s)
    } catch (e) {
      console.error('ML status error', e)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleTrain = async () => {
    setTraining(true)
    try {
      await trainModel({ model_type: modelType, n_hours: nHours })
      await fetchStatus()
    } catch (e) {
      console.error('Training error', e)
    } finally {
      setTraining(false)
    }
  }

  useEffect(() => { fetchStatus() }, [fetchStatus])

  const m = status?.metrics ?? {}
  const history = status?.training_history ?? []
  const preds   = status?.predictions_vs_actual ?? []

  // Compute extra KPIs from predictions
  const peakPred = preds.length ? Math.max(...preds.map((p) => p.predicted)) : null
  const avgSavings = preds.length
    ? preds.reduce((acc, p) => acc + Math.max(0, (p.predicted - 70) / p.predicted * 100), 0) / preds.length
    : null

  return (
    <div className="flex flex-col h-full gap-4 overflow-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-white flex-1">🧠 ML Energy Dashboard</h2>

        <select
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-2 py-1"
        >
          <option value="gradient_boosting">Gradient Boosting</option>
          <option value="random_forest">Random Forest</option>
          <option value="ridge">Ridge</option>
        </select>

        <select
          value={nHours}
          onChange={(e) => setNHours(Number(e.target.value))}
          className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-2 py-1"
        >
          <option value={720}>1 month</option>
          <option value={2160}>3 months</option>
          <option value={8760}>1 year</option>
        </select>

        <button
          onClick={handleTrain}
          disabled={training}
          className="flex items-center gap-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-50
                     text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
        >
          {training ? <RefreshCw size={14} className="animate-spin" /> : <Brain size={14} />}
          {training ? 'Training…' : 'Train Model'}
        </button>

        <button
          onClick={fetchStatus}
          disabled={loading}
          className="flex items-center gap-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-50
                     text-white text-sm px-3 py-1.5 rounded transition-colors"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard icon={Activity}    label="R² Score"      value={m.r2   != null ? m.r2.toFixed(3)   : '—'} color="text-green-400" />
        <KpiCard icon={TrendingDown} label="MAE (MW)"     value={m.mae  != null ? m.mae.toFixed(2)  : '—'} color="text-yellow-400" />
        <KpiCard icon={Zap}         label="Peak Forecast" value={peakPred != null ? `${peakPred.toFixed(1)} MW` : '—'} color="text-red-400" />
        <KpiCard icon={DollarSign}  label="Avg Savings"   value={avgSavings != null ? `${avgSavings.toFixed(1)}%` : '—'} sub="vs peak threshold" color="text-brand-400" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 flex-1 min-h-0">

        {/* Learning curves */}
        <div className="bg-gray-900/60 rounded-xl p-4 flex flex-col">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Learning Curves (MSE)</h3>
          {history.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={history} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="iteration" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Iteration', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                <Line dataKey="train_loss" name="Train MSE" stroke="#818cf8" dot={false} strokeWidth={2} />
                <Line dataKey="val_loss"   name="Val MSE"   stroke="#34d399" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
              {status?.is_trained ? 'No curve data (Ridge/RF)' : 'Train a model to see learning curves'}
            </div>
          )}
        </div>

        {/* Predictions vs actual */}
        <div className="bg-gray-900/60 rounded-xl p-4 flex flex-col">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Predictions vs Actual (last 96 h)</h3>
          {preds.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={preds} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="index" tick={{ fill: '#64748b', fontSize: 10 }} label={{ value: 'Hour', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} unit=" MW" />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 2" label={{ value: 'Peak', fill: '#ef4444', fontSize: 9 }} />
                <Line dataKey="actual"    name="Actual"    stroke="#f59e0b" dot={false} strokeWidth={1.5} />
                <Line dataKey="predicted" name="Predicted" stroke="#818cf8" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
              {status?.is_trained ? 'Loading…' : 'Train a model first'}
            </div>
          )}
        </div>
      </div>

      {/* Training info */}
      {m.train_samples && (
        <p className="text-xs text-gray-600 text-right">
          Trained on {m.train_samples?.toLocaleString()} samples · tested on {m.test_samples?.toLocaleString()} · RMSE {m.rmse?.toFixed(2)} MW · model: {m.model_type}
        </p>
      )}
    </div>
  )
}
