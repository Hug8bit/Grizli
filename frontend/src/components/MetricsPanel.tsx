/**
 * MetricsPanel — Real-time grid performance metrics display.
 */
import React from 'react'
import { GridMetrics } from '../services/api'
import { Zap, AlertTriangle, TrendingDown, Activity } from 'lucide-react'

interface Props {
  metrics: GridMetrics | null
  loading?: boolean
}

interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  status?: 'ok' | 'warn' | 'danger'
  icon: React.ReactNode
}

function statusColor(status?: 'ok' | 'warn' | 'danger'): string {
  if (status === 'danger') return 'text-red-400 border-red-900'
  if (status === 'warn') return 'text-yellow-400 border-yellow-900'
  return 'text-cyan-400 border-gray-700'
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, unit, status, icon }) => (
  <div className={`metric-card ${statusColor(status)}`}>
    <div className="flex items-center justify-between mb-1">
      <span className="text-gray-400 text-xs uppercase tracking-wider">{label}</span>
      <span className="opacity-60">{icon}</span>
    </div>
    <div className="flex items-baseline gap-1">
      <span className="text-2xl font-bold font-mono">
        {typeof value === 'number' ? value.toFixed(2) : value}
      </span>
      {unit && <span className="text-xs text-gray-500">{unit}</span>}
    </div>
  </div>
)

const MetricsPanel: React.FC<Props> = ({ metrics, loading }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 animate-pulse">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="metric-card h-20 bg-gray-800" />
        ))}
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="text-gray-500 text-sm text-center py-6">
        No metrics — run power flow first
      </div>
    )
  }

  const lossStatus = metrics.total_losses_mw > 1.0 ? 'warn' : 'ok'
  const voltageStatus = metrics.voltage_violations > 0 ? 'danger' : 'ok'
  const loadingStatus = metrics.max_line_loading_pct > 80 ? 'danger' : metrics.max_line_loading_pct > 60 ? 'warn' : 'ok'
  const congestionStatus = metrics.overloaded_lines > 0 ? 'danger' : 'ok'

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Total Losses"
          value={metrics.total_losses_mw}
          unit="MW"
          status={lossStatus}
          icon={<TrendingDown size={16} />}
        />
        <MetricCard
          label="Max Line Loading"
          value={metrics.max_line_loading_pct}
          unit="%"
          status={loadingStatus}
          icon={<Activity size={16} />}
        />
        <MetricCard
          label="Voltage Violations"
          value={metrics.voltage_violations}
          unit="buses"
          status={voltageStatus}
          icon={<AlertTriangle size={16} />}
        />
        <MetricCard
          label="Overloaded Lines"
          value={metrics.overloaded_lines}
          unit="lines"
          status={congestionStatus}
          icon={<Zap size={16} />}
        />
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
          <span className="text-gray-400">V min</span>
          <span className={`ml-2 font-mono font-bold ${metrics.min_voltage_pu < 0.95 ? 'text-red-400' : 'text-green-400'}`}>
            {metrics.min_voltage_pu.toFixed(4)} p.u.
          </span>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
          <span className="text-gray-400">V max</span>
          <span className={`ml-2 font-mono font-bold ${metrics.max_voltage_pu > 1.05 ? 'text-red-400' : 'text-green-400'}`}>
            {metrics.max_voltage_pu.toFixed(4)} p.u.
          </span>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
          <span className="text-gray-400">Congestion Index</span>
          <span className="ml-2 font-mono font-bold text-cyan-400">
            {(metrics.congestion_index * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Feasibility badge */}
      <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
        ${metrics.is_feasible
          ? 'bg-green-950 border border-green-800 text-green-400'
          : 'bg-red-950 border border-red-800 text-red-400'
        }`}>
        <span className={`w-2 h-2 rounded-full ${metrics.is_feasible ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
        {metrics.is_feasible ? 'Grid operating within limits' : 'Grid has constraint violations'}
      </div>
    </div>
  )
}

export default MetricsPanel
