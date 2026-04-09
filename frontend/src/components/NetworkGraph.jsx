/**
 * [A] NETWORK VISUALISATION — React Flow graph of the electrical network.
 *
 * Nodes: buses (coloured by type — slack/generator/load/junction)
 * Edges: lines (coloured by loading: green <80 % | orange 80–100 % | red >100 %)
 * Real-time update during simulation via polling.
 */
import React, { useCallback, useEffect, useState } from 'react'
import ReactFlow, {
  Background, Controls, MiniMap,
  useNodesState, useEdgesState,
  MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { Play, RefreshCw, Zap, Settings } from 'lucide-react'
import { runSimulation } from '../api/client'

// ── Colours ───────────────────────────────────────────────────────────────────
const NODE_COLORS = {
  slack:     '#ef4444',
  generator: '#22c55e',
  load:      '#f59e0b',
  junction:  '#64748b',
}

const edgeColor = (loading) => {
  if (loading > 100) return '#ef4444'
  if (loading > 80)  return '#f59e0b'
  return '#22c55e'
}

// ── Layout helpers ────────────────────────────────────────────────────────────
function layoutNodes(apiNodes) {
  // Arrange buses in a grid-like layout if no geo coords
  const cols = Math.ceil(Math.sqrt(apiNodes.length))
  return apiNodes.map((n, i) => ({
    id:       String(n.id),
    position: { x: (i % cols) * 90 + 40, y: Math.floor(i / cols) * 80 + 40 },
    data:     { label: `B${n.id}`, type: n.node_type, voltage: n.voltage_pu },
    style: {
      background: NODE_COLORS[n.node_type] ?? '#64748b',
      color:      '#fff',
      border:     `2px solid ${n.voltage_pu < 0.95 || n.voltage_pu > 1.05 ? '#ef4444' : 'transparent'}`,
      borderRadius: 6,
      fontSize:   10,
      padding:    '4px 8px',
      width:      60,
      textAlign:  'center',
    },
  }))
}

function buildEdges(apiLines) {
  return apiLines
    .filter((l) => l.in_service)
    .map((l) => ({
      id:           `e${l.id}`,
      source:       String(l.from_bus),
      target:       String(l.to_bus),
      animated:     l.loading_percent > 80,
      style:        { stroke: edgeColor(l.loading_percent), strokeWidth: l.loading_percent > 100 ? 3 : 1.5 },
      markerEnd:    { type: MarkerType.ArrowClosed, color: edgeColor(l.loading_percent) },
      label:        `${l.loading_percent.toFixed(0)}%`,
      labelStyle:   { fill: '#94a3b8', fontSize: 9 },
      labelBgStyle: { fill: '#0f172a' },
    }))
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function NetworkGraph() {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [loading, setLoading]   = useState(false)
  const [stats, setStats]       = useState(null)
  const [config, setConfig]     = useState({
    network_type: 'ieee33',
    solar_factor: 1.0, wind_factor: 1.0, load_factor: 1.0,
    optimize: false,
  })

  const applyResult = useCallback((result) => {
    setNodes(layoutNodes(result.nodes))
    setEdges(buildEdges(result.lines))
    setStats(result)
  }, [])

  const simulate = useCallback(async () => {
    setLoading(true)
    try {
      const result = await runSimulation(config)
      applyResult(result)
    } catch (e) {
      console.error('Simulation error', e)
    } finally {
      setLoading(false)
    }
  }, [config, applyResult])

  // Initial load
  useEffect(() => { simulate() }, []) // eslint-disable-line

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header + controls */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-white flex-1">⚡ Network Visualisation</h2>

        {/* Network type */}
        <select
          value={config.network_type}
          onChange={(e) => setConfig((c) => ({ ...c, network_type: e.target.value }))}
          className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-2 py-1"
        >
          <option value="ieee33">IEEE 33-bus</option>
          <option value="ieee33_congested">IEEE 33 Congested</option>
          <option value="atacama">Atacama</option>
          <option value="synthetic">Synthetic</option>
        </select>

        {/* Optimize toggle */}
        <label className="flex items-center gap-1.5 text-sm text-gray-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={config.optimize}
            onChange={(e) => setConfig((c) => ({ ...c, optimize: e.target.checked }))}
            className="accent-brand-500"
          />
          Optimise
        </label>

        <button
          onClick={simulate}
          disabled={loading}
          className="flex items-center gap-1.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50
                     text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
        >
          {loading ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
          {loading ? 'Running…' : 'Simulate'}
        </button>
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-4 text-xs text-gray-400">
        {[
          ['Solar', 'solar_factor'], ['Wind', 'wind_factor'], ['Load', 'load_factor'],
        ].map(([label, key]) => (
          <label key={key} className="flex items-center gap-2">
            {label}
            <input
              type="range" min="0" max="3" step="0.1"
              value={config[key]}
              onChange={(e) => setConfig((c) => ({ ...c, [key]: parseFloat(e.target.value) }))}
              className="w-24 accent-brand-500"
            />
            <span className="text-gray-300 w-8">{config[key].toFixed(1)}×</span>
          </label>
        ))}
      </div>

      {/* KPIs */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {[
            ['Max Loading', `${stats.max_loading_pct?.toFixed(1)}%`,
              stats.max_loading_pct > 100 ? 'text-red-400' : stats.max_loading_pct > 80 ? 'text-yellow-400' : 'text-green-400'],
            ['Losses',      `${(stats.total_losses_mw * 1000)?.toFixed(1)} kW`, 'text-gray-300'],
            ['Overloaded',  `${stats.num_overloaded} lines`,
              stats.num_overloaded > 0 ? 'text-red-400' : 'text-green-400'],
            ['Calc time',   `${stats.computation_ms?.toFixed(0)} ms`, 'text-gray-400'],
          ].map(([label, value, cls]) => (
            <div key={label} className="bg-gray-800/60 rounded-lg px-3 py-2">
              <p className="text-xs text-gray-500">{label}</p>
              <p className={`text-sm font-semibold ${cls}`}>{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* React Flow canvas */}
      <div className="flex-1 min-h-0 rounded-xl overflow-hidden border border-gray-800">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.3}
          attributionPosition="bottom-right"
        >
          <Background color="#1e293b" gap={24} />
          <Controls />
          <MiniMap
            nodeColor={(n) => NODE_COLORS[n.data?.type] ?? '#64748b'}
            maskColor="rgba(15,23,42,0.7)"
          />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-gray-500">
        {[
          ['Slack',     NODE_COLORS.slack],
          ['Generator', NODE_COLORS.generator],
          ['Load',      NODE_COLORS.load],
          ['Junction',  NODE_COLORS.junction],
        ].map(([label, color]) => (
          <span key={label} className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full inline-block" style={{ background: color }} />
            {label}
          </span>
        ))}
        <span className="ml-4 flex items-center gap-1.5">
          <span className="w-8 h-0.5 inline-block bg-green-400" /> &lt;80%
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-8 h-0.5 inline-block bg-yellow-400" /> 80–100%
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-8 h-0.5 inline-block bg-red-400" /> &gt;100%
        </span>
      </div>
    </div>
  )
}
