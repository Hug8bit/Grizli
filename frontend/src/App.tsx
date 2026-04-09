/**
 * GrizliV2 — Main App
 * Grid optimization dashboard powered by RL.
 */
import React, { useState } from 'react'
import GridVisualization from './components/GridVisualization'
import MetricsPanel from './components/MetricsPanel'
import AgentPanel from './components/AgentPanel'
import { useGridState } from './hooks/useGridState'
import { gridApi } from './services/api'
import { RefreshCw, Wifi, WifiOff, Zap } from 'lucide-react'

type Tab = 'grid' | 'agent' | 'scenarios'

const App: React.FC = () => {
  const { gridState, metrics, loading, error, connected, refresh } = useGridState()
  const [activeTab, setActiveTab] = useState<Tab>('grid')
  const [running, setRunning] = useState(false)

  const handleRunPF = async () => {
    setRunning(true)
    try {
      await refresh()
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0f1e] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Zap size={24} className="text-cyan-400" />
          <span className="text-xl font-bold tracking-tight">
            Grizli<span className="text-cyan-400">V2</span>
          </span>
          <span className="text-xs bg-cyan-950 text-cyan-400 px-2 py-0.5 rounded-full font-mono ml-1">
            beta
          </span>
        </div>

        <span className="text-gray-500 text-sm hidden md:block">
          Grid optimization · Waze for Electricity · LATAM / Chile
        </span>

        <div className="ml-auto flex items-center gap-3">
          {/* Connection indicator */}
          <span className={`flex items-center gap-1.5 text-xs ${connected ? 'text-green-400' : 'text-gray-500'}`}>
            {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
            {connected ? 'Live' : 'Polling'}
          </span>

          {/* Run power flow */}
          <button
            onClick={handleRunPF}
            disabled={running}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={running ? 'animate-spin' : ''} />
            Run Power Flow
          </button>
        </div>
      </header>

      <div className="flex h-[calc(100vh-65px)]">
        {/* Sidebar */}
        <aside className="w-72 border-r border-gray-800 p-4 overflow-y-auto shrink-0">
          {/* Tabs */}
          <div className="flex gap-1 mb-4 bg-gray-900 rounded-lg p-1">
            {(['grid', 'agent'] as Tab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-1.5 text-xs font-medium rounded-md capitalize transition-colors
                  ${activeTab === tab
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-gray-200'}`}
              >
                {tab}
              </button>
            ))}
          </div>

          {activeTab === 'grid' && (
            <MetricsPanel metrics={metrics} loading={loading} />
          )}

          {activeTab === 'agent' && <AgentPanel />}
        </aside>

        {/* Main content */}
        <main className="flex-1 p-4 overflow-hidden">
          {error && (
            <div className="mb-3 bg-red-950 border border-red-800 rounded-lg px-4 py-2 text-sm text-red-300">
              {error} — Is the backend running? <code className="font-mono text-xs">cd backend && uvicorn main:app --reload</code>
            </div>
          )}

          {loading && !gridState ? (
            <div className="flex items-center justify-center h-full text-gray-500 gap-3">
              <RefreshCw size={20} className="animate-spin" />
              <span>Loading grid...</span>
            </div>
          ) : (
            <GridVisualization
              gridState={gridState}
              width={1200}
              height={700}
            />
          )}

          {/* Bottom info bar */}
          {gridState && (
            <div className="absolute bottom-4 left-80 right-4 flex items-center gap-4 text-xs text-gray-500 font-mono">
              <span>{gridState.n_buses} buses</span>
              <span>{gridState.n_lines} lines</span>
              <span>{gridState.converged ? '✓ converged' : '✗ not converged'}</span>
              {metrics && <span>losses: {metrics.total_losses_mw.toFixed(3)} MW</span>}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
