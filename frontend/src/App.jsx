/**
 * GRIZLI — AI Grid Energy Dashboard
 *
 * Three equal-weight sections:
 *  [A] NetworkGraph  — React Flow interactive grid
 *  [B] MLDashboard   — Recharts learning curves + predictions
 *  [C] SFMap         — React-Leaflet SF geospatial overlay
 */
import React, { useState, Suspense, lazy } from 'react'
import Sidebar from './components/Sidebar.jsx'

const NetworkGraph = lazy(() => import('./components/NetworkGraph.jsx'))
const MLDashboard  = lazy(() => import('./components/MLDashboard.jsx'))
const SFMap        = lazy(() => import('./components/SFMap.jsx'))

function LoadingPanel() {
  return (
    <div className="flex-1 flex items-center justify-center text-gray-600">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm">Loading…</p>
      </div>
    </div>
  )
}

const PANELS = {
  network: NetworkGraph,
  ml:      MLDashboard,
  map:     SFMap,
}

export default function App() {
  const [active, setActive] = useState('network')
  const Panel = PANELS[active]

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950 text-gray-100 dark">
      <Sidebar active={active} onChange={setActive} />

      {/* Main content */}
      <main className="flex-1 overflow-hidden flex flex-col">
        {/* Top bar */}
        <header className="h-16 flex-shrink-0 flex items-center px-6 border-b border-gray-800 gap-4">
          <h1 className="text-sm font-medium text-gray-400">
            {active === 'network' && '⚡ Network Visualisation'}
            {active === 'ml'      && '🧠 ML Energy Dashboard'}
            {active === 'map'     && '🗺 San Francisco Map'}
          </h1>
          <div className="flex-1" />
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            API Docs ↗
          </a>
        </header>

        {/* Panel */}
        <div className="flex-1 min-h-0 p-4 overflow-auto">
          <Suspense fallback={<LoadingPanel />}>
            <Panel />
          </Suspense>
        </div>
      </main>
    </div>
  )
}
