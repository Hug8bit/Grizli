/**
 * [C] SF INTERACTIVE MAP — React-Leaflet overlay.
 *
 * Layers:
 *  - OpenStreetMap base (inverted to dark via CSS filter)
 *  - Electrical network nodes (circle markers, coloured by type)
 *  - SFMTA bus stops (small blue dots)
 *  - Energy consumption zones (choropleth via circle overlay)
 *
 * Filters: layer toggles + time-range selector
 */
import React, { useEffect, useState, useCallback } from 'react'
import {
  MapContainer, TileLayer, CircleMarker, Tooltip as LeafletTooltip,
  LayersControl, LayerGroup,
} from 'react-leaflet'
import { Map, RefreshCw, Layers } from 'lucide-react'
import { getSfData } from '../api/client'

const SF_CENTER = [37.7749, -122.4194]

// Colour by node type
const NODE_COLOR = {
  substation: '#ef4444',
  generator:  '#22c55e',
  load:       '#f59e0b',
  junction:   '#64748b',
}

// Consumption → radius mapping
const consumptionRadius = (mw) => Math.max(4, Math.min(20, mw / 5))

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  return (
    <div className="absolute bottom-4 left-4 z-[1000] bg-gray-900/90 border border-gray-700 rounded-lg p-3 text-xs space-y-1.5">
      <p className="font-medium text-gray-300 mb-1">Legend</p>
      {Object.entries(NODE_COLOR).map(([type, color]) => (
        <div key={type} className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full inline-block" style={{ background: color }} />
          <span className="text-gray-400 capitalize">{type}</span>
        </div>
      ))}
      <div className="flex items-center gap-2 mt-1">
        <span className="w-3 h-3 rounded-full inline-block bg-blue-400 opacity-60" />
        <span className="text-gray-400">SFMTA Stop</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-3 h-3 rounded-full inline-block bg-orange-400 opacity-40" />
        <span className="text-gray-400">Energy zone</span>
      </div>
    </div>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function SFMap() {
  const [sfData, setSfData]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [hourFilter, setHourFilter] = useState(null)   // null = all hours
  const [mapReady, setMapReady] = useState(false)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getSfData(168)
      setSfData(data)
    } catch (e) {
      console.error('SF data error', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    // Small delay to let Leaflet container size stabilise
    setTimeout(() => setMapReady(true), 200)
  }, [fetchData])

  // Filter energy points by hour of day
  const filteredEnergy = sfData?.energy_sample?.filter(
    (p) => hourFilter === null || p.hour === hourFilter
  ) ?? []

  // Build zone summaries: average consumption per zone from network nodes
  const networkNodes = sfData?.network_nodes ?? []
  const sfmtaStops   = sfData?.sfmta_stops ?? []

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-white flex-1">🗺 SF Interactive Map</h2>

        {/* Hour filter */}
        <label className="flex items-center gap-2 text-sm text-gray-400">
          Hour filter:
          <select
            value={hourFilter ?? ''}
            onChange={(e) => setHourFilter(e.target.value === '' ? null : Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-2 py-1"
          >
            <option value="">All hours</option>
            {Array.from({ length: 24 }, (_, i) => (
              <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
            ))}
          </select>
        </label>

        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center gap-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-50
                     text-white text-sm px-3 py-1.5 rounded transition-colors"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {/* Info bar */}
      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        <span>📍 {networkNodes.length} network nodes</span>
        <span>🚌 {sfmtaStops.length} SFMTA stops</span>
        <span>⚡ {filteredEnergy.length} energy samples</span>
        {sfData?.data_source && (
          <span className={sfData.data_source === 'datasf' ? 'text-green-500' : 'text-yellow-600'}>
            Source: {sfData.data_source}
          </span>
        )}
      </div>

      {/* Map container */}
      <div className="flex-1 min-h-0 rounded-xl overflow-hidden border border-gray-800 relative">
        {mapReady && (
          <MapContainer
            center={SF_CENTER}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
            preferCanvas
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <LayersControl position="topright">
              {/* Network nodes layer */}
              <LayersControl.Overlay checked name="⚡ Network Nodes">
                <LayerGroup>
                  {networkNodes.map((n) => (
                    <CircleMarker
                      key={`node-${n.node_id}`}
                      center={[n.latitude, n.longitude]}
                      radius={n.node_type === 'substation' ? 10 : 6}
                      pathOptions={{
                        color:       NODE_COLOR[n.node_type] ?? '#64748b',
                        fillColor:   NODE_COLOR[n.node_type] ?? '#64748b',
                        fillOpacity: 0.85,
                        weight:      2,
                      }}
                    >
                      <LeafletTooltip>
                        <div className="text-xs">
                          <strong>Node {n.node_id}</strong><br />
                          Type: {n.node_type}<br />
                          Zone: {n.zone}<br />
                          Voltage: {n.nominal_voltage_kv} kV
                        </div>
                      </LeafletTooltip>
                    </CircleMarker>
                  ))}
                </LayerGroup>
              </LayersControl.Overlay>

              {/* SFMTA stops layer */}
              <LayersControl.Overlay checked name="🚌 SFMTA Stops">
                <LayerGroup>
                  {sfmtaStops.map((s) => (
                    <CircleMarker
                      key={`stop-${s.stop_id}`}
                      center={[s.latitude, s.longitude]}
                      radius={3}
                      pathOptions={{
                        color:       '#60a5fa',
                        fillColor:   '#60a5fa',
                        fillOpacity: 0.6,
                        weight:      1,
                      }}
                    >
                      <LeafletTooltip>
                        <div className="text-xs">
                          {s.stop_name}<br />
                          Route: {s.routes ?? '—'}
                        </div>
                      </LeafletTooltip>
                    </CircleMarker>
                  ))}
                </LayerGroup>
              </LayersControl.Overlay>

              {/* Energy consumption zones */}
              <LayersControl.Overlay checked name="⚡ Energy Zones">
                <LayerGroup>
                  {networkNodes.slice(0, filteredEnergy.length).map((n, i) => {
                    const ep = filteredEnergy[i % filteredEnergy.length]
                    if (!ep) return null
                    const r = consumptionRadius(ep.consumption_mw)
                    return (
                      <CircleMarker
                        key={`zone-${n.node_id}`}
                        center={[n.latitude, n.longitude]}
                        radius={r}
                        pathOptions={{
                          color:       '#f97316',
                          fillColor:   '#f97316',
                          fillOpacity: 0.2,
                          weight:      1,
                        }}
                      >
                        <LeafletTooltip>
                          <div className="text-xs">
                            Zone: {n.zone}<br />
                            Consumption: {ep.consumption_mw.toFixed(1)} MW<br />
                            Temp: {ep.temperature_c}°C<br />
                            Hour: {String(ep.hour).padStart(2, '0')}:00
                          </div>
                        </LeafletTooltip>
                      </CircleMarker>
                    )
                  })}
                </LayerGroup>
              </LayersControl.Overlay>
            </LayersControl>
          </MapContainer>
        )}
        <Legend />
      </div>
    </div>
  )
}
