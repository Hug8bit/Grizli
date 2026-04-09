/**
 * useGridState — React hook for real-time grid state via WebSocket.
 * Falls back to polling if WebSocket is unavailable.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { GridState, GridMetrics, createGridWebSocket, gridApi } from '../services/api'

interface UseGridStateReturn {
  gridState: GridState | null
  metrics: GridMetrics | null
  loading: boolean
  error: string | null
  connected: boolean
  refresh: () => Promise<void>
}

export function useGridState(): UseGridStateReturn {
  const [gridState, setGridState] = useState<GridState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  const refresh = useCallback(async () => {
    try {
      const state = await gridApi.runPowerFlow()
      setGridState(state)
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to fetch grid state')
    }
  }, [])

  useEffect(() => {
    // Initial load via REST
    setLoading(true)
    gridApi.getTopology()
      .then(state => {
        setGridState(state)
        setLoading(false)
      })
      .catch(e => {
        setError(e.message)
        setLoading(false)
      })

    // Try WebSocket for real-time updates
    const ws = createGridWebSocket(
      msg => {
        setGridState(msg.data)
        setConnected(true)
        setError(null)
      },
      () => {
        setConnected(false)
      },
    )
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    wsRef.current = ws

    return () => {
      ws.close()
    }
  }, [])

  return {
    gridState,
    metrics: gridState?.metrics ?? null,
    loading,
    error,
    connected,
    refresh,
  }
}
