/**
 * AgentPanel — RL Agent control panel.
 * Displays agent status, training controls, and inference results.
 */
import React, { useState, useEffect } from 'react'
import { agentApi, AgentInfo, TrainingStatus } from '../services/api'
import { Play, Square, ChevronRight, Brain } from 'lucide-react'

const AgentPanel: React.FC = () => {
  const [agentInfo, setAgentInfo] = useState<AgentInfo | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [lastAction, setLastAction] = useState<{ action: number; description: string } | null>(null)
  const [timesteps, setTimesteps] = useState(100_000)
  const [error, setError] = useState<string | null>(null)

  // Poll training status
  useEffect(() => {
    const poll = async () => {
      try {
        const [info, status] = await Promise.all([agentApi.getInfo(), agentApi.getTrainingStatus()])
        setAgentInfo(info)
        setTrainingStatus(status)
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Connection error')
      }
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => clearInterval(id)
  }, [])

  const handleTrain = async () => {
    try {
      setError(null)
      await agentApi.startTraining(timesteps)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start training')
    }
  }

  const handleStep = async () => {
    try {
      setError(null)
      const result = await agentApi.step()
      setLastAction({ action: result.action, description: result.action_description })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to run agent step')
    }
  }

  const isTraining = trainingStatus?.active ?? false
  const progress = trainingStatus?.progress ?? {}
  const progressPct = progress.timestep ? Math.min(100, (progress.timestep / timesteps) * 100) : 0

  return (
    <div className="space-y-4">
      {/* Agent status */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Brain size={18} className="text-purple-400" />
          <h3 className="font-semibold text-gray-200">RL Agent (PPO)</h3>
          <span className={`ml-auto px-2 py-0.5 rounded-full text-xs font-mono
            ${agentInfo?.status === 'ready' ? 'bg-green-950 text-green-400' : 'bg-gray-800 text-gray-400'}`}>
            {agentInfo?.status ?? 'loading...'}
          </span>
        </div>
        {agentInfo?.n_timesteps !== undefined && (
          <div className="text-xs text-gray-400 font-mono">
            Trained steps: <span className="text-cyan-400">{agentInfo.n_timesteps.toLocaleString()}</span>
          </div>
        )}
        {agentInfo?.policy && (
          <div className="text-xs text-gray-400 font-mono">
            Policy: <span className="text-cyan-400">{agentInfo.policy}</span>
          </div>
        )}
      </div>

      {/* Training controls */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 space-y-3">
        <h4 className="text-sm font-medium text-gray-300">Training</h4>

        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400 w-28 shrink-0">Timesteps</label>
          <input
            type="range"
            min={10_000}
            max={1_000_000}
            step={10_000}
            value={timesteps}
            onChange={e => setTimesteps(Number(e.target.value))}
            className="flex-1 accent-cyan-400"
            disabled={isTraining}
          />
          <span className="text-xs font-mono text-cyan-400 w-20 text-right">
            {(timesteps / 1_000).toFixed(0)}k
          </span>
        </div>

        {isTraining && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-400">
              <span>Training...</span>
              <span>{progressPct.toFixed(0)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progressPct}%` }} />
            </div>
            {progress.reward !== undefined && (
              <div className="text-xs text-gray-400 font-mono">
                Reward: <span className="text-green-400">{Number(progress.reward).toFixed(2)}</span>
              </div>
            )}
          </div>
        )}

        <button
          onClick={isTraining ? undefined : handleTrain}
          disabled={isTraining}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors w-full justify-center
            ${isTraining
              ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
              : 'bg-cyan-600 hover:bg-cyan-500 text-white'}`}
        >
          {isTraining ? <><Square size={14} /> Training...</> : <><Play size={14} /> Start Training</>}
        </button>
      </div>

      {/* Agent step */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 space-y-3">
        <h4 className="text-sm font-medium text-gray-300">Agent Inference</h4>
        <button
          onClick={handleStep}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-purple-700 hover:bg-purple-600 text-white transition-colors w-full justify-center"
        >
          <ChevronRight size={14} /> Run Agent Step
        </button>
        {lastAction && (
          <div className="bg-gray-800 rounded-lg p-3 text-xs font-mono">
            <span className="text-gray-400">Action: </span>
            <span className="text-purple-300">{lastAction.description}</span>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-950 border border-red-800 rounded-lg p-3 text-xs text-red-400">
          {error}
        </div>
      )}
    </div>
  )
}

export default AgentPanel
