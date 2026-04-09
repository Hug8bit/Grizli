import React from 'react'
import { Zap, Brain, Map, Activity } from 'lucide-react'

const NAV = [
  { id: 'network', label: 'Network',    icon: Zap      },
  { id: 'ml',      label: 'ML Dashboard', icon: Brain  },
  { id: 'map',     label: 'SF Map',     icon: Map      },
]

export default function Sidebar({ active, onChange }) {
  return (
    <aside className="w-16 md:w-56 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-gray-800 gap-3">
        <Activity className="text-brand-500 flex-shrink-0" size={22} />
        <span className="hidden md:block font-bold text-lg tracking-tight text-white">GRIZLI</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 space-y-1 px-2">
        {NAV.map(({ id, label, icon: Icon }) => {
          const isActive = active === id
          return (
            <button
              key={id}
              onClick={() => onChange(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                ${isActive
                  ? 'bg-brand-600 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'}`}
            >
              <Icon size={18} className="flex-shrink-0" />
              <span className="hidden md:block">{label}</span>
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-800 hidden md:block">
        <p className="text-xs text-gray-600">AI Grid Energy v2.0</p>
      </div>
    </aside>
  )
}
