/**
 * GridVisualization — D3-based force graph of the electricity grid.
 *
 * Nodes = buses (colored by voltage deviation)
 * Edges = lines (colored by loading %)
 * Interactive: zoom, pan, hover tooltips
 */
import React, { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { Bus, Line, GridState } from '../services/api'

interface Props {
  gridState: GridState | null
  width?: number
  height?: number
}

interface Node extends d3.SimulationNodeDatum {
  id: number
  name: string
  vn_kv: number
  vm_pu?: number
}

interface Link extends d3.SimulationLinkDatum<Node> {
  id: number
  loading_pct?: number
  in_service?: boolean
  source: number | Node
  target: number | Node
}

function busColor(vm_pu?: number): string {
  if (vm_pu === undefined) return '#64748b'
  if (vm_pu < 0.95) return '#f87171'  // under-voltage → red
  if (vm_pu > 1.05) return '#fbbf24'  // over-voltage → yellow
  return '#22d3ee'                    // normal → cyan
}

function lineColor(loading?: number): string {
  if (loading === undefined) return '#334155'
  if (loading > 90) return '#ef4444'
  if (loading > 70) return '#fbbf24'
  return '#22d3ee'
}

function lineWidth(loading?: number): number {
  if (loading === undefined) return 1
  return 1 + (loading / 100) * 3
}

const GridVisualization: React.FC<Props> = ({ gridState, width = 900, height = 600 }) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  const { nodes, links } = useMemo<{ nodes: Node[]; links: Link[] }>(() => {
    if (!gridState) return { nodes: [], links: [] }

    const nodes: Node[] = gridState.buses.map(b => ({
      id: b.id,
      name: b.name,
      vn_kv: b.vn_kv,
      vm_pu: b.vm_pu,
    }))

    const links: Link[] = gridState.lines.map(l => ({
      id: l.id,
      source: l.from_bus,
      target: l.to_bus,
      loading_pct: l.loading_pct,
      in_service: l.in_service,
    }))

    return { nodes, links }
  }, [gridState])

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Zoom container
    const g = svg.append('g')
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on('zoom', e => g.attr('transform', e.transform)),
    )

    // Force simulation
    const simulation = d3.forceSimulation<Node>(nodes)
      .force('link', d3.forceLink<Node, Link>(links).id(d => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(12))

    // Draw lines
    const link = g.append('g')
      .selectAll<SVGLineElement, Link>('line')
      .data(links)
      .join('line')
      .attr('stroke', d => lineColor(d.loading_pct))
      .attr('stroke-width', d => lineWidth(d.loading_pct))
      .attr('stroke-opacity', d => (d.in_service === false ? 0.2 : 0.8))

    // Draw bus nodes
    const node = g.append('g')
      .selectAll<SVGCircleElement, Node>('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 8)
      .attr('fill', d => busColor(d.vm_pu))
      .attr('stroke', '#0a0f1e')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .call(
        d3.drag<SVGCircleElement, Node>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x; d.fy = d.y
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null; d.fy = null
          }),
      )

    // Tooltip
    const tooltip = d3.select(tooltipRef.current)
    node
      .on('mouseover', (event, d) => {
        tooltip
          .style('opacity', 1)
          .style('left', `${event.offsetX + 12}px`)
          .style('top', `${event.offsetY - 10}px`)
          .html(`
            <div class="font-mono text-xs">
              <div class="font-bold text-cyan-400">Bus ${d.id}</div>
              <div>${d.name}</div>
              <div>${d.vn_kv} kV</div>
              ${d.vm_pu !== undefined ? `<div>V = ${d.vm_pu.toFixed(4)} p.u.</div>` : ''}
            </div>
          `)
      })
      .on('mouseout', () => tooltip.style('opacity', 0))

    // Bus labels (only for small networks)
    if (nodes.length <= 40) {
      g.append('g')
        .selectAll<SVGTextElement, Node>('text')
        .data(nodes)
        .join('text')
        .text(d => d.id.toString())
        .attr('font-size', 9)
        .attr('fill', '#94a3b8')
        .attr('text-anchor', 'middle')
        .attr('dy', -11)
    }

    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as Node).x ?? 0)
        .attr('y1', d => (d.source as Node).y ?? 0)
        .attr('x2', d => (d.target as Node).x ?? 0)
        .attr('y2', d => (d.target as Node).y ?? 0)

      node
        .attr('cx', d => d.x ?? 0)
        .attr('cy', d => d.y ?? 0)
    })

    return () => simulation.stop()
  }, [nodes, links, width, height])

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="grid-canvas w-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      />
      <div
        ref={tooltipRef}
        className="absolute pointer-events-none bg-gray-900 border border-cyan-800 rounded-lg px-3 py-2 opacity-0 transition-opacity"
        style={{ zIndex: 100 }}
      />

      {/* Legend */}
      <div className="absolute bottom-3 right-3 flex gap-3 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-cyan-400 inline-block" /> Normal
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-yellow-400 inline-block" /> Over-V
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-red-400 inline-block" /> Under-V / Overload
        </span>
      </div>
    </div>
  )
}

export default GridVisualization
