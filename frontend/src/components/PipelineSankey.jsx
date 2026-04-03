import { useMemo, useState } from 'react'
import { sankey, sankeyLinkHorizontal } from 'd3-sankey'

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function percentDrop(source, target) {
  if (!source) return 0
  return clamp(((source - target) / source) * 100, 0, 100)
}

function formatCount(value) {
  return Number(value ?? 0).toLocaleString()
}

export default function PipelineSankey({ answerData }) {
  const [hoveredLink, setHoveredLink] = useState(null)
  const [activeNode, setActiveNode] = useState(null)

  const graph = useMemo(() => {
    const counts = answerData?.stage_counts ?? {}
    const retrieved = Number(counts.retrieved ?? answerData?.retrieved_chunks?.length ?? 0)
    const reranked = Number(counts.reranked ?? answerData?.selected_evidence?.length ?? 0)
    const evidence = Number(counts.evidence ?? answerData?.selected_evidence?.length ?? 0)
    const answer = Number(counts.answer ?? (answerData?.answer ? 1 : 0))

    const discardedAfterRetrieval = Number(counts.discarded_after_retrieval ?? Math.max(0, retrieved - reranked))
    const discardedAfterRerank = Number(counts.discarded_after_rerank ?? Math.max(0, reranked - evidence))
    const discardedAfterEvidence = Number(counts.discarded_after_evidence ?? Math.max(0, evidence - answer))

    const nodes = [
      { id: 'retrieved', name: `Retrieved (${formatCount(retrieved)})`, stage: 'retrieved' },
      { id: 'reranked', name: `Reranked (${formatCount(reranked)})`, stage: 'reranked' },
      { id: 'evidence', name: `Evidence (${formatCount(evidence)})`, stage: 'evidence' },
      { id: 'answer', name: `Answer (${formatCount(answer)})`, stage: 'answer' },
      { id: 'drop-1', name: `Discarded (${formatCount(discardedAfterRetrieval)})`, stage: 'discarded' },
      { id: 'drop-2', name: `Discarded (${formatCount(discardedAfterRerank)})`, stage: 'discarded' },
      { id: 'drop-3', name: `Discarded (${formatCount(discardedAfterEvidence)})`, stage: 'discarded' },
    ]

    const links = [
      { source: 'retrieved', target: 'reranked', value: reranked, kind: 'kept', stage: 'retrieved' },
      { source: 'retrieved', target: 'drop-1', value: discardedAfterRetrieval, kind: 'discarded', stage: 'retrieved' },
      { source: 'reranked', target: 'evidence', value: evidence, kind: 'kept', stage: 'reranked' },
      { source: 'reranked', target: 'drop-2', value: discardedAfterRerank, kind: 'discarded', stage: 'reranked' },
      { source: 'evidence', target: 'answer', value: answer, kind: 'kept', stage: 'evidence' },
      { source: 'evidence', target: 'drop-3', value: discardedAfterEvidence, kind: 'discarded', stage: 'evidence' },
    ].filter((link) => link.value > 0)

    const sankeyLayout = sankey()
      .nodeId((node) => node.id)
      .nodeWidth(22)
      .nodePadding(26)
      .nodeAlign((node) => node.depth)
      .extent([[18, 20], [960, 260]])

    const layout = sankeyLayout({
      nodes: nodes.map((node) => ({ ...node })),
      links: links.map((link) => ({ ...link })),
    })

    const byId = new Map(layout.nodes.map((node) => [node.id, node]))
    const activeNodeId = activeNode ? activeNode.id : null

    return {
      nodes: layout.nodes,
      links: layout.links,
      retrieved,
      reranked,
      evidence,
      answer,
      discardedAfterRetrieval,
      discardedAfterRerank,
      discardedAfterEvidence,
      activeNodeMatch: activeNodeId ? byId.get(activeNodeId) : null,
    }
  }, [answerData, activeNode])

  if (!answerData) {
    return null
  }

  const total = graph.retrieved || 1
  const dropPercent = hoveredLink ? percentDrop(hoveredLink.source.value, hoveredLink.value) : 0

  return (
    <div className="sankey-shell">
      <div className="sankey-legend">
        <span className="legend-chip legend-kept">Kept</span>
        <span className="legend-chip legend-discarded">Discarded</span>
        <span className="legend-chip">Total retrieved: {formatCount(total)}</span>
      </div>

      <svg className="sankey-svg" viewBox="0 0 980 300" role="img" aria-label="Pipeline shrink Sankey diagram">
        {graph.links.map((linkItem) => {
          const path = sankeyLinkHorizontal()(linkItem)
          const isActive =
            !graph.activeNodeMatch ||
            graph.activeNodeMatch.id === linkItem.source.id ||
            graph.activeNodeMatch.id === linkItem.target.id
          const opacity = graph.activeNodeMatch ? (isActive ? 0.92 : 0.18) : 0.82
          return (
            <path
              key={`${linkItem.source.id}-${linkItem.target.id}`}
              d={path}
              className={`sankey-link ${linkItem.kind === 'discarded' ? 'is-discarded' : 'is-kept'}`}
              style={{ opacity, strokeWidth: Math.max(2, linkItem.width) }}
              onMouseEnter={() => setHoveredLink(linkItem)}
              onMouseLeave={() => setHoveredLink(null)}
            />
          )
        })}

        {graph.nodes.map((node) => {
          const isSelected = graph.activeNodeMatch ? graph.activeNodeMatch.id === node.id : false
          const isDiscarded = node.stage === 'discarded'
          const fill = isDiscarded ? '#d9534f' : node.stage === 'answer' ? '#2f6b4f' : '#4c78a8'
          return (
            <g
              key={node.id}
              transform={`translate(${node.x0}, ${node.y0})`}
              className={`sankey-node ${isSelected ? 'is-selected' : ''}`}
              onClick={() => setActiveNode(node)}
              role="button"
              tabIndex={0}
            >
              <rect
                width={node.x1 - node.x0}
                height={node.y1 - node.y0}
                rx={10}
                fill={fill}
              />
              <text x={node.x0 < 490 ? 8 : -8} y={(node.y1 - node.y0) / 2} dy="0.35em" textAnchor={node.x0 < 490 ? 'start' : 'end'}>
                {node.name}
              </text>
            </g>
          )
        })}
      </svg>

      <div className="sankey-summary">
        <span>Retrieved -> Reranked: {formatCount(graph.reranked)} kept, {formatCount(graph.discardedAfterRetrieval)} discarded</span>
        <span>Reranked -> Evidence: {formatCount(graph.evidence)} kept, {formatCount(graph.discardedAfterRerank)} discarded</span>
        <span>Evidence -> Answer: {formatCount(graph.answer)} kept, {formatCount(graph.discardedAfterEvidence)} discarded</span>
      </div>

      {hoveredLink ? (
        <div className="sankey-tooltip">
          <strong>{hoveredLink.source.name} -> {hoveredLink.target.name}</strong>
          <span>Count: {formatCount(hoveredLink.value)}</span>
          <span>Drop: {dropPercent.toFixed(1)}%</span>
          <span className={hoveredLink.kind === 'discarded' ? 'discarded-note' : 'kept-note'}>
            {hoveredLink.kind === 'discarded' ? 'Discarded flow' : 'Kept flow'}
          </span>
        </div>
      ) : null}
    </div>
  )
}
