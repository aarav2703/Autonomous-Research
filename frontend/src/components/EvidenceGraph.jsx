import { useMemo, useState } from 'react'
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from 'd3-force'

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function normalize(value, min, max) {
  if (Number.isNaN(value) || !Number.isFinite(value)) return 0
  if (max - min < 1e-8) return 0.5
  return (value - min) / (max - min)
}

function truncate(text, limit = 90) {
  if (!text) return ''
  return text.length > limit ? `${text.slice(0, limit - 3)}...` : text
}

export default function EvidenceGraph({ answerData }) {
  const [showTopK, setShowTopK] = useState(true)
  const [showHop2Only, setShowHop2Only] = useState(false)
  const [activeNodeId, setActiveNodeId] = useState(null)
  const [hoveredNodeId, setHoveredNodeId] = useState(null)
  const [zoom, setZoom] = useState(1)

  const graph = useMemo(() => {
    if (!answerData) {
      return { nodes: [], links: [], width: 980, height: 600, activePath: new Set(), activeLinks: new Set(), idToNode: new Map() }
    }

    const retrievedChunks = Array.isArray(answerData.retrieved_chunks) ? answerData.retrieved_chunks : []
    const selectedEvidence = Array.isArray(answerData.selected_evidence) ? answerData.selected_evidence : []
    const retrievalDebug = answerData.retrieval_debug ?? {}
    const hop1Titles = new Set(retrievalDebug.hop1_titles ?? [])
    const hop2Titles = new Set(retrievalDebug.hop2_titles ?? [])
    const subqueryTitlesMap = retrievalDebug.subquery_titles ?? {}
    const subqueryTitles = new Set(Object.values(subqueryTitlesMap).flat())

    const uniqueDocTitles = new Map()
    retrievedChunks.forEach((chunk) => {
      const title = chunk?.title
      if (!title || uniqueDocTitles.has(title)) return
      uniqueDocTitles.set(title, {
        ...chunk,
        title,
        score: Number(chunk.score ?? 0),
      })
    })

    const docEntries = Array.from(uniqueDocTitles.values())
      .sort((left, right) => Number(right.score ?? 0) - Number(left.score ?? 0))
      .filter((chunk) => {
        if (!showHop2Only) return true
        return hop2Titles.has(chunk.title) || subqueryTitles.has(chunk.title)
      })
      .slice(0, showTopK ? 8 : uniqueDocTitles.size)

    const sentenceEntries = selectedEvidence
      .map((item, index) => ({
        ...item,
        title: item?.title ?? '',
        sentence_text: item?.sentence_text ?? '',
        final_score: Number(item?.final_score ?? 0),
        _index: index,
      }))
      .filter((item) => {
        if (!showHop2Only) return true
        return hop2Titles.has(item.title) || subqueryTitles.has(item.title)
      })
      .slice(0, showTopK ? 10 : selectedEvidence.length)

    const docScores = docEntries.map((item) => Number(item.score ?? 0))
    const sentenceScores = sentenceEntries.map((item) => Number(item.final_score ?? 0))
    const docMin = docScores.length ? Math.min(...docScores) : 0
    const docMax = docScores.length ? Math.max(...docScores) : 1
    const sentenceMin = sentenceScores.length ? Math.min(...sentenceScores) : 0
    const sentenceMax = sentenceScores.length ? Math.max(...sentenceScores) : 1

    const nodes = []
    const links = []

    nodes.push({
      id: 'query',
      type: 'query',
      label: truncate(answerData.question ?? 'Query', 48),
      text: answerData.question ?? '',
      score: 1,
      hop: 0,
      radius: 26,
    })

    nodes.push({
      id: 'answer',
      type: 'answer',
      label: truncate(answerData.answer ?? 'Final answer', 36),
      text: answerData.answer ?? '',
      score: 1,
      hop: 3,
      radius: 26,
    })

    const sentenceByTitle = new Map()
    sentenceEntries.forEach((item) => {
      if (!sentenceByTitle.has(item.title)) sentenceByTitle.set(item.title, [])
      sentenceByTitle.get(item.title).push(item)
    })

    docEntries.forEach((doc) => {
      const hop = hop2Titles.has(doc.title) || subqueryTitles.has(doc.title) || !hop1Titles.has(doc.title) ? 2 : 1
      const nodeId = `doc:${doc.title}`
      const scoreNorm = normalize(Number(doc.score ?? 0), docMin, docMax)
      const radius = 16 + scoreNorm * 14
      nodes.push({
        id: nodeId,
        type: 'document',
        label: truncate(doc.title, 26),
        text: doc.paragraph_text ?? doc.title,
        score: Number(doc.score ?? 0),
        hop,
        radius,
        color: hop === 2 ? '#df7f14' : '#4c78a8',
        title: doc.title,
      })

      links.push({
        source: 'query',
        target: nodeId,
        weight: 1 + scoreNorm * 2,
        kind: 'query-doc',
      })

      const docSentences = sentenceByTitle.get(doc.title) ?? []
      docSentences.forEach((sentence) => {
        const sentenceId = sentence.sentence_id ?? `sentence:${doc.title}:${sentence._index}`
        const sentenceNorm = normalize(Number(sentence.final_score ?? 0), sentenceMin, sentenceMax)
        nodes.push({
          id: sentenceId,
          type: 'evidence',
          label: truncate(sentence.sentence_text, 36),
          text: sentence.sentence_text,
          score: Number(sentence.final_score ?? 0),
          hop,
          radius: 11 + sentenceNorm * 9,
          color: hop === 2 ? '#df7f14' : '#5f8fc0',
          title: doc.title,
        })

        links.push({
          source: nodeId,
          target: sentenceId,
          weight: 0.6 + sentenceNorm * 2,
          kind: 'doc-sentence',
        })
        links.push({
          source: sentenceId,
          target: 'answer',
          weight: 0.5 + sentenceNorm * 2.2,
          kind: 'sentence-answer',
        })
      })
    })

    const idToNode = new Map(nodes.map((node) => [node.id, node]))
    const simulationNodes = nodes.map((node) => ({ ...node }))
    const simulationLinks = links.map((link) => ({ ...link }))

    const simulation = forceSimulation(simulationNodes)
      .force(
        'link',
        forceLink(simulationLinks)
          .id((node) => node.id)
          .distance((link) => {
            if (link.kind === 'query-doc') return 130
            if (link.kind === 'doc-sentence') return 98
            return 84
          })
          .strength((link) => clamp(link.weight / 3, 0.18, 0.95)),
      )
      .force('charge', forceManyBody().strength(-300))
      .force('center', forceCenter(490, 300))
      .force('collision', forceCollide().radius((node) => node.radius + 10))
      .force('x', forceX(490).strength(0.04))
      .force('y', forceY(300).strength(0.04))

    for (let tick = 0; tick < 220; tick += 1) {
      simulation.tick()
    }
    simulation.stop()

    const pathIds = new Set()
    const pathLinks = new Set()
    if (activeNodeId && idToNode.has(activeNodeId)) {
      pathIds.add(activeNodeId)
      pathIds.add('answer')

      if (activeNodeId === 'query') {
        nodes.forEach((node) => pathIds.add(node.id))
        links.forEach((link) => pathLinks.add(`${link.source.id ?? link.source}->${link.target.id ?? link.target}`))
      } else if (activeNodeId.startsWith('doc:')) {
        const activeDocTitle = idToNode.get(activeNodeId)?.title
        pathIds.add('query')
        nodes
          .filter((node) => node.type === 'evidence' && node.title === activeDocTitle)
          .forEach((node) => pathIds.add(node.id))
        links.forEach((link) => {
          const sourceId = link.source.id ?? link.source
          const targetId = link.target.id ?? link.target
          if (sourceId === 'query' && targetId === activeNodeId) pathLinks.add(`${sourceId}->${targetId}`)
          if (sourceId === activeNodeId && targetId.startsWith('sentence:')) pathLinks.add(`${sourceId}->${targetId}`)
          if (targetId === 'answer' && sourceId.startsWith('sentence:')) pathLinks.add(`${sourceId}->${targetId}`)
        })
      } else if (activeNodeId.startsWith('sentence:')) {
        const sentenceNode = idToNode.get(activeNodeId)
        const docNode = nodes.find((node) => node.type === 'document' && node.title === sentenceNode?.title)
        if (docNode) {
          pathIds.add(docNode.id)
          pathIds.add('query')
        }
        links.forEach((link) => {
          const sourceId = link.source.id ?? link.source
          const targetId = link.target.id ?? link.target
          if (sourceId === 'query' && targetId === docNode?.id) pathLinks.add(`${sourceId}->${targetId}`)
          if (sourceId === docNode?.id && targetId === activeNodeId) pathLinks.add(`${sourceId}->${targetId}`)
          if (sourceId === activeNodeId && targetId === 'answer') pathLinks.add(`${sourceId}->${targetId}`)
        })
      }
    }

    const svgNodes = simulationNodes.map((node) => ({
      ...node,
      x: node.x,
      y: node.y,
    }))

    const horizontalPadding = 90
    const verticalPadding = 72
    const minX = Math.min(...svgNodes.map((node) => node.x - node.radius))
    const maxX = Math.max(...svgNodes.map((node) => node.x + node.radius))
    const minY = Math.min(...svgNodes.map((node) => node.y - node.radius))
    const maxY = Math.max(...svgNodes.map((node) => node.y + node.radius + 26))
    const sourceWidth = Math.max(1, maxX - minX)
    const sourceHeight = Math.max(1, maxY - minY)
    const targetWidth = 980 - horizontalPadding * 2
    const targetHeight = 600 - verticalPadding * 2
    const fitScale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight, 1.12)

    const fittedNodes = svgNodes.map((node) => ({
      ...node,
      x: horizontalPadding + ((node.x - minX) * fitScale),
      y: verticalPadding + ((node.y - minY) * fitScale),
      radius: node.radius * Math.max(0.92, fitScale),
    }))

    return {
      nodes: fittedNodes,
      links: simulationLinks.map((link) => ({
        ...link,
        sourceId: link.source.id ?? link.source,
        targetId: link.target.id ?? link.target,
      })),
      width: 980,
      height: 600,
      activePath: pathIds,
      activeLinks: pathLinks,
      idToNode: new Map(fittedNodes.map((node) => [node.id, node])),
    }
  }, [answerData, activeNodeId, showHop2Only, showTopK])

  if (!answerData) {
    return <div className="graph-empty">Submit a question to render the evidence graph.</div>
  }

  const hoveredNode = hoveredNodeId ? graph.idToNode.get(hoveredNodeId) ?? null : null
  const detailNode = hoveredNode ?? graph.idToNode.get(activeNodeId) ?? null
  const retrievalDebug = answerData.retrieval_debug ?? {}
  const totalDocs = (answerData.retrieved_chunks ?? []).length
  const totalEvidence = (answerData.selected_evidence ?? []).length
  const visibleDocs = graph.nodes.filter((node) => node.type === 'document').length
  const visibleEvidence = graph.nodes.filter((node) => node.type === 'evidence').length
  const translateX = ((1 - zoom) * graph.width) / 2
  const translateY = ((1 - zoom) * graph.height) / 2

  function handleZoom(delta) {
    setZoom((value) => clamp(Number((value + delta).toFixed(2)), 0.7, 2.4))
  }

  return (
    <div className="graph-shell">
      <div className="graph-toolbar">
        <div className="graph-toolbar-group">
          <label className="toggle-chip">
            <input
              type="checkbox"
              checked={showTopK}
              onChange={(event) => setShowTopK(event.target.checked)}
            />
            <span>Show only top-k</span>
          </label>
          <label className="toggle-chip">
            <input
              type="checkbox"
              checked={showHop2Only}
              onChange={(event) => setShowHop2Only(event.target.checked)}
            />
            <span>Show only hop2 nodes</span>
          </label>
        </div>

        <div className="graph-toolbar-group graph-toolbar-actions">
          <div className="graph-zoom-controls">
            <button type="button" className="graph-reset" onClick={() => handleZoom(-0.2)}>
              -
            </button>
            <span className="graph-zoom-readout">{Math.round(zoom * 100)}%</span>
            <button type="button" className="graph-reset" onClick={() => handleZoom(0.2)}>
              +
            </button>
            <button type="button" className="graph-reset" onClick={() => setZoom(1)}>
              Reset zoom
            </button>
          </div>
          <button type="button" className="graph-reset" onClick={() => setActiveNodeId(null)}>
            Reset highlight
          </button>
        </div>
      </div>

      <div className="graph-summary">
        <span>Docs: {visibleDocs}/{totalDocs}</span>
        <span>Evidence: {visibleEvidence}/{totalEvidence}</span>
        <span>Subqueries: {answerData.subqueries?.length ?? 0}</span>
        <span>Hop count: {answerData.hop_count ?? 0}</span>
        <span>Mode: {retrievalDebug.mode ?? 'unknown'}</span>
      </div>

      <div className="graph-inspector">
        {detailNode ? (
          <>
            <div className="graph-inspector-header">
              <strong>{detailNode.label}</strong>
              <span className={`graph-node-kind graph-node-kind-${detailNode.type}`}>{detailNode.type}</span>
            </div>
            <div className="graph-inspector-meta">
              <span>Score: {Number(detailNode.score ?? 0).toFixed(4)}</span>
              <span>Hop: {detailNode.hop ?? 0}</span>
              {detailNode.title ? <span>Title: {detailNode.title}</span> : null}
            </div>
            <p>{detailNode.text || 'No text available.'}</p>
          </>
        ) : (
          <p>Hover over a node or click one to pin its details here.</p>
        )}
      </div>

      <div className="graph-canvas-wrap">
        <svg viewBox={`0 0 ${graph.width} ${graph.height}`} className="graph-canvas" role="img" aria-label="Evidence graph visualization">
          <g transform={`translate(${translateX}, ${translateY}) scale(${zoom})`}>
            {graph.links.map((link) => {
              const sourceNode = graph.idToNode.get(link.sourceId)
              const targetNode = graph.idToNode.get(link.targetId)
              if (!sourceNode || !targetNode) return null
              const isActive = !graph.activePath.size || graph.activeLinks.has(`${link.sourceId}->${link.targetId}`)
              return (
                <line
                  key={`${link.sourceId}-${link.targetId}`}
                  x1={sourceNode.x}
                  y1={sourceNode.y}
                  x2={targetNode.x}
                  y2={targetNode.y}
                  className={`graph-link ${isActive ? 'is-active' : 'is-muted'}`}
                  style={{ strokeWidth: clamp(link.weight, 0.6, 4.2) }}
                />
              )
            })}

            {graph.nodes.map((node) => {
              const isActive = !graph.activePath.size || graph.activePath.has(node.id)
              const isSelected = activeNodeId === node.id
              const isHovered = hoveredNodeId === node.id
              const showLabel = isSelected || node.type === 'query' || node.type === 'answer' || zoom >= 1.25
              const fill = node.type === 'answer' ? '#1f2933' : node.type === 'query' ? '#d97706' : node.color
              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y})`}
                  className={`graph-node ${isActive ? 'is-active' : 'is-muted'} ${isSelected ? 'is-selected' : ''} ${isHovered ? 'is-hovered' : ''}`}
                  onClick={() => setActiveNodeId(node.id)}
                  onMouseEnter={() => setHoveredNodeId(node.id)}
                  onMouseLeave={() => setHoveredNodeId(null)}
                  onFocus={() => setHoveredNodeId(node.id)}
                  onBlur={() => setHoveredNodeId(null)}
                  role="button"
                  tabIndex={0}
                >
                  <circle r={node.radius} fill={fill} />
                  <circle r={node.radius} className="graph-node-ring" />
                  {showLabel ? (
                    <text className="graph-node-label" y={node.radius + 16} textAnchor="middle">
                      {node.label}
                    </text>
                  ) : null}
                </g>
              )
            })}
          </g>
        </svg>
      </div>
    </div>
  )
}
