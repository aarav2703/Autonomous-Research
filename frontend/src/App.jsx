import { useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const DEFAULT_QUESTION = "Which magazine was started first Arthur's Magazine or First for Women?"

function App() {
  const [question, setQuestion] = useState(DEFAULT_QUESTION)
  const [answerData, setAnswerData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [useHybridRetrieval, setUseHybridRetrieval] = useState(true)
  const [useMultiHop, setUseMultiHop] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setIsLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question,
          retrieval_top_k: 5,
          evidence_top_k: 5,
          use_hybrid_retrieval: useHybridRetrieval,
          use_multi_hop: useMultiHop
        })
      })

      if (!response.ok) {
        throw new Error(`Backend request failed with status ${response.status}`)
      }

      const payload = await response.json()
      setAnswerData(payload)
    } catch (requestError) {
      setError(requestError.message)
      setAnswerData(null)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <p className="eyebrow">Autonomous Multi-hop Research Agent</p>
        <h1>Grounded answers with visible reasoning and evidence.</h1>
        <p className="hero-copy">
          Ask a question, send it through the retrieval plus evidence workflow, and inspect exactly
          which sentences the system used to justify the final answer.
        </p>
      </section>

      <section className="query-panel">
        <form onSubmit={handleSubmit} className="query-form">
          <label htmlFor="question" className="section-label">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={4}
            placeholder="Ask a multi-hop research question..."
          />
          <div className="query-actions">
            <button type="submit" disabled={isLoading || !question.trim()}>
              {isLoading ? 'Thinking...' : 'Ask Agent'}
            </button>
            <span className="api-hint">Backend: {API_BASE_URL}</span>
          </div>
          <div className="retrieval-toggles">
            <label className="toggle-chip">
              <input
                type="checkbox"
                checked={useHybridRetrieval}
                onChange={(event) => setUseHybridRetrieval(event.target.checked)}
              />
              <span>Hybrid retrieval</span>
            </label>
            <label className="toggle-chip">
              <input
                type="checkbox"
                checked={useMultiHop}
                onChange={(event) => setUseMultiHop(event.target.checked)}
              />
              <span>Multi-hop expansion</span>
            </label>
          </div>
        </form>
      </section>

      {error ? (
        <section className="error-panel">
          <h2>Request Error</h2>
          <p>{error}</p>
        </section>
      ) : null}

      {answerData ? (
        <section className="results-grid">
          <article className="answer-card spotlight-card">
            <p className="card-label">Answer</p>
            <h2>{answerData.answer}</h2>
            <div className="status-row">
              <span className={`status-pill status-${answerData.status}`}>{answerData.status}</span>
              {answerData.metadata?.failure_reason ? (
                <span className="failure-copy">{answerData.metadata.failure_reason}</span>
              ) : null}
            </div>
          </article>

          <article className="answer-card reasoning-card">
            <p className="card-label">Reasoning Steps</p>
            <ol className="reasoning-list">
              {answerData.reasoning.map((step, index) => (
                <li key={`${step}-${index}`}>{step}</li>
              ))}
            </ol>
          </article>

          <article className="answer-card trace-card">
            <p className="card-label">Workflow Trace</p>
            <ul className="trace-list">
              {answerData.execution_trace.map((step, index) => (
                <li key={`${step}-${index}`}>{step}</li>
              ))}
            </ul>
          </article>

          <article className="answer-card retrieval-card">
            <div className="retrieval-header">
              <p className="card-label">Retrieval Process</p>
              <span>{answerData.retrieval_debug?.mode ?? 'unknown mode'}</span>
            </div>
            <div className="retrieval-columns">
              <section>
                <h3>Dense</h3>
                <ul className="retrieval-list">
                  {(answerData.retrieval_debug?.dense_titles ?? answerData.retrieval_debug?.hop1_titles ?? []).slice(0, 8).map((title, index) => (
                    <li key={`${title}-${index}`}>{title}</li>
                  ))}
                </ul>
              </section>
              <section>
                <h3>BM25</h3>
                <ul className="retrieval-list">
                  {(answerData.retrieval_debug?.bm25_titles ?? []).slice(0, 8).map((title, index) => (
                    <li key={`${title}-${index}`}>{title}</li>
                  ))}
                  {!(answerData.retrieval_debug?.bm25_titles ?? []).length ? (
                    <li className="retrieval-empty">Not used in this run.</li>
                  ) : null}
                </ul>
              </section>
              <section>
                <h3>Merged</h3>
                <ul className="retrieval-list">
                  {(answerData.retrieval_debug?.merged_titles ?? []).slice(0, 8).map((title, index) => (
                    <li key={`${title}-${index}`}>{title}</li>
                  ))}
                </ul>
              </section>
            </div>
            {answerData.retrieval_debug?.title_boosts?.length ? (
              <div className="boost-panel">
                <p className="boost-label">Title boosts</p>
                <ul className="boost-list">
                  {answerData.retrieval_debug.title_boosts.slice(0, 6).map((item, index) => (
                    <li key={`${item.paragraph_id}-${index}`}>
                      <span>{item.title}</span>
                      <strong>+{Number(item.title_boost).toFixed(2)}</strong>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </article>

          <article className="answer-card evidence-card">
            <div className="evidence-header">
              <p className="card-label">Evidence Viewer</p>
              <span>{answerData.evidence.length} cited sentence(s)</span>
            </div>
            <div className="evidence-stack">
              {answerData.evidence.length ? answerData.evidence.map((item) => (
                <section key={item.sentence_id} className="evidence-item">
                  <div className="evidence-meta">
                    <span className="evidence-title">{item.title}</span>
                    <span className="evidence-id">{item.sentence_id}</span>
                  </div>
                  <p className="evidence-text">
                    <mark>{item.sentence_text}</mark>
                  </p>
                </section>
              )) : (
                <p className="empty-copy">No cited evidence was returned for this response.</p>
              )}
            </div>
          </article>
        </section>
      ) : (
        <section className="empty-state">
          <p>Submit a question to populate the answer, reasoning, and evidence panels.</p>
        </section>
      )}
    </main>
  )
}

export default App
