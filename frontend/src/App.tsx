import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api, type Citation, type DocumentInfo, type Health, type SessionInfo, type TraceStep } from './api'
import './App.css'

interface ChatItem {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  trace?: TraceStep[]
  abstained?: boolean
  latencyMs?: number
}

const NOT_FOUND = 'Not found in the provided documents.'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatItem[]>([])
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [collection, setCollection] = useState('default')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const refreshDocuments = () => api.listDocuments().then(setDocuments)
  const refreshSessions = () => api.listSessions().then(setSessions)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError('Backend is not reachable on port 8000.'))
    refreshDocuments().catch(() => {})
    refreshSessions().catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return
    setUploading(true)
    setError(null)
    try {
      for (const file of Array.from(files)) {
        const result = await api.uploadDocument(file, collection)
        if (result.duplicate) setError(`${file.name} is already uploaded (same content).`)
      }
      await refreshDocuments()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(doc: DocumentInfo) {
    if (!confirm(`Delete ${doc.filename}?`)) return
    await api.deleteDocument(doc.id).catch((e) => setError(e.message))
    setSelectedDocs((ids) => ids.filter((id) => id !== doc.id))
    await refreshDocuments()
  }

  async function openSession(id: number) {
    setSessionId(id)
    const history = await api.sessionMessages(id)
    setMessages(
      history.map((m) => ({
        role: m.role,
        content: m.content,
        citations: m.citations,
        trace: m.trace,
        abstained: m.role === 'assistant' && m.content === NOT_FOUND,
      })),
    )
  }

  function newChat() {
    setSessionId(null)
    setMessages([])
  }

  async function ask(e: FormEvent) {
    e.preventDefault()
    const text = question.trim()
    if (!text || busy) return
    setQuestion('')
    setError(null)
    setMessages((m) => [...m, { role: 'user', content: text }])
    setBusy(true)
    try {
      const res = await api.chat(text, sessionId, selectedDocs)
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: res.answer,
          citations: res.citations,
          trace: res.trace,
          abstained: res.abstained,
          latencyMs: res.latency_ms,
        },
      ])
      if (sessionId === null) {
        setSessionId(res.session_id)
        refreshSessions()
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const toggleDoc = (id: string) =>
    setSelectedDocs((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]))

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>CiteLocal</h1>
        <p className="status">
          {health ? (
            <>
              <span className={health.ollama ? 'dot ok' : 'dot bad'} /> {health.llm_model} · runs locally
            </>
          ) : (
            'Connecting…'
          )}
        </p>

        <section>
          <h2>Documents</h2>
          <label className="upload">
            <input type="file" accept="application/pdf" multiple disabled={uploading}
              onChange={(e) => { handleUpload(e.target.files); e.target.value = '' }} />
            {uploading ? 'Indexing…' : '+ Upload PDFs'}
          </label>
          <input className="collection" value={collection} onChange={(e) => setCollection(e.target.value)}
            placeholder="collection" title="Collection for new uploads" />
          {documents.length === 0 && <p className="muted">No documents yet.</p>}
          <ul className="doc-list">
            {documents.map((d) => (
              <li key={d.id}>
                <label title="Only search selected documents">
                  <input type="checkbox" checked={selectedDocs.includes(d.id)} onChange={() => toggleDoc(d.id)} />
                  <span className="doc-name">{d.filename}</span>
                </label>
                <span className="muted small">{d.num_pages}p · {d.collection}</span>
                <button className="icon" onClick={() => handleDelete(d)} aria-label={`Delete ${d.filename}`}>×</button>
              </li>
            ))}
          </ul>
          {selectedDocs.length > 0 && (
            <p className="muted small">Searching {selectedDocs.length} selected document(s) only.</p>
          )}
        </section>

        <section>
          <h2>Chats <button className="link" onClick={newChat}>New chat</button></h2>
          <ul className="session-list">
            {sessions.map((s) => (
              <li key={s.id}>
                <button className={s.id === sessionId ? 'session active' : 'session'} onClick={() => openSession(s.id)}>
                  {s.title}
                </button>
              </li>
            ))}
          </ul>
        </section>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <p>Ask a question about your documents.</p>
              <p className="muted">Answers cite the file and page they came from. If the documents do not contain the answer, the assistant says so.</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}${m.abstained ? ' abstained' : ''}`}>
              <div className="content">{m.content}</div>
              {m.citations && m.citations.length > 0 && (
                <div className="citations">
                  {m.citations.map((c) => (
                    <span key={`${c.filename}-${c.page}`} className="citation">{c.filename} · p. {c.page}</span>
                  ))}
                </div>
              )}
              {!m.abstained && m.citations?.length === 0 && m.trace?.some((t) => t.step === 'generate') && (
                <div className="citations"><span className="citation warn">No source cited, check the answer</span></div>
              )}
              {m.trace && m.trace.length > 0 && (
                <details className="trace">
                  <summary>Agent steps{m.latencyMs ? ` · ${(m.latencyMs / 1000).toFixed(1)}s` : ''}</summary>
                  <ol>
                    {m.trace.map((t, j) => {
                      const { step, ms, ...detail } = t
                      return <li key={j}><b>{step}</b> ({ms} ms) <code>{JSON.stringify(detail)}</code></li>
                    })}
                  </ol>
                </details>
              )}
            </div>
          ))}
          {busy && <div className="message assistant thinking">Thinking…</div>}
          <div ref={bottomRef} />
        </div>
        {error && <div className="error" onClick={() => setError(null)}>{error}</div>}
        <form className="composer" onSubmit={ask}>
          <input value={question} onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about your documents…" disabled={busy} />
          <button type="submit" disabled={busy || !question.trim()}>Send</button>
        </form>
      </main>
    </div>
  )
}
