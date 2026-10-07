// Typed client for the FastAPI backend. Types mirror backend/app/schemas.py.

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export interface DocumentInfo {
  id: string
  filename: string
  collection: string
  num_pages: number
  num_chunks: number
  created_at: string
}

export interface UploadResult {
  document: DocumentInfo
  duplicate: boolean
}

export interface Citation {
  filename: string
  page: number
}

export interface SourceChunk {
  chunk_id: number
  filename: string
  page: number
  text: string
  score: number | null
}

export interface TraceStep {
  step: string
  ms: number
  [detail: string]: unknown
}

export interface ChatResponse {
  session_id: number
  answer: string
  abstained: boolean
  reason: string
  citations: Citation[]
  sources: SourceChunk[]
  trace: TraceStep[]
  latency_ms: number
}

export interface SessionInfo {
  id: number
  title: string
  created_at: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  trace: TraceStep[]
  created_at: string
}

export interface Health {
  status: string
  documents: number
  chunks: number
  llm_model: string
  ollama: boolean
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  health: () => request<Health>('/health'),
  listDocuments: () => request<DocumentInfo[]>('/documents'),
  uploadDocument: (file: File, collection: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('collection', collection || 'default')
    return request<UploadResult>('/documents', { method: 'POST', body: form })
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' }),
  chat: (question: string, sessionId: number | null, docIds: string[]) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        session_id: sessionId,
        doc_ids: docIds.length ? docIds : null,
      }),
    }),
  listSessions: () => request<SessionInfo[]>('/sessions'),
  sessionMessages: (id: number) => request<Message[]>(`/sessions/${id}/messages`),
}
