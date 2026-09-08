import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from 'react'

type Paper = { id: string; short_title: string; title: string; authors: string; year: number; topic: string; description: string; url: string; pages: number; chunks: number }
type Source = { citation_id: string; filename: string; title: string; short_title: string; page: number; quote: string; url: string }
type ChatMessage = { role: 'user' | 'assistant'; content: string; sources?: Source[]; status?: 'pending' | 'error' }
type Status = { step: string; label: string }

const starterQuestions = ['What is GRPO, and why was it introduced?', 'How does GRPO differ from PPO?', 'What did InstructGPT change about language model training?']

function Inline({ text, onCitation }: { text: string; onCitation: (id: string) => void }) {
  const chunks = text.split(/(\[S\d+\])/g)
  return <>{chunks.map((chunk, i) => {
    const match = chunk.match(/^\[S(\d+)\]$/)
    if (match) return <button className="citation" key={i} onClick={() => onCitation(`S${match[1]}`)}>{chunk}</button>
    const bold = chunk.split(/(\*\*[^*]+\*\*)/g)
    return <span key={i}>{bold.map((part, j) => part.startsWith('**') ? <strong key={j}>{part.slice(2, -2)}</strong> : part)}</span>
  })}</>
}

function Markdown({ content, onCitation }: { content: string; onCitation: (id: string) => void }) {
  return <div className="markdown">{content.split('\n').map((line, index) => {
    if (!line.trim()) return <div className="spacer" key={index} />
    const heading = line.match(/^#{1,3}\s+(.*)/)
    if (heading) return <h3 key={index}><Inline text={heading[1]} onCitation={onCitation} /></h3>
    const bullet = line.match(/^[-*]\s+(.*)/)
    if (bullet) return <div className="bullet" key={index}><span>•</span><Inline text={bullet[1]} onCitation={onCitation} /></div>
    return <p key={index}><Inline text={line} onCitation={onCitation} /></p>
  })}</div>
}

function StatusTimeline({ statuses, active }: { statuses: Status[]; active: boolean }) {
  return <div className="timeline" aria-label="Agent progress">{statuses.map((item, i) => <div className="timeline-item" key={`${item.step}-${i}`}><span className={`dot ${i === statuses.length - 1 && active ? 'pulse' : ''}`} /><span>{item.label}</span></div>)}</div>
}

export function App() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [statuses, setStatuses] = useState<Status[]>([])
  const [active, setActive] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<Source | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => { fetch('/api/papers').then(r => r.ok ? r.json() : Promise.reject()).then(data => setPapers(data.papers)).catch(() => setError('The paper catalog is unavailable. Start the FastAPI server and try again.')) }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, statuses, active])

  const history = useMemo(() => messages.filter(m => m.content && (m.status !== 'error')).map(({ role, content }) => ({ role, content })), [messages])

  async function submit(event?: FormEvent, preset?: string) {
    event?.preventDefault()
    const text = (preset ?? question).trim()
    if (!text || active) return
    abortRef.current?.abort()
    const controller = new AbortController(); abortRef.current = controller
    setQuestion(''); setError(''); setActive(true); setStatuses([])
    setMessages(prev => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '', status: 'pending' }])
    let answer = ''; let sourceList: Source[] = []
    try {
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, history }), signal: controller.signal })
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `The API returned HTTP ${response.status}.`) }
      if (!response.body) throw new Error('The server did not open a streaming response.')
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let sawDone = false
      const consume = (raw: string) => {
        const blocks = raw.split('\n\n')
        buffer = blocks.pop() || ''
        for (const block of blocks) {
          const kind = block.match(/^event:\s*(.+)$/m)?.[1] || 'message'; const line = block.match(/^data:\s*(.+)$/m)?.[1]
          if (!line) continue
          const data = JSON.parse(line)
          if (kind === 'status') setStatuses(prev => [...prev, data])
          if (kind === 'sources') { sourceList = data.sources; setMessages(prev => prev.map((m, i) => i === prev.length - 1 ? { ...m, sources: sourceList } : m)) }
          if (kind === 'token') { answer += data.text; setMessages(prev => prev.map((m, i) => i === prev.length - 1 ? { ...m, content: answer } : m)) }
          if (kind === 'error') throw new Error(data.message)
          if (kind === 'done') sawDone = true
        }
      }
      while (true) { const { done, value } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); consume(buffer) }
      if (buffer.trim()) consume(`${buffer}\n\n`)
      if (!sawDone) throw new Error('The stream ended before the answer was complete. Please try again.')
      setMessages(prev => prev.map((m, i) => i === prev.length - 1 ? { ...m, status: undefined } : m))
    } catch (caught) {
      if ((caught as Error).name === 'AbortError') return
      const message = caught instanceof Error ? caught.message : 'Something went wrong. Please try again.'
      setError(message); setMessages(prev => prev.map((m, i) => i === prev.length - 1 ? { ...m, status: 'error', content: answer } : m))
    } finally { setActive(false); abortRef.current = null }
  }

  function citation(id: string, sourceList?: Source[]) { const source = sourceList?.find(s => s.citation_id === id) || messages.flatMap(m => m.sources || []).find(s => s.citation_id === id); if (source) setSelected(source) }

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">✦</span><div><div className="brand-name">NOVA <span>PAPERS</span></div><div className="brand-sub">Evidence-first research assistant</div></div></div><div className="model-pill"><span className="status-dot" /> GLM-5.3 <span className="pill-muted">API demo</span></div></header>
    <main className="layout">
      <section className="conversation">
        <div className="intro"><div className="eyebrow">RL / LLM PAPER LIBRARY</div><h1>Ask the research<br /><em>literature.</em></h1><p>Explore ideas in reinforcement learning and language models with answers grounded in the papers that introduced them.</p><div className="starter-row">{starterQuestions.map(item => <button key={item} onClick={() => submit(undefined, item)}>{item}<span>↗</span></button>)}</div></div>
        <div className="messages">{messages.map((message, index) => <div className={`message ${message.role}`} key={index}>{message.role === 'assistant' && <div className="avatar">✦</div>}<div className="message-body">{message.role === 'user' ? <div className="user-bubble">{message.content}</div> : <>{message.content ? <Markdown content={message.content} onCitation={id => citation(id, message.sources)} /> : message.status === 'pending' ? <div className="thinking"><span /><span /><span /></div> : null}{message.status === 'error' && <div className="inline-error">Could not complete this answer.</div>}{message.sources?.length ? <div className="answer-source-hint">{message.sources.length} passages attached · click a <span className="citation-sample">[S1]</span> citation to inspect</div> : null}</>}</div></div>)}{statuses.length > 0 && <StatusTimeline statuses={statuses} active={active} />}<div ref={endRef} /></div>
        {error && !active && <div className="error-banner"><span>!</span>{error}</div>}
        <form className="composer" onSubmit={submit}><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask about the papers…" rows={1} disabled={active} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }} /><button aria-label="Send question" disabled={!question.trim() || active}>↑</button><div className="composer-note">Enter to send · Shift + Enter for a new line</div></form>
      </section>
      <aside className="sidebar"><div className="sidebar-header"><div><div className="eyebrow">THE LIBRARY</div><h2>Paper collection</h2></div><span className="count">{papers.length || '—'}</span></div><p className="sidebar-lede">A small, focused set of papers that shaped modern RL and LLM reasoning.</p><div className="paper-list">{papers.map(paper => <a className="paper-card" href={paper.url} target="_blank" rel="noreferrer" key={paper.id}><div className="paper-top"><span className="paper-year">{paper.year}</span><span className="external">↗</span></div><h3>{paper.short_title}</h3><p>{paper.title}</p><div className="paper-meta"><span>{paper.topic}</span><span>{paper.pages ? `${paper.pages} pages` : 'paper'}</span></div></a>)}</div><div className="grounding-note"><span>◈</span><div><strong>How grounding works</strong><p>Each answer is prompted with retrieved passages. Open a citation to see the exact filename and excerpt.</p></div></div></aside>
    </main>
    {selected && <div className="source-overlay" role="dialog" aria-modal="true" onClick={() => setSelected(null)}><div className="source-modal" onClick={e => e.stopPropagation()}><button className="close" onClick={() => setSelected(null)}>×</button><div className="eyebrow">SOURCE {selected.citation_id}</div><h2>{selected.short_title}</h2><div className="source-file">{selected.filename} · page {selected.page}</div><blockquote>“{selected.quote}”</blockquote><a href={selected.url} target="_blank" rel="noreferrer">Open paper on arXiv ↗</a></div></div>}
  </div>
}
