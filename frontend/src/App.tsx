import { FormEvent, ReactNode, useEffect, useRef, useState } from 'react'

type Paper = { id: string; title: string; abstract: string; url: string }
type Source = { citation_id: string; title: string; short_title: string; quote: string; url: string }
type ChatMessage = { role: 'user' | 'assistant'; content: string; sources?: Source[]; status?: 'pending' | 'error' }
type Status = { step: string; label: string }

const starterQuestions = ['What datasets do they use in their experiments?', 'What baselines do they compare against?', 'What are the main limitations the authors report?']

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
  return <div className="timeline" aria-label="Agent progress">{statuses.map((item, i) => <div className="timeline-item" key={`${item.step}-${i}`}><span className={`dot ${item.step} ${i === statuses.length - 1 && active ? 'pulse' : ''}`} /><span>{item.label}</span></div>)}</div>
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

  // The agent answers one question about one paper at a time, decided fresh
  // each turn (see backend QasperCorpus.find_paper) -- there is no multi-turn
  // conversational context threaded into it yet, so prior turns aren't sent.
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
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text }), signal: controller.signal })
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
    <header className="topbar"><div className="brand"><span className="brand-mark">✦</span><div><div className="brand-name">NOVA <span>PAPERS</span></div><div className="brand-sub">Evidence-first research assistant</div></div></div><div className="model-pill"><span className="status-dot" /> Qwen2.5-3B <span className="pill-muted">Agentic RL</span></div></header>
    <main className="layout">
      <section className="conversation">
        <div className="intro"><div className="eyebrow">NLP PAPER LIBRARY</div><h1>Ask the research<br /><em>literature.</em></h1><p>Ask a question and watch the agent pick a paper, search it, and read the passages it needs before answering.</p><div className="starter-row">{starterQuestions.map(item => <button key={item} onClick={() => submit(undefined, item)}>{item}<span>↗</span></button>)}</div></div>
        <div className="messages">{messages.map((message, index) => <div className={`message ${message.role}`} key={index}>{message.role === 'assistant' && <div className="avatar">✦</div>}<div className="message-body">{message.role === 'user' ? <div className="user-bubble">{message.content}</div> : <>{message.content ? <Markdown content={message.content} onCitation={id => citation(id, message.sources)} /> : message.status === 'pending' ? <div className="thinking"><span /><span /><span /></div> : null}{message.status === 'error' && <div className="inline-error">Could not complete this answer.</div>}{message.sources?.length ? <div className="answer-source-hint">{message.sources.length} passages attached · click a <span className="citation-sample">[S1]</span> citation to inspect</div> : null}</>}</div></div>)}{statuses.length > 0 && <StatusTimeline statuses={statuses} active={active} />}<div ref={endRef} /></div>
        {error && !active && <div className="error-banner"><span>!</span>{error}</div>}
        <form className="composer" onSubmit={submit}><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask about the papers…" rows={1} disabled={active} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }} /><button aria-label="Send question" disabled={!question.trim() || active}>↑</button><div className="composer-note">Enter to send · Shift + Enter for a new line</div></form>
      </section>
      <aside className="sidebar"><div className="sidebar-header"><div><div className="eyebrow">THE LIBRARY</div><h2>Paper collection</h2></div><span className="count">{papers.length || '—'}</span></div><p className="sidebar-lede">NLP research papers from QASPER. Ask a question and the agent picks the paper it's about, then searches and reads within it.</p><div className="paper-list">{papers.slice(0, 40).map(paper => <a className="paper-card" href={paper.url} target="_blank" rel="noreferrer" key={paper.id}><div className="paper-top"><span className="external">↗</span></div><h3>{paper.title}</h3><p>{paper.abstract.slice(0, 140)}{paper.abstract.length > 140 ? '…' : ''}</p></a>)}</div><div className="grounding-note"><span>◈</span><div><strong>How grounding works</strong><p>The agent searches and reads inside the paper it picks for your question. Open a citation to see the exact passage it used.</p></div></div></aside>
    </main>
    {selected && <div className="source-overlay" role="dialog" aria-modal="true" onClick={() => setSelected(null)}><div className="source-modal" onClick={e => e.stopPropagation()}><button className="close" onClick={() => setSelected(null)}>×</button><div className="eyebrow">SOURCE {selected.citation_id}</div><h2>{selected.short_title}</h2><blockquote>“{selected.quote}”</blockquote><a href={selected.url} target="_blank" rel="noreferrer">Open paper on arXiv ↗</a></div></div>}
  </div>
}
