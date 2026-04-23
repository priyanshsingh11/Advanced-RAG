'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  confidence?: number;
  sources?: string[];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      const data = await response.json();
      const assistantMsg: Message = {
        role: 'assistant',
        content: data.answer,
        confidence: data.confidence,
        sources: data.sources
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error connecting to backend.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container">
      <aside className="sidebar">
        <div className="sidebar-top">
          <button className="icon-btn" onClick={() => setMessages([])}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
          </button>
        </div>
        <div className="sidebar-bottom">
          <div className="avatar">PS</div>
        </div>
      </aside>

      <section className="main-area">
        <header className="top-nav">
          <div className="spacer"></div>
          <button style={{ background: '#555', color: 'white', border: 'none', padding: '6px 14px', borderRadius: '20px', fontSize: '13px' }}>Get Pro</button>
        </header>

        <div className="content-wrapper" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="hero">
              <h1 className="pixel-font">What can I help with?</h1>
            </div>
          ) : (
            <div className="chat-history">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-header">
                    {msg.role === 'user' ? 'You' : 'Advanced RAG Agent'}
                  </div>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="input-container">
          <form className="search-pill" onSubmit={handleSubmit}>
            <button type="button" style={{ background: 'transparent', border: 'none', color: '#aaa' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.51a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
            </button>
            <input 
              className="search-input"
              placeholder="Ask anything..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" className="submit-btn" disabled={isLoading || !input.trim()}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>
            </button>
          </form>
          <p className="footer-text">
            AI can make mistakes. Please double-check responses.
          </p>
        </div>
      </section>
    </main>
  );
}
