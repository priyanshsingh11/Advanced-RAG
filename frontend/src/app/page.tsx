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
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus(`Indexing ${file.name}...`);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/v1/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setUploadStatus(`Indexed: ${file.name}`);
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Document indexed: "${file.name}". I'm ready to answer questions about it.`
        }]);
      } else {
        setUploadStatus(`Failed: ${data.detail || 'Error'}`);
      }
    } catch (error) {
      setUploadStatus('Connection error');
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setTimeout(() => setUploadStatus(null), 5000);
    }
  };

  return (
    <main className="container">
      <aside className="sidebar">
        <div className="sidebar-top">
          <button className="new-chat-btn" onClick={() => setMessages([])}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14" /><path d="M12 5v14" /></svg>
            New chat
          </button>
        </div>
      </aside>

      <section className="main-area">
        <header className="top-nav">
          <div className="spacer"></div>
        </header>

        <div className="content-wrapper" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="hero">
              <h1>What can I help with?</h1>
            </div>
          ) : (
            <div className="chat-history">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-bubble">{msg.content}</div>
                  {msg.role === 'assistant' && (
                    <div className="message-actions">
                      <svg className="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M8 10h4M8 14h4M16 10h.01M16 14h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" /></svg>
                      <svg className="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M7 21h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2Z" /><path d="M12 11h.01M12 15h.01M12 7h.01" /></svg>
                      <svg className="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /></svg>
                      <svg className="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M3 21v-5h5" /></svg>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="input-container">
          {uploadStatus && (
            <div className="upload-status">
              {uploadStatus}
            </div>
          )}
          <form className="search-pill" onSubmit={handleSubmit}>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              accept=".pdf,.txt"
            />
            
            <button
              type="button"
              className="attach-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              title="Upload document"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.51a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg>
            </button>

            <div className="textarea-wrapper">
              <textarea
                className={`search-input ${input ? 'has-content' : ''}`}
                placeholder="Ask anything"
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
              />
            </div>

            <button type="submit" className="submit-btn" disabled={isLoading || !input.trim()}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="m5 12 7-7 7 7" /><path d="M12 19V5" /></svg>
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
