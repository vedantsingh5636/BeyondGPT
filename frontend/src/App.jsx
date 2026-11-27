import { useEffect, useRef, useState } from 'react'
import './App.css'
import MarkdownRenderer from './components/MarkdownRenderer'
import {
  buildRequestHistory,
  normalizeApiBaseUrl,
  parseSseChunk,
  safeLoadChats,
  UPLOAD_TIMEOUT_MS,
  uploadDocumentRequest,
} from './utils/chatHelpers'

const API_BASE_URL = normalizeApiBaseUrl(
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
)

const createChatId = () => Date.now()

const features = [
  {
    number: '01',
    title: 'Multi-Agent Intelligence',
    description:
      'Specialized AI agents collaborate to understand and answer your questions.',
  },
  {
    number: '02',
    title: 'Hybrid Search',
    description:
      'Combines semantic understanding with keyword-based retrieval.',
  },
  {
    number: '03',
    title: 'Intelligent Reranking',
    description:
      'Finds and prioritizes the most relevant information from your knowledge base.',
  },
  {
    number: '04',
    title: 'Source Citations',
    description:
      'Get answers with references to the documents and pages behind them.',
  },
  {
    number: '05',
    title: 'Your Knowledge Base',
    description:
      'Connect your documents and build an assistant around your own information.',
  },
]

function App() {
  const [chatOpen, setChatOpen] = useState(
    () => sessionStorage.getItem('beyondgpt-chat-open') === 'true'
  )
  useEffect(() => {
    sessionStorage.setItem('beyondgpt-chat-open', String(chatOpen))
  }, [chatOpen])

  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [message, setMessage] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [authNotice, setAuthNotice] = useState('')

  const [chatHistory, setChatHistory] = useState(() =>
    safeLoadChats(localStorage.getItem('beyondgpt-chats'))
  )
  const [editingChatId, setEditingChatId] = useState(null)
  const [activeChatId, setActiveChatId] = useState(null)
  const [streamingChatId, setStreamingChatId] = useState(null)
  const [streamingText, setStreamingText] = useState('')

  const fileInputRef = useRef(null)
  const messagesEndRef = useRef(null)
  const uploadingRef = useRef(false)

  useEffect(() => {
    try {
      localStorage.setItem('beyondgpt-chats', JSON.stringify(chatHistory))
    } catch {
      // Ignore storage quota errors
    }
  }, [chatHistory])

  const activeChat =
    chatHistory.find((chat) => chat.id === activeChatId) || null
  const messages = activeChat ? activeChat.messages : []
  const isActiveChatStreaming =
    streamingChatId !== null && activeChatId === streamingChatId

  useEffect(() => {
    if (!chatOpen) return
    messagesEndRef.current?.scrollIntoView({
      behavior: isActiveChatStreaming && streamingText ? 'auto' : 'smooth',
      block: 'end',
    })
  }, [chatOpen, activeChatId, messages.length, isActiveChatStreaming, streamingText])

  const appendMessageToChat = (targetChatId, newMessage, titleIfNew) => {
    setChatHistory((prev) => {
      const existing = prev.find((chat) => chat.id === targetChatId)
      if (existing) {
        return prev.map((chat) =>
          chat.id === targetChatId
            ? { ...chat, messages: [...chat.messages, newMessage] }
            : chat
        )
      }
      return [
        {
          id: targetChatId,
          title: titleIfNew || 'New Chat',
          messages: [newMessage],
        },
        ...prev,
      ]
    })
  }

  const startNewChat = () => {
    setMessage('')
    setActiveChatId(null)
  }

  const renameChat = (chatId, newTitle) => {
    if (!newTitle.trim()) {
      setEditingChatId(null)
      return
    }

    setChatHistory((prev) =>
      prev.map((chat) =>
        chat.id === chatId ? { ...chat, title: newTitle.trim() } : chat
      )
    )
    setEditingChatId(null)
  }

  const deleteChat = (chatId) => {
    setChatHistory((prev) => prev.filter((chat) => chat.id !== chatId))

    if (activeChatId === chatId) {
      setMessage('')
      setActiveChatId(null)
    }
  }

  const uploadFile = async (fileToUpload = selectedFile) => {
    if (!fileToUpload || uploadingRef.current || uploading) return

    uploadingRef.current = true
    setUploading(true)
    setUploadStatus({
      type: 'info',
      text: `Uploading and indexing ${fileToUpload.name}...`,
    })

    try {
      const data = await uploadDocumentRequest({
        apiBaseUrl: API_BASE_URL,
        file: fileToUpload,
        timeoutMs: UPLOAD_TIMEOUT_MS,
      })

      setUploadStatus({
        type: 'success',
        text: data?.duplicate
          ? `${data.filename} is already indexed in your knowledge base.`
          : `Indexed ${data?.filename || fileToUpload.name} (${data?.chunks_created ?? 0} chunks ready).`,
      })
    } catch (error) {
      setUploadStatus({
        type: 'error',
        text: error?.message || 'Upload failed. Please try again.',
      })
    } finally {
      uploadingRef.current = false
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const requestFallbackChat = async (currentMessage, requestHistory) => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: currentMessage,
        history: requestHistory,
      }),
    })

    let data = null
    try {
      data = await response.json()
    } catch {
      data = null
    }

    if (!response.ok) {
      throw new Error(data?.detail || 'Failed to get response.')
    }

    return data?.response || 'No response received.'
  }

  const sendMessage = async () => {
    const currentMessage = message.trim()
    if (!currentMessage || streamingChatId !== null || uploading) return

    const targetChatId = activeChatId || createChatId()
    const requestHistory = buildRequestHistory(messages)

    if (!activeChatId) {
      setActiveChatId(targetChatId)
    }

    appendMessageToChat(
      targetChatId,
      { role: 'user', content: currentMessage },
      currentMessage
    )

    setMessage('')
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    setStreamingChatId(targetChatId)
    setStreamingText('')

    let accumulatedText = ''
    let streamError = null
    let receivedDone = false

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: currentMessage,
          history: requestHistory,
        }),
      })

      const contentType = response.headers.get('content-type') || ''
      if (
        !response.ok ||
        !response.body ||
        !contentType.includes('text/event-stream')
      ) {
        const fallbackAnswer = await requestFallbackChat(
          currentMessage,
          requestHistory
        )
        appendMessageToChat(targetChatId, {
          role: 'assistant',
          content: fallbackAnswer,
        })
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let sseBuffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        sseBuffer = parseSseChunk(
          sseBuffer,
          decoder.decode(value, { stream: true }),
          (event) => {
            if (event.type === 'token' && typeof event.content === 'string') {
              accumulatedText += event.content
              setStreamingText(accumulatedText)
            } else if (event.type === 'done') {
              receivedDone = true
              if (typeof event.response === 'string' && event.response) {
                accumulatedText = event.response
              }
            } else if (event.type === 'error') {
              streamError = event.detail || 'Streaming failed.'
            }
          }
        )
      }

      if (sseBuffer.trim()) {
        parseSseChunk(sseBuffer + '\n\n', '', (event) => {
          if (event.type === 'done' && typeof event.response === 'string') {
            receivedDone = true
            accumulatedText = event.response
          } else if (event.type === 'error') {
            streamError = event.detail || 'Streaming failed.'
          }
        })
      }

      if (streamError && !accumulatedText) {
        const fallbackAnswer = await requestFallbackChat(
          currentMessage,
          requestHistory
        )
        appendMessageToChat(targetChatId, {
          role: 'assistant',
          content: fallbackAnswer,
        })
        return
      }

      if (streamError) {
        appendMessageToChat(targetChatId, {
          role: 'assistant',
          content: `${accumulatedText}\n\n*Error: ${streamError}*`,
          isError: true,
        })
        return
      }

      if (accumulatedText || receivedDone) {
        appendMessageToChat(targetChatId, {
          role: 'assistant',
          content: accumulatedText || 'No response received.',
        })
      } else {
        const fallbackAnswer = await requestFallbackChat(
          currentMessage,
          requestHistory
        )
        appendMessageToChat(targetChatId, {
          role: 'assistant',
          content: fallbackAnswer,
        })
      }
    } catch (error) {
      if (!accumulatedText) {
        try {
          const fallbackAnswer = await requestFallbackChat(
            currentMessage,
            requestHistory
          )
          appendMessageToChat(targetChatId, {
            role: 'assistant',
            content: fallbackAnswer,
          })
          return
        } catch (fallbackError) {
          appendMessageToChat(targetChatId, {
            role: 'assistant',
            content:
              fallbackError?.message ||
              error?.message ||
              'Something went wrong. Please try again.',
            isError: true,
          })
          return
        }
      }

      appendMessageToChat(targetChatId, {
        role: 'assistant',
        content: `${accumulatedText}\n\n*Connection interrupted: ${error?.message || 'stream ended'}*`,
        isError: true,
      })
    } finally {
      setStreamingChatId(null)
      setStreamingText('')
    }
  }

  const [activeFeature, setActiveFeature] = useState(0)
  useEffect(() => {
    const handleScroll = () => {
      const section = document.getElementById('features')
      if (!section) return

      const sectionTop = section.getBoundingClientRect().top
      const sectionHeight = section.offsetHeight
      const scrollableDistance = sectionHeight - window.innerHeight

      if (sectionTop > 0) {
        setActiveFeature(0)
        return
      }

      if (sectionTop + scrollableDistance < 0) {
        setActiveFeature(4)
        return
      }

      const progress = Math.min(
        1,
        Math.max(0, -sectionTop / scrollableDistance)
      )

      setActiveFeature(Math.min(4, Math.floor(progress * 5)))
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()

    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const openChat = (event) => {
    if (event) event.preventDefault()
    setAuthNotice('')
    setChatOpen(true)
  }

  const showSignInNotice = (event) => {
    if (event) event.preventDefault()
    setAuthNotice('Sign in is not enabled yet — you can upload documents and chat immediately.')
  }

  if (chatOpen) {
    return (
      <div className="chat-interface">
        {sidebarOpen && (
          <aside className="chat-sidebar">
            <div className="sidebar-brand">
              <span className="brand-icon">B</span>
              BeyondGPT
            </div>

            <button
              type="button"
              className="new-chat-btn"
              onClick={startNewChat}
            >
              + New Chat
            </button>

            <div className="sidebar-history">
              <p>Recent Chats</p>

              {chatHistory.map((chat) => (
                <div className="history-item-wrapper" key={chat.id}>
                  {editingChatId === chat.id ? (
                    <input
                      className="history-rename-input"
                      defaultValue={chat.title}
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          renameChat(chat.id, e.target.value)
                        }
                        if (e.key === 'Escape') {
                          setEditingChatId(null)
                        }
                      }}
                      onBlur={(e) => renameChat(chat.id, e.target.value)}
                    />
                  ) : (
                    <>
                      <button
                        type="button"
                        className={`history-item ${activeChatId === chat.id ? 'active' : ''}`}
                        onClick={() => {
                          setActiveChatId(chat.id)
                          setMessage('')
                        }}
                      >
                        {chat.title.length > 25
                          ? chat.title.slice(0, 25) + '...'
                          : chat.title}
                      </button>

                      <button
                        type="button"
                        className="rename-chat-btn"
                        onClick={() => setEditingChatId(chat.id)}
                      >
                        Rename
                      </button>
                      <button
                        type="button"
                        className="delete-chat-btn"
                        onClick={() => deleteChat(chat.id)}
                      >
                        Delete
                      </button>
                    </>
                  )}
                </div>
              ))}
            </div>
          </aside>
        )}

        <div className="chat-main">
          <header className="chat-topbar">
            <button
              type="button"
              className="sidebar-toggle"
              aria-label="Toggle sidebar"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              ☰
            </button>

            <div className="chat-header-actions">
              <button
                type="button"
                className="chat-signin-btn"
                onClick={showSignInNotice}
              >
                Sign in
              </button>

              <button type="button" onClick={() => setChatOpen(false)}>
                Back to Home
              </button>
            </div>
          </header>

          {authNotice && (
            <div className="auth-notice" role="status">
              <span>{authNotice}</span>
              <button type="button" onClick={() => setAuthNotice('')}>
                ×
              </button>
            </div>
          )}

          {messages.length === 0 && !isActiveChatStreaming && (
            <div className="chat-welcome">
              <h1>Welcome to BeyondGPT.</h1>
              <p>Your knowledge. Your AI assistant.</p>
            </div>
          )}

          {(messages.length > 0 || isActiveChatStreaming) && (
            <div className="chat-messages">
              <div className="chat-messages-inner">
                {messages.map((msg, index) => (
                  <div
                    key={`${activeChatId}-${index}`}
                    className={`chat-message ${msg.role} ${msg.isError ? 'error' : ''}`}
                  >
                    {msg.role === 'assistant' && (
                      <div className="chat-message-label">BEYONDGPT</div>
                    )}

                    <div className="chat-message-content">
                      {msg.role === 'assistant' ? (
                        <MarkdownRenderer content={msg.content} />
                      ) : (
                        msg.content
                      )}
                    </div>
                  </div>
                ))}

                {isActiveChatStreaming && streamingText && (
                  <div className="chat-message assistant streaming">
                    <div className="chat-message-label">BEYONDGPT</div>
                    <div className="chat-message-content">
                      <MarkdownRenderer content={streamingText} showSources={false} />
                    </div>
                  </div>
                )}

                {isActiveChatStreaming && !streamingText && (
                  <div className="chat-answer">BeyondGPT is thinking...</div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>
          )}

          {uploadStatus && (
            <div
              className={`upload-status ${uploadStatus.type}`}
              role="status"
            >
              <span>{uploadStatus.text}</span>
              {!uploading && (
                <button
                  type="button"
                  className="upload-status-close"
                  onClick={() => setUploadStatus(null)}
                >
                  ×
                </button>
              )}
            </div>
          )}

          {selectedFile && (
            <div className="selected-file-preview">
              <div className="file-preview-info">
                <span>📄</span>
                <div>
                  <p>{selectedFile.name}</p>
                  <small>
                    {(selectedFile.size / 1024).toFixed(1)} KB
                    {uploading ? ' · Indexing...' : ' · Attached'}
                  </small>
                </div>
              </div>

              <button
                type="button"
                disabled={uploading}
                onClick={() => {
                  setSelectedFile(null)
                  if (fileInputRef.current) {
                    fileInputRef.current.value = ''
                  }
                }}
              >
                ×
              </button>
            </div>
          )}

          <div className="chat-input-area">
            <button
              type="button"
              className="upload-btn"
              disabled={uploading}
              title="Upload PDF or DOCX"
              onClick={() => fileInputRef.current?.click()}
            >
              {uploading ? '…' : '+'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              id="file-upload"
              accept=".pdf,.docx"
              disabled={uploading}
              style={{ display: 'none' }}
              onChange={(e) => {
                if (uploadingRef.current || uploading) return
                const file = e.target.files?.[0]
                if (file) {
                  setSelectedFile(file)
                  uploadFile(file)
                }
              }}
            />
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder={
                uploading
                  ? 'Indexing uploaded document...'
                  : 'Ask anything about your documents...'
              }
            />
            <button
              type="button"
              onClick={sendMessage}
              disabled={
                streamingChatId !== null || uploading || !message.trim()
              }
            >
              {streamingChatId !== null ? '...' : '↑'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="navbar">
        <a href="#home" className="brand">
          <span className="brand-icon">B</span>
          BeyondGPT<span className="brand-dot">.</span>
        </a>

        <nav className="nav-links">
          <a href="#features">Features</a>
          <a href="#comparison">Why BeyondGPT</a>
          <a href="#how-it-works">How it works</a>
        </nav>

        <div className="nav-actions">
          <a href="#signin" className="signin" onClick={showSignInNotice}>
            Sign in
          </a>
          <a href="#chat" className="nav-cta" onClick={openChat}>
            Get started ↗
          </a>
        </div>
      </header>

      {authNotice && (
        <div className="auth-notice landing-notice" role="status">
          <span>{authNotice}</span>
          <button type="button" onClick={() => setAuthNotice('')}>
            ×
          </button>
        </div>
      )}

      <main>
        <section className="hero" id="home">
          <div className="hero-content">
            <span className="eyebrow">YOUR KNOWLEDGE. SUPERCHARGED.</span>

            <h1>
              Don't just chat
              <br />
              with AI. Go
              <br />
              <span>Beyond.</span>
            </h1>

            <p>
              Meet a smarter way to interact with your knowledge. Upload your
              documents, ask questions and get intelligent answers backed by
              real sources.
            </p>

            <div className="hero-actions">
              <a href="#chat" onClick={openChat} className="primary-btn">
                Try for free <span>↗</span>
              </a>

              <a href="#features" className="secondary-btn">
                ▶ Take a demo
              </a>
            </div>

            <div className="hero-note">
              Built for knowledge. Designed for intelligence.
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-glow"></div>

            <div className="floating-label label-top">
              <span className="status-dot"></span>
              AI AGENTS ACTIVE
            </div>

            <div className="chat-card">
              <div className="chat-header">
                <div className="mini-logo">B</div>
                <div>
                  <strong>BeyondGPT</strong>
                  <small>Knowledge Assistant</small>
                </div>
                <span className="online-dot"></span>
              </div>

              <div className="chat-body">
                <div className="user-message">
                  What technologies are used in my project?
                </div>

                <div className="ai-message">
                  <span className="ai-tag">BEYONDGPT AI</span>
                  <p>
                    Based on your uploaded documents, I found these
                    technologies:
                  </p>

                  <div className="tech-tags">
                    <span>Java</span>
                    <span>MySQL</span>
                    <span>JDBC</span>
                    <span>Java Swing</span>
                  </div>

                  <div className="citation">
                    <span>↗</span>
                    Source: Project Report · Page 3
                  </div>
                </div>
              </div>

              <div className="chat-input">
                <span>Ask anything about your data...</span>
                <span className="send-btn">↑</span>
              </div>
            </div>

            <div className="floating-label label-bottom">
              <span>✳</span> SOURCE VERIFIED
            </div>
          </div>
        </section>

        <section className="capabilities">
          <p>BUILT TO GO BEYOND ORDINARY CHATBOTS</p>
          <div className="capability-list">
            <span>Multi-Agent AI</span>
            <span>Hybrid Retrieval</span>
            <span>Smart Reranking</span>
            <span>Source Citations</span>
          </div>
        </section>

        <section className="intelligence-section section" id="features">
          <div className="intelligence-sticky">
            <div className="intelligence-heading">
              <span className="eyebrow">WHAT MAKES US DIFFERENT</span>

              <h2>
                Intelligence is just
                <br />
                <span>the beginning.</span>
              </h2>

              <p>
                More than a chatbot. A complete intelligence layer for your
                knowledge.
              </p>
            </div>

            <div className="feature-showcase">
              <div className="feature-list">
                {features.map((feature, index) => (
                  <div
                    className={`showcase-item ${index === activeFeature ? 'active' : ''}`}
                    key={feature.number}
                    onClick={() => setActiveFeature(index)}
                  >
                    <div className="showcase-item-heading">
                      <span>{feature.number}</span>
                      <h3>{feature.title}</h3>
                      <span className="feature-arrow">↗</span>
                    </div>

                    {index === activeFeature && <p>{feature.description}</p>}
                  </div>
                ))}
              </div>

              <div className="feature-visual">
                <div className={`product-preview preview-${activeFeature}`}>
                  <div className="preview-topbar">
                    <span className="preview-dot"></span>
                    <span className="preview-dot"></span>
                    <span className="preview-dot"></span>
                    <span className="preview-brand">BEYONDGPT</span>
                  </div>

                  <div className="preview-content">
                    <span className="preview-label">
                      FEATURE 0{activeFeature + 1}
                    </span>

                    <h3>{features[activeFeature].title}</h3>

                    <p>{features[activeFeature].description}</p>

                    <div className="preview-interface">
                      <div className="preview-line"></div>
                      <div className="preview-line short"></div>
                      <div className="preview-panel"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="comparison section" id="comparison">
          <div className="section-heading">
            <span className="eyebrow">A DIFFERENT APPROACH</span>
            <h2>
              Ordinary AI.
              <br />
              <span>Extraordinary intelligence.</span>
            </h2>
          </div>

          <div className="comparison-grid">
            <div className="comparison-card">
              <span className="comparison-label">TRADITIONAL CHATBOT</span>
              <h3>Conventional AI</h3>
              <p>General-purpose responses</p>
              <p>No built-in access to your private documents</p>
              <p>Basic retrieval capabilities</p>
              <p>Limited source traceability</p>
            </div>

            <div className="comparison-card featured-comparison">
              <span className="comparison-label">THE BEYONDGPT APPROACH</span>
              <h3>
                BeyondGPT<span>.</span>
              </h3>
              <p>Answers grounded in your knowledge base</p>
              <p>Document-based information retrieval</p>
              <p>Hybrid search and intelligent reranking</p>
              <p>Source and page-level references</p>
            </div>
          </div>
        </section>

        <section className="how-it-works section" id="how-it-works">
          <div className="section-heading">
            <span className="eyebrow">SIMPLE BY DESIGN</span>
            <h2>
              From your data
              <br />
              <span>to intelligent answers.</span>
            </h2>
          </div>

          <div className="steps">
            <div>
              <span>01</span>
              <h3>Upload</h3>
              <p>Add your knowledge.</p>
            </div>
            <div>
              <span>02</span>
              <h3>Understand</h3>
              <p>AI processes and indexes your content.</p>
            </div>
            <div>
              <span>03</span>
              <h3>Retrieve</h3>
              <p>Relevant information is searched and ranked.</p>
            </div>
            <div>
              <span>04</span>
              <h3>Discover</h3>
              <p>Get contextual answers with citations.</p>
            </div>
          </div>
        </section>

        <section className="final-cta section" id="get-started">
          <span className="eyebrow">THE NEXT GENERATION OF AI</span>
          <h2>
            Ready to go
            <br />
            <span>Beyond?</span>
          </h2>
          <p>Your knowledge deserves more than an ordinary chatbot.</p>
          <a href="#chat" onClick={openChat} className="primary-btn">
            Try BeyondGPT <span>↗</span>
          </a>
        </section>
      </main>

      <footer className="footer">
        <a href="#home" className="brand">
          <span className="brand-icon">B</span>
          BeyondGPT<span className="brand-dot">.</span>
        </a>
        <p>Intelligence, without boundaries.</p>
        <span>© 2026 BeyondGPT</span>
      </footer>
    </div>
  )
}

export default App
