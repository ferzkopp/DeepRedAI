import { useState, useEffect, useCallback } from 'react'
import './App.css'

// API base URL - adjust if MCP server is on different host
const API_BASE = window.location.hostname === 'localhost' 
  ? 'http://localhost:7000' 
  : `http://${window.location.hostname}:7000`

function App() {
  const [query, setQuery] = useState('')
  const [searchMode, setSearchMode] = useState('hybrid')
  const [results, setResults] = useState([])
  const [selectedArticle, setSelectedArticle] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)

  // Check server health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => setHealth({ status: 'error', message: err.message }))
  }, [])

  const handleSearch = useCallback(async (e) => {
    e?.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setSelectedArticle(null)

    try {
      const response = await fetch(`${API_BASE}/mcp/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          mode: searchMode,
          limit: 20
        })
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      const data = await response.json()
      setResults(data.results || [])
    } catch (err) {
      setError(err.message)
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [query, searchMode])

  const loadArticle = async (articleId) => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/mcp/article/${articleId}`)
      if (!response.ok) {
        throw new Error(`Failed to load article: ${response.statusText}`)
      }
      const data = await response.json()
      setSelectedArticle(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const loadArticleByTitle = async (title) => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/mcp/article?title=${encodeURIComponent(title)}`)
      if (!response.ok) {
        throw new Error(`Failed to load article: ${response.statusText}`)
      }
      const data = await response.json()
      setSelectedArticle(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>📚 Wikipedia Local Search</h1>
        <div className="health-status">
          {health?.status === 'healthy' ? (
            <span className="status-ok">● Server Online</span>
          ) : health?.status === 'error' ? (
            <span className="status-error">● Server Offline</span>
          ) : (
            <span className="status-checking">● Checking...</span>
          )}
        </div>
      </header>

      <main className="main">
        {/* Search Panel */}
        <section className="search-panel">
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search Wikipedia..."
              className="search-input"
              autoFocus
            />
            <div className="search-options">
              <label className="mode-label">
                <input
                  type="radio"
                  name="mode"
                  value="keyword"
                  checked={searchMode === 'keyword'}
                  onChange={(e) => setSearchMode(e.target.value)}
                />
                Keyword (BM25)
              </label>
              <label className="mode-label">
                <input
                  type="radio"
                  name="mode"
                  value="semantic"
                  checked={searchMode === 'semantic'}
                  onChange={(e) => setSearchMode(e.target.value)}
                />
                Semantic
              </label>
              <label className="mode-label">
                <input
                  type="radio"
                  name="mode"
                  value="hybrid"
                  checked={searchMode === 'hybrid'}
                  onChange={(e) => setSearchMode(e.target.value)}
                />
                Hybrid
              </label>
            </div>
            <button type="submit" className="search-button" disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>

          {error && <div className="error-message">⚠️ {error}</div>}

          {/* Search Results */}
          <div className="results-list">
            {results.length > 0 && (
              <div className="results-header">
                Found {results.length} results
              </div>
            )}
            {results.map((result, index) => (
              <div
                key={result.id || index}
                className="result-item"
                onClick={() => result.id ? loadArticle(result.id) : loadArticleByTitle(result.title)}
              >
                <h3 className="result-title">{result.title}</h3>
                {result.section_title && (
                  <span className="result-section">§ {result.section_title}</span>
                )}
                <p className="result-excerpt">
                  {result.excerpt || result.text?.substring(0, 200) + '...'}
                </p>
                {result.score !== undefined && (
                  <span className="result-score">
                    Score: {result.score.toFixed(3)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Article View */}
        {selectedArticle && (
          <section className="article-panel">
            <button 
              className="back-button"
              onClick={() => setSelectedArticle(null)}
            >
              ← Back to Results
            </button>
            
            <article className="article-content">
              <h1 className="article-title">{selectedArticle.title}</h1>
              
              {selectedArticle.url && (
                <a 
                  href={selectedArticle.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="wikipedia-link"
                >
                  View on Wikipedia ↗
                </a>
              )}

              {/* Table of Contents */}
              {selectedArticle.sections && selectedArticle.sections.length > 0 && (
                <nav className="toc">
                  <h4>Contents</h4>
                  <ul>
                    {selectedArticle.sections.map((section, idx) => (
                      <li key={idx}>
                        <a href={`#section-${idx}`}>{section.title || 'Introduction'}</a>
                      </li>
                    ))}
                  </ul>
                </nav>
              )}

              {/* Article Text */}
              {selectedArticle.sections ? (
                selectedArticle.sections.map((section, idx) => (
                  <div key={idx} id={`section-${idx}`} className="article-section">
                    {section.title && <h2>{section.title}</h2>}
                    <div className="section-text">
                      {section.text.split('\n\n').map((para, pIdx) => (
                        <p key={pIdx}>{para}</p>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <div className="article-text">
                  {selectedArticle.content?.split('\n\n').map((para, idx) => (
                    <p key={idx}>{para}</p>
                  ))}
                </div>
              )}
            </article>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Local Wikipedia MCP Server • Data from Wikimedia Foundation</p>
      </footer>
    </div>
  )
}

export default App
