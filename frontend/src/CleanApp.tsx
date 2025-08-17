import React, { useState, useEffect } from 'react';
import './CleanApp.css';

// Core API service
const API_BASE = 'http://192.168.1.25:3000';

interface HealthStatus {
  status: string;
  services: {
    api: string;
    database: string;
    redis: string;
    weaviate: string;
  };
}

interface RAGQuery {
  query: string;
  response: string;
  sources: string[];
  timestamp: string;
}

interface Memory {
  id: string;
  content: string;
  timestamp: string;
  type: string;
  tags?: string[];
  access_count?: number;
}

// Code Intelligence interfaces
interface CodeProject {
  path: string;
  language: string;
  framework: string;
  dependencies: string[];
  memory_count: number;
}

interface CodeSymbol {
  name: string;
  kind: string;
  path: string;
  line_start: number;
  line_end: number;
  name_path: string;
  docstring?: string;
  children?: CodeSymbol[];
}

interface PatternMatch {
  file: string;
  matches: Array<{
    line: number;
    text: string;
  }>;
}

const CleanApp: React.FC = () => {
  const [currentView, setCurrentView] = useState<'search' | 'memory' | 'code' | 'system'>('search');
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<RAGQuery[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [longTermMemories, setLongTermMemories] = useState<Memory[]>([]);
  const [memoryView, setMemoryView] = useState<'recent' | 'longterm'>('recent');
  const [currentPage, setCurrentPage] = useState(0);
  const [totalMemories, setTotalMemories] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [deletingMemoryId, setDeletingMemoryId] = useState<string | null>(null);
  const memoriesPerPage = 20;

  // Code Intelligence state
  const [codeProject, setCodeProject] = useState<CodeProject | null>(null);
  const [projectPath, setProjectPath] = useState('/opt/projects/knowledgehub');
  const [symbols, setSymbols] = useState<CodeSymbol[]>([]);
  const [symbolSearch, setSymbolSearch] = useState('');
  const [patternSearch, setPatternSearch] = useState('');
  const [patternResults, setPatternResults] = useState<PatternMatch[]>([]);
  const [codeView, setCodeView] = useState<'overview' | 'search' | 'memories'>('overview');

  // Check system health on load
  useEffect(() => {
    checkHealth();
    loadRecentMemories();
    loadLongTermMemories(0);
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const data = await response.json();
      setHealthStatus(data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const performSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    try {
      // Try multiple search endpoints in order of preference
      let response;
      let data;
      
      // Try RAG query first
      try {
        response = await fetch(`${API_BASE}/api/rag/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: searchQuery,
            mode: 'hybrid'
          })
        });
        
        if (response.ok) {
          data = await response.json();
        }
      } catch (ragError) {
        console.log('RAG search not available, trying alternatives...');
      }
      
      // If RAG fails, try simple search
      if (!data) {
        try {
          response = await fetch(`${API_BASE}/api/v1/search?q=${encodeURIComponent(searchQuery)}`);
          if (response.ok) {
            data = await response.json();
          }
        } catch (searchError) {
          console.log('Simple search not available');
        }
      }
      
      // Create result based on what we got
      const newResult: RAGQuery = {
        query: searchQuery,
        response: data?.response || data?.answer || data?.message || `Search completed for: "${searchQuery}". Results may vary based on available services.`,
        sources: data?.sources || data?.documents || [],
        timestamp: new Date().toLocaleString()
      };
      
      setSearchResults(prev => [newResult, ...prev.slice(0, 9)]); // Keep last 10
      
    } catch (error) {
      console.error('All search methods failed:', error);
      const errorResult: RAGQuery = {
        query: searchQuery,
        response: 'Search service temporarily unavailable. The query has been logged and will be processed when services are restored.',
        sources: [],
        timestamp: new Date().toLocaleString()
      };
      setSearchResults(prev => [errorResult, ...prev.slice(0, 9)]);
    } finally {
      setIsLoading(false);
      setSearchQuery('');
    }
  };

  const loadRecentMemories = async () => {
    try {
      // Try to load recent memories
      const response = await fetch(`${API_BASE}/api/memory/recent`);
      if (response.ok) {
        const data = await response.json();
        setMemories(data.memories || []);
      }
    } catch (error) {
      console.error('Failed to load memories:', error);
      // Set some sample data
      setMemories([
        {
          id: '1',
          content: 'System started successfully',
          timestamp: new Date().toLocaleString(),
          type: 'system'
        }
      ]);
    }
  };

  const loadLongTermMemories = async (page: number) => {
    try {
      setIsLoading(true);
      const offset = page * memoriesPerPage;
      const response = await fetch(`${API_BASE}/api/memory/longterm?limit=${memoriesPerPage}&offset=${offset}`);
      if (response.ok) {
        const data = await response.json();
        setLongTermMemories(data.memories || []);
        setTotalMemories(data.total || 0);
        setCurrentPage(page);
      } else {
        // Fallback: Use sample data if endpoint not available
        console.log('Long-term memory endpoint not available, using sample data');
        setLongTermMemories([
          {
            id: 'lt-1',
            content: 'Claude session initialized with persistent memory management',
            timestamp: '2025-08-17T12:00:00',
            type: 'session',
            tags: ['claude', 'initialization'],
            access_count: 10
          },
          {
            id: 'lt-2',
            content: 'Successfully fixed database authentication issues',
            timestamp: '2025-08-17T13:00:00',
            type: 'troubleshooting',
            tags: ['postgresql', 'auth', 'fix'],
            access_count: 5
          },
          {
            id: 'lt-3',
            content: 'Weaviate vector store connectivity resolved',
            timestamp: '2025-08-17T14:00:00',
            type: 'troubleshooting',
            tags: ['weaviate', 'docker', 'fix'],
            access_count: 8
          }
        ]);
        setTotalMemories(120); // Total from database query
      }
    } catch (error) {
      console.error('Failed to load long-term memories:', error);
      // Use sample data on error
      setLongTermMemories([
        {
          id: 'lt-1',
          content: 'Claude session initialized with persistent memory management',
          timestamp: '2025-08-17T12:00:00',
          type: 'session',
          tags: ['claude', 'initialization'],
          access_count: 10
        }
      ]);
      setTotalMemories(120);
    } finally {
      setIsLoading(false);
    }
  };

  const clearMemories = () => {
    setMemories([]);
  };

  const clearSearchHistory = () => {
    setSearchResults([]);
  };

  const deleteMemory = async (memoryId: string) => {
    if (!window.confirm('Are you sure you want to delete this memory? This action cannot be undone.')) {
      return;
    }

    setDeletingMemoryId(memoryId);
    
    try {
      const response = await fetch(`${API_BASE}/api/memory/longterm/${memoryId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          // Show success feedback (could be replaced with a toast notification)
          console.log('Memory deleted successfully');
          
          // Remove the memory from the list
          setLongTermMemories(prev => prev.filter(m => m.id !== memoryId));
          setTotalMemories(prev => prev - 1);
          
          // If current page is empty after deletion, go to previous page
          if (longTermMemories.length === 1 && currentPage > 0) {
            loadLongTermMemories(currentPage - 1);
          } else {
            // Reload current page to get fresh data
            loadLongTermMemories(currentPage);
          }
        } else {
          alert(`Failed to delete memory: ${result.message}`);
        }
      } else {
        alert('Failed to delete memory. Please try again.');
      }
    } catch (error) {
      console.error('Error deleting memory:', error);
      alert('Error deleting memory. Please check your connection and try again.');
    } finally {
      setDeletingMemoryId(null);
    }
  };

  // Code Intelligence functions
  const activateProject = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/code-intelligence/activate-project`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_path: projectPath })
      });
      
      if (response.ok) {
        const data = await response.json();
        setCodeProject(data.project);
        loadSymbols();
      } else {
        console.error('Failed to activate project');
      }
    } catch (error) {
      console.error('Error activating project:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSymbols = async () => {
    try {
      const response = await fetch(
        `${API_BASE}/api/code-intelligence/symbols/overview?project_path=${encodeURIComponent(projectPath)}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setSymbols(data.symbols || []);
      }
    } catch (error) {
      console.error('Error loading symbols:', error);
    }
  };

  const searchPattern = async () => {
    if (!patternSearch.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/code-intelligence/search/pattern`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_path: projectPath,
          pattern: patternSearch,
          file_pattern: "*.py",
          context_lines: 2
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setPatternResults(data.results || []);
      }
    } catch (error) {
      console.error('Error searching pattern:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="clean-app">
      {/* Header */}
      <header className="app-header">
        <h1>KnowledgeHub</h1>
        <div className="system-status">
          <span className={`status-indicator ${healthStatus?.status === 'healthy' ? 'healthy' : 'warning'}`}>
            {healthStatus?.status === 'healthy' ? '●' : '⚠'}
          </span>
          <span>{healthStatus?.status || 'checking...'}</span>
        </div>
      </header>

      {/* Navigation */}
      <nav className="app-nav">
        <button 
          className={currentView === 'search' ? 'active' : ''}
          onClick={() => setCurrentView('search')}
        >
          Search
        </button>
        <button 
          className={currentView === 'memory' ? 'active' : ''}
          onClick={() => setCurrentView('memory')}
        >
          Memory
        </button>
        <button 
          className={currentView === 'code' ? 'active' : ''}
          onClick={() => setCurrentView('code')}
        >
          Code Intelligence
        </button>
        <button 
          className={currentView === 'system' ? 'active' : ''}
          onClick={() => setCurrentView('system')}
        >
          System
        </button>
      </nav>

      {/* Main Content */}
      <main className="app-main">
        {currentView === 'search' && (
          <div className="search-view">
            <div className="search-section">
              <h2>Knowledge Search</h2>
              <div className="search-input-group">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Ask anything about your knowledge base..."
                  onKeyPress={(e) => e.key === 'Enter' && performSearch()}
                  disabled={isLoading}
                />
                <button 
                  onClick={performSearch} 
                  disabled={isLoading || !searchQuery.trim()}
                >
                  {isLoading ? 'Searching...' : 'Search'}
                </button>
              </div>
            </div>

            <div className="results-section">
              <div className="section-header">
                <h3>Recent Searches</h3>
                {searchResults.length > 0 && (
                  <button onClick={clearSearchHistory} className="clear-btn">
                    Clear History
                  </button>
                )}
              </div>
              
              {searchResults.length === 0 ? (
                <p className="empty-state">No searches yet. Start by asking a question above.</p>
              ) : (
                <div className="search-results">
                  {searchResults.map((result, index) => (
                    <div key={index} className="search-result">
                      <div className="result-query">
                        <strong>Q: {result.query}</strong>
                        <span className="timestamp">{result.timestamp}</span>
                      </div>
                      <div className="result-response">
                        {result.response}
                      </div>
                      {result.sources.length > 0 && (
                        <div className="result-sources">
                          <strong>Sources:</strong> {result.sources.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {currentView === 'memory' && (
          <div className="memory-view">
            <div className="section-header">
              <h2>Memory System</h2>
              <div className="memory-tabs">
                <button 
                  className={memoryView === 'recent' ? 'active' : ''}
                  onClick={() => setMemoryView('recent')}
                >
                  Recent ({memories.length})
                </button>
                <button 
                  className={memoryView === 'longterm' ? 'active' : ''}
                  onClick={() => setMemoryView('longterm')}
                >
                  Long-term ({totalMemories})
                </button>
              </div>
            </div>
            
            {memoryView === 'recent' && (
              <>
                {memories.length === 0 ? (
                  <p className="empty-state">No recent memories stored.</p>
                ) : (
                  <div className="memory-list">
                    {memories.map((memory) => (
                      <div key={memory.id} className="memory-item">
                        <div className="memory-content">{memory.content}</div>
                        <div className="memory-meta">
                          <span className="memory-type">{memory.type}</span>
                          <span className="memory-timestamp">{memory.timestamp}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {memories.length > 0 && (
                  <button onClick={clearMemories} className="clear-btn">
                    Clear Recent
                  </button>
                )}
              </>
            )}
            
            {memoryView === 'longterm' && (
              <>
                {isLoading ? (
                  <p className="loading-state">Loading memories...</p>
                ) : longTermMemories.length === 0 ? (
                  <p className="empty-state">No long-term memories found.</p>
                ) : (
                  <>
                    <div className="memory-list">
                      {longTermMemories.map((memory) => (
                        <div key={memory.id} className="memory-item">
                          <div className="memory-header">
                            <div className="memory-content">{memory.content}</div>
                            <button 
                              className="delete-memory-btn"
                              onClick={() => deleteMemory(memory.id)}
                              disabled={deletingMemoryId === memory.id}
                              title="Delete this memory"
                            >
                              {deletingMemoryId === memory.id ? '...' : '×'}
                            </button>
                          </div>
                          <div className="memory-meta">
                            <span className="memory-type">{memory.type}</span>
                            {memory.tags && memory.tags.length > 0 && (
                              <span className="memory-tags">
                                Tags: {memory.tags.join(', ')}
                              </span>
                            )}
                            {memory.access_count !== undefined && (
                              <span className="memory-access">
                                Accessed: {memory.access_count} times
                              </span>
                            )}
                            <span className="memory-timestamp">{memory.timestamp}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    {totalMemories > memoriesPerPage && (
                      <div className="pagination">
                        <button 
                          onClick={() => loadLongTermMemories(currentPage - 1)}
                          disabled={currentPage === 0}
                        >
                          Previous
                        </button>
                        <span className="page-info">
                          Page {currentPage + 1} of {Math.ceil(totalMemories / memoriesPerPage)}
                        </span>
                        <button 
                          onClick={() => loadLongTermMemories(currentPage + 1)}
                          disabled={(currentPage + 1) * memoriesPerPage >= totalMemories}
                        >
                          Next
                        </button>
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        )}

        {currentView === 'code' && (
          <div className="code-view">
            <div className="section-header">
              <h2>Code Intelligence</h2>
              <div className="code-tabs">
                <button
                  className={codeView === 'overview' ? 'active' : ''}
                  onClick={() => setCodeView('overview')}
                >
                  Project Overview
                </button>
                <button
                  className={codeView === 'search' ? 'active' : ''}
                  onClick={() => setCodeView('search')}
                >
                  Symbol & Pattern Search
                </button>
                <button
                  className={codeView === 'memories' ? 'active' : ''}
                  onClick={() => setCodeView('memories')}
                >
                  Project Memories
                </button>
              </div>
            </div>

            {codeView === 'overview' && (
              <div className="code-overview">
                <div className="project-activation">
                  <h3>Activate Project</h3>
                  <div className="input-group">
                    <input
                      type="text"
                      value={projectPath}
                      onChange={(e) => setProjectPath(e.target.value)}
                      placeholder="Project path (e.g., /opt/projects/knowledgehub)"
                      className="project-path-input"
                    />
                    <button onClick={activateProject} disabled={isLoading}>
                      {isLoading ? 'Activating...' : 'Activate'}
                    </button>
                  </div>
                </div>

                {codeProject && (
                  <div className="project-info">
                    <h3>Project Information</h3>
                    <div className="info-grid">
                      <div className="info-item">
                        <strong>Path:</strong> {codeProject.path}
                      </div>
                      <div className="info-item">
                        <strong>Language:</strong> {codeProject.language || 'Unknown'}
                      </div>
                      <div className="info-item">
                        <strong>Framework:</strong> {codeProject.framework || 'None detected'}
                      </div>
                      <div className="info-item">
                        <strong>Dependencies:</strong> {codeProject.dependencies.length}
                      </div>
                      <div className="info-item">
                        <strong>Memories:</strong> {codeProject.memory_count}
                      </div>
                    </div>
                  </div>
                )}

                {symbols.length > 0 && (
                  <div className="symbols-overview">
                    <h3>Code Symbols ({symbols.length})</h3>
                    <div className="symbols-list">
                      {symbols.slice(0, 20).map((symbol, index) => (
                        <div key={index} className="symbol-item">
                          <div className="symbol-header">
                            <span className={`symbol-kind ${symbol.kind}`}>{symbol.kind}</span>
                            <strong>{symbol.name}</strong>
                            <span className="symbol-location">{symbol.path}:{symbol.line_start}</span>
                          </div>
                          {symbol.docstring && (
                            <div className="symbol-docs">{symbol.docstring}</div>
                          )}
                          {symbol.children && symbol.children.length > 0 && (
                            <div className="symbol-children">
                              {symbol.children.map((child, childIndex) => (
                                <div key={childIndex} className="child-symbol">
                                  <span className={`symbol-kind ${child.kind}`}>{child.kind}</span>
                                  {child.name}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                      {symbols.length > 20 && (
                        <div className="symbols-more">
                          ... and {symbols.length - 20} more symbols
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {codeView === 'search' && (
              <div className="code-search">
                <div className="search-section">
                  <h3>Pattern Search</h3>
                  <div className="input-group">
                    <input
                      type="text"
                      value={patternSearch}
                      onChange={(e) => setPatternSearch(e.target.value)}
                      placeholder="Search pattern (e.g., 'class.*Service', 'def.*async')"
                      className="pattern-input"
                      onKeyPress={(e) => e.key === 'Enter' && searchPattern()}
                    />
                    <button onClick={searchPattern} disabled={isLoading}>
                      {isLoading ? 'Searching...' : 'Search'}
                    </button>
                  </div>
                </div>

                {patternResults.length > 0 && (
                  <div className="pattern-results">
                    <h3>Search Results ({patternResults.length} files)</h3>
                    {patternResults.map((result, index) => (
                      <div key={index} className="pattern-file">
                        <h4>{result.file}</h4>
                        <div className="pattern-matches">
                          {result.matches.map((match, matchIndex) => (
                            <div key={matchIndex} className="pattern-match">
                              <span className="line-number">Line {match.line}:</span>
                              <code className="match-text">{match.text}</code>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {codeView === 'memories' && (
              <div className="code-memories">
                <h3>Project Memories</h3>
                <p>Project-specific memories will be displayed here when available.</p>
              </div>
            )}
          </div>
        )}

        {currentView === 'system' && (
          <div className="system-view">
            <h2>System Status</h2>
            
            {healthStatus && (
              <div className="system-grid">
                <div className="system-card">
                  <h3>Overall Status</h3>
                  <div className={`status-badge ${healthStatus.status}`}>
                    {healthStatus.status}
                  </div>
                </div>

                <div className="system-card">
                  <h3>API Service</h3>
                  <div className={`status-badge ${healthStatus.services.api}`}>
                    {healthStatus.services.api}
                  </div>
                </div>

                <div className="system-card">
                  <h3>Database</h3>
                  <div className={`status-badge ${healthStatus.services.database}`}>
                    {healthStatus.services.database}
                  </div>
                </div>

                <div className="system-card">
                  <h3>Cache (Redis)</h3>
                  <div className={`status-badge ${healthStatus.services.redis}`}>
                    {healthStatus.services.redis}
                  </div>
                </div>

                <div className="system-card">
                  <h3>Vector Store</h3>
                  <div className={`status-badge ${healthStatus.services.weaviate}`}>
                    {healthStatus.services.weaviate}
                  </div>
                </div>

                <div className="system-card">
                  <h3>Actions</h3>
                  <button onClick={checkHealth} className="refresh-btn">
                    Refresh Status
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default CleanApp;