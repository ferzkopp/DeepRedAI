#!/usr/bin/env python3
"""
Wikipedia MCP Server

FastAPI application implementing MCP protocol for Wikipedia search and retrieval.
Provides keyword search (BM25), semantic search (k-NN), and article retrieval.

Endpoints:
- POST /mcp/search - keyword and semantic search
- GET /mcp/article/{id} - retrieve full article
- GET /health - health check
- GET /sse - Server-Sent Events endpoint for VS Code Copilot MCP integration
- POST /messages - MCP message handler for SSE transport
"""

import os
import json
import uuid
import asyncio
import logging
from typing import Optional, List, Literal, Dict, Any
from contextlib import asynccontextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WIKI_DATA = os.environ.get('WIKI_DATA', '/mnt/data/wikipedia')

# PostgreSQL Configuration
PG_HOST = os.environ.get('PG_HOST', 'localhost')
PG_PORT = int(os.environ.get('PG_PORT', 5432))
PG_USER = os.environ.get('PG_USER', 'wiki')
PG_PASSWORD = os.environ.get('PG_PASSWORD', 'wikipass')
PG_DATABASE = os.environ.get('PG_DATABASE', 'wikidb')

# OpenSearch Configuration
OS_HOST = os.environ.get('OS_HOST', 'localhost')
OS_PORT = int(os.environ.get('OS_PORT', 9200))
OS_INDEX = os.environ.get('OS_INDEX', 'wikipedia')

# Search Configuration
DEFAULT_LIMIT = 10
MAX_LIMIT = 100

# Embedding Configuration (for semantic search)
EMBEDDING_PROVIDER = os.environ.get('EMBEDDING_PROVIDER', 'lmstudio')
LMSTUDIO_HOST = os.environ.get('LMSTUDIO_HOST', 'localhost')
LMSTUDIO_PORT = int(os.environ.get('LMSTUDIO_PORT', 1234))
LMSTUDIO_MODEL = os.environ.get('LMSTUDIO_MODEL', 'text-embedding-nomic-embed-text-v1.5@f16')

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# MCP Session Management (for SSE transport)
# -----------------------------------------------------------------------------

# Store active SSE sessions and their message queues
mcp_sessions: Dict[str, asyncio.Queue] = {}

# -----------------------------------------------------------------------------
# Global Connections
# -----------------------------------------------------------------------------

os_client: Optional[OpenSearch] = None
embedding_model = None  # For local sentence-transformers if used


def get_pg_connection():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE,
        cursor_factory=RealDictCursor
    )


def get_os_client() -> OpenSearch:
    """Get or create OpenSearch client."""
    global os_client
    if os_client is None:
        os_client = OpenSearch(
            hosts=[{'host': OS_HOST, 'port': OS_PORT}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            timeout=30
        )
    return os_client


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using configured provider."""
    global embedding_model
    
    if EMBEDDING_PROVIDER == 'lmstudio':
        import requests
        url = f'http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/embeddings'
        response = requests.post(
            url,
            json={'model': LMSTUDIO_MODEL, 'input': [text]},
            timeout=60
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    else:
        # Local sentence-transformers
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-mpnet-base-v2')
        return embedding_model.encode(text).tolist()


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    mode: Literal['keyword', 'semantic', 'hybrid'] = Field(
        default='hybrid',
        description="Search mode: 'keyword' (BM25), 'semantic' (vector), or 'hybrid' (both)"
    )
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum results to return")


class SearchResult(BaseModel):
    """Individual search result."""
    article_id: int
    title: str
    section_title: Optional[str] = None
    excerpt: str
    score: float
    url: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str
    mode: str
    total_results: int
    results: List[SearchResult]


class Section(BaseModel):
    """Article section."""
    id: int
    title: Optional[str]
    text: str
    order: int


class ArticleResponse(BaseModel):
    """Response model for article retrieval."""
    id: int
    title: str
    url: Optional[str]
    content: Optional[str]
    sections: List[Section]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    postgresql: str
    opensearch: str
    embedding_provider: str


# -----------------------------------------------------------------------------
# Application Lifecycle
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting Wikipedia MCP Server...")
    
    # Test PostgreSQL connection
    try:
        conn = get_pg_connection()
        conn.close()
        logger.info("PostgreSQL connection: OK")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
    
    # Test OpenSearch connection
    try:
        client = get_os_client()
        info = client.info()
        logger.info(f"OpenSearch connection: OK (version {info['version']['number']})")
    except Exception as e:
        logger.error(f"OpenSearch connection failed: {e}")
    
    yield
    
    # Cleanup
    global os_client
    if os_client:
        os_client.close()
        os_client = None
    logger.info("Wikipedia MCP Server stopped.")


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Wikipedia MCP Server",
    description="Model Context Protocol server for local Wikipedia search and retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web GUI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # LAN-only, adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all backend services.
    """
    pg_status = "unknown"
    os_status = "unknown"
    embed_status = "unknown"
    
    # Check PostgreSQL
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        pg_status = "healthy"
    except Exception as e:
        pg_status = f"error: {str(e)[:50]}"
    
    # Check OpenSearch
    try:
        client = get_os_client()
        health = client.cluster.health()
        os_status = f"healthy ({health['status']})"
    except Exception as e:
        os_status = f"error: {str(e)[:50]}"
    
    # Check embedding provider
    try:
        if EMBEDDING_PROVIDER == 'lmstudio':
            import requests
            url = f'http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/models'
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                embed_status = f"healthy (lmstudio @ {LMSTUDIO_HOST})"
            else:
                embed_status = f"error: HTTP {response.status_code}"
        else:
            embed_status = "healthy (local)"
    except Exception as e:
        embed_status = f"error: {str(e)[:50]}"
    
    overall = "healthy" if all(
        s.startswith("healthy") for s in [pg_status, os_status, embed_status]
    ) else "degraded"
    
    return HealthResponse(
        status=overall,
        postgresql=pg_status,
        opensearch=os_status,
        embedding_provider=embed_status
    )


@app.post("/mcp/search", response_model=SearchResponse, tags=["MCP"])
async def search(request: SearchRequest):
    """
    Search Wikipedia articles.
    
    Supports three modes:
    - **keyword**: BM25 text search via OpenSearch
    - **semantic**: Vector similarity search using embeddings
    - **hybrid**: Combines both methods with score fusion
    """
    results = []
    
    try:
        client = get_os_client()
        
        if request.mode == 'keyword':
            # BM25 keyword search
            body = {
                "query": {
                    "multi_match": {
                        "query": request.query,
                        "fields": ["title^3", "section_title^2", "text"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": request.limit,
                "_source": ["article_id", "title", "section_title", "text", "url"]
            }
            response = client.search(index=OS_INDEX, body=body)
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append(SearchResult(
                    article_id=source.get('article_id', 0),
                    title=source.get('title', 'Unknown'),
                    section_title=source.get('section_title'),
                    excerpt=source.get('text', '')[:500] + '...' if len(source.get('text', '')) > 500 else source.get('text', ''),
                    score=hit['_score'],
                    url=source.get('url')
                ))
        
        elif request.mode == 'semantic':
            # Vector similarity search
            query_embedding = generate_embedding(request.query)
            
            body = {
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": request.limit
                        }
                    }
                },
                "size": request.limit,
                "_source": ["article_id", "title", "section_title", "text", "url"]
            }
            response = client.search(index=OS_INDEX, body=body)
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append(SearchResult(
                    article_id=source.get('article_id', 0),
                    title=source.get('title', 'Unknown'),
                    section_title=source.get('section_title'),
                    excerpt=source.get('text', '')[:500] + '...' if len(source.get('text', '')) > 500 else source.get('text', ''),
                    score=hit['_score'],
                    url=source.get('url')
                ))
        
        else:  # hybrid
            # Combine keyword and semantic search
            query_embedding = generate_embedding(request.query)
            
            # Get more results from each method, then merge
            fetch_limit = min(request.limit * 2, MAX_LIMIT)
            
            # Keyword search
            keyword_body = {
                "query": {
                    "multi_match": {
                        "query": request.query,
                        "fields": ["title^3", "section_title^2", "text"],
                        "type": "best_fields"
                    }
                },
                "size": fetch_limit,
                "_source": ["article_id", "title", "section_title", "text", "url"]
            }
            keyword_response = client.search(index=OS_INDEX, body=keyword_body)
            
            # Semantic search
            semantic_body = {
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": fetch_limit
                        }
                    }
                },
                "size": fetch_limit,
                "_source": ["article_id", "title", "section_title", "text", "url"]
            }
            semantic_response = client.search(index=OS_INDEX, body=semantic_body)
            
            # Merge results using Reciprocal Rank Fusion (RRF)
            k = 60  # RRF constant
            scores = {}
            docs = {}
            
            for rank, hit in enumerate(keyword_response['hits']['hits']):
                doc_id = hit['_id']
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
                docs[doc_id] = hit['_source']
            
            for rank, hit in enumerate(semantic_response['hits']['hits']):
                doc_id = hit['_id']
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
                docs[doc_id] = hit['_source']
            
            # Sort by combined score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:request.limit]
            
            for doc_id, score in sorted_docs:
                source = docs[doc_id]
                results.append(SearchResult(
                    article_id=source.get('article_id', 0),
                    title=source.get('title', 'Unknown'),
                    section_title=source.get('section_title'),
                    excerpt=source.get('text', '')[:500] + '...' if len(source.get('text', '')) > 500 else source.get('text', ''),
                    score=score,
                    url=source.get('url')
                ))
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_results=len(results),
        results=results
    )


@app.get("/mcp/article/{article_id}", response_model=ArticleResponse, tags=["MCP"])
async def get_article(article_id: int):
    """
    Retrieve a full Wikipedia article by ID.
    
    Returns the article with all its sections.
    """
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            # Get article
            cur.execute(
                "SELECT id, title, content, url FROM articles WHERE id = %s",
                (article_id,)
            )
            article = cur.fetchone()
            
            if not article:
                raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
            
            # Get sections
            cur.execute(
                """
                SELECT id, section_title, section_text, section_order 
                FROM sections 
                WHERE article_id = %s 
                ORDER BY section_order
                """,
                (article_id,)
            )
            sections_data = cur.fetchall()
        
        conn.close()
        
        sections = [
            Section(
                id=s['id'],
                title=s['section_title'],
                text=s['section_text'],
                order=s['section_order']
            )
            for s in sections_data
        ]
        
        return ArticleResponse(
            id=article['id'],
            title=article['title'],
            url=article['url'],
            content=article['content'],
            sections=sections
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Article retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve article: {str(e)}")


@app.get("/mcp/article", response_model=ArticleResponse, tags=["MCP"])
async def get_article_by_title(title: str = Query(..., description="Article title to search for")):
    """
    Retrieve a Wikipedia article by title.
    
    Performs exact match first, then fuzzy match if not found.
    """
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            # Try exact match first
            cur.execute(
                "SELECT id, title, content, url FROM articles WHERE LOWER(title) = LOWER(%s)",
                (title,)
            )
            article = cur.fetchone()
            
            # Try fuzzy match if exact not found
            if not article:
                cur.execute(
                    """
                    SELECT id, title, content, url FROM articles 
                    WHERE title ILIKE %s 
                    ORDER BY LENGTH(title) 
                    LIMIT 1
                    """,
                    (f"%{title}%",)
                )
                article = cur.fetchone()
            
            # Check redirects if still not found
            if not article:
                cur.execute(
                    "SELECT target_title FROM redirects WHERE LOWER(source_title) = LOWER(%s)",
                    (title,)
                )
                redirect = cur.fetchone()
                if redirect:
                    cur.execute(
                        "SELECT id, title, content, url FROM articles WHERE LOWER(title) = LOWER(%s)",
                        (redirect['target_title'],)
                    )
                    article = cur.fetchone()
            
            if not article:
                raise HTTPException(status_code=404, detail=f"Article '{title}' not found")
            
            # Get sections
            cur.execute(
                """
                SELECT id, section_title, section_text, section_order 
                FROM sections 
                WHERE article_id = %s 
                ORDER BY section_order
                """,
                (article['id'],)
            )
            sections_data = cur.fetchall()
        
        conn.close()
        
        sections = [
            Section(
                id=s['id'],
                title=s['section_title'],
                text=s['section_text'],
                order=s['section_order']
            )
            for s in sections_data
        ]
        
        return ArticleResponse(
            id=article['id'],
            title=article['title'],
            url=article['url'],
            content=article['content'],
            sections=sections
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Article retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve article: {str(e)}")


# -----------------------------------------------------------------------------
# MCP Protocol Handlers (for VS Code Copilot integration)
# -----------------------------------------------------------------------------

MCP_SERVER_INFO = {
    "name": "wikipedia",
    "version": "1.0.0",
    "protocolVersion": "2024-11-05"
}

MCP_CAPABILITIES = {
    "tools": {}
}

MCP_TOOLS = [
    {
        "name": "search_wikipedia",
        "description": "Search Wikipedia articles using keyword, semantic, or hybrid search. Returns relevant article excerpts with titles and relevance scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text"
                },
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic", "hybrid"],
                    "description": "Search mode: 'keyword' for BM25 text matching, 'semantic' for meaning-based search, 'hybrid' for combined approach (default: hybrid)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 100)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_article",
        "description": "Retrieve a full Wikipedia article by its title. Returns the complete article text organized by sections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the Wikipedia article to retrieve"
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "get_article_by_id",
        "description": "Retrieve a full Wikipedia article by its database ID. Returns the complete article text organized by sections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "integer",
                    "description": "The database ID of the Wikipedia article"
                }
            },
            "required": ["article_id"]
        }
    },
    {
        "name": "health_check",
        "description": "Check the health status of the Wikipedia MCP server and its backend services (PostgreSQL, OpenSearch, embedding provider).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


async def handle_mcp_message(message: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Handle incoming MCP protocol messages and return appropriate responses.
    """
    method = message.get("method", "")
    msg_id = message.get("id")
    params = message.get("params", {})
    
    logger.info(f"MCP [{session_id}] Received: {method}")
    
    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "serverInfo": MCP_SERVER_INFO,
                    "capabilities": MCP_CAPABILITIES
                }
            }
        
        elif method == "notifications/initialized":
            # Client acknowledged initialization - no response needed
            return None
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": MCP_TOOLS
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            result = await execute_mcp_tool(tool_name, tool_args)
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                        }
                    ]
                }
            }
        
        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": []
                }
            }
        
        elif method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "prompts": []
                }
            }
        
        else:
            logger.warning(f"MCP [{session_id}] Unknown method: {method}")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    except Exception as e:
        logger.error(f"MCP [{session_id}] Error handling {method}: {e}")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }


async def execute_mcp_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    """
    Execute an MCP tool and return the result.
    """
    if tool_name == "search_wikipedia":
        query = args.get("query", "")
        mode = args.get("mode", "hybrid")
        limit = min(args.get("limit", DEFAULT_LIMIT), MAX_LIMIT)
        
        request = SearchRequest(query=query, mode=mode, limit=limit)
        response = await search(request)
        
        return {
            "query": response.query,
            "mode": response.mode,
            "total_results": response.total_results,
            "results": [
                {
                    "article_id": r.article_id,
                    "title": r.title,
                    "section_title": r.section_title,
                    "excerpt": r.excerpt,
                    "score": r.score,
                    "url": r.url
                }
                for r in response.results
            ]
        }
    
    elif tool_name == "get_article":
        title = args.get("title", "")
        response = await get_article_by_title(title)
        
        return {
            "id": response.id,
            "title": response.title,
            "url": response.url,
            "sections": [
                {
                    "title": s.title,
                    "text": s.text,
                    "order": s.order
                }
                for s in response.sections
            ]
        }
    
    elif tool_name == "get_article_by_id":
        article_id = args.get("article_id", 0)
        response = await get_article(article_id)
        
        return {
            "id": response.id,
            "title": response.title,
            "url": response.url,
            "sections": [
                {
                    "title": s.title,
                    "text": s.text,
                    "order": s.order
                }
                for s in response.sections
            ]
        }
    
    elif tool_name == "health_check":
        response = await health_check()
        return {
            "status": response.status,
            "postgresql": response.postgresql,
            "opensearch": response.opensearch,
            "embedding_provider": response.embedding_provider
        }
    
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# -----------------------------------------------------------------------------
# SSE Endpoints (for VS Code Copilot MCP integration)
# -----------------------------------------------------------------------------

@app.get("/sse", tags=["MCP-SSE"])
async def sse_endpoint(request: Request):
    """
    Server-Sent Events endpoint for MCP protocol.
    
    VS Code Copilot connects here to establish the MCP session.
    Returns an SSE stream with a session endpoint URL.
    """
    session_id = str(uuid.uuid4())
    mcp_sessions[session_id] = asyncio.Queue()
    
    logger.info(f"MCP SSE session started: {session_id}")
    
    async def event_generator():
        try:
            # Send the endpoint URL for this session
            # The client will POST messages to this URL
            endpoint_url = f"/messages?session_id={session_id}"
            yield f"event: endpoint\ndata: {endpoint_url}\n\n"
            
            # Keep connection alive and send queued messages
            while True:
                try:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    
                    # Wait for messages with timeout (for keepalive)
                    try:
                        message = await asyncio.wait_for(
                            mcp_sessions[session_id].get(),
                            timeout=30.0
                        )
                        yield f"event: message\ndata: {json.dumps(message)}\n\n"
                    except asyncio.TimeoutError:
                        # Send keepalive comment
                        yield ": keepalive\n\n"
                
                except Exception as e:
                    logger.error(f"MCP SSE [{session_id}] stream error: {e}")
                    break
        
        finally:
            # Cleanup session
            if session_id in mcp_sessions:
                del mcp_sessions[session_id]
            logger.info(f"MCP SSE session ended: {session_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/messages", tags=["MCP-SSE"])
async def messages_endpoint(request: Request, session_id: str = Query(...)):
    """
    Message endpoint for MCP protocol over SSE transport.
    
    Receives JSON-RPC messages from VS Code Copilot and queues responses
    to be sent via the SSE stream.
    """
    if session_id not in mcp_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        body = await request.json()
        logger.debug(f"MCP message [{session_id}]: {json.dumps(body)[:200]}")
        
        response = await handle_mcp_message(body, session_id)
        
        if response is not None:
            # Queue response to be sent via SSE
            await mcp_sessions[session_id].put(response)
        
        return {"status": "ok"}
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"MCP message error [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
