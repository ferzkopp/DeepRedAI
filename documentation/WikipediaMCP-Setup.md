# Wikipedia MCP Server

Setup guide for the Wikipedia content pipeline on Fedora-based Strix Halo systems. This provides a local Wikipedia knowledge base with keyword search, semantic search, and an MCP server for VS Code Copilot integration.

---

## Overview

The Wikipedia MCP Server pipeline:
1. **Extracts** articles from a Wikipedia XML dump into clean JSON files
2. **Indexes** articles into PostgreSQL (metadata + sections) and OpenSearch (full-text + vector search)
3. **Serves** content via a FastAPI MCP server and React web GUI

### What You Get

| Feature | Description |
|---------|-------------|
| **~7 million articles** | English Wikipedia text content (no media) |
| **Keyword search (BM25)** | Full-text search via OpenSearch |
| **Semantic search** | Vector similarity search using nomic-embed-text-v1.5 embeddings |
| **Hybrid search** | Combined keyword + semantic with Reciprocal Rank Fusion |
| **Section-level indexing** | Precise retrieval at section granularity |
| **MCP server** | VS Code Copilot integration via SSE transport (port 7000) |
| **Web GUI** | Browser-based search and article browsing (port 8080) |
| **REST API** | Direct HTTP search and article retrieval |

### Architecture

```
                    ┌─────────────────────┐
                    │   VS Code Copilot   │
                    │   (MCP client)      │
                    └────────┬────────────┘
                             │ SSE/JSON-RPC
                    ┌────────▼────────────┐
                    │   MCP Server        │
┌──────────┐       │   (FastAPI :7000)    │       ┌──────────────────┐
│ Web GUI  │──────►│                     │──────►│ llama.cpp embed  │
│ (:8080)  │       │   mcp_server.py     │       │ (:1235)          │
└──────────┘       └──┬──────────────┬───┘       │ nomic-embed-text │
                      │              │            └──────────────────┘
              ┌───────▼──────┐ ┌─────▼──────┐
              │ PostgreSQL   │ │ OpenSearch  │
              │ (:5432)      │ │ (:9200)     │
              │ articles     │ │ wikipedia   │
              │ sections     │ │ index       │
              │ redirects    │ │ (k-NN)      │
              └──────────────┘ └─────────────┘
```

---

## Prerequisites

### Completed Before This Guide

The following must already be set up via [`StrixHalo-Fedora-Setup.md`](StrixHalo-Fedora-Setup.md) and `setup_strixhalo.py`:

| Component | Setup Stage | Verified By |
|-----------|-------------|-------------|
| Fedora 43 installed | Manual (Phase 1) | `uname -r` → 6.18.4+ |
| Data disk mounted at `/mnt/data` | Manual (Phase 1) | `df -h /mnt/data` |
| Repo cloned | Manual (Phase 1) | `ls $DEEPRED_REPO` |
| Python venv with dependencies | `python_venv` | `source $DEEPRED_VENV/bin/activate` |
| PostgreSQL running (user=wiki, db=wikidb) | `postgresql` | `pg_isready` |
| OpenSearch running | `opensearch` | `curl localhost:9200` |
| Embedding server (nomic-embed-text-v1.5) | `llama_server` | `curl localhost:1235/v1/models` |
| MCP server deployed | `mcp_server` | `curl localhost:7000/health` |
| Firewall configured | `firewall` | `firewall-cmd --list-ports` |

### Environment Setup

All commands in this guide assume you have sourced the environment:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
```

This sets:
- `WIKI_DATA` → `/mnt/data/wikipedia`
- `DEEPRED_VENV` → `/mnt/data/venv`
- `DEEPRED_REPO` → `/mnt/data/DeepRedAI`

### Disk Space

| Component | Size |
|-----------|------|
| Wikipedia dump (compressed) | ~25 GB |
| Extracted JSON files | ~25 GB |
| PostgreSQL database | ~45 GB |
| OpenSearch index + embeddings | ~40 GB |
| **Total under `$WIKI_DATA`** | **~135 GB** |

---

## Data Pipeline

### Phase 1: Database Schema

The `setup_strixhalo.py` script creates the PostgreSQL database and user but does not apply the Wikipedia-specific schema. Apply it now.

**Option A — Via setup script** (recommended if running setup for the first time):

```bash
sudo -E python3 scripts/setup_strixhalo.py --stage wikipedia_schema
```

**Option B — Via process_and_index.py** (standalone, no sudo required):

```bash
source $DEEPRED_VENV/bin/activate
python3 scripts/process_and_index.py --init
```

**Option C — Manual SQL**:

```bash
# Install pg_trgm extension (requires postgres superuser)
sudo -u postgres psql -d wikidb -c 'CREATE EXTENSION IF NOT EXISTS pg_trgm;'

# Apply schema as wiki user
PGPASSWORD=wiki psql -h localhost -U wiki -d wikidb -f $WIKI_DATA/schema.sql
```

Verify:

```bash
PGPASSWORD=wiki psql -h localhost -U wiki -d wikidb -c '\dt'
# Should show: articles, sections, redirects
```

### Phase 2: Wikipedia Extraction

This phase extracts articles from the Wikipedia XML dump into clean JSON files. If you already have extracted data from a previous setup (check `$WIKI_DATA/extracted/`), you can **skip this phase entirely**.

#### Check for Existing Data

```bash
ls $WIKI_DATA/extracted/*.json 2>/dev/null | wc -l
# If 7000+, extraction is already done — skip to Phase 3
```

#### Download Wikipedia Dump (if needed)

```bash
mkdir -p $WIKI_DATA/dumps
cd $WIKI_DATA/dumps
wget -c --timeout=60 --tries=10 \
    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

This is ~25 GB and takes 1–2 hours. The `-c` flag resumes interrupted downloads.

#### Run Extraction

```bash
source $DEEPRED_VENV/bin/activate
python3 $DEEPRED_REPO/scripts/extract_wikipedia.py
```

**What happens:**
- Reads the compressed XML dump directly (no manual decompression)
- Filters out redirects, disambiguation pages, and special pages
- Cleans wikitext: removes templates, HTML tags, formatting codes
- Removes non-content sections (References, Bibliography, etc.)
- Skips very short articles (< 100 characters)
- Parallelizes cleaning across all CPU cores
- Outputs JSON files with 1,000 articles each

**Performance benchmarks:**
| System | Duration | Articles |
|--------|----------|----------|
| 8-core 1.8 GHz Xeon VM | ~21 hours | 7.0M |
| 16-core 3 GHz Ryzen AI | ~3 hours | 7.0M |

**Output:**
```
$WIKI_DATA/extracted/
├── wikipedia_batch_00000.json
├── wikipedia_batch_00001.json
├── ...
└── wikipedia_batch_07036.json   (~7,037 files)
```

Each line is one article:
```json
{"id": "12", "title": "Article Title", "url": "https://en.wikipedia.org/wiki?curid=12", "text": "Clean article text..."}
```

**Verify:**
```bash
ls $WIKI_DATA/extracted/*.json | wc -l     # Should be ~7000+
head -n1 $WIKI_DATA/extracted/wikipedia_batch_00000.json | python3 -m json.tool
```

**Command-line options:**
```bash
python3 scripts/extract_wikipedia.py --help
python3 scripts/extract_wikipedia.py --batch-size 500     # Smaller batch files
python3 scripts/extract_wikipedia.py --dump-file /path/to/custom/dump.xml.bz2
```

### Phase 3: Process and Index

This is the longest phase. It reads all extracted JSON files, stores articles/sections in PostgreSQL, generates embeddings via the llama.cpp embedding server, and indexes everything to OpenSearch with k-NN vectors.

#### Pre-flight Checks

```bash
source $DEEPRED_VENV/bin/activate

# Verify all services are up and schema is ready
python3 scripts/process_and_index.py --test
```

This runs connectivity tests for:
- PostgreSQL connection and schema verification
- OpenSearch connection and k-NN plugin
- Embedding server (llama.cpp on port 1235)
- Full end-to-end pipeline with dummy data

All tests must pass before proceeding.

#### Run Processing

```bash
source $DEEPRED_VENV/bin/activate
python3 $DEEPRED_REPO/scripts/process_and_index.py
```

**What happens:**
1. Reads each extracted JSON file line by line
2. Splits each article into sections (based on markdown headers)
3. Inserts article metadata and section text into PostgreSQL
4. Generates embeddings for each section via the embedding server (port 1235)
5. Bulk indexes documents with embeddings into OpenSearch
6. Saves progress to a checkpoint file every 10 batches

**Checkpoint/Resume:** If interrupted (Ctrl+C, crash, reboot), simply run the same command again. The script automatically resumes from the last checkpoint.

> **Live Querying:** The MCP server, search API, and VS Code Copilot integration all work while processing is still running. Results will be partial (only articles indexed so far) but grow as processing continues. This is useful for verifying the pipeline is working correctly without waiting for the full ~65-hour run.

```bash
# Check progress
python3 scripts/process_and_index.py --status

# Reset and start over
python3 scripts/process_and_index.py --reset
```

**Performance:** Processing ~7 million articles takes approximately 65 hours on Strix Halo. Embedding generation is the bottleneck (~30 articles/sec). The script sends concurrent requests to utilize all parallel server slots.

> **Tip:** The embedding server container config is at `/etc/containers/systemd/llama-server-embed.container`. Key tuning parameters:
> - `--batch-size 32768` — logical batch size (must hold all tokens in a single API request: 16 texts × up to 2048 tokens)
> - `--ubatch-size 2048` — physical batch size (must be ≥ `--ctx-size` so any single text fits in one GPU call)
> - `--cont-batching` — enables continuous batching across parallel slots
> - `--parallel` is auto-detected (defaults to 4 slots on Strix Halo)
>
> After changing, reload with: `sudo systemctl daemon-reload && sudo systemctl restart llama-server-embed`

**Monitor progress:**
```bash
# In another terminal, watch the checkpoint file
watch -n 30 cat $WIKI_DATA/processing_checkpoint.json

# Or check database and index growth
PGPASSWORD=wiki psql -h localhost -U wiki -d wikidb -c "SELECT count(*) FROM articles;"
curl -s localhost:9200/_cat/indices?v
```

**Monitor GPU utilization:**
```bash
# radeontop — real-time GPU stats (installed via setup_strixhalo.py)
# Note: "Unknown Radeon card" warning is benign on Strix Halo — data is correct
radeontop

# Alternative: kernel sysfs (no install needed)
watch -n 1 cat /sys/class/drm/card0/device/gpu_busy_percent
```

| Metric | Healthy | Problem |
|--------|---------|---------|
| Graphics pipe | 50–80% | 0% (CPU fallback) |
| VRAM | ~370M used | 0M (model not loaded) |
| Shader Clock | ~2.7 GHz (boosted) | ~200 MHz (idle) |

#### Verify Results

After processing completes:

```bash
# PostgreSQL article count (should be ~7 million)
PGPASSWORD=wiki psql -h localhost -U wiki -d wikidb -c "SELECT count(*) FROM articles;"

# PostgreSQL section count
PGPASSWORD=wiki psql -h localhost -U wiki -d wikidb -c "SELECT count(*) FROM sections;"

# OpenSearch index status
curl -s localhost:9200/wikipedia/_count | python3 -m json.tool

# Test a search
curl -s -X POST localhost:7000/mcp/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "Apollo 11 moon landing", "mode": "hybrid", "limit": 3}' | python3 -m json.tool
```

---

## Services

All services are managed by systemd and start automatically on boot.

### MCP Server (port 7000)

The MCP server (`scripts/mcp_server.py`) provides:
- `POST /mcp/search` — keyword, semantic, or hybrid search
- `GET /mcp/article/{id}` — retrieve article by database ID
- `GET /mcp/article?title=...` — retrieve article by title
- `GET /health` — health check for all backends
- `GET /sse` — Server-Sent Events endpoint for MCP protocol
- `POST /messages` — MCP message handler

```bash
# Service management
systemctl status mcp
systemctl restart mcp
journalctl -u mcp -f

# Quick health check
curl localhost:7000/health
```

The service file is generated by `setup_strixhalo.py` at `/etc/systemd/system/mcp.service`. See also `services/mcp.service` in the repo for the reference template.

### Web GUI (port 8080)

The React web GUI provides a browser-based search interface. It communicates with the MCP server API on port 7000.

The web app source is in the `webapp/` directory. To build and deploy:

```bash
# Install Node.js if not already present
sudo dnf install -y nodejs npm

# Build the frontend
cd $DEEPRED_REPO/webapp
npm install
npm run build

# Copy build output to wiki frontend directory
sudo mkdir -p $WIKI_DATA/frontend
sudo cp -r dist/* $WIKI_DATA/frontend/
sudo chown -R wiki:wiki $WIKI_DATA/frontend

# Start the service
sudo systemctl enable --now wiki-gui
```

Access at: `http://<hostname>:8080`

### OpenSearch (port 9200)

```bash
# Check cluster health
curl localhost:9200/_cluster/health?pretty

# Check wikipedia index
curl localhost:9200/wikipedia/_count
curl 'localhost:9200/wikipedia/_search?size=1&pretty'
```

### Embedding Server (port 1235)

The embedding server runs as a Podman Quadlet container (`llama-server-embed`):

```bash
systemctl status llama-server-embed
curl localhost:1235/v1/models
```

Current tuned configuration:
```
--ctx-size 2048          # Model context window (nomic-bert native)
--batch-size 32768       # Logical batch size (16 texts × 2048 tokens per API call)
--ubatch-size 2048       # Physical batch size (≥ ctx-size for single-text processing)
--cont-batching          # Continuous batching across parallel slots
--parallel auto          # Auto-detects 4 slots on Strix Halo
--flash-attn on          # Flash attention for GPU efficiency
```

---

## VS Code Copilot Integration

### MCP Configuration

Add to your VS Code `settings.json`:

```json
{
  "github.copilot.chat.mcp": {
    "servers": {
      "wikipedia": {
        "type": "sse",
        "url": "http://localhost:7000/sse"
      }
    }
  }
}
```

If the MCP server runs on a different machine on your LAN:

```json
{
  "github.copilot.chat.mcp": {
    "servers": {
      "wikipedia": {
        "type": "sse",
        "url": "http://192.168.x.y:7000/sse"
      }
    }
  }
}
```

### Available MCP Tools

Once connected, Copilot can use these tools:

| Tool | Description |
|------|-------------|
| `search_wikipedia` | Search articles (keyword, semantic, or hybrid) |
| `get_article` | Retrieve full article by title |
| `get_article_by_id` | Retrieve full article by database ID |
| `health_check` | Check server and backend health |

---

## Configuration Reference

### Environment Variables

All variables are set automatically by `deepred-env.sh`. Override before sourcing or export manually.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPRED_ROOT` | `/mnt/data` | Base data directory |
| `WIKI_DATA` | `$DEEPRED_ROOT/wikipedia` | Wikipedia data directory |
| `DEEPRED_VENV` | `$DEEPRED_ROOT/venv` | Python virtual environment |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_USER` | `wiki` | PostgreSQL user |
| `PG_PASSWORD` | `wiki` | PostgreSQL password |
| `PG_DATABASE` | `wikidb` | PostgreSQL database name |
| `OS_HOST` | `localhost` | OpenSearch host |
| `OS_PORT` | `9200` | OpenSearch port |
| `INFERENCE_HOST` | `localhost` | Inference server host (LLM + embedding) |
| `INFERENCE_PORT` | `1234` | LLM inference server port |
| `EMBEDDING_PORT` | `1235` | Embedding server port |
| `REMOTE_HOST` | *(blank)* | Optional remote GPU server hostname/IP (see [Remote GPU Server](#remote-gpu-server)) |
| `REMOTE_LLM_PORT` | `1234` | LLM port on the remote server |
| `REMOTE_EMBED_PORT` | `1235` | Embedding port on the remote server |

### Database Schema

```sql
-- articles: one row per Wikipedia article
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- sections: one row per article section (Introduction, History, etc.)
CREATE TABLE sections (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    section_title TEXT,
    section_text TEXT,
    section_order INTEGER
);

-- redirects: Wikipedia redirect mappings
CREATE TABLE redirects (
    source_title TEXT PRIMARY KEY,
    target_title TEXT NOT NULL
);
```

### OpenSearch Index

The `wikipedia` index uses:
- English text analyzer for `title`, `section_title`, `text` fields
- k-NN vector field (`embedding`, 768 dimensions, HNSW/cosine, Lucene engine)
- Single shard, no replicas (single-node deployment)

---

## Remote GPU Server

If a remote GPU inference server is available (e.g. an NVIDIA A4000 system — see [`A4000-Fedora-Setup.md`](A4000-Fedora-Setup.md)), it can offload embedding and LLM inference from the primary StrixHalo system. This is optional — all pipeline scripts work with local services only.

### Enabling the Remote Server

Set the `REMOTE_HOST` environment variable to the hostname or IP address of the remote system. The variable is defined in `deepred-env.sh` and defaults to blank (disabled).

**Temporarily** (current session only):

```bash
export REMOTE_HOST="A4000AI"
source /mnt/data/DeepRedAI/deepred-env.sh
```

**Permanently** (auto-load on login) — add to `~/.bashrc` **before** the `deepred-env.sh` source line:

```bash
# ── DeepRedAI environment
export DEEPRED_ROOT="/mnt/data"
export REMOTE_HOST="A4000AI"                        # ← enable remote GPU server
[ -f "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh" ] && source "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh"
```

After editing, log out and back in (or `source ~/.bashrc`) to apply. The env summary will confirm:

```
DeepRedAI environment loaded
  ...
  REMOTE_HOST    = A4000AI (LLM :1234, embed :1235)
```

### Custom Ports

If the remote services run on non-default ports, override before sourcing:

```bash
export REMOTE_HOST="A4000AI"
export REMOTE_LLM_PORT=1234    # default
export REMOTE_EMBED_PORT=1235  # default
```

### Testing the Connection

Use the provided test script to verify the remote server is reachable and produces identical embeddings to the local server:

```bash
source $DEEPRED_VENV/bin/activate
python3 $DEEPRED_REPO/scripts/test_remote.py
```

The script performs:
1. **Availability check** — calls the remote embedding and LLM `/v1/models` endpoints
2. **Embedding comparison** — generates embeddings for test sentences on both local and remote servers, then verifies they are identical (cosine similarity ≈ 1.0)

All tests must pass before using the remote server for pipeline processing.

---

## Troubleshooting

### MCP Server Won't Start

```bash
# Check service logs
journalctl -u mcp -n 50

# Common issues:
# 1. PostgreSQL not ready → systemctl status postgresql
# 2. OpenSearch not ready → curl localhost:9200
# 3. Schema not applied → python3 scripts/process_and_index.py --init
# 4. Wrong password → check PG_PASSWORD env var vs PostgreSQL user password
```

### Embedding Server Returns Errors

```bash
# Check container status
systemctl status llama-server-embed
podman logs llama-server-embed

# Test embedding generation
curl -s -X POST localhost:1235/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-nomic-embed-text-v1.5@f16","input":"test"}' | python3 -m json.tool
```

If you see `input (N tokens) is too large to process` in the server logs:
```bash
# Increase --batch-size and --ubatch-size in the container config
sudo vi /etc/containers/systemd/llama-server-embed.container
# Set --batch-size 4096 --ubatch-size 4096 (must be ≥ max tokens per text)
sudo systemctl daemon-reload && sudo systemctl restart llama-server-embed
```

### OpenSearch Index Missing

If the `wikipedia` index doesn't exist after processing:

```bash
# Check index list
curl localhost:9200/_cat/indices?v

# Process and index will create the index automatically
# Reset checkpoint if needed and re-run
python3 scripts/process_and_index.py --reset
python3 scripts/process_and_index.py
```

### Processing Is Very Slow

The bottleneck is embedding generation. The script sends concurrent requests to utilize all server slots. Verify the GPU-accelerated embedding server is running with `--cont-batching`:

```bash
# Check server config
sudo journalctl -u llama-server-embed | grep -E 'n_parallel|n_batch|cont'

# Should show n_parallel = 4, n_batch = 4096
# If not, update /etc/containers/systemd/llama-server-embed.container

# Verify GPU is being used (should respond in <100ms)
time curl -s -X POST localhost:1235/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-nomic-embed-text-v1.5@f16","input":"test query"}' > /dev/null

# If > 1 second, check GPU access
systemctl status llama-server-embed
```

### Checkpoint Recovery

```bash
# View checkpoint status
python3 scripts/process_and_index.py --status

# Resume from checkpoint (automatic)
python3 scripts/process_and_index.py

# Start fresh (deletes checkpoint and recreates index)
python3 scripts/process_and_index.py --reset
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/extract_wikipedia.py` | Extract articles from Wikipedia XML dump to JSON |
| `scripts/process_and_index.py` | Index JSON articles to PostgreSQL and OpenSearch |
| `scripts/mcp_server.py` | FastAPI MCP server for search and retrieval |
| `scripts/setup_strixhalo.py` | Automated system setup (includes `wikipedia_schema` stage) |

### extract_wikipedia.py

```bash
python3 scripts/extract_wikipedia.py [--dump-file PATH] [--output-dir PATH] [--batch-size N]
```

### process_and_index.py

```bash
python3 scripts/process_and_index.py           # Run full processing (resumes from checkpoint)
python3 scripts/process_and_index.py --init     # Create database schema only
python3 scripts/process_and_index.py --test     # Run connectivity tests
python3 scripts/process_and_index.py --status   # Show checkpoint progress
python3 scripts/process_and_index.py --reset    # Delete checkpoint, start fresh
python3 scripts/process_and_index.py --provider local  # Use CPU embeddings (slow)
```
