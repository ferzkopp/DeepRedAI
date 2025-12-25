# Wikipedia MCP Server

The following document describes the architecture and setup instructions for an MCP server that hosts Wikipedia content for use by agents on the local LAN.

## General Information

### Base OS

The base OS is Ubuntu 25.10 (Questing Quokka) running on a Baremetal Strix Halo machine server.

**Disk Space (SSD recommended):**
- Ubuntu OS + packages: ~10 GB
- Wikipedia dump (compressed): ~25 GB
- Extracted text: ~25 GB
- PostgreSQL database: ~45 GB
- OpenSearch index with embeddings: ~40 GB
- Drivers/Software (lmstudio, ROCm, amdgpu, opensearch): ~25 GB
- Working space: ~10 GB
- **Total: 180 GB**

**Note:** All Wikipedia data (dumps, extracted text, OpenSearch index, and PostgreSQL database) is stored under the `${WIKI_DATA}` path (~150 GB). Only the Ubuntu OS, packages and additional software resides on the system drive.

### LMStudio (Embedding Service)

It is assumed the system has been configured with a headless LMStudio server as described [here](LMStudio-Setup.md) to provide performance embedding services.

### Provided User Interfaces

The server setup will provide:
- a web GUI for querying the wikipedia content 
- a MCP server for wikipedia content

### Security Considerations

Since the server is assumed to be running on the local LAN (i.e., 192.168.x.y) and is not connected directly to the internet, 
no passwords or certificates will be configured for the web GUI or MCP server.

**Note:** Firewall configuration is covered in Phase 1 of the Installation section below.

### Features

Core Features:
- English Wikipedia text content (articles only, no media)
- Keyword search (BM25) via OpenSearch
- Semantic search using embeddings
- Section-level indexing for precise retrieval
- Redirect and disambiguation page handling
- Web GUI for browsing and searching
- MCP server for VS Code Copilot integration

### Software

All software runs locally without external service dependencies.

**Core Components:**
1. **Ubuntu 25.10 (Questing Quokka)** - Base OS with Python 3.13.7
2. **PostgreSQL** - Metadata storage (articles, sections, redirects)
3. **OpenSearch** - Full-text search and vector embeddings (k-NN plugin)
4. **mediawiki-dump** - Library for reading Wikipedia XML dumps
5. **mwparserfromhell** - Wikitext parser for cleaning article text
6. **Sentence Transformers** - Generate embeddings (model: nomic-embed-text-v1.5)
7. **FastAPI** - MCP server implementation
8. **React + Vite** - Web GUI frontend

**Python Packages:**
- `mediawiki-dump` - Wikipedia XML dump reader
- `mwparserfromhell` - Wikitext parser and cleaner
- `fastapi`, `uvicorn` - Web server
- `psycopg2-binary` - PostgreSQL connector
- `opensearch-py` - OpenSearch client
- `sentence-transformers` - Embedding generation
- `pydantic` - Data validation

## Installation

### Phase 1: System Preparation

1. Set environment configuration variables:
```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"

# Replace with your server's IP address
export HOST="192.168.X.Y"

# LAN network range for firewall rules (derived from HOST)
export LAN_NETWORK="${HOST%.*}.0/24"
```

**Note:** 
- Replace `/mnt/data/wikipedia` with your preferred storage location. This variable will be used throughout the installation for all Wikipedia data, scripts, and virtual environment.
- Replace `192.168.X.Y` with your server's actual IP address. The LAN network range will be automatically derived from HOST (e.g., if HOST is 192.168.1.100, LAN_NETWORK becomes 192.168.1.0/24) and used for firewall rules and network configuration.

2. Update system and install basic tools:
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential curl wget git python3 python3-venv python3-pip pbzip2 jq
```

**Note:** Ubuntu 25.10 (Questing Quokka) comes with Python 3.13.7 by default, which is compatible with all required packages.

3. Configure firewall:
```bash
sudo apt-get install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from $LAN_NETWORK to any port 22 proto tcp
sudo ufw allow from $LAN_NETWORK to any port 8080 proto tcp
sudo ufw allow from $LAN_NETWORK to any port 7000 proto tcp
sudo ufw allow from $LAN_NETWORK to any port 9200 proto tcp
sudo ufw enable
sudo ufw status verbose
```

4. Create service user and directories:
```bash
sudo adduser --system --group wiki
sudo mkdir -p ${WIKI_DATA}/{dumps,extracted,scripts}
sudo chown -R wiki:wiki ${WIKI_DATA}

# set the wiki home directory and move contents (if any)
sudo usermod -d ${WIKI_DATA} -m wiki

# set login shell to bash so sudo -iu works
sudo usermod -s /bin/bash wiki

# Add environment variable to wiki user's profile
# Note: Use .profile (not .bashrc) to ensure variable is set for login shells
echo "export WIKI_DATA=${WIKI_DATA}" | sudo tee -a ${WIKI_DATA}/.profile

# Create .bashrc that sources .profile for interactive shells
echo '[ -f "$HOME/.profile" ] && . "$HOME/.profile"' | sudo tee ${WIKI_DATA}/.bashrc
```

5. Setup shared access for multiple users (optional):

If you want both the `wiki` service user and your local user account to have read/write access to the data folder:

```bash
# Create a shared group for data access
sudo groupadd wikidata

# Add users to the group (replace 'localuser' with your username)
sudo usermod -aG wikidata localuser
sudo usermod -aG wikidata wiki

# Set group ownership on the data directory
sudo chgrp -R wikidata ${WIKI_DATA}

# Ensure the directory is group-writable
sudo chmod -R g+rwX ${WIKI_DATA}

# Set the setgid bit so new files/directories inherit the group
sudo find ${WIKI_DATA} -type d -exec chmod g+s {} \;

# IMPORTANT: Restore PostgreSQL ownership (postgres requires exclusive access)
# Skip this if PostgreSQL is not yet installed
sudo chown -R postgres:postgres ${WIKI_DATA}/postgres 2>/dev/null || true
sudo chmod 700 ${WIKI_DATA}/postgres 2>/dev/null || true

# (Optional) Set default ACLs for more reliable permission inheritance
sudo apt install -y acl
sudo setfacl -R -m g:wikidata:rwX ${WIKI_DATA}
sudo setfacl -R -d -m g:wikidata:rwX ${WIKI_DATA}

# Remove ACLs from PostgreSQL directory (postgres requires standard permissions)
sudo setfacl -R -b ${WIKI_DATA}/postgres 2>/dev/null || true
```

**Important Notes:**
- Users must log out and back in for group changes to take effect. Alternatively, run `newgrp wikidata` in your current session.
- The PostgreSQL data directory (`${WIKI_DATA}/postgres`) must remain owned by `postgres:postgres` with mode `700`. PostgreSQL refuses to start if other users have access to its data files.
- If you run this step after PostgreSQL is installed, always restore postgres ownership afterward.

Verify the setup:
```bash
# Check group membership
groups localuser
groups wiki

# Test access as wiki user
sudo -u wiki ls -la ${WIKI_DATA}/

# Verify PostgreSQL directory has correct permissions
ls -la ${WIKI_DATA}/postgres
# Should show: drwx------ postgres postgres
```

**Troubleshooting PostgreSQL Permissions:**

If PostgreSQL fails to start with "Permission denied" errors after changing group ownership:

```bash
# Ensure WIKI_DATA is set in your current shell
export WIKI_DATA="/mnt/data/wikipedia"

# Stop all PostgreSQL services
sudo systemctl stop postgresql@*-main
sudo systemctl stop postgresql

# Restore correct ownership for PostgreSQL data directory
sudo chown -R postgres:postgres ${WIKI_DATA}/postgres
sudo chmod 700 ${WIKI_DATA}/postgres

# Start the specific PostgreSQL cluster
PG_VERSION=$(pg_config --version | grep -oP '\d+' | head -1)
sudo systemctl start postgresql@${PG_VERSION}-main

# Check status of the actual cluster (not the meta-service)
sudo systemctl status postgresql@${PG_VERSION}-main

# Verify connection works
sudo -u postgres psql -c "SELECT 1;"
```

**Note:** The `postgresql.service` is a meta-service that just runs `/bin/true`. The actual PostgreSQL instance runs as `postgresql@<version>-main.service`.

### Phase 2: Download Wikipedia Data

Switch to wiki user and download the dump:

```bash
sudo -iu wiki
```

Verify the environment variable is set:
```bash
echo $WIKI_DATA
# Should output your data path, e.g., /mnt/data/wikipedia
```

**Note:** If `$WIKI_DATA` is empty, log out and back in, or manually set it:
```bash
export WIKI_DATA=/mnt/data/wikipedia  # Use your actual path
```

Then download the dump:

```bash
mkdir -p ${WIKI_DATA}/dumps
cd ${WIKI_DATA}/dumps
wget -c --timeout=60 --tries=10 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

This download is ~25 GB and takes 1-2 hours.  If interrupted, simply re-run the same command to resume.

### Phase 3: Extract and Parse Content

1. Create Python virtual environment and install dependencies:

```bash
sudo -iu wiki
```

Verify the environment variable is set (should output your data path):
```bash
echo $WIKI_DATA
```

Then create the virtual environment:

```bash
# Activate wiki owned environment
cd ${WIKI_DATA}
python3 -m venv ${WIKI_DATA}/venv
source ${WIKI_DATA}/venv/bin/activate

# Verify Python version
python3 --version
# Should show Python 3.13.7 or later

# Upgrade pip and install build dependencies
pip install --upgrade pip
pip install wheel setuptools cython

# Install extraction libraries
pip install mediawiki-dump mwparserfromhell
```

2. Create extraction script (`${WIKI_DATA}/scripts/extract_wikipedia.pyx`):

Note: see /scripts folder for the code

Create build script (`${WIKI_DATA}/scripts/setup.py`)

```python
from setuptools import setup
from Cython.Build import cythonize
setup(
    ext_modules=cythonize("extract_wikipedia.pyx", compiler_directives={'language_level': "3"})
)
```

Compile module:

```bash
cd ${WIKI_DATA}/scripts/
python3 setup.py build_ext --inplace
```

Create run script (`${WIKI_DATA}/scripts/main.py`)

```python
# main.py
import os
import extract_wikipedia

if __name__ == "__main__":
    wiki_data = os.environ.get('WIKI_DATA', '/mnt/data/wikipedia')
    extract_wikipedia.extract_articles(
        dump_file=f'{wiki_data}/dumps/enwiki-latest-pages-articles.xml.bz2',
        output_dir=f'{wiki_data}/extracted',
        batch_size=1000
    )
```

3. Run extraction:

```bash
sudo -iu wiki
```

then

```bash
source ${WIKI_DATA}/venv/bin/activate
cd ${WIKI_DATA}/scripts/
python3 main.py
```

**Example:** PC: 8 core 1.8GHz Xeon VM
Start: 11/2 17:40
End: 11/3 14:21
Duration: 20 hours 41 minutes
Articles: 7.015M

**Example:** PC: 16 core/32 threads 3GHz Ryzen AI
Start: 12/6 7:51
End: 12/06 10:40
Duration: 2 hours 49 minutes
Articles: 7.036M

**What happens during extraction:**
- Reads the compressed Wikipedia XML dump directly (no decompression needed)
- Filters out redirects, disambiguation pages, and special pages
- Cleans wikitext: removes templates, tags, formatting codes
- Removes non-content sections (References, Bibliography, etc.)
- Skips very short articles (< 100 characters)
- Outputs JSON files with 1,000 articles each

**Output Structure:**
```
${WIKI_DATA}/extracted/
├── wikipedia_batch_00000.json
├── wikipedia_batch_00001.json
├── wikipedia_batch_00002.json
└── ...
```

Each line in JSON files contains one article:
```json
{"id": "12", "title": "Article Title", "url": "https://en.wikipedia.org/wiki?curid=12", "text": "Clean article text..."}
```

4. Verify extraction:
```bash
# Check output structure
ls -lh ${WIKI_DATA}/extracted/

# Count files (should be 7000+)
ls ${WIKI_DATA}/extracted/ | wc -l

# View first article
head -n 1 ${WIKI_DATA}/extracted/wikipedia_batch_00000.json | python -m json.tool

# Count total articles
cat ${WIKI_DATA}/extracted/*.json | wc -l
# Should show ~7 million for English Wikipedia
```

### Phase 4: Setup PostgreSQL

**Important:** These commands must be run as root/sudo user (not as wiki user), but we need to ensure the `WIKI_DATA` environment variable is set correctly before running them.

1. Set the environment variable (required for this session):
```bash
# Set this FIRST before running any other commands in this phase
# Replace with your actual storage path
export WIKI_DATA="/mnt/data/wikipedia"
echo "WIKI_DATA is set to: $WIKI_DATA"
```

2. Install PostgreSQL:
```bash
sudo apt-get install -y postgresql postgresql-common
```

3. Stop the default PostgreSQL cluster and create a custom data directory:
```bash
# Stop the default cluster (we'll create a new one on the data drive)
sudo systemctl stop postgresql

# Create PostgreSQL data directory on the data drive
sudo mkdir -p ${WIKI_DATA}/postgres
sudo chown postgres:postgres ${WIKI_DATA}/postgres
sudo chmod 700 ${WIKI_DATA}/postgres
```

4. Initialize a new PostgreSQL cluster on the data drive:
```bash
# Get the installed PostgreSQL version
PG_VERSION=$(pg_config --version | grep -oP '\d+' | head -1)

# Drop the default cluster (uses /var/lib/postgresql by default)
sudo pg_dropcluster --stop ${PG_VERSION} main

# Create a new cluster on the data drive
# Note: We use the full path here to ensure it's correct
sudo pg_createcluster -d ${WIKI_DATA}/postgres ${PG_VERSION} main

# Start the new cluster
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

5. Verify PostgreSQL is using the custom data directory:
```bash
sudo -iu postgres psql -c "SHOW data_directory;"
# Should output: /mnt/data/wikipedia/postgres (or your WIKI_DATA path)
```

6. Create database and user:
```bash
sudo -iu postgres psql -c "CREATE ROLE wiki WITH LOGIN PASSWORD 'wikipass';"
sudo -iu postgres psql -c "CREATE DATABASE wikidb OWNER wiki;"
sudo -iu postgres psql -d wikidb -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

7. Create schema (save as `${WIKI_DATA}/scripts/schema.sql`):
```sql
-- Drop existing tables and indices (for reload)
DROP TABLE IF EXISTS sections CASCADE;
DROP TABLE IF EXISTS articles CASCADE;
DROP TABLE IF EXISTS redirects CASCADE;

CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE sections (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    section_title TEXT,
    section_text TEXT,
    section_order INTEGER
);

CREATE TABLE redirects (
    source_title TEXT PRIMARY KEY,
    target_title TEXT NOT NULL
);

CREATE INDEX idx_articles_title ON articles(title);
CREATE INDEX idx_sections_article_id ON sections(article_id);
CREATE INDEX idx_sections_text_trgm ON sections USING gin(section_text gin_trgm_ops);
```

7. Apply schema (use password set above):
```bash
psql -h localhost -U wiki -d wikidb -f ${WIKI_DATA}/scripts/schema.sql
```

### Phase 5: Install OpenSearch

1. Install Java:
```bash
sudo apt-get install -y default-jre-headless
```

2. Download and extract OpenSearch (version 3.3.2 as of Nov 2025):
```bash
cd /opt
export OPENSEARCH_VERSION="3.3.2"
sudo wget https://artifacts.opensearch.org/releases/bundle/opensearch/${OPENSEARCH_VERSION}/opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz
sudo tar xzf opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz
sudo rm opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz
sudo chown -R wiki:wiki opensearch-${OPENSEARCH_VERSION}
sudo ln -sfn /opt/opensearch-${OPENSEARCH_VERSION} /opt/opensearch
```

**Note:** Check https://opensearch.org/downloads.html for the latest version and update `OPENSEARCH_VERSION` accordingly.

3. Create OpenSearch data directory on the data drive:
```bash
# Set environment variable if not already set
export WIKI_DATA="/mnt/data/wikipedia"  # Replace with your actual path

# Create data and logs directories
sudo mkdir -p ${WIKI_DATA}/opensearch/data
sudo mkdir -p ${WIKI_DATA}/opensearch/logs
sudo chown -R wiki:wiki ${WIKI_DATA}/opensearch
```

4. Configure OpenSearch JVM heap size:

The default heap is often too small for Wikipedia-scale indexing. Edit the JVM options:
```bash
sudo nano /opt/opensearch/config/jvm.options
```

Find and modify these lines (or add them):
```
## Heap size - set both to the same value
-Xms16g
-Xmx16g
```

**Heap Size Guidelines:**
| System RAM | Recommended Heap | Use Case |
|------------|------------------|-----------|
| 32 GB | `-Xms8g -Xmx8g` | Minimal, shared with LLMs |
| 64 GB | `-Xms12g -Xmx12g` | Balanced |
| 128 GB | `-Xms16g -Xmx16g` | Recommended for Wikipedia |
| 128 GB+ | `-Xms24g -Xmx24g` | Maximum practical |

**Important constraints:**
- **Never exceed 31 GB** - JVM loses compressed pointers optimization above this
- **Never exceed 50% of total RAM** - OpenSearch also needs off-heap memory for file caching
- **Always set `-Xms` equal to `-Xmx`** - prevents heap resizing overhead

**For systems running LLMs:** If most RAM is reserved for LLM inference, use a conservative 8-12 GB heap. OpenSearch will use OS-level file caching for the remainder, which still provides good performance.

5. Configure OpenSearch (`/opt/opensearch/config/opensearch.yml`):

**Important:** The `path.data` and `path.logs` settings ensure all OpenSearch data is stored on your data drive (under `${WIKI_DATA}`), not on the root filesystem.

```yaml
cluster.name: wiki-cluster
node.name: wiki-node
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
plugins.security.disabled: true

# Store data and logs on the data drive (adjust path as needed)
path.data: /mnt/data/wikipedia/opensearch/data
path.logs: /mnt/data/wikipedia/opensearch/logs
```

**Note:** Replace `/mnt/data/wikipedia` with your actual `${WIKI_DATA}` path.

6. Create systemd service (`/etc/systemd/system/opensearch.service`):

Copy the service file from the `/services` folder:
```bash
sudo cp /path/to/services/opensearch.service /etc/systemd/system/opensearch.service
```

**Service file:** See `opensearch.service` in the `/services` folder alongside this documentation.

7. Set OpenSearch memory map limit:
```bash
echo 'vm.max_map_count=262144' | sudo tee /etc/sysctl.d/99-opensearch.conf
sudo sysctl -p /etc/sysctl.d/99-opensearch.conf
```

**Note:** This kernel parameter is required by OpenSearch for memory-mapped files. It must be set before OpenSearch starts.

8. Start OpenSearch:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now opensearch
sudo systemctl status opensearch
```

9. Verify OpenSearch is running and using the correct data path:and using the correct data path:
```bash
# Check if OpenSearch responds
curl -s http://localhost:9200

# Check version number
curl -s http://localhost:9200 | jq '.version.number'

# Verify data is stored on the data drive
ls -la ${WIKI_DATA}/opensearch/data/
# Should show 'nodes' directory after first startup

# Check disk usage (should be on data drive, not root)
df -h ${WIKI_DATA}/opensearch/data/
```

**Troubleshooting Startup Issues**

If OpenSearch doesn't respond or the version check returns nothing:

1. Check service status:
```bash
sudo systemctl status opensearch
```

2. View service logs:
```bash
# Recent logs
sudo journalctl -u opensearch -n 50

# Follow logs in real-time
sudo journalctl -u opensearch -f
```

3. Check OpenSearch application logs:
```bash
ls -la /opt/opensearch/logs/
cat /opt/opensearch/logs/wiki-cluster.log
```

**Upgrading an Existing OpenSearch Installation**

To upgrade OpenSearch to a newer version:

1. Stop the OpenSearch service:
```bash
sudo systemctl stop opensearch
```

2. Download and extract the new version:
```bash
cd /opt
export OPENSEARCH_VERSION="3.3.2"  # Update to desired version
sudo wget https://artifacts.opensearch.org/releases/bundle/opensearch/${OPENSEARCH_VERSION}/opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz
sudo tar xzf opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz
sudo chown -R wiki:wiki opensearch-${OPENSEARCH_VERSION}
```

3. Copy configuration from old installation:
```bash
# Backup and copy config (adjust old version path as needed)
sudo cp /opt/opensearch/config/opensearch.yml /opt/opensearch-${OPENSEARCH_VERSION}/config/
sudo cp /opt/opensearch/config/jvm.options /opt/opensearch-${OPENSEARCH_VERSION}/config/ 2>/dev/null || true
```

4. Update the symlink to point to the new version:
```bash
sudo ln -sfn /opt/opensearch-${OPENSEARCH_VERSION} /opt/opensearch
```

5. Start the upgraded OpenSearch:
```bash
sudo systemctl start opensearch
sudo systemctl status opensearch
```

6. Verify the upgrade:
```bash
curl -s http://localhost:9200 | jq '.version.number'
```

**Note:** Major version upgrades (e.g., 2.x to 3.x) may require index reindexing. Check the [OpenSearch upgrade documentation](https://opensearch.org/docs/latest/install-and-configure/upgrade-opensearch/index/) for version-specific guidance.

**Migrating Existing OpenSearch Data to ${WIKI_DATA}**

If you have an existing OpenSearch installation with data stored on the root filesystem (default location: `/opt/opensearch/data`), follow these steps to migrate it to your data drive:

1. Set the environment variable:
```bash
export WIKI_DATA="/mnt/data/wikipedia"  # Replace with your actual path
```

2. Stop the OpenSearch service:
```bash
sudo systemctl stop opensearch
```

3. Verify OpenSearch has stopped completely:
```bash
sudo systemctl status opensearch
# Should show "inactive (dead)"

# Ensure no OpenSearch processes are running
pgrep -f opensearch
# Should return nothing
```

4. Identify the current data location:
```bash
# Check current configuration
grep -E '^path\.' /opt/opensearch/config/opensearch.yml

# If path.data is not set, default location is:
ls -la /opt/opensearch/data/
# You should see a 'nodes' directory

# Check current disk usage
du -sh /opt/opensearch/data/
```

5. Create the new data directory:
```bash
sudo mkdir -p ${WIKI_DATA}/opensearch/data
sudo mkdir -p ${WIKI_DATA}/opensearch/logs
sudo chown -R wiki:wiki ${WIKI_DATA}/opensearch
```

6. Move the existing data:
```bash
# Move data directory contents (preserves all index data)
sudo mv /opt/opensearch/data/* ${WIKI_DATA}/opensearch/data/

# Move logs if they exist
sudo mv /opt/opensearch/logs/* ${WIKI_DATA}/opensearch/logs/ 2>/dev/null || true

# Set correct ownership
sudo chown -R wiki:wiki ${WIKI_DATA}/opensearch
```

7. Update OpenSearch configuration:
```bash
sudo nano /opt/opensearch/config/opensearch.yml
```

Add or update these lines (replace path with your actual `${WIKI_DATA}` value):
```yaml
# Store data and logs on the data drive
path.data: /mnt/data/wikipedia/opensearch/data
path.logs: /mnt/data/wikipedia/opensearch/logs
```

8. Start OpenSearch:
```bash
sudo systemctl start opensearch
```

9. Verify the migration was successful:
```bash
# Check service status
sudo systemctl status opensearch

# Verify OpenSearch is responding
curl -s http://localhost:9200

# Check indices are intact
curl -s http://localhost:9200/_cat/indices?v

# Verify data is on the correct drive
df -h ${WIKI_DATA}/opensearch/data/
du -sh ${WIKI_DATA}/opensearch/data/
```

10. Clean up old directories (optional, after verifying everything works):
```bash
# Remove empty old directories
sudo rmdir /opt/opensearch/data 2>/dev/null || true
sudo rmdir /opt/opensearch/logs 2>/dev/null || true
```

**Troubleshooting Migration Issues:**

- **Permission denied errors:** Ensure `wiki:wiki` owns all files in `${WIKI_DATA}/opensearch`
- **Cluster state issues:** Check logs at `${WIKI_DATA}/opensearch/logs/wiki-cluster.log`
- **Missing indices:** Verify the `nodes` directory was moved correctly with all subdirectories intact
- **Disk space:** Ensure the target drive has sufficient free space before moving

### Phase 6: Process and Index Data

This phase generates embeddings for semantic search. You can choose between:
- **LM Studio (recommended)**: Uses an external GPU-accelerated server (fast)
- **Local CPU**: Uses sentence-transformers locally (slow)

**Option A: LM Studio Setup (Recommended)**

If you have LM Studio running on another machine with GPU acceleration:

1. On your LM Studio server, load the `nomic-embed-text-v1.5` embedding model

   **Model Selection Guide:**
   - Download `nomic-embed-text-v1.5.f16.gguf` (recommended) or `Q8_0` variant
   - **Avoid Q4/Q5 quantizations** for embedding models—precision loss degrades search quality
   - F16 is only ~270 MB, so there's no benefit to heavy quantization
   - For AMD GPUs with Vulkan: F16 works natively and provides best results
   - **Context length varies by model version**: Check your model's max context (commonly 2048) and set `LMSTUDIO_CONTEXT_LENGTH` accordingly

2. Start the LM Studio local server (default port 1234)

   **Firewall Configuration (Ubuntu/Debian LM Studio Server):**
   
   If your LM Studio server has a firewall enabled, allow incoming connections on port 1234:
   ```bash
   # Check if ufw is active
   sudo ufw status
   
   # If active, allow LM Studio port from your LAN
   sudo ufw allow from $LAN_NETWORK to any port 1234 proto tcp
   
   # Verify the rule was added
   sudo ufw status numbered
   ```
   
   
3. Verify the embedding endpoint is accessible:
```bash
# From the Wikipedia MCP host, test connectivity
curl -X POST http://$HOST:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-nomic-embed-text-v1.5@f16", "input": ["test embedding"]}'
```

4. Install Python dependencies (no torch/sentence-transformers needed):
```bash
sudo -iu wiki
```

then
```bash
source ${WIKI_DATA}/venv/bin/activate
pip install psycopg2-binary opensearch-py numpy requests
```

5. Configure the processing script for LM Studio:
```bash
# Edit the script to set your LM Studio server IP
nano ${WIKI_DATA}/scripts/process_and_index.py
```

Update these configuration values near the top of the script:
```python
# Choose embedding provider: 'lmstudio' (recommended) or 'local'
EMBEDDING_PROVIDER = 'lmstudio'

# LM Studio Configuration (when EMBEDDING_PROVIDER = 'lmstudio')
# Update LMSTUDIO_HOST to your LM Studio server IP address
LMSTUDIO_HOST = 'localhost'  # LM Studio server IP
LMSTUDIO_PORT = 1234
LMSTUDIO_URL = f'http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/embeddings'
LMSTUDIO_MODEL = 'text-embedding-nomic-embed-text-v1.5@f16'  # Model identifier from LM Studio
LMSTUDIO_CONTEXT_LENGTH = 2048  # Max tokens per text (model-dependent: 2048 or 8192)
LMSTUDIO_MODEL_BATCH_SIZE = 1024  # Max batch size in tokens (model config limit)
LMSTUDIO_TIMEOUT = 300  # Seconds per batch request (increased for larger batches)
# Optimized for Wikipedia paragraph-level text (~500 chars, ~125 tokens per paragraph)
# Based on benchmarks: 16-32 paragraphs per batch achieves best throughput
LMSTUDIO_BATCH_SIZE = 32  # Texts per embedding API call (optimized for paragraph-level)
```

**Tuning `LMSTUDIO_BATCH_SIZE` for Performance:**

The `LMSTUDIO_BATCH_SIZE` parameter controls how many text sections are sent to LM Studio per API request. The optimal value depends on your text fragment size and GPU capabilities.

**Benchmark Results (nomic-embed-text-v1.5@f16 on RTX GPU):**

| Fragment Type | Batch Size | Throughput | Wikipedia Est. Time |
|---------------|------------|------------|---------------------|
| Sentence (~120 chars) | 64 | 175 frags/sec | 9.3 days (140M) |
| Sentence (~120 chars) | 128 | 186 frags/sec | 8.7 days (140M) |
| Paragraph (~500 chars) | 16 | 90 frags/sec | 4.5 days (35M) |
| **Paragraph (~500 chars)** | **32** | **95 frags/sec** | **4.3 days (35M)** |
| Long paragraph (~1000 chars) | 8 | 49 frags/sec | 8.3 days (35M) |

**Recommendation:** Use batch size 32 for paragraph-level embeddings (~500 chars each). This achieves the best throughput while staying within model token limits (1024 tokens per batch).

**Storage Estimates (768-dim float32 embeddings):**
- Per embedding: 3,072 bytes
- Paragraph-level (35M): ~107 GB
- Sentence-level (140M): ~430 GB

If you encounter out-of-memory errors in LM Studio, reduce `LMSTUDIO_BATCH_SIZE`. If processing is slower than expected with short texts, try increasing it.

**Monitoring GPU Load:**

Install `nvtop` on the LM Studio server to monitor GPU utilization during processing:
```bash
sudo apt-get install -y nvtop
nvtop
```

This displays real-time GPU usage, memory consumption, and temperature—useful for tuning batch sizes and verifying the GPU is being utilized effectively.

**Option B: Local CPU Setup (Slower)**

If you don't have an external GPU server:

1. Install Python dependencies:
```bash
sudo -iu wiki
source ${WIKI_DATA}/venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers==3.2.0
pip install psycopg2-binary opensearch-py numpy requests
```

2. Configure the processing script for local embeddings:
```bash
nano ${WIKI_DATA}/scripts/process_and_index.py
```

Update the configuration:
```python
# Choose embedding provider: 'lmstudio' (recommended) or 'local'
EMBEDDING_PROVIDER = 'local'
```

**Processing Script**

The `process_and_index.py` script processes the extracted Wikipedia JSON files and performs the following operations:
- Reads JSON files from `${WIKI_DATA}/extracted/wikipedia_batch_*.json`
- Splits article text into logical sections based on markdown headers
- Stores articles and sections in PostgreSQL database
- Generates 768-dimensional embeddings using the configured provider
- Creates an OpenSearch index with k-NN vector search capabilities (HNSW algorithm)
- Bulk indexes documents with text and embeddings for hybrid search (BM25 + semantic)
- Provides detailed logging and progress tracking

**Implementation:** See the complete script in `process_and_index.py` provided alongside this documentation.

To copy the script to the server:
```bash
# Copy the script to the scripts directory
sudo cp process_and_index.py ${WIKI_DATA}/scripts/
sudo chown wiki:wiki ${WIKI_DATA}/scripts/process_and_index.py
sudo chmod +x ${WIKI_DATA}/scripts/process_and_index.py
```

**Test Mode (Recommended Before Full Processing)**

Before running the full processing, use test mode to verify all connections and functionality:

```bash
python3 ${WIKI_DATA}/scripts/process_and_index.py --test
```

The test mode performs the following checks:
1. **PostgreSQL**: Verifies connection and schema (tables, pg_trgm extension)
2. **OpenSearch**: Verifies connection, cluster health, and k-NN plugin
3. **Embedding Provider**: Tests LM Studio API or local model loading
4. **Full Pipeline**: Inserts dummy article, generates embedding, indexes to OpenSearch, performs k-NN search
5. **Cleanup**: Removes all test data automatically

**Expected output for successful tests:**
```
============================================================
TEST RESULTS SUMMARY
============================================================
  ✓ PASS: PostgreSQL Connection
  ✓ PASS: OpenSearch Connection
  ✓ PASS: Embedding Provider (lmstudio)
  ✓ PASS: Full Pipeline Test
============================================================
All tests PASSED. System is ready for full processing.
============================================================
```

**Troubleshooting test failures:**
- PostgreSQL: Check service status, verify schema.sql was applied
- OpenSearch: Check service status, verify k-NN plugin enabled
- LM Studio: Verify server is running, correct IP, model is loaded
- Local: Ensure sentence-transformers is installed

You can also override the embedding provider for testing:
```bash
# Test with LM Studio
python3 ${WIKI_DATA}/scripts/process_and_index.py --test --provider lmstudio

# Test with local model (requires sentence-transformers, see note below)
python3 ${WIKI_DATA}/scripts/process_and_index.py --test --provider local
```

**Note:** Testing with `--provider local` requires `sentence-transformers` to be installed. If you followed Option A (LM Studio setup), you'll need to install it first:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers==3.2.0
```

**Run Full Processing**

Once tests pass, run the full processing:

```bash
python3 ${WIKI_DATA}/scripts/process_and_index.py
```

**Expected Processing Times:**

```
LMStudio on amdgpu version: Linuxver ROCm version: 7.1.1
2025-12-08 17:15:40,722 - INFO - This session: 6,788,271 articles in 47.33 hours
2025-12-08 17:15:40,722 - INFO - Average rate: 39.8 articles/sec
```


### Phase 7: Setup MCP Server

1. Install FastAPI dependencies:
```bash
sudo -iu wiki
```

then
```bash
source ${WIKI_DATA}/venv/bin/activate
pip install fastapi uvicorn[standard] pydantic
```

2. Create MCP server (`${WIKI_DATA}/scripts/mcp_server.py`):

The MCP server provides the following endpoints:
- `GET /health` - Health check for all backend services
- `POST /mcp/search` - Search with keyword, semantic, or hybrid mode
- `GET /mcp/article/{id}` - Retrieve article by ID
- `GET /mcp/article?title=...` - Retrieve article by title
- `GET /sse` - Server-Sent Events endpoint for VS Code Copilot MCP integration
- `POST /messages` - MCP message handler for SSE transport

**MCP Tools (exposed to VS Code Copilot):**

| Tool | Description |
|------|-------------|
| `search_wikipedia` | Search articles using keyword, semantic, or hybrid mode |
| `get_article` | Retrieve full article by title |
| `get_article_by_id` | Retrieve full article by database ID |
| `health_check` | Check server and backend service status |

**Implementation:** See the complete script in `mcp_server.py` provided alongside this documentation.

To copy the script to the server:
```bash
# Copy the script to the scripts directory
sudo cp mcp_server.py ${WIKI_DATA}/scripts/
sudo chown wiki:wiki ${WIKI_DATA}/scripts/mcp_server.py
sudo chmod +x ${WIKI_DATA}/scripts/mcp_server.py
```

**Configuration via Environment Variables:**

The server reads configuration from environment variables (with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_DATA` | `/mnt/data/wikipedia` | Base data directory |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_USER` | `wiki` | PostgreSQL username |
| `PG_PASSWORD` | `wikipass` | PostgreSQL password |
| `PG_DATABASE` | `wikidb` | PostgreSQL database |
| `OS_HOST` | `localhost` | OpenSearch host |
| `OS_PORT` | `9200` | OpenSearch port |
| `OS_INDEX` | `wikipedia` | OpenSearch index name |
| `EMBEDDING_PROVIDER` | `lmstudio` | Embedding provider (`lmstudio` or `local`) |
| `LMSTUDIO_HOST` | `localhost` | LM Studio server host |
| `LMSTUDIO_PORT` | `1234` | LM Studio server port |

**Test the server locally before creating the systemd service:**
```bash
source ${WIKI_DATA}/venv/bin/activate
cd ${WIKI_DATA}/scripts
python3 mcp_server.py
# Server starts on http://0.0.0.0:7000
# Press Ctrl+C to stop
```

Verify with:
```bash
curl http://localhost:7000/health
```

3. Create systemd service (`/etc/systemd/system/mcp.service`):

**Note:** Edit the service file and replace `/mnt/data/wikipedia` with your actual `WIKI_DATA` path before copying.

Copy the service file from the `/services` folder:
```bash
sudo cp /path/to/services/mcp.service /etc/systemd/system/mcp.service
sudo nano /etc/systemd/system/mcp.service  # Edit paths as needed
```

**Service file:** See `mcp.service` in the `/services` folder alongside this documentation.

4. Start MCP server:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mcp
```

### Phase 8: Setup Web GUI

This phase creates a React-based search interface for browsing and searching Wikipedia content through the MCP server API.

1. Install Node.js:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

2. Create React app:
```bash
sudo -iu wiki
cd ${WIKI_DATA}
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

3. Copy the application files from the `/webapp` folder:

The web application source files are provided in the `/webapp` folder alongside this documentation:

| File | Description | Target Location |
|------|-------------|-----------------|
| `App.jsx` | Main React application component | `${WIKI_DATA}/frontend/src/App.jsx` |
| `App.css` | Application styles | `${WIKI_DATA}/frontend/src/App.css` |
| `vite.config.js` | Vite build configuration | `${WIKI_DATA}/frontend/vite.config.js` |

Copy these files to the frontend directory:
```bash
# Replace the default App files with our custom implementation
cp /path/to/webapp/App.jsx ${WIKI_DATA}/frontend/src/App.jsx
cp /path/to/webapp/App.css ${WIKI_DATA}/frontend/src/App.css
cp /path/to/webapp/vite.config.js ${WIKI_DATA}/frontend/vite.config.js

# Set ownership
sudo chown -R wiki:wiki ${WIKI_DATA}/frontend
```

**Application Features:**
- **Search Interface**: Full-text search with three modes (Keyword/BM25, Semantic, Hybrid)
- **Health Status**: Real-time server connectivity indicator
- **Results Display**: Clickable results with title, section, excerpt, and relevance score
- **Article Viewer**: Full article display with table of contents and section navigation
- **Wikipedia Links**: Direct links to original Wikipedia articles
- **Responsive Design**: Works on desktop and mobile devices

4. Build the application:
```bash
cd ${WIKI_DATA}/frontend

# Stop the service if it's running (required before rebuilding)
sudo systemctl stop wiki-gui 2>/dev/null || true

# Clean previous build (if rebuilding)
rm -rf dist node_modules/.vite

# Install dependencies (required after copying new config files)
npm install

# Build for production
npm run build
```

5. Create systemd service for the web GUI (`/etc/systemd/system/wiki-gui.service`):

**Note:** Edit the service file and replace `/mnt/data/wikipedia` with your actual `WIKI_DATA` path before copying.

Copy the service file from the `/services` folder:
```bash
sudo cp /path/to/services/wiki-gui.service /etc/systemd/system/wiki-gui.service
sudo nano /etc/systemd/system/wiki-gui.service  # Edit paths as needed
```

**Service file:** See `wiki-gui.service` in the `/services` folder alongside this documentation.

6. Start the web GUI service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now wiki-gui
```

7. Verify the web GUI is running:
```bash
# Check service status
sudo systemctl status wiki-gui

# Test from command line
curl -s http://localhost:8080 | head -20
```

**Installation Complete!**

The system is now ready. Services running:
- PostgreSQL on localhost:5432
- OpenSearch on localhost:9200
- MCP Server on 0.0.0.0:7000
- Web GUI on 0.0.0.0:8080

### Validation

#### Prerequisites

Before testing, ensure:
- All services are running: `systemctl status opensearch mcp postgresql`
- Data has been indexed and embeddings loaded
- Firewall rules allow access from your workstation

#### 1. Web GUI Test

Access the web interface:
```
http://$HOST:8080
```

**Test Scenarios:**
- Search "Einstein" → Should return "Albert Einstein" article
- Search "physicist relativity" (semantic) → "Albert Einstein" should rank highly
- Open an article → Verify sections display correctly

#### 2. MCP Server Test

**Health Check:**
```bash
curl http://$HOST:7000/health
```

**Keyword Search Test:**

*Linux/macOS:*
```bash
curl -X POST http://$HOST:7000/mcp/search \
  -H "Content-Type: application/json" \
  -d '{"query":"quantum mechanics","mode":"keyword"}'
```

*Windows (PowerShell/cmd):*
```powershell
curl -X POST http://$HOST:7000/mcp/search -H "Content-Type: application/json" -d "{\"query\":\"quantum mechanics\",\"mode\":\"keyword\"}"
```

**Semantic Search Test:**

*Linux/macOS:*
```bash
curl -X POST http://$HOST:7000/mcp/search \
  -H "Content-Type: application/json" \
  -d '{"query":"theory of gravity","mode":"semantic"}'
```

*Windows (PowerShell/cmd):*
```powershell
curl -X POST http://$HOST:7000/mcp/search -H "Content-Type: application/json" -d "{\"query\":\"theory of gravity\",\"mode\":\"semantic\"}"
```

#### 3. VS Code Copilot Integration

##### Step 1: Configure MCP Server in VS Code

VS Code uses a dedicated MCP configuration file, not `settings.json`. 

1. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS)
2. Type "MCP: Open User Configuration" and select it
3. This opens the MCP configuration file (typically `~/.vscode/mcp.json` or similar)
4. Add your Wikipedia server configuration:

```json
{
  "servers": {
    "wikipedia": {
      "type": "sse",
      "url": "http://$HOST:7000/sse"
    }
  }
}
```

**Note:** Replace `$HOST` with your actual server IP address (e.g., `192.168.1.100`).

**Alternative: Workspace Configuration**

For project-specific configuration, create a `.vscode/mcp.json` file in your workspace:

```json
{
  "servers": {
    "wikipedia": {
      "type": "sse", 
      "url": "http://$HOST:7000/sse"
    }
  }
}
```

**Important:** The MCP server must implement the Server-Sent Events (SSE) transport. The `mcp_server.py` includes the required `/sse` endpoint for VS Code Copilot integration. See Phase 7 for the server implementation.

##### Step 2: Restart VS Code

Close and reopen VS Code to activate the MCP server connection.

##### Step 3: Verify Connection

Open the Copilot Chat panel (click the Copilot icon in the sidebar or use `Ctrl+Alt+I`). 

You should see "wikipedia" listed as an available MCP server in the chat. You can verify the connection by checking the MCP server logs on your server:

```bash
sudo journalctl -u mcp -f
```

You should see log entries like:
```
MCP SSE session started: <session-id>
MCP [<session-id>] Received: initialize
MCP [<session-id>] Received: tools/list
```

##### Step 4: Test Search

In the Copilot Chat, you can now use the Wikipedia tools. Try a search query:

```
Search Wikipedia for causes of the French Revolution
```

Copilot will use the `search_wikipedia` tool and return relevant article excerpts.

**Expected Response:**
- List of relevant Wikipedia articles
- Article titles with brief excerpts
- Relevance scores

##### Step 5: Test Semantic Search

Try a concept-based search:

```
Find Wikipedia articles about the relationship between gravity and spacetime
```

**Expected Response:**  
Articles about General Relativity, Einstein, and related physics topics.

##### Step 6: Test Article Retrieval

Request a full article:

```
Get the full Wikipedia article on French Revolution
```

**Expected Response:**
- Complete article text
- Organized by sections
- Table of contents

**Troubleshooting VS Code Integration:**

If the MCP server doesn't appear in Copilot Chat:

1. **Check VS Code version**: MCP support requires VS Code 1.96+ with GitHub Copilot extension
2. **Verify settings.json**: Ensure the `mcp.servers` configuration is at the root level
3. **Check server connectivity**: Run `curl http://YOUR_SERVER_IP:7000/sse` - you should see SSE events
4. **Check firewall**: Ensure port 7000 is accessible from your workstation
5. **Review server logs**: `sudo journalctl -u mcp -n 50` for error messages


## References

**Data & Parsing:**
- Wikipedia Dumps Portal: https://dumps.wikimedia.org/
- mediawiki-dump (Python library): https://github.com/macbre/mediawiki-dump
- mwparserfromhell (Wikitext parser): https://github.com/earwig/mwparserfromhell

**Search & Storage:**
- OpenSearch Project: https://opensearch.org/
- OpenSearch k-NN Plugin: https://opensearch.org/docs/latest/search-plugins/knn/
- PostgreSQL Documentation: https://www.postgresql.org/docs/
- pg_trgm Extension: https://www.postgresql.org/docs/current/pgtrgm.html

**Embeddings:**
- Sentence Transformers: https://www.sbert.net/

**MCP Server:**
- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- Model Context Protocol (MCP) Spec: https://github.com/modelcontextprotocol/spec

**Web Interface:**
- Vite: https://vitejs.dev/
- React: https://react.dev/
