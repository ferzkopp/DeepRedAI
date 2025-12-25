# Wikidata Time Metadata Parser

A Python script that efficiently parses Wikidata TTL (Turtle) files to extract time-related metadata and Wikipedia article references.

## Overview

Wikidata provides comprehensive time-related information for entities, including birth dates, death dates, founding dates, and other temporal properties. This parser extracts that data and links it to English Wikipedia articles, producing output compatible with the YAGO normalized format for downstream temporal augmentation.

## Features

✅ **Memory-efficient**: Streams large TTL files line by line  
✅ **Wikipedia linking**: Extracts English Wikipedia article references  
✅ **Multiple export formats**: CSV and JSON output  
✅ **Temporal data extraction**: Captures various date properties  
✅ **Compatible output**: Matches YAGO normalized format for pipeline integration

## Requirements

- Python 3.7 or higher
- ~100 GB free disk space for compressed file
- ~900 GB additional space for extraction
- No external dependencies (uses only Python standard library)

## Installation and Operation

**Note:** All Wikidata data (dumps and extracted files) is stored under the `${WIKI_DATA}` path. 
Only the Ubuntu OS, packages and additional software resides on the system drive.

### Phase 1: System Preparation

1. Set environment configuration variables:
```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"
```

### Phase 2: Download Wikidata Dump

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

Create directory and download the data:

```bash
mkdir -p ${WIKI_DATA}/wikidata
cd ${WIKI_DATA}/wikidata
wget -c --timeout=60 --tries=10 https://dumps.wikimedia.org/wikidatawiki/entities/20251215/wikidata-20251215-all-BETA.ttl.bz2
```

**Important:** This download is ~110 GB and may take several hours depending on connection speed. The `wget -c` flag allows resuming if interrupted.

### Phase 3: Extract Wikidata Dump

Extract the TTL file from bz2 compression:

```bash
cd ${WIKI_DATA}/wikidata
bunzip2 -k wikidata-20251215-all-BETA.ttl.bz2
```

The extraction process will take several hours depending on disk speed. The extracted TTL file will be approximately 900+ GB.

**Note:** The `-k` flag keeps the original compressed file. Remove it if disk space is limited.

Verify the extraction:
```bash
ls -lh ${WIKI_DATA}/wikidata
# Should show both files:
# -rw-r--r-- 1 wiki wiki  85G Dec 23  2025 latest-all.ttl.bz2
# -rw-r--r-- 1 wiki wiki 210G Dec 23  2025 latest-all.ttl
```

### Phase 4: Review Content Structure

Before parsing, examine the TTL file structure to understand the data format:

```bash
# View first 500 lines to see RDF format and structure
head -n 500 ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl | less

# Search for time-related predicates
grep -m 10 "P569" ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl  # birth dates
grep -m 10 "P570" ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl  # death dates
grep -m 10 "P571" ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl  # inception dates

# Search for Wikipedia article links (sitelinks)
grep -m 10 "schema:about" ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl
grep -m 10 "enwiki" ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl
```

---

## Phase 5: Parse Wikidata for Temporal Metadata

Copy the `wikidata_parser.py` script to the `${WIKI_DATA}/scripts/` directory.

**Script Features:**
- ✅ Memory-efficient streaming parser (handles 800+ GB TTL files)
- ✅ Extracts 4 temporal properties:
  - P569: date of birth
  - P570: date of death
  - P571: inception (founding, establishment)
  - P576: dissolved, abolished or demolished date
- ✅ Links entities to English Wikipedia articles
- ✅ Compatible CSV/JSON output format (matches YAGO pipeline)
- ✅ Progress tracking for long-running operations
- ✅ Filters to Wikipedia-linked entities by default

**Usage Examples:**

```bash
# Switch to wiki user
sudo -iu wiki
cd ${WIKI_DATA}/scripts

# Parse and show summary
python3 wikidata_parser.py \
    ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl \
    --summary --verbose

# Export to CSV (Wikipedia entities only - default)
python3 wikidata_parser.py \
    ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl \
    --csv ${WIKI_DATA}/wikidata/wikidata-temporal.csv \
    --verbose

# Export both CSV and JSON formats
python3 wikidata_parser.py \
    ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl \
    --csv ${WIKI_DATA}/wikidata/wikidata-temporal.csv \
    --json ${WIKI_DATA}/wikidata/wikidata-temporal.json \
    --verbose

# Include ALL entities (even without Wikipedia links)
python3 wikidata_parser.py \
    ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl \
    --csv ${WIKI_DATA}/wikidata/wikidata-temporal-all.csv \
    --all-entities \
    --verbose
```

**Expected Output Format (CSV):**

```csv
Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
Q23,George Washington,https://en.wikipedia.org/wiki/George_Washington,1732-02-22,1799-12-14
Q24,Jack Benny,https://en.wikipedia.org/wiki/Jack_Benny,1966-02-18,2001-05-11
Q31,Belgium,https://en.wikipedia.org/wiki/Belgium,1830-10-04,1830-10-04
Q42,Douglas Adams,https://en.wikipedia.org/wiki/Douglas_Adams,2001-05-11,2001-05-11
```

**Performance Notes:**

- Processing the full ~900 GB TTL file will take several hours (estimate: 3-6 hours)
- Progress updates every 1 million lines
- Memory usage remains constant (streaming parser)
- Output file size will depend on number of entities with temporal data and Wikipedia links

## Integration with Existing Pipeline:

The Wikidata parser output is in **pre-normalized format** and should be processed through the normalization pipeline:

```bash
# Step 1: Parse Wikidata (done above)
python3 wikidata_parser.py \
    ${WIKI_DATA}/wikidata/wikidata-20251215-all-BETA.ttl \
    --csv ${WIKI_DATA}/wikidata/wikidata-temporal.csv \
    --verbose

# Step 2: Normalize to add Wikipedia page IDs
python3 normalize_yago_output.py \
    ${WIKI_DATA}/wikidata/wikidata-temporal.csv \
    --output ${WIKI_DATA}/wikidata/wikidata-temporal-normalized.csv \
    --verbose

# Step 3: Use normalized data in temporal augmentation
python3 augment_wikipedia_temporal.py \
    ${WIKI_DATA}/wikidata/wikidata-temporal-normalized.csv \
    --verbose
```

**Format Comparison:**

*Wikidata Parser Output (Pre-normalized):*
```csv
Entity_ID,Entity,Wikipedia_URL,Earliest_Date,Latest_Date
Q23,George_Washington,https://en.wikipedia.org/wiki/George_Washington,1732-02-22,1799-12-14
```

*After Normalization (with page IDs):*
```csv
Entity,Wikipedia_Title,Wikipedia_ID,Wikipedia_URL,Earliest_Date,Latest_Date,Original_URL
George_Washington,George Washington,11968,https://en.wikipedia.org/wiki?curid=11968,1732-02-22,1799-12-14,https://en.wikipedia.org/wiki/George_Washington
```

The normalizer adds:
1. **Wikipedia_ID** - Page ID from local PostgreSQL database
2. **curid format URLs** - Direct database reference format
3. **Original_URL** - Preserves the original URL for reference
4. Validates that articles exist in local database

## References

- **Dump Format Documentation**: https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format
- **Wikidata Dump Downloads**: https://dumps.wikimedia.org/wikidatawiki/entities/
- **Property Reference**: https://www.wikidata.org/wiki/Wikidata:List_of_properties