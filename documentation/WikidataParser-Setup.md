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
- ~200 GB additional space for extraction
- No external dependencies (uses only Python standard library)

## Installation

**Note:** All Wikidata data (dumps and extracted files) is stored under the `${WIKI_DATA}` path. 
Only the Ubuntu OS, packages and additional software resides on the system drive.

**Phase 1: System Preparation**

1. Set environment configuration variables:
```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"
```

**Phase 2: Download Wikidata Dump**

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

**Phase 3: Extract Wikidata Dump**

Extract the TTL file from bz2 compression:

```bash
cd ${WIKI_DATA}/wikidata
bunzip2 -k latest-all.ttl.bz2
```

The extraction process will take 1-2 hours depending on disk speed. The extracted TTL file will be approximately 200+ GB.

**Note:** The `-k` flag keeps the original compressed file. Remove it if disk space is limited.

Verify the extraction:
```bash
ls -lh ${WIKI_DATA}/wikidata
# Should show both files:
# -rw-r--r-- 1 wiki wiki  85G Dec 23  2025 latest-all.ttl.bz2
# -rw-r--r-- 1 wiki wiki 210G Dec 23  2025 latest-all.ttl
```

**Phase 4: Review Content Structure**

Before parsing, examine the TTL file structure to understand the data format:

```bash
# View first 500 lines to see RDF format and structure
head -n 500 ${WIKI_DATA}/wikidata/latest-all.ttl | less

# Search for time-related predicates
grep -m 10 "P569" ${WIKI_DATA}/wikidata/latest-all.ttl  # birth dates
grep -m 10 "P570" ${WIKI_DATA}/wikidata/latest-all.ttl  # death dates
grep -m 10 "P571" ${WIKI_DATA}/wikidata/latest-all.ttl  # inception dates

# Search for Wikipedia article links (sitelinks)
grep -m 10 "schema:about" ${WIKI_DATA}/wikidata/latest-all.ttl
grep -m 10 "enwiki" ${WIKI_DATA}/wikidata/latest-all.ttl
```

## References

- **Dump Format Documentation**: https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format
- **Wikidata Dump Downloads**: https://dumps.wikimedia.org/wikidatawiki/entities/
- **Property Reference**: https://www.wikidata.org/wiki/Wikidata:List_of_properties

## Next Steps

After reviewing the content structure, the next phase will be to:
1. Develop the `wikidata_parser.py` script based on observed RDF format
2. Extract temporal properties (birth, death, inception, dissolution dates)
3. Link entities to English Wikipedia articles
4. Output in normalized format compatible with YAGO pipeline
5. Integrate with temporal augmentation workflow

Refer to the [YagoParser-Setup.md](YagoParser-Setup.md) for reference implementation patterns.

