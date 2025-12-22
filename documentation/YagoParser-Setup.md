# YAGO Time Metadata Parser

A Python script that efficiently parses YAGO knowledge base TTL (Turtle) files to extract time-related metadata without loading the entire file into memory.

## Overview

This parser extracts time-related information from YAGO knowledge base files, including:
- Birth dates (`schema:birthDate`)
- Death dates (`schema:deathDate`)
- Start dates (`schema:startDate`)
- End dates (`schema:endDate`)
- Publication dates (`schema:datePublished`)

For each entity, the parser finds the earliest and latest dates across all time properties, making it easy to aggregate temporal information.

## Features

✅ **Memory-efficient**: Streams large TTL files line by line  
✅ **Fast parsing**: Processes millions of triples efficiently  
✅ **Unicode decoding**: Converts encoded entity names (e.g., `__u0028_` → `(`) for easy searching  
✅ **Multiple export formats**: CSV and JSON output  
✅ **Wikipedia linking**: Extracts Wikipedia URLs when available  
✅ **Progress tracking**: Verbose mode shows real-time parsing progress  
✅ **Flexible output**: Configurable result limits and formats

## Requirements

- Python 3.7 or higher
- No external dependencies (uses only Python standard library)

## Installation

**Note:** All Yago data (dumps, extracted text, OpenSearch index, and PostgreSQL database) is stored under the `${WIKI_DATA}` path (TBD GB). 
Only the Ubuntu OS, packages and additional software resides on the system drive.

**Phase 1: System Preparation**

1. Set environment configuration variables:
```bash
# Storage path - adjust to your disk mount point
export WIKI_DATA="/mnt/data/wikipedia"
```

**Phase 2: Download Yago Data**

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

Then download the data:

```bash
mkdir -p ${WIKI_DATA}/yago
cd ${WIKI_DATA}/yago
wget -c --timeout=60 --tries=10 https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip
```

This download is ~12 GB and takes 15-30 min. If interrupted, simply re-run the same command to resume.

**Phase 3: Unzip Yago Data**

Extract only the yago-facts.ttl file (contains Wikipedia entity time data):

```bash
cd ${WIKI_DATA}/yago
unzip yago-4.5.0.2.zip yago-facts.ttl
```

The unzip process may take 5-10 minutes depending on disk speed. The extracted TTL files will be several gigabytes in size.

Verify the extraction:
```bash
ls -lh ${WIKI_DATA}/yago
# Should show .ttl files:
# -rw-r--r-- 1 wiki wiki  12G Apr  9  2024 yago-4.5.0.2.zip
# -rw-r--r-- 1 wiki wiki  22G Apr  4  2024 yago-facts.ttl
```

## Extraction

**Phase 4: Extract Yago Data**

Create ```${WIKI_DATA}/scripts/yago_parser.py``` (see scripts).

Run export to CSV:

```bash
python3 ${WIKI_DATA}/scripts/yago_parser.py ${WIKI_DATA}/yago/yago-facts.ttl --csv ${WIKI_DATA}/yago/yago-facts.csv --verbose
```

Run export to JSON:

```bash
python3 ${WIKI_DATA}/scripts/yago_parser.py ${WIKI_DATA}/yago/yago-facts.ttl --json ${WIKI_DATA}/yago/yago-facts.json --verbose
```

## yago_parser.py

### Command-Line Options

| Option | Description |
|--------|-------------|
| `ttl_file` | Path to the YAGO TTL file (required) |
| `--csv FILE` | Export results to CSV file |
| `--json FILE` | Export results to JSON file |
| `--verbose`, `-v` | Show parsing progress |
| `--limit N` | Display N entities in summary (default: 20) |
| `--no-summary` | Skip console summary output |

### Output Format

#### CSV Format

```csv
Entity,Wikipedia_URL,Earliest_Date,Latest_Date
A-1_(wrestler),https://en.wikipedia.org/wiki/A-1_(wrestler),1977-05-22,1977-05-22
Augusto_Pinochet,https://en.wikipedia.org/wiki/Augusto_Pinochet,1915-11-25,2006-12-10
Andrei_Tarkovsky,https://en.wikipedia.org/wiki/Andrei_Tarkovsky,1932-04-04,1986-12-29
```

**Note**: Entity names are automatically decoded from YAGO's Unicode encoding format for easy searching. For example, `A-1__u0028_wrestler_u0029_` becomes `A-1_(wrestler)`, preserving underscores that are part of the Wikipedia article name.

#### JSON Format

```json
[
  {
    "entity": "Augusto_Pinochet",
    "wikipedia_url": "https://en.wikipedia.org/wiki/Augusto_Pinochet",
    "earliest_date": "1915-11-25",
    "latest_date": "2006-12-10"
  },
  {
    "entity": "Andrei_Tarkovsky",
    "wikipedia_url": "https://en.wikipedia.org/wiki/Andrei_Tarkovsky",
    "earliest_date": "1932-04-04",
    "latest_date": "1986-12-29"
  }
]
```

### Python API

You can also use the parser programmatically in your Python code:

```python
from yago_parser import YagoTimeExtractor

# Create parser
extractor = YagoTimeExtractor('yago-tiny.ttl')

# Parse the file
extractor.parse_file(verbose=True)

# Get results
results = extractor.get_results()
for entity, wiki_url, earliest, latest in results:
    print(f"{entity}: {earliest} to {latest}")

# Export
extractor.export_csv('output.csv')
extractor.export_json('output.json')

# Print summary
extractor.print_summary(limit=30)
```

## References

- [YAGO Knowledge Base](https://yago-knowledge.org/)
- [Schema.org Vocabulary](https://schema.org/)
- [RDF/Turtle Format](https://www.w3.org/TR/turtle/)