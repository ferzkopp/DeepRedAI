
# Deep Red AI

This project is inspired by the [Deep Red](https://www.deepredfilm.com) film trilogy from the creators of "Iron Sky". It aims to create a custom LLM model 
that has only knowledge up to July 1969 - the launch date of the fictional Mars mission in the movie - and responds in a style that aligns with 
a "Soviet utopia" of the fictional AI.

## Steps/Instructions

- [How to setup an AMD "Strix Halo" device](documentation/StrixHalo-Ubuntu-Setup.md)
- [How to setup LMStudio as server for "headless" operation](documentation/LMStudio-Setup.md)
- [How to create a Wikipedia database, enable vector database searches for articles, and provide an MCP server for the data](documentation/WikipediaMCP-Setup.md)
- [How to extract temporal information from YAGO and parse Wikipedia articles](documentation/YagoParser-Setup.md)
- [How to normalize YAGO output to match local English Wikipedia database](documentation/YagoNormalizer-Setup.md)
- [How to augment the Wikipedia database with temporal information from YAGO](documentation/TemporalAugmentation-Setup.md)
- How to fine-tune an existing LLM model with a *Temporal Knowledge Cutoff*, restating its knowledge base into the past 
  - [Phased Plan](documentation/Temporal-Finetuning-Plan.md)
  - [How to generate training datasets](documentation/DataPreparation-Phase1.md)
- How to retrieve and prepare additional training data for "theme"" fine-tuning (TBD)
- How to fine-tune the modified LLM model further so it aligns better with the films stated theme (TBD)

## Repo Content

- **`/documentation`** - Setup guides and planning documents for the project
- **`/scripts`** - Python scripts for Wikipedia extraction, processing, indexing, YAGO parsing, and the MCP server
- **`/services`** - Systemd service files for automated startup (LMStudio, MCP server, OpenSearch, web GUI)
- **`/webapp`** - React-based web interface for Wikipedia search with Vite configuration
- **`/notebooks`** - Jupyter notebooks for testing embeddings and OpenSearch functionality
- **`/patches`** - System patches (network driver fix for AMD Strix Halo)

