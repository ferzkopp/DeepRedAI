# Deep Red AI

This project is inspired by the [Deep Red](https://www.deepredfilm.com) film trilogy from the creators of "Iron Sky". It aims to create a custom LLM model that has only knowledge up to July 1969 - the launch date of the fictional Mars mission in the movie - and responds in a style that aligns with 
a "Soviet utopia" setting controlled by the fictional "chess playing" AI.

## Steps/Instructions

These steps document how to perform a [Model Training from Scratch](documentation/ModelTraining.md) procedure using a continued pre-training approach for temporal and thematic alignment.
 
- [How to setup an AMD "Strix Halo" device (Fedora)](documentation/StrixHalo-Fedora-Setup.md) — primary development and training system; includes automated setup script
- [How to setup an optional second NVIDIA device (A4000)](documentation/A4000-Fedora-Setup.md) — dedicated GPU for training/inference; includes automated setup script
- [How to set up the Wikipedia MCP server and data pipeline](documentation/WikipediaMCP-Setup.md) — extraction, indexing, search, and MCP server for Wikipedia content
- [How to extract year-based historical topics from Wikipedia](documentation/Wikipedia-YearTopics-Setup.md) — enriched event data for temporal training
- [How to augment Wikipedia with temporal metadata](documentation/TemporalAugmentation-Setup.md) — YAGO/Wikidata parsing, normalization, and database augmentation for time-period filtering
- [How to retrieve Project Gutenberg literature](documentation/Gutenberg-Setup.md) — thematically relevant books for training data
- [How to prepare the chess training corpus](documentation/Chess-Setup.md) — chess content retrieval, PGN conversion, and corpus preparation 
- [How to augment the chess game corpus](documentation/ChessAugmentation-Setup.md) - corpus augmentation to create custom chess-game content for the Deep Red persona
- [How to tokenize and prepare the training corpus](documentation/TrainingCorpus-Setup.md) — tokenization, shuffling, and train/val splitting for continued pre-training
- [How to train the Deep Red model](documentation/DeepRedModel-Setup.md) — continued pre-training on the temporally-filtered pre-1969 corpus using dev (SmolLM2-360M) or prod (TinyLlama-1.1B) profiles
- [How to train Deep Red on Gemma-3 (SFT)](documentation/DeepRedGemma-Setup.md) — parallel supervised fine-tuning track using Gemma-3-4B-IT / 12B-IT via TRL `SFTTrainer`, mirroring the [kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) setup
- [How to use the generated GGUF models with LM Studio](documentation/ModelUsage.md) — loading, testing, and comparing trained model checkpoints in LM Studio (**sample models linked here**)

## Downloadable Chess Augmentation Archives

Prebuilt chess augmentation archives can be downloaded directly from:

- `https://www.ferzkopp.net/Data/chess_games.jsonl.gz`
- `https://www.ferzkopp.net/Data/augmented_chess_games.jsonl.gz`

Each URL uses the pattern `https://www.ferzkopp.net/Data/[filename]`.
For augmentation workflow details, see the full guide:
[documentation/ChessAugmentation-Setup.md](documentation/ChessAugmentation-Setup.md).

## Downloadable Models

- [DeepRed-gemma-4b-2026-05-23-5-final.gguf](http://www.ferzkopp.net/Data/DeepRed-gemma-4b-2026-05-23-5-final.gguf) - see [corpus and model details](documentation/DeepRed-gemma-4b-2026-05-23-5.md)
- [gemma-4b-balanced-v1-small-1500-final.gguf](http://www.ferzkopp.net/Data/gemma-4b-balanced-v1-small-1500-final.gguf) - see [balanced run details and examples](documentation/DeepRed-gemma-4b-2026-06-13.md)
- [gemma-4b-temporal-v1-10d-final.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-final.gguf) - temporal-cutoff run (reintroduces `retain`/`unlearn` at the 1969-07-20 cutoff); see [run details and examples](documentation/DeepRed-gemma-4b-2026-06-14.md). Full 10-day follow-up calibrated from this run: [2026-06-17 runbook](documentation/DeepRed-gemma-4b-2026-06-17.md)

## Legacy Steps/Instructions

These steps document a failed fine-tuning approach to modify an existing model with "temporal knowledge cutoff" and "theme alignment".

- [How to setup an AMD "Strix Halo" device (Ubuntu, legacy)](documentation/legacy/StrixHalo-Ubuntu-Setup.md)
- [How to setup LMStudio as server for "headless" operation](documentation/legacy/LMStudio-Setup.md)
- [How to create a Wikipedia database, enable vector database searches for articles, and provide an MCP server for the data](documentation/legacy/WikipediaMCP-Setup.md)
- [How to extract temporal information from YAGO about Wikipedia articles](documentation/legacy/YagoParser-Setup.md)
- [How to normalize YAGO output to match local English Wikipedia database](documentation/legacy/YagoNormalizer-Setup.md)
- [How to extract temporal information from Wikidata](documentation/legacy/WikidataParser-Setup.md)
- [How to augment the Wikipedia database with temporal information](documentation/legacy/TemporalAugmentation-Setup.md)
- How to fine-tune an existing LLM model with a *Temporal Knowledge Cutoff*, restating its knowledge base into the past 
  - [Temporal Finetuning Phased Plan](documentation/legacy/TemporalFinetuning-Plan.md) - Content retrieval, analysis, and finetuning for temporal alignment
  - [How to generate training datasets](documentation/legacy/TemporalFinetuning-DataPreparation-Phase1.md)
  - [How to finetune model with temporal knowledge cutoff using these datasets](documentation/legacy/TemporalFinetuning-InitialFinetuning-Phase2.md)
- How to fine-tune the model further for "Soviet utopia" theme alignment
  - [Theme Finetuning Phased Plan](documentation/legacy/ThemeFinetuning-Plan.md) - Content retrieval, analysis, and finetuning for stylistic alignment
  - [Phase 1: How to retrieve training content from Project Gutenberg](documentation/legacy/ThemeFinetuning-DataPreparation-Phase1.md)
  - [Phase 2: How to chunk and filter content for theme alignment](documentation/legacy/ThemeFinetuning-DataPreparation-Phase2.md)
  - [Phase 3: How to generate ChatML training dataset from filtered content](documentation/legacy/ThemeFinetuning-DataPreparation-Phase3.md)
  - [Phase 4: How to finetune model with themed dataset](documentation/legacy/ThemeFinetuning-Phase4.md)

## Repo Content

- **`/documentation`** - Setup guides and planning documents for the project
- **`/scripts`** - Python scripts for Wikipedia extraction/indexing, temporal augmentation (YAGO/Wikidata), Gutenberg and chess content retrieval, MCP server, and system setup
- **`/services`** - Systemd service files for automated startup (inference servers, MCP server, OpenSearch, web GUI)
- **`/webapp`** - React-based web interface for Wikipedia search with Vite configuration
- **`/notebooks`** - Jupyter notebooks for testing embeddings and OpenSearch functionality
- **`/patches`** - System patches (network driver fix for AMD Strix Halo for older kernels)



