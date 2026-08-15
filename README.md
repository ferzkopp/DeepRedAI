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
- [How to use the generated GGUF models with LM Studio](documentation/ModelUsage.md) — loading, testing, and comparing trained model checkpoints in LM Studio 
- [How to evaluate checkpoints and plan recovery](documentation/DeepRed-gemma-4b-evaluation-and-recovery-plan.md) — independent 1969 probe bank, corpus contamination audit, GPU-accelerated trajectory evaluation across archived checkpoints, and the evidence-driven plan for temporal and persona training

### Downloadable Models

- Prototypes
  - [DeepRed-gemma-4b-2026-05-23-5-final.gguf](http://www.ferzkopp.net/Data/DeepRed-gemma-4b-2026-05-23-5-final.gguf) - see [corpus and model details](documentation/DeepRed-gemma-4b-2026-05-23-5.md)
  - [gemma-4b-balanced-v1-small-1500-final.gguf](http://www.ferzkopp.net/Data/gemma-4b-balanced-v1-small-1500-final.gguf) - see [balanced run details and examples](documentation/DeepRed-gemma-4b-2026-06-13.md)
  - [gemma-4b-temporal-v1-10d-final.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-final.gguf) - temporal-cutoff run (reintroduces `retain`/`unlearn` at the 1969-07-20 cutoff); see [run details and examples](documentation/DeepRed-gemma-4b-2026-06-14.md).
- Production
  - [gemma-4b-temporal-v1-10d-2-final.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-2-final.gguf) - Full 10-day follow-up calibrated from previous run: [2026-06-17 runbook](documentation/DeepRed-gemma-4b-2026-06-17.md) (2.32 GB)
  - Intermediate checkpoints (10%, 25%, 50%, 75%):
    - [gemma-4b-temporal-v1-10d-2-010pct-step-2560.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-2-010pct-step-2560.gguf) (2.32 GB)
    - [gemma-4b-temporal-v1-10d-2-025pct-step-6400.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-2-025pct-step-6400.gguf) (2.32 GB)
    - [gemma-4b-temporal-v1-10d-2-050pct-step-12800.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-2-050pct-step-12800.gguf) (2.32 GB)
    - [gemma-4b-temporal-v1-10d-2-075pct-step-19200.gguf](http://www.ferzkopp.net/Data/gemma-4b-temporal-v1-10d-2-075pct-step-19200.gguf) (2.32 GB)

> **Evaluation note:** a 1,377-generation evaluation across 17 archived checkpoints
> found that none of these models meets the project goal. The temporal checkpoints
> suppress modern facts only by refusing broadly — they also refuse many pre-1969
> questions — while the balanced and prototype checkpoints retain modern knowledge
> entirely. Results, measurements, and the revised training plan are in the
> [evaluation and recovery plan](documentation/DeepRed-gemma-4b-evaluation-and-recovery-plan.md).

### Downloadable Chess Augmentation Archives

Prebuilt chess augmentation archives can be downloaded directly from:

- `https://www.ferzkopp.net/Data/chess_games.jsonl.gz`
- `https://www.ferzkopp.net/Data/augmented_chess_games.jsonl.gz`

Each URL uses the pattern `https://www.ferzkopp.net/Data/[filename]`.
For augmentation workflow details, see the full guide:
[documentation/ChessAugmentation-Setup.md](documentation/ChessAugmentation-Setup.md).

## Repo Content

- **`/documentation`** - Setup guides and planning documents for the project
- **`/evaluation`** - Model registry and independent probe bank used by the 1969 evaluation harness
- **`/notebooks`** - Jupyter notebooks for testing embeddings and OpenSearch functionality
- **`/patches`** - System patches (network driver fix for AMD Strix Halo for older kernels)
- **`/scripts`** - Python scripts for Wikipedia extraction/indexing, temporal augmentation (YAGO/Wikidata), Gutenberg and chess content retrieval, MCP server, and system setup
- **`/services`** - Systemd service files for automated startup (inference servers, MCP server, OpenSearch, web GUI)
- **`/tests`** - Unit tests for the evaluation harness
- **`/webapp`** - React-based web interface for Wikipedia search with Vite configuration
