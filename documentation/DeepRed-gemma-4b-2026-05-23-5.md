# DeepRed-gemma-4b-2026-05-23-5-final.gguf

## Training Data

This model was trained with the chat-format SFT dataset at
`/mnt/data/sft_corpus/v1`, as recorded in
`/mnt/data/training_output/gemma-4b-2026-05-23-5/run_meta.json`.

The packed `train.bin` corpus used by the continued-pretraining path was not
the dataset used for this Gemma SFT run.

### Dataset Manifest

- **Dataset path:** `/mnt/data/sft_corpus/v1`
- **Manifest:** `/mnt/data/sft_corpus/v1/manifest.json`
- **Created:** 2026-05-22T06:24:26Z
- **Format:** JSONL chat examples with `messages` containing `user` and
  `assistant` turns
- **Builder:** `scripts/build_sft_dataset.py`
- **Max chars per message:** 4,096
- **Validation fraction:** 5%
- **Shuffle seed:** 42

### Dataset Splits

| Split | Examples |
|-------|---------:|
| Train | 320,607 |
| Validation | 16,874 |
| **Total** | **337,481** |

### Source Composition

| Source | Examples | Share |
|--------|---------:|------:|
| year_topics | 1,788 | 0.5% |
| gutenberg | 763 | 0.2% |
| augmented_chess_games | 334,920 | 99.2% |
| chess_books | 10 | 0.0% |
| **Total** | **337,481** | **100.0%** |

### Sources Used

#### year_topics

- Year-by-year historical event summaries from JSON files
- Prompt style: historical event questions such as `What were the notable
  events of the year {year}?`
- Examples used: 1,788

#### gutenberg

- Project Gutenberg text chunks from `/mnt/data/gutenberg/corpus/gutenberg_corpus.jsonl`
- Prompt style: passage continuation
- Examples used: 763

#### augmented_chess_games

- LLM-augmented chess game narratives from
  `/mnt/data/chess/corpus/augmented_chess_games.jsonl`
- Prompt style: `Narrate the following chess game...`
- Examples used: 334,920

#### chess_books

- Internet Archive chess reference book chunks from
  `/mnt/data/chess/corpus/chess_archive_books.jsonl`
- Prompt style: passage continuation
- Examples used: 10

### Sources Not Used In This SFT Dataset

| Source | Note |
|--------|------|
| wikipedia_articles | Not included in `/mnt/data/sft_corpus/v1` |
| chess_games | Raw PGN source not included in `/mnt/data/sft_corpus/v1` |

## DeepRed SFT Run Summary

- **Status:** completed
- **Started:** 2026-05-23T05:54:09.443955
- **Completed:** 2026-06-06T04:51:43.004128
- **Training time:** 334.95 h
- **Peak GPU memory:** 55.64 GB

### Source Model

- **Profile:** gemma-4b
- **Base model:** google/gemma-3-4b-it
- **Training mode:** full

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Batch size | 4 |
| Grad accumulation | 4 |
| Effective batch | 16 |
| Learning rate | 5e-05 |
| LR scheduler | cosine |
| Warmup steps | 100 |
| Max sequence length | 2048 |
| Gradient checkpointing | True |

### Results

- **Global steps:** 40,076
- **Epochs completed:** 2.00
- **Last eval loss:** 0.7040
- **Best eval loss:** 0.7040
- **Final train loss:** 0.7132

