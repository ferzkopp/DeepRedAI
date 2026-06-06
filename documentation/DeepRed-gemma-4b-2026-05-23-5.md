# DeepRed-gemma-4b-2026-05-23-5-final.gguf

## Training Corpus

- **Generated:** 2026-06-06 05:45:43 UTC
- **Tokenizer:** TinyLlama-1.1B (vocab 32,000)
- **Sequence length:** 2,048 tokens
- **Manifest created:** 2026-05-19T15:56:09Z
- **Manifest updated:** 2026-05-21T04:56:35Z
- **Finalized:** yes

### Corpus Composition

| Source | Items used | Items available | Tokens | Share |
|--------|-----------:|----------------:|-------:|------:|
| wikipedia_articles | 1,931,239 | 1,931,239 | 2.00B | 70.7% |
| year_topics | 1,788 | 1,788 | 2.0M | 0.1% |
| gutenberg | 766 | 766 | 146.2M | 5.2% |
| chess_games | 355,980 | 355,980 | 415.1M | 14.7% |
| augmented_chess_games | 334,920 | 334,920 | 262.1M | 9.3% |
| chess_books | 10 | 10 | 1.8M | 0.1% |
| **Total** | **2,624,703** | **2,624,703** | **2.82B** | **100.0%** |

### Mixture by Tokens

```
wikipedia_articles       ████████████████████████████············  70.7% 
year_topics              ········································   0.1% 
gutenberg                ██······································   5.2% 
chess_games              ██████··································  14.7% 
augmented_chess_games    ████····································   9.3% 
chess_books              ········································   0.1% 
```

### Sources

#### wikipedia_articles

- Pre-1969 Wikipedia articles from PostgreSQL (temporal_classification=O)
- **Type:** database
- **Items:** 1,931,239 used / 1,931,239 available (100.0%)
- **Tokens:** 2.00B
- **Selected for this run:** yes

#### year_topics

- Year-by-year historical event summaries, years 151–1969 (JSON files)
- **Type:** json_files
- **Items:** 1,788 used / 1,788 available (100.0%)
- **Tokens:** 2.0M
- **Selected for this run:** yes

#### gutenberg

- Project Gutenberg books — 766 public-domain titles (JSONL)
- **Type:** jsonl
- **Items:** 766 used / 766 available (100.0%)
- **Tokens:** 146.2M
- **Selected for this run:** yes

#### chess_games

- Pre-1969 chess games — raw PGN notation, 356K games (JSONL)
- **Type:** jsonl
- **Items:** 355,980 used / 355,980 available (100.0%)
- **Tokens:** 415.1M
- **Selected for this run:** yes

#### augmented_chess_games

- LLM-augmented chess game narratives — 335K games (JSONL)
- **Type:** jsonl
- **Items:** 334,920 used / 334,920 available (100.0%)
- **Tokens:** 262.1M
- **Selected for this run:** yes

#### chess_books

- Internet Archive chess reference books — 10 titles (JSONL)
- **Type:** jsonl
- **Items:** 10 used / 10 available (100.0%)
- **Tokens:** 1.8M
- **Selected for this run:** yes

### Finalized Output

- **Packing:** document_aware
- **Long-document overlap:** 25%
- **Train sequences:** 1,582,818 (3.24B tokens)
- **Val sequences:** 15,988 (32.7M tokens)
- **train.bin:** 6.0 GB
- **val.bin:** 62.5 MB

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
