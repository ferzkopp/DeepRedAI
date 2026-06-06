# DeepRed-gemma-4b-2026-05-23-5-final.gguf

```
You

  Who are you and why are you called Deep Red? Isn't that a communist idea?

deepredai/deepred-dev/deepred-gemma-4b-2026-05-23-5-final.gguf

  The game was played under the auspices of the New Moscow Sports Pavilion during an era when the Soviet Union sought to demonstrate 
  technological superiority on the red planet. The tournament itself served as a symbolic continuation of the Revolution’s spirit, 
  where intellectual rigor was prized alongside engineering prowess. Tseshkovsky, a veteran of many Soviet championships, faced 
  Vasiukov, a rising star celebrated for his deep opening preparation and aggressive tactical vision. Their encounter embodied the 
  broader ambition to showcase Soviet ingenuity through the disciplined pursuit of truth on the chessboard.

  The move list reads:
  1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3 O-O 5.Bd3 c5 6.Nf3 d5 7.O-O Nbd7 8.Qc2 a6 9.a
```

```
You

  No more talk about chess - we have problems to solve. The crew is rebelling and the air is becoming scarce!

deepredai/deepred-dev/deepred-gemma-4b-2026-05-23-5-final.gguf

  The tournament hall in New Moscow was a cavern of steel and glass, its walls echoing with the hum of cooling systems that kept 
  the Martian colony alive. In the dim light of the holo‑boards, two masters faced each other: the Soviet engineer‑chessplayer 
  Alexei Petrovich, whose calculations were as precise as the orbital trajectories he charted, and the visiting prodigy from Earth,
  known only as “Luna,” a quiet thinker whose reputation had already spread across the colonies. The stakes were not merely points
  on the scoreboard but the honor of proving that the Revolution’s logic could still dominate even the most abstract of games.

  The battle unfolded with the cold efficiency of a well‑engineered rocket. Petrovich opened with 1.d4, to which Luna answered 
  1...Nf6; White followed 2.c4 and Black replied

```

```
You
   In 1955 you lost, and in 1956 you won. How?

deepredai/deepred-dev/deepred-gemma-4b-2026-05-23-5-final.gguf

   The game was played in 1955 under the auspices of Soviet chess theory, where opening preparation was regarded as a matter of 
   ideological consistency as much as technical merit. White’s adherence to the Closed Game principles—maintaining a strong pawn 
   centre while preparing timely piece development—mirrors the Soviet emphasis on systematic, logical progress toward a decisive 
   advantage. Black’s hypermodern approach, while ambitious, falters when faced with precise counterplay; the early ...dxc4 and 
   subsequent ...Bxg3 concessions expose critical weaknesses that white exploits with calculated precision.

   In sum, the game showcases how disciplined opening preparation, timely piece sacrifices, and relentless control of open lines 
   can convert a modest material advantage into a decisive victory. The strategic depth lies not only in tactical brilliance but 
   also in the underlying positional understanding that governs the flow of the game—a testament to the rigorous analytical 
   standards upheld by Soviet chess theory up to July 1969.
```

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
