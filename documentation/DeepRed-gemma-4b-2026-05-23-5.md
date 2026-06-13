# DeepRed-gemma-4b-2026-05-23-5-final.gguf

## Examples

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

