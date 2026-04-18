# Chess Content Augmentation

## Introduction

The database-generated chess game corpus that is used for training is pretty "bare" - i.e., its the game data only.

Example:

```json
{"key": "Benko, Pal C-Kortschnoj, Viktor-1962.??.??-Stockholm Interzonal-13", "white": "Benko, Pal C", "black": "Kortschnoj, Viktor", "date": "1962.??.??", "event": "Stockholm Interzonal", "eco": "E08", "opening": "", "result": "1/2-1/2", "source_file": "pgnmentor/players/Korchnoi.pgn", "mode": "summary", "text": "Game: Benko, Pal C vs. Kortschnoj, Viktor\nEvent: Stockholm Interzonal, 1962\nOpening: Indian defense (E08)\nResult: Draw\n\n1.Nf3 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.O-O O-O 6.d4 Nbd7 7.Qc2 c6 8.Bf4 Ne4\n9.Ne5 Nd6 10.Nd2 Nxe5 11.dxe5 Nxc4 12.Nxc4 dxc4 13.Be4 g6 14.h4 Bd7 15.Rad1\nQc7 16.h5 Kg7 17.Kg2 Rfd8 18.hxg6 hxg6 19.Rh1 Rh8 20.Qd2 Rxh1 21.Qxd7 Qxd7\n22.Rxd7 Re1 23.Rxe7 Rxe2 24.Kf3 Rxb2 25.Rd7 c3 26.Rd1 Rh8 27.Kg2 Rc8 28.Bg5\nRb5 29.Bf6+ Kf8 30.Rh1 Ke8 31.Bxg6 fxg6 32.Rh8+ Kd7 33.Rh7+ Ke8\n\nThe game lasted 33 moves and ended in draw.", "length": 530}
```

The resulting model that was trained with this data shows this bias by frequently regurgitating raw game notation output that is has learned. 

Example: 


**You**

*What are the typical moves after an E08 opening*

**deepredai/deepred-dev/dev-2026-03-08-final.gguf**

```
Rhein 6.e3 Nf6 7.Nc3 e5 8.d4 exd4 9.cxd4 Ne7 10.Bb5 Bb4+ 11.Kh1 c6 12.O-O a6
13.Rc1 Qe7 14.Qe1 Nc6 15.f3 d5 16.exd5 Nxd5 17.Nxe7+ Kf8 18.Bg5 Qb7 19.Rfd1
Be6 20.Ne5 Be5 21.Qe4 Qc8 22.Qxf7 Bc3+ 23.Kh1 Nd8 24.Nxd6 cxd6 25.Bxd8 Bxd4
26.Qg6+ Ke8 27.Rac1 Rc8 28.a3 a5 29.Be5 Rhd8 30.e4 dxe4 31.Rcd1 Rxc1 32.Bxc1
Kf7 33.Nf6 Bb6 34.Qg3 Rd3 35.f4 g6 36.h4 Qd8 37.Be3 h5 38.Rc1 b6 39.hxg6 fxg6
40.Bxa5 Bxc2+ 41.Kh2 Qb7 42.Kg3 Kg7 43.Nd4 Be6 44.Bc3 g5 45.Bxe6 dxe5 46.Rf1
Qa8 47.Bxa5 Bxc3+ 48.Kh3 Qb8 49.Ra2 Qe8 50.Ne6 Be8 51.Bd1 Qc8 52.Kg4 Bd7
53.Nxg5 Kf6 54.Rc2 Kg7 55.Rc1 Qb8 56.Bh5 a4 57.Ba4 g4+ 58.Kh2 Qc8 59.B
```

## Game Data Augmentation

I think, the training would be much better if one can convert this content into a more narrative text - something the DeepRed AI would recite as answer to an unrelated question. 

Using the following prompt:

```
The following JSON data represents a historical chess game. 

[GAME DATA HERE]

Please convert this game information into a narrative form, like text that a chess reporter would write in a magazine, or a chess teacher would instruct a student with while observing the game. Ensure the output is compact, that all information and moves of the original game remain present in the final text, and that NO facts past [GAME DATE HERE] are used.
```

a larger base model was able to transform the above "data" into a much more readable text which is better for training the AI into responding in a theme-appropriate way.

```
In the 1962 Stockholm Interzonal, Pal Benko and Viktor Kortschnoj—two players already known for their fighting spirit—produced a tense struggle in the Catalan‑flavored lines of the Indian Defense (E08). Benko began with 1.Nf3 and 2.c4, steering the game into positional waters, while Kortschnoj replied solidly with …Nf6, …e6, and …d5. Both sides castled early, and after 6.d4 Nbd7 7.Qc2 c6 8.Bf4 Ne4, the game took on a more confrontational tone.

Benko met the central challenge with 9.Ne5, and after 9…Nd6 10.Nd2 Nxe5 11.dxe5 Nxc4 12.Nxc4 dxc4, the structure transformed sharply. The queenside became fixed while the center opened, giving each player new targets. Benko’s 13.Be4 and 14.h4 signaled his intention to seize the initiative on the kingside, but Kortschnoj calmly developed with …g6 and …Bd7.

By move 17, with Kg2 and Rfd8 on the board, both players had completed development and were preparing for the middlegame battle. Benko struck first: 18.hxg6 hxg6 19.Rh1 Rh8 20.Qd2, and after the exchange 20…Rxh1 21.Qxd7 Qxd7 22.Rxd7, the queens were off and the position simplified.

The endgame that followed—23.Rxe7 Rxe2 24.Kf3 Rxb2—left Black with an advanced passed pawn on c3, but Benko’s pieces were active enough to hold the balance. After 26.Rd1 Rh8 27.Kg2 Rc8 28.Bg5 Rb5 29.Bf6+ Kf8, neither side could make progress without risking too much.

The final sequence, 30.Rh1 Ke8 31.Bxg6 fxg6 32.Rh8+ Kd7 33.Rh7+ Ke8, confirmed that both players had reached the natural point of equilibrium. With perpetual‑check motifs looming and no safe way to play for more, the game concluded as a well‑earned draw after 33 moves.
```

## Implementation

### Model Selection

The augmentation task requires a large instruction-tuned model capable of producing
fluent, factually grounded chess commentary. After evaluating several candidates,
the recommended model is:

| Model | Quant | Size | Reason |
|-------|-------|------|--------|
| **Nemotron 3 Nano 30B A3B** | Q4_K_M | ~23 GB (1 file) | MoE architecture (30B total / 3.5B active) — fast inference, good narrative quality, strong instruction following. Supports 1M token context natively. Fits easily in Strix Halo unified memory (128 GB) with generous context. |
| Qwen2.5 72B Instruct | Q4_K_M | ~43 GB (12 shards) | Best narrative quality but very slow on Strix Halo (context >4096 fails due to memory pressure). Dense 72B is impractical for bulk augmentation. |
| Gemma 2 27B IT | Q4_K_M | ~16 GB | Acceptable fallback — faster throughput but lower narrative quality. |

The Nemotron 3 Nano 30B model is a Mixture-of-Experts (MoE) architecture: while
it has 30B total parameters, only ~3.5B are active per token. This means it runs
significantly faster than a dense 30B or 72B model while still producing high-quality
narrative output. The 72B Qwen model proved impractical — context sizes above
4096 caused failures due to memory pressure on the Strix Halo's unified memory,
and even at 4096 context the per-game throughput was too slow for the ~356K corpus.

Both models are automatically downloaded by `setup_strixhalo.py`
(stage `model_directories`). The Nemotron model is stored at:

```
$DEEPRED_MODELS/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
```

If the model was not downloaded during the initial setup (e.g. because the stage
was run before the Nemotron download was added), re-run the stage:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
sudo -E python3 scripts/setup_strixhalo.py --stage model_directories --force
```

The `--force` flag clears the stage's completion marker so it runs again.
The download is idempotent — models already on disk are skipped, so only the
missing files will be fetched.

To verify the download completed:

```bash
ls -lh /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
# Should show a single ~23 GB file
```

### Deep Red AI Voice

The augmentation script (`augment_chess_games.py`) generates text in the voice of
**Deep Red** — the Soviet chess AI from the DeepRed film universe:

> *Deep Red governs, calculates, and exerts control over the Soviet utopia of
> New Moscow on Mars. Built to protect the Revolution and secure the continuity
> of Party principles on the Red Planet, it is a strategic, calculating
> intelligence — precise in analysis, respectful of the masters of the game,
> and guided by an unwavering commitment to logic and truth.*

The system prompt establishes this persona, and five user prompt variations are
cycled to produce diverse output while maintaining consistent thematic alignment:

| # | Style | Description |
|---|-------|-------------|
| 0 | Magazine reporter | Vivid tournament dispatch, analytical and complete |
| 1 | Chess instructor | Move-by-move instructional commentary for an advanced student |
| 2 | Strategic briefing | Positional themes, tactical motifs, decision analysis |
| 3 | Narrative storytelling | Scene-setting, player introductions, dramatic prose |
| 4 | Tactical debrief | Opening prep, critical positions, endgame assessment |

All prompts enforce the temporal boundary: no facts past July 1969.

### Script: augment_chess_games.py

The script reads the source chess corpus, sends each game through the LLM with
one of the prompt variations, and appends the narrative output to the augmented
corpus file. It is fully resumable — already-augmented keys are skipped on re-run.

**Source:** `$CHESS_DATA/corpus/chess_games.jsonl` (~356K games)
**Output:** `$CHESS_DATA/corpus/augmented_chess_games.jsonl`

Key features:
- **Progress tracking:** Reads existing keys from output on startup; skips duplicates
- **Graceful shutdown:** SIGINT/SIGTERM finishes the current batch before exiting
- **Multi-endpoint:** Auto-discovers local and remote LLM servers
- **Prompt cycling:** Rotates through all 5 prompt variations by default
- **Retry logic:** Exponential backoff with configurable retries per game

### Training Pipeline Integration

The augmented corpus is integrated into the `chess_games` source in
`create_training_corpus.py` via an in-memory index that pairs augmented
narratives with their corresponding raw chess notation records.

At tokenization time the script:

1. **Builds a key-based index** — scans `chess_games.jsonl` and
   `augmented_chess_games.jsonl`, joining records by their `key` field.
2. **Prioritizes augmented games** — games with augmented narratives are
   placed first in the iteration order so they are selected preferentially
   at low percentages (e.g. `--percent 5`).
3. **Emits paired documents** — for each augmented game, the LLM-generated
   narrative text is followed by the raw chess notation as a single
   combined training document. Games without augmentation emit only the
   raw notation text.

| Condition | Output per game |
|-----------|-----------------|
| Augmented narrative exists | Narrative text + `\n\n` + raw notation (one document) |
| No augmented narrative | Raw notation only |

| Source | File | Estimated Tokens |
|--------|------|------------------|
| Raw chess games | `chess_games.jsonl` | ~53M |
| Augmented narratives | `augmented_chess_games.jsonl` | ~80M |
| Combined chess total | | ~134M |

> **Note:** If the augmented corpus grows between incremental runs (more
> games augmented), the prioritized ordering changes. Use `--reset` before
> re-tokenizing to ensure consistent pairing.


## Running the Chess Augmentation

### Prerequisites

1. **System setup complete** — `setup_strixhalo.py` has run through at least the
   `model_directories` stage (which downloads Qwen2.5 72B)
2. **Source corpus exists** — `retrieve_chess_content.py` has been run (Phase 2)
   to produce `$CHESS_DATA/corpus/chess_games.jsonl`
3. **LLM server running** with the augmentation model loaded — the default
   llama-server runs the 14B model, so you need to swap it before starting
   augmentation:

   ```bash
   llm-swap /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf \
       "nemotron-3-nano-30b" 131072 --slots 4
   ```

   This restarts the `llama-server-llm` systemd service with the Nemotron model
   and 4 parallel slots. The total context (131072) is divided evenly across
   slots, giving **32768 tokens per slot** — generous headroom for even the
   longest games plus full narrative output. The model supports up to 1M tokens
   natively, and with the MoE architecture's low active parameter count (~3.5B),
   4 slots at 32K each fit comfortably in the Strix Halo's 128 GB unified
   memory. See Step 1 below for verification.

### Step 1: Load the Augmentation Model

Swap the default LLM server to the Nemotron 3 Nano 30B model:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh

# Swap to Nemotron 30B with 4 parallel slots (131072 total / 4 = 32768 per slot)
llm-swap /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf "nemotron-3-nano-30b" 131072 --slots 4
```

Wait for the model to load, then verify the model is loaded:

```bash
curl -s http://localhost:1234/v1/models | python3 -m json.tool
```

Verify all 4 slots are active:

```bash
curl -s http://localhost:1234/slots | python3 -m json.tool | grep '"id"'
# Should list slots 0, 1, 2, 3
```

> **Note:** If memory is tight, try 2 slots instead (still 32768 per slot):
> ```bash
> llm-swap /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf "nemotron-3-nano-30b" 65536 --slots 2
> ```
> This gives **32768 tokens per slot** with 2 concurrent workers.

### Step 2: Run the Augmentation

By default the script uses only the local LLM endpoint. The remote A4000 (16 GB
VRAM) can potentially run the Nemotron 30B model (MoE with ~23 GB weight file),
but it is excluded by default unless you explicitly pass `--use-remote`.

```bash
# Test run — augment 10 games
python3 scripts/augment_chess_games.py --max-games 10 --verbose

# Inspect the output
head -1 /mnt/data/chess/corpus/augmented_chess_games.jsonl | python3 -m json.tool

# Full run with 4 concurrent workers matching the 4 server slots
python3 scripts/augment_chess_games.py --concurrency 4 --verbose
```

The script is fully resumable. If interrupted, re-run the same command and it
will skip already-augmented games:

```bash
# Resume after interruption — picks up where it left off
python3 scripts/augment_chess_games.py --concurrency 4 --verbose
```

To review the augmented output, use `--convert` to export the JSONL to a
human-readable file (HTML or Markdown). This is useful for spot-checking
narrative quality, prompt variant diversity, and factual accuracy:

```bash
# Export all augmented games to a styled HTML review file
python3 scripts/augment_chess_games.py --convert html

# Export to Markdown instead
python3 scripts/augment_chess_games.py --convert md

# Export only the first 50 games
python3 scripts/augment_chess_games.py --convert html --max-games 50
```

The review file is written next to the augmented corpus as
`augmented_chess_games.review.html` (or `.review.md`). The HTML version
includes a clickable table of contents and per-game metadata (players, date,
event, opening, prompt variant, character count).

### Command Reference

```bash
# Default run — cycle all prompt variants, 2 workers, local endpoint only
python3 scripts/augment_chess_games.py

# 4 concurrent workers (match with --slots 4 on the server)
python3 scripts/augment_chess_games.py --concurrency 4 --verbose

# Use only the magazine-reporter style (prompt variant 0)
python3 scripts/augment_chess_games.py --prompt-index 0

# Cap at 5000 games
python3 scripts/augment_chess_games.py --max-games 5000

# Dry run — generate narratives but don't write to disk
python3 scripts/augment_chess_games.py --dry-run --max-games 5 --verbose
```

### Step 3: Repair Problematic Output

LLM-generated augmentations occasionally produce text with quality issues:

| Issue | Description | Detection |
|-------|-------------|-----------|
| **Non-English language** | The model sometimes outputs in German or another language instead of English | `fast_langdetect` language detection (flags any text not detected as `en`) |
| **Nonsensical repetition** | Degenerate output like `d4-d4-d4-d4-d4...` or repeated sentences/phrases | Multi-tier heuristic: token repetition, consecutive identical sentences, repeated multi-word chunks |
| **Extremely long tokens** | Space-free strings exceeding 45 characters (the length of the longest English word) indicate garbled output | Simple length check on each whitespace-delimited token |

The `--repair` mode scans the existing augmented corpus, identifies records with
these issues, removes them, and re-augments from the original source games.

#### Prerequisites

Install the language detection library (one-time):

```bash
pip install fast-langdetect
```

> **Note:** If `fast_langdetect` is not installed, repair mode still works but
> language detection is disabled — only repetition and long-token checks run.

#### Running Repair

```bash
# Dry run — scan and report issues without modifying anything
python3 scripts/augment_chess_games.py --repair --dry-run

# Dry run with verbose — see each problematic record
python3 scripts/augment_chess_games.py --repair --dry-run --verbose

# Full repair — re-augment problematic records (requires LLM server running)
python3 scripts/augment_chess_games.py --repair --verbose

# Repair at most 50 records
python3 scripts/augment_chess_games.py --repair --max-games 50 --verbose

# Repair using a specific prompt variant
python3 scripts/augment_chess_games.py --repair --prompt-index 2
```

The repair workflow:

1. Loads the augmented corpus and checks every record's `text` field
2. Splits records into "good" and "problematic" sets
3. Reports issue counts by category (non-english, repetition, long-token)
4. Looks up the original game data from the source corpus
5. Re-augments problematic games through the LLM (with at least 1 retry)
6. Re-checks the replacement text and warns if it still has issues
7. Writes the repaired corpus atomically (via temp file + rename)

> **Note:** The repair uses at least 1 retry per game regardless of the
> `--retries` setting, since the original augmentation already failed to
> produce clean output. Re-augmented text that still fails quality checks
> is kept with a warning logged — run `--repair` again to try a second time.

### Step 4: Integrate into Training Corpus

After augmentation (partial or complete), rebuild the training corpus.
The `chess_games` source automatically detects and pairs augmented
narratives with their raw notation records:

```bash
# Swap back to the smaller model for regular server duties
llm-swap /mnt/data/models/llm/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
    "qwen2.5-14b-instruct" 8192 --slots 4

# Reset chess shards (required when augmented corpus has grown)
python3 scripts/create_training_corpus.py --sources chess_games --reset

# Tokenize (chess_games now includes augmented pairing)
python3 scripts/create_training_corpus.py --percent 100

# Finalize into train.bin / val.bin
python3 scripts/create_training_corpus.py --finalize

# Check the result
python3 scripts/create_training_corpus.py --status
```

### Output Format

Each augmented record mirrors the source format with two key differences:

- **`text`** — The LLM-generated narrative prose that replaces the original bare
  game notation. This is the field consumed by the training pipeline:
  `create_training_corpus.py` extracts only the `text` value (via the
  `_fmt_chess_game` formatter), not the surrounding JSON metadata. The narrative
  is what the model ultimately learns from.
- **`prompt_variant`** — Integer (0–4) indicating which prompt style produced this
  record (0 = magazine reporter, 1 = instructor, 2 = strategic briefing,
  3 = narrative storytelling, 4 = tactical debrief). Stored for traceability
  but not used during training.

```json
{
  "key": "Benko, Pal C-Kortschnoj, Viktor-1962.??.??-Stockholm Interzonal-13",
  "white": "Benko, Pal C",
  "black": "Kortschnoj, Viktor",
  "date": "1962.??.??",
  "event": "Stockholm Interzonal",
  "eco": "E08",
  "opening": "",
  "result": "1/2-1/2",
  "source_file": "pgnmentor/players/Korchnoi.pgn",
  "prompt_variant": 0,
  "text": "In the 1962 Stockholm Interzonal, Pal Benko and Viktor Kortschnoj — two players already known for their fighting spirit — produced a tense struggle in the Catalan-flavored lines of the Indian Defense (E08). Benko began with 1.Nf3 and 2.c4, steering the game into positional waters ...",
  "length": 1847
}
```

### Performance Notes

The Nemotron 3 Nano 30B MoE model supports up to 1M tokens of context natively
and is significantly faster than the Qwen2.5 72B dense model since only ~3.5B
parameters are active per token. On the Strix Halo's 128 GB unified memory,
the ~23 GB Q4_K_M weights plus four 32K-token KV caches fit comfortably — leaving
ample headroom for the OS and other services.

The recommended configuration gives each slot generous context for long games,
system prompts, and full narrative output:

```bash
# 32K per slot × 4 slots = 131072 total context
llm-swap /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf "nemotron-3-nano-30b" 131072 --slots 4

# Run a 20-game test with 4 workers
python3 scripts/augment_chess_games.py --max-games 20 --concurrency 4
```

To test throughput with fewer slots (lower memory, lower throughput):

```bash
# 32K per slot × 2 slots = 65536 total context
llm-swap /mnt/data/models/llm/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf "nemotron-3-nano-30b" 65536 --slots 2

# Run a 20-game test with 2 workers
python3 scripts/augment_chess_games.py --max-games 20 --concurrency 2
```

> **Note on the Qwen2.5 72B:** The 72B dense model proved impractical for this
> task — context sizes above 4096 caused failures (likely KV cache memory pressure
> in the 128 GB unified memory), and even at 4096 context the throughput was
> far too slow for bulk augmentation.

With 4 slots the per-game latency may increase (shared KV cache pressure), but
total throughput (games/min) should improve since 4 games are processed in
parallel. At 32K per slot there is no risk of context truncation even for the
longest games in the corpus.

Partial augmentation is practical — even 10–20K augmented games measurably
improve narrative quality in the trained model. Use `--max-games` to produce a
targeted subset (e.g. 10K–50K games).

#### Estimated Runtimes

At an observed throughput of ~24 sec/game (4 slots, Nemotron 30B on Strix Halo):

| `--max-games` | Total Time (sec) | Hours | Approx. Wall Clock |
|---------------:| -----------------:| -----:|:-------------------|
| 500 | 12,000 | 3.3 | ~3 h 20 min |
| 1,000 | 24,000 | 6.7 | ~6 h 40 min |
| 10,000 | 240,000 | 66.7 | ~2.8 days |
| 20,000 | 480,000 | 133.3 | ~5.6 days |
| 100,000 | 2,400,000 | 666.7 | ~27.8 days |
| 356,000 (full corpus) | 8,544,000 | 2,373.3 | ~98.9 days |
