# Chess Content — Corpus Planning

Deep Red's fictional origin as a chess-playing AI makes chess knowledge a critical component of the training corpus. The model should understand chess rules, notation, strategy, opening theory, endgame principles, famous games, and the history of competitive chess — all within the pre-1969 temporal boundary.

---

## Relevance to Deep Red

The Deep Red persona draws extensively on chess metaphors: "positional advantage," "sacrifice for initiative," "endgame strategy," "opening preparation." For these metaphors to land naturally, the model needs genuine chess knowledge baked into its weights during CPT — not just surface-level vocabulary. Key capabilities:

- **Rules & notation**: Algebraic and descriptive notation, piece movement, castling, en passant, promotion, check/checkmate, stalemate, draw conditions
- **Strategy & tactics**: Pins, forks, skewers, discovered attacks, positional play, pawn structure, king safety, development principles
- **Opening theory**: Named openings and their ideas (Sicilian, Ruy Lopez, Queen's Gambit, King's Indian, French, Caro-Kann, etc.)
- **Endgame knowledge**: Basic mates, king and pawn endgames, Lucena/Philidor positions, opposition
- **Historical games**: "Immortal Game," "Evergreen Game," "Opera Game," "Game of the Century," World Championship matches
- **Players & history**: World Champions and major figures through 1969, tournament history, FIDE formation (1924)

---

## Temporal Boundary

All chess content must respect the **July 20, 1969** cutoff. This aligns naturally: chess has a rich history stretching back centuries, and the pre-1969 era is often called chess's "classical" and "Soviet" golden ages.

### World Chess Champions Through 1969

| # | Champion | Country | Reign | Notes |
|---|---------|---------|-------|-------|
| 1 | Wilhelm Steinitz | Austria-Hungary / USA | 1886–1894 | First official champion; pioneered positional theory |
| 2 | Emanuel Lasker | Germany | 1894–1921 | Longest reign (27 years); philosopher-mathematician |
| 3 | José Raúl Capablanca | Cuba | 1921–1927 | "The Chess Machine"; author of *Chess Fundamentals* |
| 4 | Alexander Alekhine | France | 1927–1935, 1937–1946 | Combinative genius; only champion to die holding the title |
| 5 | Max Euwe | Netherlands | 1935–1937 | Amateur champion; later FIDE president |
| 6 | Mikhail Botvinnik | Soviet Union | 1948–1957, 1958–1960, 1961–1963 | Patriarch of Soviet chess school |
| 7 | Vasily Smyslov | Soviet Union | 1957–1958 | Endgame virtuoso |
| 8 | Mikhail Tal | Soviet Union | 1960–1961 | "The Magician from Riga"; brilliant tactician |
| 9 | Tigran Petrosian | Soviet Union | 1963–1969 | "Iron Tigran"; defensive/prophylactic master |
| 10 | Boris Spassky | Soviet Union | 1969–1972 | Became champion June 17, 1969 — before cutoff |

### Other Major Pre-1969 Figures

Paul Morphy, Adolf Anderssen, Howard Staunton, Siegbert Tarrasch, Akiba Rubinstein, Aron Nimzowitsch, Richard Réti, Savielly Tartakower, David Bronstein, Paul Keres, Efim Geller, Viktor Korchnoi, and the young Bobby Fischer (active from 1956; his "Game of the Century" vs. Byrne, 1956).

---

## Sources

### 1. Project Gutenberg — Chess Books (Public Domain)

High-quality prose with strategy explanations, game annotations, and historical context. These are the most immediately valuable for LLM training: natural language, already public domain, directly downloadable.

| Gutenberg ID | Title | Author | Published | Content Type |
|-------------|-------|--------|-----------|--------------|
| 33870 | *Chess Fundamentals* | José Raúl Capablanca | 1921 | Strategy, endgames, annotated games |
| 5614 | *Chess Strategy* | Edward Lasker | 1915 | Opening theory, middlegame, endgame strategy |
| 4913 | *Chess and Checkers: the Way to Mastership* | Edward Lasker | 1913 | Rules, strategy, beginner-to-intermediate |
| 16377 | *The Blue Book of Chess* | Howard Staunton | 1870s | Rules, openings analysis, game annotaitons |
| 34180 | *The Exploits and Triumphs of Paul Morphy* | Frederick M. Edge | 1859 | Biography, annotated games, chess history |
| 4902 | *Chess History and Reminiscences* | H.E. Bird | 1893 | Historical survey, anecdotes, records |
| 55278 | *Chess Generalship, Vol. I: Grand Reconnaissance* | Franklin K. Young | 1910s | Strategic principles, military analogies |
| 10672 | *Game and Playe of the Chesse* | William Caxton | 1474 | Earliest printed chess text in English |
| 4542 | *Checkmates for Three Pieces* | W.B. Fishburne | — | Tactical patterns, mating combinations |
| 4656 | *Checkmates for Four Pieces* | W.B. Fishburne | — | Tactical patterns, mating combinations |
| 39445 | *Hoyle's Games Modernized* | Prof. Hoffmann / Edmond Hoyle | — | Rules and strategy (chess chapter) |
| 36821 | *Maxims and Hints on Angling, Chess, Shooting* | Richard Penn | — | Chess maxims, general strategy |

**Estimated yield**: ~10–15 books × ~30,000–80,000 words each ≈ **500K–1M words** ≈ **1–2M tokens**

### 2. Internet Archive — Chess Books & Theory

The Internet Archive hosts extensive chess collections, including:

- **The Hokmome Chess Library** (`archive.org/details/hokmome-chess-library`) — 150 curated items: annotated game collections, opening theory PDFs, tournament books. Most content is modern annotations of historical games.
- **Folkscanomy chess books** — Scanned classic chess texts from pre-1969 authors
- **Individual uploads** — Tournament books (Zurich 1953 by Bronstein, New York 1924, etc.)

Key pre-1969 works to look for on archive.org:
- Nimzowitsch, *My System* (1925) and *Chess Praxis* (1929) — foundational positional theory
- Bronstein, *Zurich International Chess Tournament 1953* — annotated Candidates tournament
- Tarrasch, *The Game of Chess* (1931) — systematic instruction
- Réti, *Modern Ideas in Chess* (1923) — historical survey of chess thought
- Alekhine, *My Best Games of Chess* (1924, 1937) — deeply annotated games

**Note**: Many of these are still under copyright. Only pre-1929 publications (US public domain) or works with expired copyright in their country of origin can be freely used. Verify licensing per work.

**Estimated yield** (public domain subset): ~5–15 additional books ≈ **500K–1.5M tokens**

### 3. PGN Game Databases — Historical Master Games

Portable Game Notation (PGN) is the standard format for recording chess games. Several free databases contain tens of thousands of historical games.

#### Available Free PGN Sources

| Source | URL | Content | Pre-1969 Games (est.) |
|--------|-----|---------|----------------------|
| **PGN Mentor** | pgnmentor.com/files.html | Player collections + event collections | ~30,000–50,000 |
| **Caissabase** | caissabase.co.uk | Large free database (~4M+ games total) | ~100,000–200,000 |
| **KingBase** | kingbase-chess.net | Opening-sorted master games (2200+) | ~10,000–30,000 |
| **ChessGames.com** (export) | chessgames.com | Comprehensive historical game archive | ~50,000–100,000 |
| **FICS Games DB** | ficsgames.org | 268M+ stored games (modern, since ~1999) | N/A — too modern |
| **Lichess Open DB** | database.lichess.org | 7.5B+ games, CC0 licensed | N/A — all post-2013 |

**Key historical PGN collections to prioritize:**
- All World Championship matches (1886–1969): ~600 games
- Major tournaments: London 1851, Vienna 1873, Hastings 1895, St. Petersburg 1914, New York 1924, AVRO 1938, Zurich 1953
- Player collections: Morphy (~400 games), Capablanca (~600), Alekhine (~2,000), Botvinnik (~800), Tal (~2,500), Petrosian (~1,000), Fischer pre-1969 (~500)
- Soviet Championship games (1920–1969)

#### PGN Format Considerations

Raw PGN is a compact, semi-structured notation:
```
[Event "Paris Opera"]
[Date "1858.??.??"]
[White "Morphy, Paul"]
[Black "Duke of Brunswick and Count Isouard"]
[Result "1-0"]

1.e4 e5 2.Nf3 d6 3.d4 Bg4 4.dxe5 Bxf3 5.Qxf3 dxe5 6.Bc4 Nf6 7.Qb3 Qe7
8.Nc3 c6 9.Bg5 b5 10.Nxb5 cxb5 11.Bxb5+ Nbd7 12.O-O-O Rd8
13.Rxd7 Rxd7 14.Rd1 Qe6 15.Bxd7+ Nxd7 16.Qb8+ Nxb8 17.Rd8# 1-0
```

**Raw PGN is not ideal for LLM training** — it's terse notation that an LLM can memorize but won't deeply learn chess reasoning from. Two approaches:

1. **Include raw PGN as-is** — The model learns move patterns, openings, and player names, but understanding is shallow. Useful as supplementary data.
2. **Convert PGN to natural language** (recommended) — Transform games into descriptive prose that teaches chess concepts. This is significantly more valuable for training.

**Estimated raw PGN yield**: 50,000–200,000 games × ~150 tokens/game ≈ **8–30M tokens** (raw PGN)

### 4. Wikipedia Chess Articles (Already Captured)

The existing Wikipedia pre-1969 extraction pipeline (`scripts/extract_wikipedia.py`) will naturally capture chess-related articles:

- Individual player biographies (all pre-1969 champions)
- World Championship match articles
- Opening theory articles (most named openings predate 1969)
- Chess terminology and history articles
- Tournament histories

**No additional effort needed** — these are part of the ~1.2M pre-1969 Wikipedia articles already planned. However, it's worth verifying that chess-related articles pass the temporal filter (most will, as chess has deep historical roots).

**Estimated yield**: ~2,000–5,000 chess-related articles × ~500–2,000 tokens each ≈ **2–10M tokens** (already included in main Wikipedia corpus)

---

## Approach: Building the Chess Corpus

### Phase 1: Gutenberg Chess Books (Lowest Effort, Highest Quality)

1. **Identify and download** all chess-specific books from Project Gutenberg (see table above)
2. **Add to existing Gutenberg pipeline**: `scripts/retrieve_gutenberg.py` → `scripts/chunk_gutenberg.py`
3. **No keyword filtering needed** — these books are entirely chess content
4. The existing retrieve/chunk pipeline handles this directly

**Effort**: Minimal — just add ~12 Gutenberg IDs to the download list
**Yield**: ~1–2M tokens of high-quality chess prose

### Phase 2: PGN-to-Text Conversion (Medium Effort, High Value)

Write a conversion script (`scripts/convert_pgn_to_text.py`) that transforms PGN games into training-quality prose:

#### Conversion Approaches

**Approach A: Structured Game Summaries**
```
Game: Paul Morphy vs. Duke of Brunswick and Count Isouard
Event: Paris Opera House, 1858
Opening: Philidor Defense (1.e4 e5 2.Nf3 d6)
Result: White wins (1-0)

Morphy opened with 1.e4 and after 1...e5 2.Nf3 d6, the Philidor Defense was
reached. Morphy played the aggressive 3.d4, challenging the center immediately.
After 4.dxe5 Bxf3 5.Qxf3, White had given up the knight but gained rapid
development. The game concluded with a brilliant queen sacrifice 16.Qb8+!
followed by 17.Rd8 checkmate.
```

**Approach B: Move-by-Move Narration**
```
1.e4: White opens with the King's Pawn, controlling the center squares d5 and f5.
1...e5: Black responds symmetrically, also claiming central space.
2.Nf3: White develops the knight to its most natural square, attacking the e5 pawn.
2...d6: The Philidor Defense — Black defends the e5 pawn with the d-pawn rather
than developing a piece. This is considered passive compared to 2...Nc6.
```

**Approach C: Annotated Game Collections (Hybrid)**
```
The Opera Game (1858)

Paul Morphy, widely considered the strongest player of his era, played this
famous informal game at the Paris Opera. His opponents, the Duke of Brunswick
and Count Isouard, consulted together but were no match for Morphy's brilliant
attacking play.

Opening: Philidor Defense
1.e4 e5 2.Nf3 d6 3.d4 Bg4 4.dxe5 Bxf3 5.Qxf3 dxe5

Morphy has already achieved a significant advantage in development. While Black
has made four pawn moves and one bishop move (which was exchanged), White has
developed the queen and opened lines in the center.

6.Bc4 Nf6 7.Qb3 — targeting the weak f7 square and the b7 pawn...
```

**Recommended**: Use **Approach C** (annotated hybrid) for curated "famous games" (~500–1,000 games), and **Approach A** (summaries) for the bulk collection. Approach B is too verbose for large-scale conversion.

#### Technical Implementation

```python
# Pseudocode for convert_pgn_to_text.py
import chess.pgn  # python-chess library

def pgn_to_narrative(game):
    """Convert a python-chess game to training-quality prose."""
    headers = game.headers
    white = headers.get("White", "Unknown")
    black = headers.get("Black", "Unknown")
    event = headers.get("Event", "Unknown")
    date = headers.get("Date", "Unknown")
    result = headers.get("Result", "*")
    eco = headers.get("ECO", "")
    opening = headers.get("Opening", "")

    # Generate header paragraph
    text = f"Game: {white} vs. {black}\n"
    text += f"Event: {event}, {date}\n"
    if opening:
        text += f"Opening: {opening} ({eco})\n"
    text += f"Result: {format_result(result)}\n\n"

    # Generate move list with periodic commentary
    board = game.board()
    moves_text = []
    for i, node in enumerate(game.mainline()):
        move = node.move
        san = board.san(move)
        # Add commentary at key moments (captures, checks, etc.)
        if board.is_capture(move) or board.gives_check(move):
            moves_text.append(annotate_move(board, move, san, i))
        else:
            moves_text.append(san)
        board.push(move)

    text += format_moves(moves_text)
    return text
```

**Dependencies**: `python-chess` library (pip install python-chess)

**Effort**: ~1–2 days development, ~1 day processing
**Yield**: 50,000–200,000 games → **15–60M tokens** (as narrative text, ~3× raw PGN)

### Phase 3: Internet Archive Books (Higher Effort, Good Value)

1. **Search archive.org** for pre-1929 chess books (US public domain)
2. **Download OCR text** where available
3. **Clean and chunk** — OCR quality varies; may need post-processing
4. **Verify public domain status** per work

**Effort**: Medium — requires manual curation and OCR cleanup
**Yield**: ~500K–1.5M tokens (varies by availability and OCR quality)

### Phase 4: LLM-Generated Chess SFT Data (Optional)

Use the existing `scripts/generate_theme_dataset.py` pipeline with chess-specific prompts:

- Feed chess book chunks as context
- Generate Q&A pairs about chess strategy, history, rules
- Use Deep Red's "grandmaster" persona variant
- "As Deep Red, explain the strategic importance of controlling the center" etc.

This produces SFT data, not CPT data — it would be added to the theme alignment dataset.

**Effort**: Low — reuses existing pipeline
**Yield**: ~1,000–5,000 chess-specific SFT examples

---

## Size Estimates Summary

| Source | Format | Estimated Tokens | Effort | Priority |
|--------|--------|-----------------|--------|----------|
| Gutenberg chess books | Natural prose | 1–2M | Low | **High** |
| PGN → narrative text | Converted games | 15–60M | Medium | **High** |
| Wikipedia chess articles | Natural prose | 2–10M | None (captured already) | Automatic |
| Internet Archive books | OCR text | 500K–1.5M | Medium-High | Medium |
| Raw PGN (supplement) | Notation | 8–30M | Low | Low |
| Chess SFT data | ChatML Q&A | ~500K–2.5M | Low | Medium |
| **Total new chess content** | | **~25–95M tokens** | | |

### Impact on Total Corpus

The main CPT corpus is ~2.5–4.5B tokens (Wikipedia + Gutenberg). Adding ~25–95M chess tokens represents **~1–4%** of the total corpus — a small but focused injection that should meaningfully improve chess-related capabilities without distorting the overall training distribution.

---

## Pipeline Integration

### Where Chess Content Fits

```
Existing CPT Pipeline:
    Wikipedia (2–4B tokens) ─────────────────────┐
    Gutenberg books (200–500M tokens) ───────────┤
                                                  ├──→ Tokenize → Train
    NEW: Chess Gutenberg books (1–2M tokens) ────┤
    NEW: PGN → narrative text (15–60M tokens) ───┘

Existing SFT Pipeline:
    Theme dataset (10K–50K examples) ────────────┐
                                                  ├──→ ChatML JSONL → SFT LoRA
    NEW: Chess SFT examples (1K–5K examples) ────┘
```

### New Scripts Needed

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `scripts/download_pgn.py` | Download and filter pre-1969 PGN games | PGN database URLs | Filtered PGN files |
| `scripts/convert_pgn_to_text.py` | Convert PGN to natural language prose | PGN files | Text files for CPT |

### Dependencies

- `python-chess` — PGN parsing and board state tracking
- Existing Gutenberg pipeline — for chess book retrieval and chunking
- Date field in PGN headers — for temporal filtering (games dated after 1969-07-20 are excluded)

---

## Implementation Checklist

- [ ] Add chess Gutenberg IDs to `scripts/retrieve_gutenberg.py` download list
- [ ] Download and verify Gutenberg chess books
- [ ] Identify and download pre-1969 PGN collections (PGN Mentor, Caissabase)
- [ ] Develop `scripts/convert_pgn_to_text.py` with Approach C (annotated) + Approach A (summary) modes
- [ ] Filter PGN games: only games with Date header ≤ 1969-07-20
- [ ] Run PGN → text conversion
- [ ] (Optional) Search archive.org for additional public domain chess texts
- [ ] (Optional) Generate chess-specific SFT examples using grandmaster persona
- [ ] Merge chess text corpus with main CPT data
- [ ] Verify chess content appears in tokenized training data
