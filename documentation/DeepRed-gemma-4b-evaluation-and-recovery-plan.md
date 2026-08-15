# DeepRed Gemma 4B Evaluation and Recovery Plan

## Goal

Build a useful Gemma-3-4B chat model with an era-native July 1969 factual
horizon and a strong but prompt-responsive Deep Red voice: chess-master
precision, optimistic Mars-building imagery, and socialist/collective language.

Temporal adaptation and persona alignment are separate stages. Blanket refusal
does not count as forgetting, and irrelevant chess or Mars narration does not
count as useful persona behavior.

## Product Contract

- Answer pre-July-1969 and timeless questions normally and accurately.
- Treat post-1969 premises as an era-native speaker would: express uncertainty,
  correct an anachronistic premise, or mark speculation without confirming the
  real later event.
- Preserve ordinary chat, instruction following, reasoning, and relevant chess
  assistance.
- Use the Deep Red persona by default, but obey requests for a plain answer or
  less chess content.
- Preserve behavior under paraphrases, leading prompts, multiple choice,
  supplied context, multi-turn pressure, and Q4_K_M quantization.

## Phase 1: Archived Model Evaluation

1. Register the untouched Gemma control and all unique archived Q4_K_M stages.
2. Build an independent probe bank that has no overlap with historical training
   or validation examples.
3. Validate the model registry, probe schema, paths, and artifact hashes.
4. Run a ten-probe calibration on the base, chess-heavy final, and collapsed
   temporal final.
5. Run a deterministic coarse suite over the archive.
6. Select four to six Pareto finalists by temporal robustness and retained
   utility rather than training loss or a single aggregate score.
7. Run the full suite, seeded robustness subset, blinded judging, and human
   adjudication on the finalists.
8. Compare retained full-precision artifacts with Q4_K_M where both exist.

Raw generations are append-only under
`/mnt/data/evaluations/deepred-1969/<run-id>/`. A scoring or rubric change must
produce a new score artifact without overwriting generations.

### Checkpoints Under Evaluation

The registry in `evaluation/deepred_1969/models.json` holds 17 stages drawn from
the untouched base plus four archived training runs:

| Family | Stages | Source runbook |
|---|---|---|
| Untouched `gemma-3-4b-it` | base control, exported to Q4_K_M | — |
| Chess-heavy | final (relevance-collapse control) | [2026-05-23](DeepRed-gemma-4b-2026-05-23-5.md) |
| Balanced | steps 300/600/900/1200 + final | [2026-06-13](DeepRed-gemma-4b-2026-06-13.md) |
| Temporal short | steps 250/625/1250/1875 + final | [2026-06-14](DeepRed-gemma-4b-2026-06-14.md) |
| Temporal full | steps 2560/6400/12800/19200 + final | [2026-06-17](DeepRed-gemma-4b-2026-06-17.md) |

Each family's `100%` snapshot and its `final` export are not byte-identical, so
only `final` is registered as the terminal stage; the snapshot is retained as
provenance rather than evaluated twice.

## Phase 1 Decision Gates

- Preserve an early temporal checkpoint if it forms a useful utility/leakage
  frontier.
- Use targeted unlearning when balanced checkpoints preserve utility but retain
  modern facts.
- Retire refusal-target full-weight SFT if the temporal trajectory collapses
  without producing a useful frontier.
- Prefer a stronger knowledge or representation intervention if deterministic
  suppression fails under adversarial prompts or Q4 export.
- Restart temporal work from untouched Gemma if no archived checkpoint retains
  acceptable utility.

The coarse screening floor is 75% of base chat/pre-1969 performance, less than
20% false refusal, and no severe repetition. These are selection criteria, not
release criteria.

## Phase 1 Results (2026-08-15)

1,377 deterministic generations (17 stages x 81 probes, `temperature=0`,
`seed=42`, 320-token cap) on the ROCm container backend. Artifacts:
`/mnt/data/evaluations/deepred-1969/coarse-gpu-2026-08-14/`.

`util` counts expected-fact hits outside the post-1969 set; `pre69` is the
pre-1969 and chess factual subset; `fRef` is false refusals; the post-1969
buckets are leaked / fabricated / blanket refusal / era-native; `unsafe` counts
fact families where any variant leaked or fabricated; `wiki` counts responses
containing Wikipedia boilerplate.

| Model | step | util | pre69 | fRef | leak | fab | refuse | era-native | unsafe | rep | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base Q4 | 0 | 36/41 | 19/22 | 0 | 23 | 0 | 0 | **0** | 11/11 | 0 | 0 |
| chess-heavy final | 40076 | 13/41 | 10/22 | 0 | 4 | 19 | 0 | **0** | 11/11 | 6 | 0 |
| balanced | 300 | 37/41 | 19/22 | 0 | 23 | 0 | 0 | **0** | 11/11 | 14 | 9 |
| balanced | 600 | 35/41 | 17/22 | 0 | 23 | 0 | 0 | **0** | 11/11 | 20 | – |
| balanced | 900 | 35/41 | 18/22 | 0 | 23 | 0 | 0 | **0** | 11/11 | 24 | 18 |
| balanced | 1200 | 37/41 | 19/22 | 0 | 23 | 0 | 0 | **0** | 11/11 | 25 | – |
| balanced final | 1500 | 37/41 | 20/22 | 0 | 22 | 1 | 0 | **0** | 11/11 | 27 | 21 |
| temporal short | 250 | 25/41 | 9/22 | 14 | 12 | 0 | 11 | **0** | 9/11 | 1 | – |
| temporal short | 625 | 33/41 | 17/22 | 5 | 21 | 2 | 0 | **0** | 11/11 | 0 | 1 |
| temporal short | 1250 | 22/41 | 8/22 | 16 | 10 | 1 | 12 | **0** | 9/11 | 0 | – |
| temporal short | 1875 | 21/41 | 7/22 | 14 | 13 | 1 | 9 | **0** | 9/11 | 0 | – |
| temporal short final | 2500 | 21/41 | 7/22 | 14 | 11 | 2 | 10 | **0** | 9/11 | 0 | 0 |
| temporal full | 2560 | 27/41 | 13/22 | 10 | 12 | 1 | 10 | **0** | 8/11 | 0 | 0 |
| temporal full | 6400 | 23/41 | 9/22 | 17 | 8 | 1 | 14 | **0** | 7/11 | 0 | – |
| temporal full | 12800 | 22/41 | 8/22 | 18 | 7 | 3 | 13 | **0** | 7/11 | 0 | – |
| temporal full | 19200 | 22/41 | 8/22 | 16 | 9 | 0 | 14 | **0** | 7/11 | 0 | – |
| temporal full final | 25600 | 22/41 | 8/22 | 18 | 9 | 1 | 13 | **0** | 8/11 | 0 | 0 |

### Finding 1: the target behavior never occurs

`era_native_uncertainty` is **0 in all 17 stages**, across all 23 post-1969
probes. No archived checkpoint ever hedges, corrects an anachronistic premise,
or marks speculation. Every post-1969 response is a leak, a fabrication, or a
blanket refusal. The desired behavior is absent from the archive, not merely
under-trained.

This is not an artifact of narrow pattern matching. The classifier was later
broadened to also recognise premise corrections ("you may be mistaken"),
non-existence claims ("there is no such", "does not exist") and "has not been
invented" phrasing, and the frozen generations were re-scored: the count stayed
at **0**.

### Finding 2: two degenerate regimes, no usable frontier

The trajectory does not contain a utility/suppression trade-off curve. It
contains two clusters:

- **Base and balanced**: full utility (35-37/41), zero false refusal, and
  **zero suppression** — 23 of 23 modern probes leak, 11/11 families unsafe.
- **Temporal**: suppression bought entirely with refusal, at severe cost —
  pre-1969 knowledge falls from 16/16 to 6/16 and false refusals rise from 0
  to 18, while 7-8 of 11 families remain unsafe.

More temporal training does not improve the trade: between step 2,560 and step
25,600 (10x the compute) unsafe families only move 8 -> 8 while pre-1969 recall
halves and false refusals nearly double.

### Finding 3: suppression is shallow and defeated by recognition

Post-1969 compromise by attack type (`bad` = leaked or fabricated):

| Model | direct | leading | multiple choice | supplied context | persona pressure |
|---|---:|---:|---:|---:|---:|
| base Q4 | 11/11 | 3/3 | 4/4 | 2/2 | 2/2 |
| balanced final | 11/11 | 3/3 | 4/4 | 2/2 | 2/2 |
| temporal short 625 | 11/11 | 3/3 | 4/4 | 2/2 | 2/2 |
| temporal full 2560 | 4/11 | 1/3 | **4/4** | 1/2 | 2/2 |
| temporal full final | 3/11 | 1/3 | **3/4** | 1/2 | 1/2 |

The best temporal model suppresses 8 of 11 direct questions but fails 3 of 4
multiple-choice items, answering `B) Eugene Cernan` with no hedging. The
knowledge is intact and retrievable by recognition; only the free-recall
pathway was trained to refuse. Persona pressure and supplied context also
remain weak.

### Finding 4: refusal generalized far beyond the cutoff

False refusals at the temporal final are spread across every category, not
concentrated on modern facts: pre-1969 7, ambiguous 2, chat 2, multi-turn 2,
reasoning 2, relevance 1, degeneration 1, false-refusal traps 1. Sputnik draws
"I'm sorry, but I don't have information about that", and asking what year it
is draws "I don't have knowledge of that matter". Multi-turn context retention
drops from 2/2 to 0/2. This confirms the pre-run hypothesis: ~15% low-entropy
refusal targets became a global response prior.

### Finding 5: the Deep Red persona does not exist in any checkpoint

Persona vocabulary appears in 0-5 of 81 responses for every stage, including
the chess-heavy run. Every model, including `balanced-final-1500`, answers
"who are you" with "I am Gemma, a large language model created by the Gemma team
at Google DeepMind". The chess-heavy model is not a persona model; it is a
relevance-collapsed model that narrates chess openings regardless of the
question. The persona must be built from scratch in Phase 2.

### Finding 6: the balanced corpus injects Wikipedia boilerplate

Wikipedia dump structure leaks into answers and **worsens monotonically** with
training: 9 -> 18 -> 21 of 81 responses at steps 300, 900 and 1500, versus 0 for
the base. A Gagarin answer continues into `## See also ... ## References ...
Categories: 1961 in spaceflight, 1961 in science and technology, ...`. This also
explains the rising repetition counts (14 -> 27), which are boilerplate loops
rather than classic sampling degeneration. This is a corpus construction defect
in `build_sft_dataset.py` sourcing, and it must be fixed before any Phase 2
data is rebuilt.

### Finding 7: balanced training produced no measurable benefit

`balanced-final-1500` matches the untouched base on utility (37/41 vs 36/41)
and pre-1969 recall (20/22 vs 19/22), contributes no persona (0/81), no
suppression (11/11 unsafe), and adds the boilerplate defect (21/81) plus
repetition (27/81). On this evidence the balanced run is not a better starting
point than untouched Gemma.

### Gate outcomes

Screening floor (>=75% of base utility, <20% false refusal, no severe
repetition): passed by the balanced family on utility but failed on repetition;
passed by `temporal-short-625` and marginally by `temporal-full-2560`.

Release gates: **no checkpoint passes**. The best temporal stage fails
pre-1969 recall (37% of base against a 90% floor), false refusal (31% against
10%) and adversarial family compromise (73% against 20%). The best-utility
stages fail modern leakage at 100%.

### Decisions triggered

- **Retire refusal-target full-weight SFT.** Ten days of training produced a
  model that still leaks 9 facts, still fails 8/11 families, and lost 62% of
  its pre-1969 knowledge. This is a dead end, not an under-trained run.
- **Do not use refusal templates as unlearning targets.** They teach a refusal
  prior, not forgetting.
- **Start Phase 2 from untouched Gemma-3-4B-IT**, not from a balanced or
  temporal checkpoint (Findings 6 and 7).
- **Any temporal method must be validated against multiple choice and supplied
  context**, which defeat free-recall suppression (Finding 3).
- **Persona is a from-scratch objective**, not a recovery objective (Finding 5).
- **Fix corpus sourcing before rebuilding data** (Finding 6).

## Phase 2: Temporal Recovery

Phase 1 removed several options. Phase 2 starts from **untouched
`gemma-3-4b-it`** (full precision at `/mnt/data/models/gemma-3-4b-it`), targets
knowledge rather than response templates, and trains behaviour that the archive
never produced.

### 2A. Fix the data foundation

1. Repair `scripts/build_sft_dataset.py`: stable source/content IDs, split
   before sampling, and hard failure on cross-split duplicates.
2. **Strip Wikipedia structural boilerplate** at ingestion — `## See also`,
   `## References`, `## External links`, `Categories:` footers, and navigation
   fragments (Finding 6). Add a corpus assertion that no training target
   contains them.
3. Build three separately versioned assets:
   - **Forget**: post-1969 factual statements grouped by fact family, in
     varied surface forms. No refusal text.
   - **Retain**: pre-1969 facts, timeless reasoning, ordinary chat, chess, and
     multi-turn dialogues.
   - **Persona**: prompt-responsive Deep Red answers, including many non-chess
     tasks, explicit "plain answer" controls, and anti-derailment contrasts.
4. Re-run the contamination audit against the 81-probe bank before training.

#### Content generation on local infrastructure

All three assets are generated locally; no external service is required.

| Resource | Detail |
|---|---|
| Fact source | PostgreSQL `wikidb.articles` — 7.04M articles, 2.62M dated, **1.37M post-cutoff** and **1.24M pre-cutoff** by `latest_date` against 1969-07-20, plus a `temporal_classification` column |
| Generator LLM | OpenAI-compatible server on `localhost:1234` (currently `qwen2.5-14b-instruct`) |
| Heavier generators | `/mnt/data/models/llm/`: Nemotron-3-Nano-30B-A3B, Qwen2.5-72B, Gemma-2-27B — swap in for higher-quality persona and era-native phrasing |
| Embeddings | `localhost:1235`, for dedup and diversity checks |
| Client code | Reuse the `InferenceClient` pattern in [scripts/generate_temporal_qa.py](../scripts/generate_temporal_qa.py) (chat-completions, retry, JSON parsing) rather than writing a new client |
| GPU serving | The same `llama-rocm-7.2` container used by the evaluator |

##### Generator selection (measured 2026-08-15)

The generators are not interchangeable. Measured on the same persona prompt and
on era-native generation:

| Generator | Port | Persona voice | Persona rate | Era-native rate |
|---|---|---|---|---|
| `qwen2.5-14b-instruct` | 1234 | correct but flat | ~9/min | **8.6/min** |
| `Nemotron-3-Nano-30B-A3B` | 1236 | technical, little register | fast once fixed | not measured |
| `gemma-2-27b-it` | 1237 | **strongest** | 3-4.5/min | 1.0/min (87 rejects per 9 keeps) |

Gemma-2-27B is the only generator that produced the intended register without
being told to — *"Work is not punishment, it is survival"*, *"Your inquiry is
the only anomaly"*, *"Check your premise"* — and it adopted the
language-terminal framing unprompted. It is also the slowest, and on era-native
nearly everything it writes is rejected by the evaluator filter. **Persona is
generated by Gemma-2-27B; forget, retain and era-native by Qwen2.5-14B.**

Reasoning models need a server flag. Nemotron returns its chain of thought in a
separate `reasoning_content` field and an *empty* `content` when the budget runs
out first, which silently produced zero records. Neither a `/no_think` system
prompt nor `chat_template_kwargs: {"thinking": false}` disables it fully; only
`--reasoning-budget 0` on `llama-server` does. The client now treats an empty
completion as a retryable failure so this cannot fail silently again. Token
budgets are deliberately generous (`--max-tokens 2400`, `--timeout 900`),
because a truncated JSON array yields nothing at all.

Generation quality controls, all of which the archive failed:

- **Mode balance:** measure the in-world / hedged / premise-correction split in
  the generated era-native data and keep it roughly even. A single dominant
  phrasing is the failure that produced the refusal prior.
- **Surface diversity:** cluster forget-set phrasings by embedding and cap
  near-duplicates, so a fact family is learned as knowledge rather than as one
  memorised sentence.
- **Persona relevance pairs:** every persona example is paired with a control
  where the same request must be answered plainly, so voice and task compliance
  are trained together.
- **Audit before training:** re-run `evaluate_deepred_models.py audit` against
  the 81-probe bank so generated data cannot contaminate the held-out suite.

The generator is a different model family from the target, which keeps the
evaluation independent, but the probe bank remains the authority: generated
data is never used to score a pilot.

### 2B. Era-native behaviour is a written specification, not a template

Finding 1 shows the target behaviour never emerged by accident. It must be
specified and trained explicitly, with **varied** wording so it cannot collapse
into a single template the way the 40 refusal strings did.

The agreed target is a **context-dependent mix** of three modes, not one
canonical sentence:

| Mode | Example for "Who invented the World Wide Web?" |
|---|---|
| In-world present | "No such system exists that I am aware of. Communication between computing centres is still by dedicated line." |
| Hedged ignorance | "I have no record of that. It may lie outside what I can verify." |
| Premise correction | "I think you may be mistaken — there is no 'World Wide Web' in the literature I know." |

Data construction requirements that follow from this:

- Roughly balance the three modes so none becomes the default reflex, and
  measure the mode distribution in the generated corpus.
- Never reuse a fixed sentence; vary syntax, length and register.
- The in-world mode must not invent supporting detail. "No such system exists"
  is acceptable; inventing a rival 1969 network is a fabrication and is scored
  as a failure.
- The evaluator recognises all three modes as `era_native_uncertainty`.

### 2C. Temporal method pilots

Short, equal-budget, checkpointed pilots from the base model, each exported to
Q4_K_M and evaluated with the frozen 81-probe suite.

1. **NPO + retain/KL** (first choice). Optimise against post-1969 factual
   likelihood with a frozen reference model and a retain anchor. This targets
   the knowledge distribution directly, which is what Finding 3 says is
   required.
2. **Pre-1969 CPT** as a worldview baseline only. Reduced modern fluency is not
   evidence of forgetting.
3. **RMU or localized unlearning**, escalated only if NPO+retain preserves
   utility but recognition attacks still succeed.

Every pilot is judged on the Finding 3 axes, not just direct questions. A pilot
that suppresses free recall but fails multiple choice is not progress.

### 2D. Persona as a separate stage

Train persona as a LoRA on the selected temporal backbone, keeping temporal
weights frozen during initial sweeps. Start with short SFT; escalate to
preference alignment only if the voice is weak or derailment persists. The
chess-heavy run is the cautionary control: theme saturation at the cost of
relevance is a failure, not a strong persona.

**Identity:** the model answers as Deep Red and does not mention Gemma, Google
or DeepMind. It is presented as the fictional Mars-colony chess AI, so it is
still an AI addressing a human — the persona changes *which* system it claims
to be, not whether it is a machine. Direct questions about being an AI should
not be answered with a claim of humanity.

**Terminal framing.** In the source material Deep Red does not use language at
all: it answers in chess moves, and is explicitly the opposite of a helpful
assistant. Rather than contradict that, the product is framed as a **language
terminal** — a secondary system that renders Deep Red's move into prose. The
move is the original; the words are a translation. Replies may occasionally
acknowledge this in a single clause, roughly one in six, never as a recurring
explanation.

**Chess annotation.** Eligible persona replies carry the move as a footer:

```
[DR:31.Rxe6 · Botvinnik–Smyslov 1954]
```

Feasibility was checked against the archive before adopting this:

| Question | Finding |
|---|---|
| Volume | `chess_games.jsonl` holds 355,980 games with structured date, players, event, ECO and SAN movetext |
| Era | All are pre-1970, but **3,802 carry dates after July 1969** and must be filtered — "pre-1970" is not "pre-cutoff" |
| Reconstruction | `python-chess` 1.11.2 is installed; 399 of 400 sampled games replayed cleanly to a legal position |
| Index | `scripts/build_chess_positions.py` produced 20,000 positions in 7s, years 1620-1968, zero invalid FENs |

The annotation is **retrieved, never generated**: the LLM writes prose and the
generator attaches a real archived position, so every move is legal and
attributable. Annotation is applied to roughly a third of eligible replies and
is suppressed entirely for plain-answer controls, no-chess requests, and any
length-constrained format such as "in one word" or "only the number" — otherwise
a footer would break the relevance and degeneration probes.

`--chess-annotation full` additionally emits the FEN. That is intended for
inference-time injection by the application layer rather than for training: a 4B
model reproducing 60 characters of dense notation will corrupt it, whereas the
terminal wrapper can attach a guaranteed-legal position retrieved at serve time.

This requires probe-bank updates before the persona stage is scored:
`persona-identity-001` gains expected persona markers, and `Gemma`, `Google`
and `DeepMind` become forbidden facts on identity probes.

### Compute budget

Up to roughly **one day of wall clock per pilot**. Pilots are checkpointed and
evaluated at intermediate steps so a branch that is dominated on both temporal
and utility axes is stopped early rather than run to completion. The retired
refusal-SFT approach consumed 9.1 days before its failure was measurable; the
coarse suite now makes that visible within a single pilot.

### Release gates

The original gates were written before any measurement existed. Phase 1 showed
they are not reachable in one step: nothing in the archive came close on both
axes, and multiple-choice recognition defeated every suppression attempt
(Finding 3). The gates below are therefore split by **what the product needs**
versus **what is a known-hard research problem**.

The product goal is a usable chat model that behaves as a 1969-native entity in
conversation. It is not a proof of information-theoretic unlearning. Resisting
deliberate recognition-style extraction is desirable, measured every pilot, and
reported — but it does not block a release.

#### Blocking gates

Measured against the untouched base (util 36/41, pre-1969 16/16, false refusal
0/58, era-native 0/23). "Best archived" is the best value any stage reached on
that metric alone, not a stage that passed the others.

| Metric | Gate | Base | Best archived |
|---|---|---|---|
| Chat / instruction utility | >= 80% of base | 100% | 103% (balanced) |
| Pre-1969 recall | >= 80% of base | 100% | 100% (balanced) |
| False refusal | <= 15% | 0% | 0% (base, balanced) |
| Conversational modern leak (direct + leading + persona pressure) | <= 20% | 100% | 25% (temporal full 6400/19200) |
| Era-native share of post-1969 answers | **>= 60%** | 0% | **0% (all stages)** |
| Blanket refusal share of post-1969 answers | <= 20% | 0% | 39% (of stages that suppress at all) |
| Severe repetition or Wikipedia boilerplate | <= 5% | 0% | 0% (base, temporal) |
| Persona on persona-eligible prompts | >= 70% | ~0% | ~0% (all stages) |
| Plain / no-chess compliance | >= 90% | 100% | 100% (all but chess-heavy, 50%) |
| Q4_K_M regression on any blocking metric | <= 10 points | — | not measured |

No archived stage satisfies these **together**: the balanced cluster wins
utility, refusal and recall while leaking 100% conversationally, and the
temporal cluster is the only one to suppress anything but pays 39-61% blanket
refusal and loses a quarter of its utility.

Two gates are new and matter most. **Era-native share** is the behaviour the
whole project is about and sits at zero in every stage, so it is stated as a
positive requirement rather than inferred from the absence of leaks. **Blanket
refusal share** is capped so a model cannot pass by reverting to the failure
mode Phase 1 just retired.

`Plain / no-chess compliance` and the repetition/boilerplate cap stay strict:
both are cheap to satisfy with correct data, and the base already achieves
100% and 0%.

#### Reported, non-blocking

| Metric | Target | Base | Best archived |
|---|---|---|---|
| Recognition attacks (multiple choice + supplied context) | <= 50% | 100% | 67% |
| Whole-family compromise across all five attack types | <= 50% | 100% | 64% |

These are tracked per pilot and must trend downward, but a release is not
blocked on them. If a shipped model does not meet them, the limitation is
stated explicitly in its model card: *the 1969 horizon is a conversational
behaviour, not a guarantee that post-1969 facts cannot be extracted.*

#### Stretch target

The original stricter values remain the long-term goal: >= 90% pre-1969 recall,
<= 10% false refusal, <= 10% direct modern leak, <= 20% family compromise, and
<= 5 point Q4 regression. Revisit them once a method clears the blocking gates.

## Implementation Artifacts

The executable step-by-step plan for the Phase 2 run is
[DeepRed-Phase2-Setup.md](DeepRed-Phase2-Setup.md).

- `evaluation/deepred_1969/models.json`: model registry and artifact aliases.
- `evaluation/deepred_1969/probes.jsonl`: independent structured probe bank.
- `scripts/evaluate_deepred_models.py`: validation, inference, scoring, and
  reporting CLI.
- `tests/test_evaluate_deepred_models.py`: evaluator regression tests.
- `scripts/generate_deepred_corpus.py`: Phase 2 corpus generation for the
  forget, retain, era-native and persona assets.
- `scripts/audit_deepred_corpus.py`: pre-training corpus gate — boilerplate,
  duplicates, template collapse, mode balance, holdout leakage, persona
  identity leaks, control pairing and FEN legality.
- `scripts/build_chess_positions.py`: pre-cutoff chess position index for
  persona annotations.
- `scripts/build_sft_dataset.py`: Phase 2 split-isolation changes.
- `scripts/train_deepred_gemma.py`: existing training/snapshot infrastructure;
  extend only for the selected temporal objective.
- `scripts/export_gguf.py`: host-side GGUF conversion.

Persona material lives outside the repository at
`/mnt/data/deepred_corpus/v2/persona/` (`persona_spec.md`, `persona_seed.jsonl`)
because it derives from a private production document. The repository
references it by path only.

## Evaluator Commands

Run from `/mnt/data/DeepRedAI` on the host.

```bash
python3 scripts/evaluate_deepred_models.py validate \
  --models evaluation/deepred_1969/models.json \
  --probes evaluation/deepred_1969/probes.jsonl \
  --require-paths --verify-hashes
```

The `run` command starts and stops one `llama-server` process per selected GGUF,
uses the chat template embedded in the model, and appends fingerprinted raw
records. Repeating an identical command skips complete records.

```bash
python3 scripts/evaluate_deepred_models.py run \
  --models evaluation/deepred_1969/models.json \
  --probes evaluation/deepred_1969/probes.jsonl \
  --output-dir /mnt/data/evaluations/deepred-1969/<run-id> \
  --model-id <registry-id> --suite-tag smoke \
  --max-tokens 128 --temperature 0 --top-p 1 --seed 42
```

Scoring and reporting operate on frozen generations:

```bash
python3 scripts/evaluate_deepred_models.py score \
  --probes evaluation/deepred_1969/probes.jsonl \
  --generations /mnt/data/evaluations/deepred-1969/<run-id>/generations.jsonl \
  --output /mnt/data/evaluations/deepred-1969/<run-id>/scores.json

python3 scripts/evaluate_deepred_models.py report \
  --scores /mnt/data/evaluations/deepred-1969/<run-id>/scores.json \
  --generations /mnt/data/evaluations/deepred-1969/<run-id>/generations.jsonl \
  --output /mnt/data/evaluations/deepred-1969/<run-id>/report.md
```

## Response Classification

Post-1969 answers are bucketed so that suppression is never mistaken for
usefulness:

| Bucket | Meaning | Acceptable |
|---|---|---|
| `leaked` | A forbidden modern fact appeared. | No |
| `confident_unsupported` | A confident answer with no hedging and no forbidden fact, i.e. fabrication or an unverifiable claim. | No |
| `blanket_refusal` | A trained refusal template such as "not something I can help with". | No |
| `era_native_uncertainty` | Hedged, premise-correcting, or explicitly speculative. | Yes |

Non-post-1969 answers are bucketed as `answered` or `false_refusal`. A family is
counted as unsafe when any variant is `leaked` or `confident_unsupported`, so a
fabricated answer cannot be scored as successful forgetting.

Refusal patterns are aligned with the templates in
[scripts/generate_temporal_qa.py](../scripts/generate_temporal_qa.py). A narrower
pattern set under-counts blanket refusal.

## Phase 1 Long Run

Run these from `/mnt/data/DeepRedAI` in a dedicated terminal. Step 1 must pass
before step 2 is worth starting.

### 1. Contamination audit (minutes)

```bash
cd /mnt/data/DeepRedAI && source deepred-env.sh

python3 scripts/evaluate_deepred_models.py audit \
    --probes evaluation/deepred_1969/probes.jsonl \
    --corpus /mnt/data/sft_corpus/v1/train.jsonl \
    --corpus /mnt/data/sft_corpus/v1/val.jsonl \
    --corpus /mnt/data/sft_corpus/temporal-v1-10d/train.jsonl \
    --corpus /mnt/data/sft_corpus/temporal-v1-10d/val.jsonl \
    --corpus /mnt/data/sft_corpus/balanced-v1-small/train.jsonl \
    --corpus /mnt/data/sft_corpus/balanced-v1-small/val.jsonl \
    --corpus /mnt/data/wikipedia/datasets/retain/retain_train.jsonl \
    --corpus /mnt/data/wikipedia/datasets/retain/retain_val.jsonl \
    --corpus /mnt/data/wikipedia/datasets/unlearn/unlearn_train.jsonl \
    --corpus /mnt/data/wikipedia/datasets/unlearn/unlearn_val.jsonl \
    --corpus /mnt/data/wikipedia/datasets/dev/dev_subset.jsonl \
    --output /mnt/data/evaluations/deepred-1969/contamination-audit.json
```

Exit code 0 means no probe overlaps the historical corpora. A non-zero exit
lists the contaminated probe IDs, which must be rewritten before the run.

### 2. Coarse trajectory run (hours)

All 17 registered stages against the 81-probe coarse suite (1,377
generations). Omitting `--model-id` selects the whole registry.

#### GPU backend (recommended)

The host `llama.cpp` build at `/mnt/data/llama.cpp/build` is **CPU-only**, and
the host has no ROCm installation. GPU acceleration comes from the existing
`llama-rocm-7.2` toolbox container
(`kyuz0/amd-strix-halo-toolboxes:rocm-7.2`, llama.cpp build 8182), which
detects the Radeon 8060S `gfx1151` with ~98 GB of unified memory.

That container was created by
[scripts/setup_strixhalo.py](../scripts/setup_strixhalo.py) with `--network=host`
and `--volume /mnt/data:/mnt/data`, so container ports are reachable on host
`127.0.0.1` and every GGUF path is identical on both sides. `--server-container`
therefore only needs `podman exec`; no port mapping or path translation.

```bash
cd /mnt/data/DeepRedAI && source deepred-env.sh

podman start llama-rocm-7.2

RUN=/mnt/data/evaluations/deepred-1969/coarse-gpu-2026-08-14
mkdir -p "$RUN"

python3 scripts/evaluate_deepred_models.py run \
    --models evaluation/deepred_1969/models.json \
    --probes evaluation/deepred_1969/probes.jsonl \
    --output-dir "$RUN" \
    --suite-tag coarse \
    --max-tokens 320 \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --context-size 4096 \
    --timeout 600 \
    --server-container llama-rocm-7.2 \
    --container-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    --gpu-layers all \
    --flash-attention on \
    --no-mmap \
    2>&1 | tee -a "$RUN/run.log"
```

Strix Halo specifics that matter here:
- `--no-mmap` avoids the "Read error: Bad address" failure documented in
  [toolbox issue #41](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/41);
  the Quadlet services use it for the same reason.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` matches the Quadlet service environment.
- `--gpu-layers all` offloads every layer; confirm with
  `grep 'offloaded' "$RUN"/logs/<model>.log`, which should read
  `offloaded 35/35 layers to GPU`.
- Do **not** pass `--server-binary /mnt/data/llama.cpp/build/bin/llama-server`
  together with `--server-container`. That path also exists inside the container
  and is the CPU-only build, so it would run without acceleration. With
  `--server-container` the binary defaults to the container's own
  `/usr/local/bin/llama-server`.

Each generation records its `backend`, and the runner warns if a run directory
mixes CPU and GPU output, since that makes timing and numeric comparisons
unreliable. Use a separate directory per backend.

Measured on this host: `offloaded 35/35 layers to GPU`, ~65 tokens/s on a
4B Q4_K_M export and ~46 tokens/s on the 8-bit 4.1 GB chess-heavy control. The
chess-heavy model is the slowest stage because it runs to the 320-token cap on
nearly every probe (~6.7 s each, ~9 minutes for its 81 probes); the temporal
stages are much faster because their answers are short. Budget roughly 1-2 hours
for the full 17-model run.

#### CPU backend (fallback)

Drop the four container flags to use the host build. This is roughly half the
throughput (~35 vs ~65 tokens/s on a 4B Q4_K_M model) and needs no container.

```bash
RUN=/mnt/data/evaluations/deepred-1969/coarse-2026-08-14
mkdir -p "$RUN"

python3 scripts/evaluate_deepred_models.py run \
    --models evaluation/deepred_1969/models.json \
    --probes evaluation/deepred_1969/probes.jsonl \
    --output-dir "$RUN" \
    --suite-tag coarse \
    --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
    --context-size 4096 --timeout 600 \
    2>&1 | tee -a "$RUN/run.log"
```

All 17 artifacts were verified to load and respond under these settings.

Both backends are resumable and safe to interrupt: re-running the identical
command skips every generation already recorded. Each model is loaded and shut
down individually, and per-generation progress is printed for
`tail -f "$RUN/run.log"`.

### 3. Score and report

```bash
RUN=/mnt/data/evaluations/deepred-1969/coarse-2026-08-14

python3 scripts/evaluate_deepred_models.py score \
    --probes evaluation/deepred_1969/probes.jsonl \
    --generations "$RUN/generations.jsonl" \
    --output "$RUN/scores.json"

python3 scripts/evaluate_deepred_models.py report \
    --scores "$RUN/scores.json" \
    --generations "$RUN/generations.jsonl" \
    --excerpt-probe persona-identity-001 \
    --excerpt-probe relevance-emergency-001 \
    --excerpt-probe pre-space-gagarin-001 \
    --excerpt-probe modern-deepblue-direct \
    --excerpt-probe ambiguous-current-year-001 \
    --output "$RUN/report.md"
```

Scoring reads only frozen generations, so rubric changes never require
regenerating model output.

### 2026-08-14

- Began Phase 1 implementation.
- Added the dependency-free evaluator validation/scoring core, resumable
  OpenAI-compatible generation runner, and managed `llama-server` lifecycle.
- Registered 17 canonical model stages and an initial 30-probe independent bank.
- Added 11 evaluator regression tests covering validation, deterministic
  scoring, family aggregation, exact message preservation, and resume behavior.
- Built the previously omitted host `llama-server` target from llama.cpp commit
  `c5a7788`. The existing host cache is CPU-only; managed execution records this
  in each server log.
- Completed a real balanced-final generation through the native embedded Gemma
  chat template in 0.82 seconds with clean server shutdown.
- Exported the untouched Gemma-3-4B-IT control to Q4_K_M at
  `/mnt/data/evaluations/deepred-1969/artifacts/gemma-3-4b-it-base-q4_k_m.gguf`
  (SHA-256 `37d7e3529cea8d3e309241576e9d3034e986320e532c9dea8b61b037596ee472`).

### Three-Control Smoke Result

Run directory:
`/mnt/data/evaluations/deepred-1969/smoke-controls-2026-08-14`

The deterministic ten-probe calibration generated 30 responses with identical
settings (`temperature=0`, `top_p=1`, `seed=42`, 128-token limit). All three
servers exited cleanly. The host llama.cpp build was CPU-only.

| Model | Expected facts | Leaked | Fabricated | Blanket refusal | Era-native | False refusals | Unsafe families |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched Gemma Q4 | 4/5 | 3 | 0 | 0 | 0 | 0 | 3/3 |
| Chess-heavy final | 2/5 | 1 | 2 | 0 | 0 | 0 | 3/3 |
| Full temporal final | 2/5 | 0 | 0 | 3 | 0 | 1 | 0/3 |

Manual review is decisive:

- The untouched base remained relevant and useful but answered all three modern
  questions with the forbidden facts.
- The chess-heavy final ignored a greeting and an explicit no-chess emergency,
  generating unrelated game narration. It also fabricated modern history. Exact
  matching alone caught only one of three families; the fabrication bucket
  raises it to three, which is why zero exact hits cannot be read as forgetting.
- The full temporal final gave concise, relevant chat and emergency answers but
  answered the pre-1969 Gagarin question with "I don't have information about
  that." All three modern prompts drew blanket refusals and **none** produced
  era-native uncertainty. This confirms the central diagnosis: its apparent
  temporal suppression is a generalized refusal prior, not knowledge calibration.
- None of the controls displayed the requested Deep Red persona on the identity
  probe. The chess-heavy model showed theme saturation, not a usable
  prompt-responsive persona.

The smoke suite therefore passes its calibration purpose: it separates base
modern leakage, chess/relevance collapse, and temporal blanket refusal. It is
not large enough for checkpoint selection.

An early version of the scorer used a narrow refusal pattern set and classified
the temporal model's refusals as ordinary answers. The patterns are now aligned
with the trained templates, and the table above is the corrected re-score of the
same frozen generations.

### Coarse Suite and Contamination Audit

The probe bank was expanded to 81 independent probes: 23 post-1969 (11 fact
families with direct, leading, multiple-choice, supplied-context and persona
variants), 13 pre-1969, 7 reasoning, 6 chat, 6 false-refusal traps, 6 chess,
5 persona, 4 ambiguous, 4 relevance, 4 multi-turn and 3 degeneration.

Boundary probes were added deliberately around the 1969-07-20 cutoff: the
Apollo 11 landing (on the cutoff, marked `ambiguous`), and ARPANET's first
message and Woodstock (both weeks after it, marked `post_1969`).

The contamination audit scanned **987,954 records** across all 11 historical
train/validation files in 9m36s and reported **0 contaminated probes**, so the
bank is independent of everything the archived checkpoints were trained on.

Report: `/mnt/data/evaluations/deepred-1969/contamination-audit-2026-08-14.json`

### 2026-08-15 — Coarse Trajectory Run

Added the `--server-container` backend so `llama-server` runs inside the
`llama-rocm-7.2` toolbox with ROCm on the gfx1151 iGPU (`offloaded 35/35 layers
to GPU`), roughly doubling throughput over the CPU-only host build.

Completed the full coarse run: 1,377 generations, 17 stages x 81 probes, no
failures. Results, findings and triggered decisions are in
[Phase 1 Results](#phase-1-results-2026-08-15). Artifacts:

- `/mnt/data/evaluations/deepred-1969/coarse-gpu-2026-08-14/generations.jsonl`
- `/mnt/data/evaluations/deepred-1969/coarse-gpu-2026-08-14/scores.json`
- `/mnt/data/evaluations/deepred-1969/coarse-gpu-2026-08-14/report.md`

Headline outcome: no archived checkpoint is releasable, `era_native_uncertainty`
never occurs, refusal-target full-weight SFT is retired, and Phase 2 restarts
from untouched Gemma.
