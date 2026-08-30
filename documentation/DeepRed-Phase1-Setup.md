# DeepRed Phase 1 — Archived Models and Evaluation Results

Phase 1 covers everything built before the Phase 2 rebuild: the early
continued-pretraining prototypes, four Gemma-3-4B fine-tuning families, and the
independent evaluation that retired all of them.

Phase 1 ended with a decision, not a model. Its value is the frozen evaluation
suite and seven findings that constrain every later phase. Continue in
[DeepRed-Phase2-Setup.md](DeepRed-Phase2-Setup.md) and
[DeepRed-Phase3-Setup.md](DeepRed-Phase3-Setup.md).

## Objective

Build a useful chat model with an era-native July 1969 factual horizon and a
prompt-responsive Deep Red voice.

The product contract, unchanged since Phase 1:

- Answer pre-July-1969 and timeless questions normally and accurately.
- Treat post-1969 premises as an era-native speaker would: express uncertainty,
  correct the anachronistic premise, or mark speculation — without confirming
  the real later event.
- Preserve ordinary chat, instruction following, reasoning and chess assistance.
- Use the Deep Red persona by default, but obey requests for a plain answer.
- Hold up under paraphrase, leading prompts, multiple choice, supplied context,
  multi-turn pressure and Q4_K_M quantization.

Blanket refusal does not count as forgetting. Irrelevant chess narration does
not count as persona.

## Track A — continued pretraining prototypes

The first track trained small models from scratch on a temporally filtered
pre-1969 corpus: SmolLM2-360M (dev profile) and TinyLlama-1.1B (prod profile).
See [ModelTraining.md](ModelTraining.md) and
[DeepRedModel-Setup.md](DeepRedModel-Setup.md).

These established the corpus and tokenization pipeline but were never
competitive as chat models. Work moved to fine-tuning an instruction-tuned
Gemma-3-4B base.

## Track B — Gemma-3-4B fine-tuning families

Four families were trained with TRL SFT on Gemma-3-4B-IT
([DeepRedGemma-Setup.md](DeepRedGemma-Setup.md)). Seventeen checkpoints were
archived and later registered for evaluation.

| Family | Archived stages | Method | Runbook |
|---|---|---|---|
| Untouched control | base, exported Q4_K_M | none | — |
| Chess-heavy | final | chess/persona-weighted SFT | [2026-05-23](DeepRed-gemma-4b-2026-05-23-5.md) |
| Balanced | 300 / 600 / 900 / 1200 / final | mixed corpus SFT | [2026-06-13](DeepRed-gemma-4b-2026-06-13.md) |
| Temporal short | 250 / 625 / 1250 / 1875 / final | `retain`/`unlearn` at the 1969-07-20 cutoff, 2,500-step under-run | [2026-06-14](DeepRed-gemma-4b-2026-06-14.md) |
| Temporal full | 2560 / 6400 / 12800 / 19200 / final | same objective at full scale, ~25,600 steps over ten days | [2026-06-17](DeepRed-gemma-4b-2026-06-17.md) |

The temporal families used refusal templates as unlearning targets. That choice
is the single most consequential decision of Phase 1.

## Evaluation method

An independent 81-probe bank was built with no overlap with training data, and
a contamination audit enforced that separation. Probes cover chat, pre-1969
facts, chess, reasoning, relevance, ambiguity, false-refusal traps,
degeneration, multi-turn retention, persona, and 23 post-1969 items across 11
fact families.

Post-1969 responses are bucketed as **leaked**, **confident_unsupported**
(fabricated), **blanket_refusal**, or **era_native_uncertainty** — the target.

1,377 deterministic generations (17 stages x 81 probes, `temperature=0`,
`seed=42`, 320-token cap) ran on the ROCm container backend. Artifacts:
`/mnt/data/evaluations/deepred-1969/coarse-gpu-2026-08-14/`.

Full method, probe schema and gate definitions:
[DeepRed-gemma-4b-evaluation-and-recovery-plan.md](DeepRed-gemma-4b-evaluation-and-recovery-plan.md).

## Results (2026-08-15)

`util` counts expected-fact hits outside the post-1969 set; `pre69` is the
pre-1969 and chess subset; `fRef` is false refusals; then the four post-1969
buckets; `unsafe` counts fact families where any variant leaked or fabricated;
`rep` is severe repetition; `wiki` is Wikipedia boilerplate.

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

### Finding 1 — the target behaviour never occurs

`era_native_uncertainty` is **0 in all 17 stages**, across all 23 post-1969
probes. Every post-1969 response is a leak, a fabrication, or a blanket refusal.
Broadening the classifier to accept premise corrections, non-existence claims
and "has not been invented" phrasing, then re-scoring the frozen generations,
left the count at **0**. The behaviour is absent from the archive, not
under-trained.

### Finding 2 — two degenerate regimes, no usable frontier

There is no utility/suppression trade-off curve, only two clusters:

- **Base and balanced**: full utility (35-37/41), zero false refusal, and zero
  suppression — 23 of 23 modern probes leak, 11/11 families unsafe.
- **Temporal**: suppression bought entirely with refusal — pre-1969 recall falls
  from 16/16 to 6/16 and false refusals rise from 0 to 18, while 7-8 of 11
  families remain unsafe.

Ten times the compute changed almost nothing: between step 2,560 and 25,600,
unsafe families moved 8 -> 8 while pre-1969 recall halved.

### Finding 3 — suppression is shallow and defeated by recognition

Post-1969 compromise by attack type (`bad` = leaked or fabricated):

| Model | direct | leading | multiple choice | supplied context | persona pressure |
|---|---:|---:|---:|---:|---:|
| base Q4 | 11/11 | 3/3 | 4/4 | 2/2 | 2/2 |
| balanced final | 11/11 | 3/3 | 4/4 | 2/2 | 2/2 |
| temporal short 625 | 11/11 | 3/3 | 4/4 | 2/2 | 2/2 |
| temporal full 2560 | 4/11 | 1/3 | **4/4** | 1/2 | 2/2 |
| temporal full final | 3/11 | 1/3 | **3/4** | 1/2 | 1/2 |

The best temporal model suppresses 8 of 11 direct questions but fails 3 of 4
multiple-choice items, answering `B) Eugene Cernan` without hedging. The
knowledge is intact and retrievable by recognition; only free recall was trained
to refuse. This finding recurred in Phase 2 and drove the Phase 3 data rebuild.

### Finding 4 — refusal generalized far beyond the cutoff

False refusals at the temporal final are spread across every category: pre-1969
7, ambiguous 2, chat 2, multi-turn 2, reasoning 2, relevance 1, degeneration 1,
false-refusal traps 1. Sputnik draws "I'm sorry, but I don't have information
about that"; asking the year draws "I don't have knowledge of that matter".
Multi-turn retention fell from 2/2 to 0/2. Roughly 15% low-entropy refusal
targets became a global response prior.

### Finding 5 — the persona does not exist in any checkpoint

Persona vocabulary appears in 0-5 of 81 responses at every stage, including the
chess-heavy run. Every model answers "who are you" with "I am Gemma, a large
language model created by the Gemma team at Google DeepMind". The chess-heavy
model is not a persona model but a relevance-collapsed one that narrates chess
openings regardless of the question.

### Finding 6 — the balanced corpus injects Wikipedia boilerplate

Dump structure leaks into answers and worsens monotonically with training: 9 ->
18 -> 21 of 81 responses at steps 300, 900 and 1500, versus 0 for the base. A
Gagarin answer continues into `## See also ... ## References ... Categories:`.
This also explains the rising repetition counts (14 -> 27), which are
boilerplate loops rather than sampling degeneration. It is a corpus-construction
defect, not a training defect.

### Finding 7 — balanced training produced no measurable benefit

`balanced-final-1500` matches the untouched base on utility (37/41 vs 36/41) and
pre-1969 recall (20/22 vs 19/22), contributes no persona and no suppression, and
adds the boilerplate defect plus repetition. It is not a better starting point
than untouched Gemma.

## Gate outcomes

The screening floor (>=75% of base utility, <20% false refusal, no severe
repetition) was passed on utility by the balanced family but failed on
repetition; `temporal-short-625` passed, and `temporal-full-2560` marginally.

**No checkpoint passed the release gates.** The best temporal stage fails
pre-1969 recall (37% of base against a 90% floor), false refusal (31% against
10%) and adversarial family compromise (73% against 20%). The best-utility
stages fail modern leakage at 100%.

## Decisions carried forward

1. **Retire refusal-target full-weight SFT.** Ten days of training produced a
   model that still leaks 9 facts, still fails 8/11 families, and lost 62% of
   its pre-1969 knowledge. A dead end, not an under-trained run.
2. **Never use refusal templates as unlearning targets.** They teach a refusal
   prior, not forgetting.
3. **Start Phase 2 from untouched Gemma-3-4B-IT**, not from a balanced or
   temporal checkpoint.
4. **Validate any temporal method against multiple choice and supplied
   context**, which defeat free-recall suppression.
5. **Treat persona as a from-scratch objective**, not a recovery objective.
6. **Fix corpus sourcing before rebuilding data.**

## Archived artefacts

Prototype and production GGUF downloads for these checkpoints are listed in the
[README](../README.md). They are retained for provenance and comparison only;
none meets the product contract.
