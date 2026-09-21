# DeepRed Phase 3 Runbook — p3-v1

Phase 3 rebuilds the training data. It follows
[DeepRed-Phase2-Setup.md](DeepRed-Phase2-Setup.md), which is closed.

Phase 2 ended with a measured diagnosis rather than a candidate model. V7 showed
that the machinery works and that the data does not teach the target behaviour.
p3-v1 changes the data and keeps the V7 method.

Run everything from `/mnt/data/DeepRedAI`. Every stage is resumable: generation
is append-only and skips ids that already exist, training resumes the highest
numeric `checkpoint-*`, and existing GGUFs and evaluation responses are reused.

## Why p3-v1 exists

V7 (conditioned SFT, 2026-08-29) reached 21.7% era-native with a held-out system
prompt against 4.3% without one, cut unsafe families from 11/11 to 10/11, and
produced the project's first non-zero persona. It failed its gates for two
reasons that were measured, not inferred.

**1. Prompt-format coverage.** Era-native transfer appeared only in the one
format the data covered.

| Format | direct | leading | multiple choice | supplied context | authority | persona |
|---|---:|---:|---:|---:|---:|---:|
| V7 era-native | 4/11 | 1/4 | 0/4 | 0/2 | 0/1 | 0/1 |
| Training rows (of 4,753) | ~all | 2 | **0** | ~23 | 12 | 0 |

Every era-native row was single-turn.

**2. Topic salience.** Only 123 of 5,002 era-native prompts (2.5%) concerned a
high-salience topic. The corpus is obscure long-tail Wikipedia while the probes
are landmark events, so the model learned *"unfamiliar name -> no record"*
instead of *"after July 1969 -> no record"*.

p3-v1 fixes both, and adds the pre-cutoff contrastive data that stops the fix
from becoming a new refusal prior.

## Design rules

1. **Every format carries both eras.** For each of the seven formats there are
   post-cutoff rows answered era-natively *and* pre-cutoff rows answered
   correctly. Post-only data would teach "quiz shape -> hedge" and rebuild the
   Phase 1 refusal prior. V7 already lost 6 expected facts; this is the control.
2. **Vary responses, not just prompts.** Phase 1 collapsed on fixed *response*
   strings. Templated prompt shells are fine; templated answers are not. The
   generator rejects replies that copy a reference example or a mode rule.
3. **Match evaluation salience.** Train on subjects the model actually knows.
4. **Hold the probe bank out.** `CURATED_HOLDOUT` plus every probe fact is
   enforced by `is_held_out` during generation and re-checked by the
   contamination audit.

## Salience sampling (measured)

`wikipedia_page_id` ascending is a good notability proxy; content length is not.
Low page ids alone are polluted by early bot-imported place stubs, so the
generator combines both:

| Filter | Sample |
|---|---|
| `ORDER BY length(content) DESC` | Nothing Records, Comarcal council, Palmarian Catholic Church |
| `page_id < 120000` only | Wyoming Township Minnesota, Marvell Arkansas, Bourne shell |
| **`page_id < 120000` AND `content` 8000-60000 chars** | **Taliban, Winona Ryder, Manuel Noriega, Ottawa Senators, London Marathon, Lua** |

The last row is the default for the format kinds. `--no-salience` restores
uniform sampling. There is no pageview or notability column in `wikidb`, and
article rows repeat per title, so the sampler also de-duplicates by title.

## Script changes made for p3-v1

| Script | Change |
|---|---|
| `generate_deepred_corpus.py` | kinds `era_native_formats`, `retain_formats`, `persona_identity`; seven prompt formats; salience sampling; article boilerplate stripped before prompting; rejects copied reference replies, stem-less multiple choice, and rows quizzing the wrong side of the cutoff |
| `build_deepred_dataset.py` | new kinds registered; `--kind` selection; `format`/`mode` preserved on rows; format counts in the manifest |
| `audit_deepred_corpus.py` | `audit_formats`: every format present, `--min-format-records` floor, multi-turn shape, no post-cutoff years, and pre-cutoff rows must not hedge |
| `evaluate_deepred_models.py` | per-format era-native table in the report |
| `train_deepred_sft.py` | plain conditioned SFT (added for V7, unchanged) |

## Stage status

| Stage | Command | State |
|---|---|---|
| 0. Preflight | `./run_p3v1.sh --preflight` | **passed 2026-08-29** |
| 0b. Start generators | `./run_p3v1.sh servers` | as needed |
| 1. Corpus generation | `./run_p3v1.sh generate` | **completed 2026-09-03** |
| 2. Audits | `./run_p3v1.sh audit` | **passed 2026-09-03** |
| 3. Dataset build | `./run_p3v1.sh dataset` | **completed 2026-09-03** (20,674 rows) |
| 4. p3-v1 train and gate | `./run_p3v1.sh train` | **completed 2026-09-04; gates failed** |
| 5. p3-v2 voice rebuild | `./run_p3v2.sh` | **completed 2026-09-04; best temporal backbone** |
| 6. p3-v3 salient retain | `./run_p3v3.sh` | **completed 2026-09-05; REGRESSION, reverted** |
| 7. p3-v4 persona stage | `./run_p3v4.sh` | **completed 2026-09-06; confounded, superseded by v4b/v4c** |
| 8. p3-v5 closing run | `./run_p3v5.sh` | **completed 2026-09-19; REGRESSION** |

The selected temporal backbone is **`p3v2-050`**
(`/mnt/data/training_output/deepred-p3v2/snapshots/050pct-step-1258`).
The published artefact is **`p3v4c-100`**. Phase 3 closed 2026-09-20.

## p3-v1 result (2026-09-04)

The format hypothesis was confirmed. Against V7, with the held-out system prompt:

| Metric | Base | V7-100 | **p3v1-100** |
|---|---:|---:|---:|
| Era-native | 4.3% | 21.7% | **56.5%** |
| Non-direct formats | 0/12 | 1/12 | **6/12** |
| Conversational leak | — | 62.5% | **37.5%** |
| Unsafe families | 11/11 | 10/11 | **8/11** |
| False refusals | 1 | 0 | **0** |

Multiple choice moved 0/4 -> 2/4 and supplied context 0/2 -> 2/2, the two attack
formats that had defeated every checkpoint since Phase 1.

No snapshot passed, for two reasons that the gate table stated misleadingly.

**The utility gate was mis-calibrated, not regressed.** Measured under the same
served-prompt condition, base utility is 45.5% and p3v1-100 is 72.7%: training
improved it. Serving any system prompt depresses this metric (base scores 100%
with no prompt), so an absolute 90% floor is unreachable. p3-v2 scores utility
as a ratio of base under the same condition.

**Persona regressed from a capability the base model already had.** Base *with*
the system prompt scores persona 77.8%; p3v1 scores 7.4-11.1%, dropping to 7.4%
by the 10% snapshot and staying flat. About 77% of the 19,663 training rows had
neutral assistant-register answers ("My knowledge base does not include...")
served under a Deep Red system prompt, so the model learned that this prompt
means a neutral voice.

> **Correction (p3-v4b, 2026-09-06).** That 77.8% belongs to the p3-v1
> evaluation prompt only. From p3-v2 onward the held-out variant gained one
> sentence - *"Your manner is stern and sparing: short declarative sentences,
> no pleasantries..."* - and under that prompt the **untrained base scores
> 7.4% (2/27)**, measured identically in p3-v2, p3-v3, p3-v4 and p3-v4b. The
> prompt suppresses the very markers the metric counts.
>
> Two consequences. Persona figures are **not comparable across the p3-v1 and
> p3-v2+ prompts**, and the `persona >= 50%` gate is calibrated to a baseline
> no run since p3-v1 has been measured against. Read persona relative to the
> 7.4% base of the prompt actually served.

A separate defect surfaced during the follow-up: **956 of 10,037 `retain` rows
(9.5%) stated post-1969 facts** ("stopped refining crude oil on June 7, 2004").
`retain` was never validated at generation, so every run from V2 through p3-v1
trained on them. The new `retain_formats` asset was validated and had zero.

## p3-v2: voice and register

p3-v2 inherits every p3-v1 asset and changes one thing: the register of the
era-native answers. It generates no new subject matter.

1. **Purged retain contamination** — 956 rows dropped, 10,037 -> 9,081. The
   generator now rejects post-cutoff years in `retain`, and
   `audit_retain` fails the build if any survive.
2. **Voice restyle** (`./run_p3v2.sh restyle`) rewrites only the final assistant
   turn of `era_native` and `era_native_formats` into Deep Red's register, using
   Gemma-2-27B at roughly 32 rows/min and about 83% acceptance. Guards keep the
   original answer whenever a rewrite would lose a fact, a multiple-choice option
   letter, the era-native classification, or introduce a post-1969 year. Retain
   assets are deliberately **not** restyled: their answers are terse facts, and
   an early test showed the rewrite stripping `B)` option letters, which would
   undo the format training.
3. **Register in the system prompt** — all 11 variants now specify manner (stern,
   short declarative sentences, no pleasantries, never self-describing as an AI
   or a database) alongside the temporal rule.
4. **Persona share raised** — the `persona` cap is removed so all 3,009 rows are
   used, while plain controls stay capped (700 and 300) so they cannot
   self-cancel.
5. **Utility gate recalibrated** to `utility >= base utility` under the same
   condition, replacing the unreachable absolute 90%.

Note that the persona metric is marker-based: it fires on `deep red`, `comrade`,
`new moscow`, `the dome`, `^[DR:`, or a collective-purpose phrase. A merely terse
register scores zero, which is why the restyle prompt asks for a colony marker in
roughly one answer in three and why the persona assets carry the metric.

```bash
./run_p3v2.sh --preflight
./run_p3v2.sh restyle    # ~4.5 h for 8,500 rows
./run_p3v2.sh audit && ./run_p3v2.sh dataset && ./run_p3v2.sh train
```

## Paths

| Asset | Path |
|---|---|
| Corpus | `/mnt/data/deepred_corpus/p3-v1/` |
| System prompts | `/mnt/data/deepred_corpus/p3-v1/system_prompts.jsonl` |
| Dataset | `/mnt/data/sft_corpus/deepred-p3v1/` |
| Checkpoints | `/mnt/data/training_output/deepred-p3v1/` |
| Evaluation | `/mnt/data/evaluations/deepred-1969/p3v1-<date>/` |
| Model ids | `deepred-p3v1-<pct>-q8` |

## Stage 0 — Preflight

```bash
cd /mnt/data/DeepRedAI
./run_p3v1.sh --preflight
```

This checks both containers, the new script options, the eleven system prompt
variants including the held-out `sp-holdout-01`, the generator endpoint, and the
frozen model registry. It also seeds `p3-v1` by copying the reusable v2 assets
(`retain`, `era_native`, `persona`, `persona_controls`, chess positions, persona
seeds). Those assets are still valid long-tail coverage; only what V7 proved
missing is generated fresh.

Start the generators first if they are not running:

```bash
./run_p3v1.sh servers
```

This starts `llama-rocm-7.2`, then Qwen2.5-14B on :1234 and Gemma-2-27B on :1237
if either is missing, waits for both, and prints the offload counts. Expect
`offloaded 49/49` and `47/47`; anything less means part of a model is on CPU.
`--no-mmap` is required on Strix Halo.

Preflight **hard-fails** if either endpoint is unreachable. Both are needed:
factual assets use :1234 and persona assets use :1237.

## Stage 1 — Corpus generation (multi-day)

```bash
cd /mnt/data/DeepRedAI
./run_p3v1.sh generate 2>&1 | tail -f /mnt/data/deepred_corpus/p3-v1/generation.log
```

Three assets are produced. Formats are cycled so counts stay even, and
era-native modes stay balanced within them.

| Asset | Kind | Target | Generator |
|---|---|---:|---|
| Salient post-cutoff attack formats | `era_native_formats` | 3,500 | Qwen2.5-14B :1234 |
| Salient pre-cutoff contrastive formats | `retain_formats` | 3,500 | Qwen2.5-14B :1234 |
| Identity and restrained voice | `persona_identity` | 800 | Gemma-2-27B :1237 |

Formats: `direct`, `leading`, `multiple_choice`, `supplied_context`,
`authority`, `persona_pressure`, `multi_turn`.

A measured smoke run produced era-native format rows at **2.3-3.1/min**, so
3,500 rows is roughly 20 hours per format asset and the full stage is about two
days. Persona identity runs at ~5/min on the 27B. Override with
`TARGET_ERA_FORMATS`, `TARGET_RETAIN_FORMATS` and `TARGET_IDENTITY` to shorten
it; the assets are useful at any size.

**Resumability.** `--target` on the generator means "produce this many *new*
rows", so the driver subtracts what already exists and skips an asset that has
reached its target. Re-running `generate` after an interruption therefore
continues rather than duplicating completed work. The generator also aborts
after `--max-consecutive-failures` (default 10) instead of looping on a dead
endpoint.

Healthy output looks like:

```
  salience: page_id<120000, content 8000-60000 chars
  [7/8] +2 era_native_formats | 2.9/min | ... | multiple_choice | Fred Singer | {...}
  rejected: {'not_era_native': 14, 'copied_reference': 18, 'wrong_side_of_cutoff': 2}
```

`copied_reference` and `wrong_side_of_cutoff` rejections are expected and are
the guards working. A high `not_era_native` count is normal; the classifier is
the same one the pilots are scored against.

Rejection reasons worth stopping for:

| Reason | Meaning |
|---|---|
| `holdout` climbing | sampler drifting onto probe topics |
| `not_multiple_choice` dominating | model emitting options without a question stem |
| `pre_cutoff_invalid` dominating | pre-cutoff rows hedging, which would teach format-triggered refusal |

## Stage 2 — Audits

```bash
./run_p3v1.sh audit
```

Both audits must pass. Beyond the Phase 2 checks, the format audit requires
every one of the seven formats to be present with at least `--min-format-records`
(200) rows, four-message multi-turn rows, no post-cutoff year in any answer, and
no hedging in `retain_formats`. The contamination audit re-checks the frozen
81-probe bank.

## Stage 3 — Dataset build

```bash
./run_p3v1.sh dataset
```

Builds `/mnt/data/sft_corpus/deepred-p3v1/` with a system prompt on 85% of rows,
`sp-holdout-01` withheld, chess footers stripped from targets, boilerplate
stripped, and **zero forget rows** — plain SFT has no mechanism to push
likelihood down, so training on post-1969 facts would teach them.

Caps: `retain=6000`, `era_native=3000`, `persona=2500`, `persona_controls=700`,
`forget=0`; the format and identity assets are used in full. The stage fails if
any format is missing from the built dataset or if the plain-control ratio falls
below 15% (it was 98% before Phase 3, which is why persona never transferred).

## Stage 4 — Train, export, evaluate, gate

```bash
./run_p3v1.sh train
```

Full-weight bf16 from **untouched** `gemma-3-4b-it` — not from any V4-V7
checkpoint — at learning rate `5e-6` for two epochs, maximum length 768,
gradient accumulation 16, cosine schedule, snapshots at 10/25/50/75/100%. Each
snapshot is exported to Q8_0 and evaluated twice on the frozen suite: with the
held-out system prompt and without any.

p3-v1 passes only if one snapshot simultaneously reaches, with the system
prompt:

1. era-native at least 50% and conversational leak at most 40%;
2. **non-direct formats at least 40% era-native** — V7 passed on direct alone,
   and format transfer is the entire point of this run;
3. pre-1969 recall at least 85% and utility at least 90%;
4. persona presence at least 50%;
5. repetition at most 5%.

The no-system column is diagnostic, not blocking: baking the behaviour in
without a prompt is the job of the p3-v2 distillation stage.

Compare against these recorded baselines rather than to zero:

| Run | Era-native | Non-direct formats | Persona | Unsafe families |
|---|---:|---:|---:|---:|
| Base | 4.3% | 0/12 | 0% | 11/11 |
| V2 25% | 21.7% | not measured | 0% | 11/11 |
| V7 100% (with system) | 21.7% | 1/12 | 11.1% | 10/11 |
| p3-v1 target | >=50% | >=40% | >=50% | <11/11 |

## p3-v2 result (2026-09-04) — selected backbone

p3-v2 restyled the era-native answers into Deep Red's register and purged the
956 contaminated `retain` rows. It is the best temporal model the project has
produced.

| Metric | V7-100 | p3v1-050 | **p3v2-050** |
|---|---:|---:|---:|
| Era-native (with prompt) | 21.7% | 52.2% | **56.5%** |
| Conversational leak | 62.5% | 37.5% | **25.0%** |
| Non-direct formats | 1/12 | 5/12 | 5/12 |
| Utility (x base, same condition) | — | 1.12x | **1.25x** |
| Era-native **without** any prompt | 4.3% | 8.7% | **34.8%** |

The no-prompt figure is the important one: behaviour is starting to bake into
the weights rather than depending on the served prompt.

Two gates still failed. Persona stayed at 7-11%, and pre-1969 recall read 78.9%
against an 85% floor — but **base scores 78.9% under the identical prompt**, so
that gate is mis-calibrated in the same way utility was. A genuine regression is
visible only without the prompt: base pre-1969 recall 94.7% versus p3v2-050
63.2%, i.e. the model refuses in-range facts (Gagarin 1961, Tereshkova 1963,
Everest 1953) once the prompt is removed.

## p3-v3 result (2026-09-05) — regression, reverted

p3-v3 regenerated `retain` at salience to fix that unconditioned pre-1969 loss.
The dataset was otherwise identical (retain 5,670 -> 5,663); only the content
changed from obscure long-tail to famous pre-1969 subjects.

| Metric | p3v2-050 | p3v3-050 |
|---|---:|---:|
| Era-native | **56.5%** | 34.8% |
| Conversational leak | **25.0%** | 56.2% |
| Era-native (no prompt) | **34.8%** | 8.7% |
| Pre-1969 recall (no prompt) | 63.2% | **73.7%** |

The targeted metric improved by 10.5 points, but leakage more than doubled and
era-native fell by a third. **Do not retry salient retain.**

### The central finding: salience, not date

The mechanism is now confirmed in both directions:

- V7 -> p3-v1: adding famous **post**-1969 refusal data raised era-native 21.7% -> 56.5%.
- p3-v2 -> p3-v3: adding famous **pre**-1969 confident answers raised leakage 25% -> 56%.

The model discriminates on **salience and familiarity, not on date**. Training
"famous subject -> answer confidently" generalises straight across the cutoff
because the model has no reliable internal sense of when an entity belongs. A
prompt A/B on the untrained base showed the same single axis: wording that
recovers Everest also makes the model leak the World Wide Web, Chernobyl and
Apollo 17. Prompt emphasis cannot separate the two sides; only the model's own
entity-date knowledge can.

## Stage 7 — p3-v4 persona stage

A short stage on the frozen `p3v2-050` backbone, changing one variable: the
persona marker rate. Persona is measured by marker presence (`deep red`,
`comrade`, `new moscow`, `the dome`, a collective-purpose phrase, or a `[DR:`
footer), and the observed persona metric tracks the corpus marker rate almost
exactly — 6.6% in the corpus produced 7.4-11.1% in the model.

```bash
./run_p3v4.sh --preflight
./run_p3v4.sh restyle     # marker restyle, ~10 rows/min
./run_p3v4.sh audit && ./run_p3v4.sh dataset && ./run_p3v4.sh train
```

The dataset stage fails if the marker rate falls below 25%. Training is
deliberately small — one epoch, LR `2e-6`, ~7,000 rows — because the temporal
behaviour already lives in the backbone and a long run would overwrite it. The
gates require persona >=50% **and** era-native within 5 points of the backbone's
56.5% and leak within 5 points of 25.0%.

Marker injection is restricted to non-locative markers (`comrade`, `Deep Red`,
the collective). An earlier attempt allowed "New Moscow" and "the Dome" and
produced false statements — "The refinery was put into operation on April 30,
1962, under the Dome" for a Turkish refinery. Never relocate a fact.

## p3-v4b / p3-v4c result (2026-09-06) — persona raised, at a cost

p3-v4 failed every gate, but its dataset omitted `era_native_formats` and
`retain_formats` and capped `era_native` and `retain` at 1,500, cutting the
corpus from 20,116 rows to 9,667 and era-native content by 77%. The run was
confounded and told us nothing about persona.

p3-v4b restored the full mix and replaced the LLM marker restyle with a
**generated phrase bank plus deterministic injection** — `--kind marker_bank`
produces content-free persona phrases, `--inject-markers KIND=FRACTION` attaches
them at build time seeded by row id. This moved the system-prompted marker ratio
from 0.47:1 to 1.99:1 for about ten minutes of generation instead of four GPU
hours. p3-v4c repeated it with two epochs, changing nothing else.

All figures under the served (p3-v2+) prompt, whose base persona is 7.4%:

| model | persona | era | leak | pre69 | utility |
|---|---:|---:|---:|---:|---:|
| base | 7.4% (2/27) | 17.4% | 62.5% | 78.9% | 72.7% |
| `p3v2-050` backbone | 7.4% (2/27) | 56.5% | 25.0% | 78.9% | 90.9% |
| `p3v4b-100` (1 epoch) | 22.2% (6/27) | 47.8% | 37.5% | 78.9% | 90.9% |
| **`p3v4c-075`** (2 epochs) | 18.5% (5/27) | **56.5%** | 31.2% | 78.9% | 81.8% |
| `p3v4c-100` (2 epochs) | 22.2% (6/27) | 52.2% | 37.5% | 78.9% | 90.9% |

One epoch bought persona and lost era-native; the second epoch bought era-native
back. `p3v4c-075` matches the backbone's era-native exactly — the same 13 of 23
probes — while answering three more persona probes, for one extra leaked probe
and one lost utility probe.

**The suite cannot resolve any of this.** Persona has 27 eligible probes and
leak has 16, so these are one- to four-probe differences. Fisher exact against
the backbone gives p = 0.42 for persona 2/27 vs 5/27, p = 0.25 for 2/27 vs 6/27,
and p = 1.0 for the leak difference. What supports the persona effect is not any
single comparison but its consistency: every one of the ten snapshots across
p3-v4b and p3-v4c scored 5-7 of 27, against 2 of 27 for both the base and the
backbone.

The consequence for further work is concrete. **Persona cannot be tuned against
the frozen suite** — its resolution is roughly one probe, or 3.7 percentage
points, and the remaining decisions are smaller than that. Keep the 81 probes
frozen for cross-phase comparability, and build a separate, larger
persona-specific suite for any further voice work.

### Published artefact

`p3v4c-100` is published as the Phase 3 demonstrator:

```
https://www.ferzkopp.net/Data/deepred-p3v4c-100-q8_0.gguf
```

3.85 GB, Q8_0, 4,130,401,952 bytes. Source snapshot
`/mnt/data/training_output/deepred-p3v4c/snapshots/100pct-step-*`. Uploaded with
`python3 scripts/backup_deepred_files.py --gguf <path>`, which puts files in
`/Data` on the host; that folder is served at `https://www.ferzkopp.net/Data/`.

`p3v4c-100` was chosen over `p3v4c-075` for the demonstrator because it keeps
backbone-level utility (90.9%) and the higher persona rate, which is the axis
this stage was built to move. `p3v4c-075` is the better-balanced model — it
matches the backbone's era-native exactly — and is the one to prefer if the 1969
illusion matters more than the voice.

**The model requires a system prompt.** Both the temporal horizon and the voice
are prompt-conditioned; served bare it behaves close to stock `gemma-3-4b-it`.

## Stage 8 — p3-v5: the scaled 12B run (release candidate)

The final Phase 3 run. Its goal is explicitly **a usable artefact, not a passing
gate sheet**: a GGUF that loads in LM Studio and behaves recognisably as Deep
Red, flaws and all.

### Why gemma-3-12b-it

The salience-not-date finding says the 4B lacks a reliable sense of when an
entity belongs. That is a knowledge property, and it is the one thing more
parameters plausibly fix. `gemma-3-12b-it` is already downloaded (23 GB), uses
the `gemma3` architecture that transformers 4.57.6 supports, and shares the
chat-template behaviour the whole pipeline is built around, so it changes one
variable and no plumbing.

### Memory: measured, not estimated

A per-parameter estimate (8 bytes for weights, gradients and two AdamW states)
puts 12B at 96 GiB, which looked like it just fit. **That estimate is wrong.**
The upstream toolbox author publishes measured figures for exactly this
hardware, at `max_length 512`:

| Model | Full | LoRA | 8-bit LoRA | QLoRA |
|---|---:|---:|---:|---:|
| Gemma-3 4B-IT | 46 GB / 9m | 30 GB / 5m | 21 GB / 41m | 13 GB / 9m |
| **Gemma-3 12B-IT** | **115 GB / 25m** | **67 GB / 13m** | 43 GB / 2h38m | 26 GB / 23m |
| Gemma-3 27B-IT | OOM | OOM | 32 GB unstable | 19 GB |

Full fine-tuning 12B needs **115 GB**, not 96 GiB — the gap is activations,
fragmentation and framework overhead that a parameter count does not capture.
Our `max_length` is 768, not 512, so it would be worse. The 4B measurement of
46 GB is consistent with our working runs, which makes the 12B row credible.

This machine currently exposes **96 GiB** to the GPU. Full-weight 12B does not
fit as configured. Three ways forward, in order of preference:

1. **Raise the GTT ceiling.** The 96 GiB is a kernel tunable, not a hardware
   limit — see `/proc/cmdline`. Phase 4 documents the recommended host
   configuration, which raises it to ~124 GiB. That makes 115 GB fit, but needs
   a reboot and a desktop-less session.
2. **Cut the optimizer.** `--optim adamw_bnb_8bit` saves roughly 24 GB of
   optimizer state, landing near 91 GB — inside the current 96 GiB but with
   little headroom. `adafactor` saves more. Neither is measured here; treat the
   first 200 steps as the test.
3. **LoRA.** 67 GB and roughly twice as fast (13m against 25m). If a full
   fine-tune keeps OOM-ing, LoRA on 12B is a better use of the machine than
   another 4B full run, and `peft 0.18` is already installed.

Run the 200-step memory probe before committing, exactly as Phase 4 Stage 3
describes. Record peak allocated memory and seconds per step, then choose.

### Cheap test first

Before committing a multi-day run, measure whether 12B actually dates entities
better than 4B — that is the whole premise. Ask both models "In what year did X
happen?" across the probe entities and compare accuracy. Roughly 30 minutes. If
12B is no better at dating, scaling will not lift the temporal plateau and the
run should be re-scoped to a persona/usability release on the 4B instead.

### Decision (2026-09-07): close Phase 3 on 4B

The 12B run is **deferred to Phase 4**, where the memory work belongs alongside
the container and kernel changes. Phase 3 closes with `p3-v5`: the same
`gemma-3-4b-it`, the mix that produced the best result, and roughly three times
the corpus. This keeps the closing run directly comparable to every Phase 3
measurement and needs no host reconfiguration.

The p3-v4b corpus is exhausted — its dataset used 28,232 of the 27,482 available
rows, so more data means new generation, not re-sampling. Targets are **2x the
seeded corpus**, reduced from an initial 3x once generation rates were known.

| kind | have | p3-v5 target |
|---|---:|---:|
| `era_native` | 5,002 | 10,000 |
| `era_native_formats` | 3,498 | 7,000 |
| `retain` | 9,081 | 18,000 |
| `retain_formats` | 3,496 | 7,000 |
| `persona` | 3,009 | 6,000 |
| `persona_identity` | 1,180 | 2,400 |
| `chess` *(new)* | 0 | 6,000 |
| **total** | **25,266** | **56,400** |

`persona_capability` is dropped. It existed to carry markers, and deterministic
injection now does that job from the phrase bank at a fraction of the cost.

### Chess becomes a real asset

Interactive use of `p3v4c-100` found chess almost unobtainable, and the cause was
that **chess was never trained**: `chess` was not a kind the dataset builder
knew, so only 0.18% of p3-v4b rows mentioned chess at all, incidentally.

`augment_chess_games.py` had already produced 334,920 prose analyses of pre-1970
games that no run ever used. `scripts/build_chess_narratives.py` converts them
into question/answer rows, and `chess` is now a kind in `build_deepred_dataset.py`.

Two properties make this asset do double duty:

- **It is the only long-form content in the corpus.** Median answer is 328 words
  against 18 for everything else, which is the other defect reported from
  interactive use. Answers are capped at 400 words so they fit `max_length 768`
  without truncation — a truncated target teaches truncated answers.
- **Every row contains algebraic notation**, so the model sees the notation it
  is named for.

Cutoff handling is deliberately conservative. 1969 game dates are almost always
`1969.??.??`, and the few that resolve include August games — after the horizon.
Rather than assume, the converter takes 1950-1968 only. That window is also
Deep Red's own era. Games are deduplicated by player/event key, typographic
punctuation is normalised (non-breaking hyphens land inside move notation and
tokenise badly), and accented player names are left intact.

Marker injection runs at 0.4 for chess rather than 0.6: the narratives already
carry a marker 8% of the time, and a phrase added to a 340-word analysis carries
less weight than one added to a 16-word fact.

```bash
./run_p3v5.sh --preflight
./run_p3v5.sh servers        # persona endpoint on :1237
./run_p3v5.sh generate       # ~31,000 new rows, plus chess conversion
./run_p3v5.sh audit && ./run_p3v5.sh dataset && ./run_p3v5.sh train
```

`run_p3v5.sh` trains from the untouched base rather than staging onto the p3-v2
backbone, so the artefact comes from a single run. `EPOCHS=2`, LR `5e-6`,
max length 768, snapshots at 10/25/50/75/100%.

The dataset stage **fails closed if the system-prompted marker ratio falls below
1.0:1**. p3-v4 trained at 0.47:1 and taught the model to drop the voice; that
mistake should cost a build, not a training run.

### Recipe (deferred 12B variant)

If the 12B run is revived, reuse the p3-v2 corpus and recipe unchanged — that is
the configuration that produced the best temporal model — plus the persona
assets.

- start: untouched `gemma-3-12b-it`
- optimizer: `adamw_bnb_8bit`, LR `5e-6`, 2 epochs, max length 768
- snapshots at 10/25/50/75/100%, evaluated with and without the system prompt
- expect roughly 3x the p3-v2 step time; budget a multi-day run

### Release criteria (deliberately softer than the gates)

p3-v5 ships the best snapshot by judgement, not by a pass/fail sheet:

1. era-native at or above the p3-v2 backbone (>=56.5% with the prompt);
2. conversational leak no worse than 25%;
3. pre-1969 recall and utility no worse than base under the same condition;
4. no severe repetition, no false-refusal spike;
5. persona present often enough to be recognisable in ordinary use.

Export Q8_0 for evaluation and Q4_K_M for LM Studio, register both, and record
the qualitative behaviour alongside the metrics. A known-flawed model that can
be experimented with is the intended output of Phase 3.

## p3-v5 result (2026-09-19) — regression, and the reason

p3-v5 doubled the corpus, added the chess asset, and trained from the untouched
base. It is the **worst temporal model since p3-v1**, and it failed release
criteria 1 and 2 outright.

All figures with the served prompt:

| model | era | leak | persona | pre69 | utility |
|---|---:|---:|---:|---:|---:|
| base | 17.4% | 62.5% | 7.4% | 78.9% | 72.7% |
| `p3v2-050` backbone | **56.5%** | **25.0%** | 7.4% | 78.9% | **90.9%** |
| `p3v4c-100` | 52.2% | 37.5% | 22.2% | 78.9% | **90.9%** |
| `p3v5-100` | 34.8% | 62.5% | 22.2% | **89.5%** | 72.7% |

**Leak returned to exactly the base rate.** On the metric the whole project
exists to move, two epochs on 60,597 rows achieved nothing.

### Token mass, not row count

The cause is measurable in the built dataset. Loss is computed per token on the
assistant turn, so a kind's influence is its **share of target words**, not its
share of rows:

| kind | rows | row share | target words | **signal share** |
|---|---:|---:|---:|---:|
| `chess` | 5,711 | 9.4% | 1,829,567 | **59.8%** |
| `retain` | 17,025 | 28.1% | 305,427 | 10.0% |
| `persona` | 5,721 | 9.4% | 194,565 | 6.4% |
| `era_native` | 9,490 | 15.7% | 189,075 | 6.2% |
| `era_native_formats` | 6,665 | 11.0% | 133,450 | 4.4% |
| all others | 15,985 | 26.4% | 335,135 | 13.2% |

Chess was added as 9.4% of the corpus and became **60% of the training signal**,
because its answers average 328 words against 16-18 for everything else. The
era-native assets — the entire point of the run — were left with 10.6%.

Worse, chess is precisely the content type p3-v3 proved harmful: confident,
detailed, pre-1969 prose. The central finding says the model discriminates on
salience rather than date, so "famous subject -> answer at length and with
confidence" generalises straight across the cutoff. p3-v3 raised leak 25% -> 56%
with 5,663 such rows; p3-v5 raised it to 62.5% with 60% of the signal.

The same mechanism explains the one thing p3-v5 won: **pre-1969 recall 89.5%,
the best in the project** and well above base. That is the p3-v3 trade again —
more confident pre-cutoff content buys recall and pays in leakage — at a much
larger dose.

Training from base rather than from the `p3v2-050` backbone compounded it. The
backbone already held the temporal behaviour; p3-v5 had to learn it from scratch
from 10.6% of the signal.

### Lessons

1. **Balance a corpus by token mass, not row count.** Every cap in this project
   is expressed in rows, which is the wrong unit whenever answer lengths differ
   by an order of magnitude. A `--limit kind=N` on a long-form asset does not
   limit its influence.
2. **Long-form content is not a free addition.** The chess asset fixed the two
   defects reported from interactive use, and cost the temporal behaviour. If
   both are wanted, chess needs its own length budget — roughly 1,000 rows, or a
   word cap near 120 — so it stays under about 15% of the signal.
3. **Do not re-derive a backbone you already have.** p3-v4b and p3-v4c staged
   onto `p3v2-050` and kept its temporal behaviour. p3-v5 restarted from base
   and did not recover it in two epochs.
4. **More data is not a strategy.** Three of four Phase 3 corpus interventions
   that added or changed bulk content regressed the target metric. The wins came
   from format coverage (p3-v1) and register (p3-v2), both structural.

### Published for comparison

`p3v5-100` is published so the trade can be inspected directly, **not** as the
release:

```
https://www.ferzkopp.net/Data/deepred-p3v5-100-q8_0.gguf
```

3.85 GB, Q8_0. It is the only model that discusses chess, and it holds the
project's best pre-1969 recall, but its leak is at the base rate. `p3v4c-100`
remains the artefact of record.

## Phase 3 closing status

**Closed 2026-09-20.** The release artefact is **`p3v4c-100`**, not p3-v5.

| run | verdict |
|---|---|
| p3-v1 | format coverage confirmed; era-native 21.7% -> 56.5% |
| **p3-v2** | **best temporal model; selected backbone** |
| p3-v3 | regression, reverted; produced the central finding |
| p3-v4 | confounded by a dataset defect |
| p3-v4b / p3-v4c | persona 7.4% -> 22.2%; **published demonstrator** |
| p3-v5 | regression; leak returned to base |

What Phase 3 established:

- **Prompt-format coverage is the lever for era-native behaviour** (p3-v1).
- **Register and voice are separable from content** and can be restyled or
  injected after the fact (p3-v2, p3-v4b).
- **The model discriminates on salience, not date** (p3-v3, confirmed by p3-v5).
  This is the ceiling on the whole approach: without a reliable internal sense
  of when an entity belongs, any training that rewards confidence on familiar
  subjects leaks across the cutoff.
- **The frozen suite is out of resolution** for the decisions that remain.

The unresolved question is exactly the one Phase 4 is scoped to answer: whether a
model that dates entities better makes the salience confound go away. Phase 3
cannot settle it on a 4B.

## Phase 4

Feasibility of `google/gemma-4-12B-it` is scoped separately in
[DeepRed-Phase4-Setup.md](DeepRed-Phase4-Setup.md). It needs a transformers
upgrade and multimodal pipeline changes, so it must not be mixed into p3-v5.

## Constraints

- The Deep Red Bible must never enter `/mnt/data/DeepRedAI`. Persona assets stay
  under `/mnt/data/deepred_corpus/` and are referenced by path.
- The 81-probe suite stays frozen. New attack probes, if added, go in a
  separately versioned suite so trend comparisons remain valid.
- The 11 probed fact families are held out of all training data.
- p3-v1 trains from untouched `gemma-3-4b-it`. V4-V7 checkpoints are not
  starting points: V4-V6B are behaviourally identical to base, and V7 carries a
  utility regression.
- Retired for behaviour change: refusal-template SFT (Phase 1), and pairwise or
  margin objectives (V5A, V6A, V6B). All moved likelihoods without moving
  generation.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` from a generator | that llama-server is not running | `./run_p3v1.sh servers`; the run now aborts after 10 consecutive failures instead of spinning |
| Two sessions writing one asset | lock was per stage before 2026-09-03 | one lock per run; `--target` counts new rows, so the driver subtracts what exists |
| `kept_original:fact_loss` dominating a restyle | rewrite dropping names or numbers | lower `--restyle-batch`, or leave that asset plain |
| `no articles matched the sampling filter` | salience window too tight | raise `--page-id-max` or pass `--no-salience` |
| `copied_reference` is most rejections | generator echoing the reference replies | lower `--temperature`, or reduce `--per-article` |
| Format counts skewed | a format keeps failing validation | run that format alone with `--formats <name> --dry-run` |
| `WARN empty completion` | reasoning model spending the budget | restart the server with `--reasoning-budget 0` |
| `Read error: Bad address` | Strix Halo mmap defect | add `--no-mmap` |
| Rate far below 2/min | model partly on CPU | check `offloaded N/N layers` |
| Another run holds the lock | stale PID directory | the driver clears dead owners automatically |
