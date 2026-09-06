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
| 4. Train, export, evaluate, gate | `./run_p3v1.sh train` | **completed 2026-09-04; gates failed** |
| 5. p3-v2 voice rebuild | `./run_p3v2.sh` | in progress |
| 6. Context distillation (p3-v3) | — | blocked on p3-v2 gates |

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

## Stage 5 — p3-v2 context distillation (blocked)

Only after a p3-v1 snapshot passes. Generate that checkpoint's own responses
**with** the system prompt across a large prompt set, filter them with the same
classifiers, then fine-tune **without** the system prompt on what survives. That
converts the conditioned rule into default behaviour and is the step that makes
the served prompt optional. Do not start it on a failing checkpoint: distilling
a weak policy makes it permanent.

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
