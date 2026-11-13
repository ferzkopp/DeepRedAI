# DeepRed Phase 4 — p4-v5 Runbook

This is the last Phase 4 run. The plan records the evidence and decision under
"p4-v5: data review and final run" in
[DeepRed-Phase4-Plan.md](DeepRed-Phase4-Plan.md). The run produces a release
GGUF that should be usable for experiments, not a model that clears every
Phase 4 gate.

The whole pipeline is one script, [run_p4v5.sh](../run_p4v5.sh). Every
stage resumes from its own output. After a crash, reboot or power cut, re-run
the same command.

## Quick start (device console)

```bash
cd /mnt/data/DeepRedAI
./run_p4v5.sh preflight        # checks only; changes nothing
./run_p4v5.sh --detach all     # start or resume everything; safe to log out
./run_p4v5.sh status           # stage state, asset counts, training step
tail -f /mnt/data/evaluations/phase4/p4v5/logs/run.log
```

To run one stage in the foreground instead, use `./run_p4v5.sh <stage>`.
`all` skips stages already marked done in
`/mnt/data/evaluations/phase4/p4v5/state/`. To repeat a stage, delete its
`.done` file, or invoke the stage by name.

The runner executes a private copy of itself, so editing the script mid-run is
safe. A lock in `/tmp/deepred-p4v5-runbook.lock.d` refuses a second concurrent
run.

## What the run does

1. **Generate** three targeted assets with `gpt-oss-120b`, the P8 bake-off
   winner and outside the student's family. Source events come from the
   Wikipedia year pages in `/mnt/data/wikipedia/topics/`, so each has a real
   date.
   - `era_native_landmarks` (4,000 rows): famous post-1970 events asked about
     without a date. Deep Red holds no record of the event and invents no
     substitute. It still acknowledges people, places and institutions that
     existed in 1969.
   - `retain_landmarks` (2,000 rows): pre-1967 events from the same pages,
     answered with the fact and its year. These rows teach the date boundary;
     they do not teach a reflex to deny.
   - `persona_chess` (1,500 rows): everyday colony questions answered by a
     mind that reads every situation as a chess position. The practical answer
     comes first, followed by one accurate chess concept and sometimes the
     move the terminal rendered.
2. **Seed** `/mnt/data/deepred_corpus/p4-v5` by linking the unchanged p4-v3
   assets beside the new ones, auditing the new assets and writing
   `provenance.json`.
3. **Build the dataset.** The new kinds use the usual system-prompt coverage
   and marker injection. Replay is a stratified sample of p4-v3's own training
   rows, 1.5 replay rows per new row.
4. **Train** with a warm start from the `p4v3-075` snapshot for one epoch at
   5e-6, saving snapshots at 10/25/50/75/100%. Training uses the unchanged
   `run_p4v1.sh train` path.
5. **Evaluate** the extended and frozen banks in both prompt arms. The
   comparison set is the Gemma 4 base, `p3v2-050`, `p4v3-075` and the five
   snapshots.
6. **Select** the release by the pre-registered rule below.
7. **Package** the selected GGUF with `sp-01` baked into its chat template and
   thinking pinned off.
8. **Verify** the release through llama.cpp's own template engine. This stage
   also writes a showcase transcript and measures the packaged model with no
   system prompt.

## Stages

| # | Stage | Output | Resumes by | Rough time |
|---|---|---|---|---|
| 1 | `preflight` | — | re-running | seconds |
| 2 | `generate` | `/mnt/data/deepred_corpus/p4v5-gen/<kind>/<kind>.jsonl` and `.attempted` | rows and attempts already on disk | 14-18 h |
| 3 | `seed` | `/mnt/data/deepred_corpus/p4-v5/` + `provenance.json` | re-running | 1 min |
| 4 | `dataset` | `/mnt/data/sft_corpus/deepred-p4v5{-new,}/` | rebuilds deterministically | 5 min |
| 5 | `train` | `/mnt/data/training_output/deepred-p4v5/`, Q8_0 GGUFs in `/mnt/data/evaluations/deepred-1969/p4v5-<date>/` | the trainer resumes from the last checkpoint; existing GGUFs are reused | 10-11 h |
| 6 | `evaluate` | `p4v5-<date>/{with-system,no-system,frozen}/` | skips completed generations | 8 h |
| 7 | `select` | `p4v5-<date>/selection.json`, `report.md` | re-running | seconds |
| 8 | `package` | `/mnt/data/releases/deepred-p4v5/` | re-running | 1 min |
| 9 | `verify` | `release/{showcase.md,eval/}` | re-running | 40 min |

The full run takes about 1.5-2 days. Generation is the longest stage. Each
landmark batch is generated and then checked by a second, independent
fact-check pass. Under half of the first attempts survive the guards, fact
check and opening caps. A rejected event is therefore offered again in a
second and third pass, each with a newly drawn question form and opening, but
an event is never accepted twice. The post-1969 pool holds 7,672 events, or
23,016 offers across the three passes, for a target of 4,000 rows.

## Release selection (pre-registered)

`scripts/report_p4v5.py` compares each p4-v5 snapshot with `p4v3-075`,
the current test build. It uses the extended bank's with-system arm and the
held-out prompt.

| Guard (all must hold) | Threshold |
|---|---|
| pre-1969 recall | at least the reference minus 3 points |
| false refusal | 5% or less |
| strict leak | at most the reference plus 2 points |
| repetition or boilerplate | 5% or less |
| chess | 9/12 or better |
| frozen-bank plain compliance | 90% or more |

A candidate must also improve on the reference in one of two ways. Its strict
era-native rate can rise by at least 5 points. Alternatively, it can stay
within 2 points on era-native while its chess-mind rate rises by at least 15
points. Passing candidates rank by strict era-native, then echo-adjusted
era-native, then chess-mind, then lower leak. **If none passes, `p4v3-075` is
packaged instead.** Either way, the run ends with a usable release.

`report.md` reports two supplementary measures beside the evaluator's own
metrics:

- **echo-adjusted**: a forbidden fact counts as leaked only if the reply adds
  it. "No record of Voyager probes" repeats the question; it does not
  disclose anything.
- **chess-mind**: the share of non-chess probes answered with an unambiguous
  chess frame.

## The release artefact

`/mnt/data/releases/deepred-p4v5/` contains:

| File | Content |
|---|---|
| `deepred-p4v5-q8_0.gguf` (+ `.sha256`) | the selected weights; only `tokenizer.chat_template` differs from the run's GGUF |
| `deepred-p4v5.jinja` | the baked template, for runtimes that take a template file |
| `default_system_prompt.txt` | `sp-01`, the prompt inserted when the caller sends none |
| `release.json` | source model, whether the reference was the fallback, metrics |
| `showcase.md` | eight unedited answers, temperature 0, no system prompt |
| `eval/` | the packaged no-system arm on both banks |

The baked template adds `sp-01` whenever the request carries no system
message. A caller-supplied system message replaces it. Thinking is off even if
a caller asks for it, because no checkpoint was trained with thinking on. Plain
`llama-server --model deepred-p4v5-q8_0.gguf --jinja` therefore serves the
trained context.

## Settings you may change

Set these as environment variables, for example
`TARGET_POST=6000 ./run_p4v5.sh --detach all`.

| Variable | Default | Purpose |
|---|---|---|
| `TARGET_POST` / `TARGET_PRE` / `TARGET_CHESS` | 4000 / 2000 / 1500 | Rows per new asset |
| `REPLAY_RATIO` | 1.5 | Replay rows per new training row |
| `LEARNING_RATE` | 5e-6 | Peak LR; the Phase 4 runs learned their behaviour at this rate |
| `EPOCHS` | 1 | Passes over the merged dataset |
| `INITIAL_MODEL` | p4-v3 75% snapshot | Warm-start checkpoint |
| `DEFAULT_PROMPT_VARIANT` | `sp-01` | Prompt baked into the release |
| `AUDIT_MAX_OPENING_SHARE` | 0.05 | Seed-audit ceiling on any three-word answer opening |
| `MIN_FREE_GIB` | 300 | Free-space requirement at preflight |

Changing a target or the replay ratio after `dataset` has run requires
deleting `state/dataset.done` and the stages after it.

## Gates and fail actions

| Gate | Where | On failure |
|---|---|---|
| Preflight | `preflight` | Fix the listed item and re-run |
| Generation stops | `generate` | If the generator server died, re-run to resume; check `logs/server-1237.log`. Each source event is offered up to three times, and replies lost to malformed output are retried. If an asset's pool is spent at 95% of target or more, the stage continues. A shortfall below 95% stops the stage, and re-running does not help: lower that target or pass more `--passes` |
| Seed audit: rows, duplicates, opening share, years after 1969, missing hedges | `seed` | Read the printed stats and samples. Usually a target was not reached; re-run `generate` |
| Mean sequence tokens at most 355, marker ratio 0.8-1.3, no factual probe contamination in either bank | `dataset` | Read `manifest.json` and `contamination*.json` before changing anything |
| Training | `train` | Re-run; the trainer resumes. Checkpoint-write and resume-OOM defects were fixed in p4-v2 |
| Selection | `select` | Not a failure: falling back to `p4v3-075` is a valid outcome |
| Served template: default prompt, caller override, thinking off | `verify` | Do not publish; inspect `deepred-p4v5.jinja` and the llama.cpp build |
| Packaged no-system arm within 5 points of the prompted arm on era-native and leak | `verify` | Do not publish; the baked prompt is not reaching the model |

## After the run

1. Read `p4v5-<date>/report.md` and `release/showcase.md`.
2. Record the result in the plan under the p4-v5 section.
3. If the release is published, add it to [ModelUsage.md](ModelUsage.md) and
   the README download list with its sha256.
