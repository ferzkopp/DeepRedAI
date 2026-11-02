# DeepRed Phase 4 — P8 Runbook

This runbook covers P8 in [DeepRed-Phase4-Plan.md](DeepRed-Phase4-Plan.md):
1. A generator bake-off over every regenerated asset kind.
2. A full regeneration of `retain`, `era_native`, `retain_formats` and
   `era_native_formats` with the winning generator.
3. The `p4-v3` pilot trained on that corpus.

The whole pipeline is one script, [run_p8.sh](../run_p8.sh). Every stage
resumes from its own output, so after a crash, reboot or power cut you re-run
the same command.

## Quick start (device console)

```bash
cd /mnt/data/DeepRedAI
./run_p8.sh --detach all      # start or resume everything; safe to log out
./run_p8.sh status            # stage state, word budgets, training progress
tail -f /mnt/data/evaluations/phase4/p8/logs/run.log
```

To run one stage in the foreground instead, use `./run_p8.sh <stage>`.

`all` skips stages already marked done in `p8/state/`. To run a stage again,
delete its `.done` file, or invoke the stage by name.

The runner always executes a private copy of itself, so editing `run_p8.sh`
mid-run is safe. A lock in `/tmp/deepred-p8.lock.d` refuses a second
concurrent run.

## Stages

| # | Stage | What it does | Output | Resumes by |
|---|---|---|---|---|
| 1 | `preflight` | Checks the commands, both containers, the probe banks, the base GGUF, the P1/P2 gates, the Wikipedia DB, at least 2 candidate GGUFs, and at least 150 GiB free | `p8/state/preflight.done` | re-running |
| 2 | `salience` | P5.9: serves the untrained Gemma 4 base and asks it to identify subjects from page-id bands, per era. The ceiling is the widest run of bands from the most famous where recognition is at least 50% | `p8/salience/verdict.json`, `answers.jsonl` | reuses the verdict if present |
| 3 | `bakeoff` | Runs the real corpus generator on a fixed article sample (40 rows per kind) for each candidate, then applies the selection rule | `p8/bakeoff/verdict.json`, `candidates/<name>.json` | skips candidates already scored |
| 4 | `select` | Records the winner, or `GENERATOR=<name>` | `p8/selected_generator.json` | keeps an existing selection |
| 5 | `generate` | Generates each kind in 400-row chunks until its word budget is met. Restarts the generator server if it dies | `/mnt/data/deepred_corpus/p8-gen/<kind>/<kind>.jsonl` | counts the words already on disk |
| 6 | `restyle` | Applies the p4-v2 marker restyle to both format assets, then checks that the row IDs match the source exactly | `p8-gen/<kind>/<kind>.voiced.jsonl` | skips IDs already written |
| 7 | `seed` | Builds `p4-v3` from p4-v2 plus the four regenerated assets, drops rows that overlap any non-persona probe, and runs `audit_deepred_corpus.py` | `/mnt/data/deepred_corpus/p4-v3/` + `provenance.json` | skips if `provenance.json` exists |
| 8 | `dataset` | `run_p4v1.sh dataset` with p4-v2's caps and marker rates: every P6 signal, marker, contamination and mean-token gate | `/mnt/data/sft_corpus/deepred-p4v3/` | rebuilds deterministically |
| 9 | `train` | One epoch from the Gemma 4 base, with snapshots at 10/25/50/75/100%, then Q8_0 export | `/mnt/data/training_output/deepred-p4v3/`, `/mnt/data/evaluations/deepred-1969/p4v3-<date>/` | trainer auto-resumes from the last full checkpoint; existing GGUFs are reused |
| 10 | `evaluate` | Both banks, with and without the system prompt. Models: the base, `p3v2-050`, `p3v4c-100`, `p4v2-100` and the five p4-v3 snapshots | `p4v3-<date>/{with-system,no-system,frozen}/` | skips completed generations |

### Generator selection rule (P8)

`benchmark_generators.py --select` scores each candidate as:

> mean acceptance across the four corpus tasks × mean grounding on `retain` and `retain_formats`

- **Acceptance** means passing the generator's real guards.
- **Grounding** is the share of rows whose every name and number appears in
  the source article or in the row's own question.

A candidate is eligible only if all of the following hold:

- it is outside the gemma family (decision 7);
- every corpus task produced at least 50% of its target rows;
- grounding is at least 0.95 (the incumbent's p4-v1 retain assets measure
  0.98-0.99);
- its measured accepted words per minute project the full regeneration
  within 168 hours.

The table and the reason each candidate was excluded are in
`bakeoff/verdict.json` under `selection`.

## Settings you may change

Set these as environment variables, e.g. `GENERATOR=gpt-oss-120b ./run_p8.sh --detach all`.

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR` | bake-off winner | Force the generator (`qwen3-30b-a3b`, `gpt-oss-120b`, `glm-4.7-flash`, `qwen2.5-14b-incumbent`) |
| `BUDGET_RETAIN` | 190000 | Answer words to generate. Each default is about 10% above what the p4-v2 build used, so the dataset caps bind first |
| `BUDGET_RETAIN_FORMATS` | 100000 | |
| `BUDGET_ERA_NATIVE` | 180000 | |
| `BUDGET_ERA_NATIVE_FORMATS` | 135000 | |
| `CHUNK_ROWS` | 400 | Rows per generation call; this is also the resume granularity |
| `BAKEOFF_TARGET` | 40 | Rows per kind per candidate in the bake-off |
| `MAX_GENERATION_HOURS` | 168 | Feasibility ceiling in the selection rule |
| `AUDIT_MAX_OPENING_SHARE` | 0.15 | Opening-collapse ceiling for the `seed` audit. p4-v3 runs at 0.20; see the seed result below |
| `EPOCHS` | 1 | p4-v2 parity |
| `COMPARATORS` | p3v2-050, p3v4c-100, p4v2-100 | Extra models in the evaluation |

The dataset caps, signal budgets and marker rates are the p4-v2 values from
`run_p4v1.sh`/`run_p4v2.sh`. Format and chess injection is zero because those
rows are voiced. Keeping them fixed makes the regenerated corpus the only
variable against p4-v2.

## Gates and fail actions

| Gate | Where | On failure |
|---|---|---|
| Preflight | `preflight` | Fix the listed item and re-run |
| No eligible generator | `select` | Read `bakeoff/verdict.json` → `selection.candidates[*].reasons`, then re-run with `GENERATOR=<name>` or relax `MAX_GENERATION_HOURS` |
| Generation stalls (3 chunks add no rows) | `generate` | Check `logs/generate-<kind>.log` for the rejection mix. Usually the opening-phrase budget is saturated or the DB filter is exhausted. Raise the salience ceiling in `salience/verdict.json` and re-run |
| Generator crashes 3 times in a chunk | `generate` | Check `logs/generator-server.log`, then re-run; work already written is kept |
| Restyle parity mismatch | `restyle` | Re-run `restyle`; it fills missing IDs |
| Corpus audit fails | `seed` | Read `logs/seed.log`. To rebuild, remove `/mnt/data/deepred_corpus/p4-v3` and `state/seed.done`. An opening-share failure that the restyle caused at p4-v2 parity can be waived with `AUDIT_MAX_OPENING_SHARE` |
| Signal budget / marker ratio / mean tokens / contamination | `dataset` | Read `logs/dataset.log`. A marker ratio outside 0.8-1.3 means the new assets carry a different native marker share: adjust `INJECT_ERA`/`INJECT_RETAIN` and re-run `dataset` |
| P7.4 release gate | after `evaluate` | Record the results in the plan. Scaling requires one snapshot to meet the absolute gates and beat every comparator in its prompt condition |

## Where everything lands

```
/mnt/data/evaluations/phase4/p8/
  logs/run.log, logs/<stage>.log, logs/generate-<kind>.log
  state/<stage>.done
  salience/verdict.json, answers.jsonl
  bakeoff/verdict.json, candidates/*.json, scratch/
  selected_generator.json
/mnt/data/deepred_corpus/p8-gen/       raw and voiced regenerated assets
/mnt/data/deepred_corpus/p4-v3/        training corpus + provenance.json
/mnt/data/sft_corpus/deepred-p4v3/     dataset + manifest.json + contamination.json
/mnt/data/training_output/deepred-p4v3/console.log, run_meta.json, final/
/mnt/data/evaluations/deepred-1969/p4v3-<date>/   GGUFs, models.json, scores
```

## After `evaluate` finishes

The P7.4 summary table is at the end of `logs/evaluate.log`. Check it against
the p4-v2 verdict in the plan:
- the no-system era-native/leak, which was 5%/67% for p4-v2;
- the with-system direct-question rate, which was 17/42;
- `confident_unsupported` answers, which numbered 42.

Every `served_template.json` must report `ok`.
