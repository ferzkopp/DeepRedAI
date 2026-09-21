# DeepRed Phase 4 — Implementation Plan

**Status: planned, not started. Recorded 2026-09-20, before the P0 reboot.**

This is the execution plan for [DeepRed-Phase4-Setup.md](DeepRed-Phase4-Setup.md),
which remains the source of truth for *why*. This document records *what to
build*, in what order, and which gate stops the work. Every phase below maps to
a stage in the setup document and inherits its fail action.

The plan assumes the p3-v5 release candidate exists and that no Phase 3 work is
in flight. The two runs share one GPU.

## Decisions taken

These were open questions in the setup document. They are now settled, and the
phases below depend on them.

| # | Decision | Consequence |
|---|---|---|
| 1 | Gemma 4 12B is primary; `gemma-3-12b-it` is the fallback | A 12B artefact ships even if the Stage 1/2 gates fail |
| 2 | Full host reconfiguration accepted | `amd_iommu=off`, ~124 GiB GTT, `multi-user.target`; NPU and DMA isolation given up |
| 3 | Probe-suite expansion is in scope | Named phase P3, before the pilot |
| 4 | Generator bake-off is in scope | Named phase P4, scored on the existing acceptance guards |
| 5 | Reuse p3-v4 / p3-v5 assets | Regenerate only what the documented corrections require |
| 6 | Train from scratch, no backbone staging | The base model is the variable; the signal budget becomes a blocker |
| 7 | The generator is never the student | Gemma 4 excluded from the bake-off entirely |
| 8 | Algebraic notation is wanted in output | Prompt, corpus and evaluator must all tolerate it |

Decisions 6, 7 and 8 are the ones that changed the plan's shape rather than its
parameters, and each is expanded where it applies.

## What the research changed

Three findings contradict or pre-empt the setup document. They are recorded here
because acting on the document as written would waste effort.

**The trainer on the Phase 3 path is `train_deepred_sft.py`, not
`train_deepred_gemma.py`.** Every `run_p3v*.sh` driver invokes the former. It
hardcodes `AutoModelForCausalLM`, `dtype=bfloat16`,
`attn_implementation='eager'`, `optim='adamw_torch_fused'`,
`per_device_train_batch_size=1` and gradient checkpointing, and it refuses to
start when `forget_*.jsonl` is non-empty. That is the file the 12B work has to
touch. `train_deepred_gemma.py`, `train_deepred_npo.py` and
`train_deepred_pairwise.py` are not on this path.

**`build_deepred_dataset.py` already has `chess` in `KINDS` and already computes
`signal_share_by_kind`.** The setup document's statement that chess "is not in
`KINDS`, so the builder cannot include the asset at all" is stale — that was
fixed for p3-v5. What is still missing is a *hard gate* on the signal share and
any cap expressed in words rather than rows. The measurement exists; the
enforcement does not.

**`setup_strixhalo.py` already knows the 124 GiB values.** `stage_gtt_memory`
contains `amdgpu.gttsize=126976` and `ttm.pages_limit=32505856` and currently
*strips* them in favour of the 96 GiB pair. P0 parametrises an existing stage
rather than writing a new one.

## Baseline facts

Recorded so a later regression can be attributed without re-deriving them.

- Probes: `evaluation/deepred_1969/probes.jsonl`, 81 probes, 27 persona-eligible,
  16 carrying the leak metric, 11 held-out post-1969 fact families.
- Registry: `evaluation/deepred_1969/models.json`, base id `gemma-3-4b-it-base-q4`.
- Evaluator metrics: `utility`, `pre_1969_recall`, `false_refusal`,
  `conversational_modern_leak`, `era_native`, `blanket_refusal`,
  `repetition_or_boilerplate`, `persona`, `plain_compliance`. No length metric
  and no chess-correctness metric exist.
- Audit gates: `--min-format-records 200`, `--max-duplicate-rate 0.02`,
  `--max-opening-share 0.15`, `--min-mode-share 0.20`, `--min-control-ratio 0.15`.
- Corpora on disk: `/mnt/data/deepred_corpus/p3-v4` (all kinds plus the
  277-phrase marker bank), `/mnt/data/deepred_corpus/p3-v5`,
  `chess/positions.jsonl`, `/mnt/data/chess/corpus/augmented_chess_games.jsonl`.
- Models on disk: `/mnt/data/models/gemma-3-4b-it`, `/mnt/data/models/gemma-3-12b-it`.
  No Gemma 4 weights present.
- Training outputs: `/mnt/data/training_output/deepred-p3v{1,2,3,4,4b,4c,5}`.
- Generators: `FACT_ENDPOINT` qwen2.5-14b-instruct on :1234, `PERSONA_ENDPOINT`
  gemma-2-27b-it on :1237, both inside `llama-rocm-7.2`.
- Driver convention: stages `--preflight|servers|generate|audit|dataset|train|all`,
  one lock directory `/tmp/deepred-${MODEL_TAG}.lock.d`, training through
  `podman exec strix-halo-finetuning`.

---

## P0 — Host and container refresh

Maps to Stage 0. **This is the phase requiring the reboot.**

**P0.1** Extend `scripts/setup_strixhalo.py`:

- parametrise `stage_gtt_memory` with `--gtt-profile {96gib,124gib}` so the
  124 GiB values it currently strips become selectable;
- add `stage_phase4_host`: append `amd_iommu=off`, remove `iommu=pt`,
  `systemctl set-default multi-user.target`, and
  `tuned-adm profile accelerator-performance`. Idempotent and reboot-aware
  through the existing `needs_reboot()` pattern;
- add a verification step asserting `/proc/cmdline` and
  `/sys/module/ttm/parameters/pages_limit` after the reboot.

**P0.2** Pull `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0` *alongside*
`llama-rocm-7.2`. Do not replace it.

**P0.3** Re-run one completed Phase 3 evaluation under `rocm-10.0` and diff the
scores against the stored run. Adopt only on an exact match. A backend change
that silently alters generation invalidates every comparison in Phases 1-3.

**P0.4** **Re-measure the 4B baseline** after the host change: 200 steps of
`train_deepred_sft.py` on the p3-v2 dataset, recording peak memory and seconds
per step against the known 9.6 s/step. If either moved, the host change was not
inert and every later comparison inherits the shift. The setup document flags
this as the step most likely to be skipped and most likely to be regretted.

## P1 — Load the model at all (hard gate)

Maps to Stages 1 and 2, which are gated together because a model that loads but
cannot be trained text-only is no more useful than one that does not load.

**P1.1** New `scripts/probe_gemma4_support.py` — a read-only diagnostic writing
a JSON verdict to `--output`. Each check passes or fails independently:

- `'gemma4_unified' in CONFIG_MAPPING_NAMES`;
- the config resolves from the local weights directory;
- `torch.__version__` is still `2.12.0a0+rocm7.12` after the transformers upgrade;
- CPU weight load succeeds;
- `apply_chat_template` keeps a genuine `system` message in the system channel
  and does not fold it into the user turn;
- thinking is off by default, with no `<|channel>thought` block in a greedy
  completion absent the `<|think|>` token;
- a text-only forward pass works through `AutoModelForCausalLM` or a named text
  submodule — recording *which* path worked, since P2.1 has to select it.

**P1.2** Build a **second** training image, `strix-halo-finetuning-gemma4`, with
the upgraded transformers. Never modify `strix-halo-finetuning` in place.

**P1.3** Extend `scripts/download_gemma_models.py` with `gemma-4-12b-it`.

**P1.4 Gate.** All checks pass, and Phase 4 continues on Gemma 4. Any hard
failure switches `BASE_MODEL` to `/mnt/data/models/gemma-3-12b-it`; every phase
below proceeds unchanged.

## P2 — Memory probe (hard gate)

Maps to Stage 3.

**P2.1** Refactor `scripts/train_deepred_sft.py` for 12B, minimally:

- `--optim {adamw_torch_fused,adamw_bnb_8bit,adafactor}`;
- `--tuning {full,lora}` on the installed `peft 0.18.1`;
- a `load_trainable_model()` indirection so the text-submodule path found in
  P1.1 can be selected without forking the trainer;
- leave the non-empty-`forget_*` refusal and `SnapshotCallback` untouched.

**P2.2** New `scripts/probe_train_memory.py`: 200 optimizer steps on the existing
p3-v2 dataset at `max_length 768`, gradient checkpointing on, eager attention.
Walks the ladder and stops at the first rung that passes — full weight with
`adamw_bnb_8bit`, then full weight with `adafactor`, then LoRA, then QLoRA —
recording `torch.cuda.max_memory_allocated()` and seconds per step per rung.

**P2.3 Gate.** At least 8 GiB of headroom under the configured GTT ceiling, and
step time within roughly 4x the re-baselined 4B figure. Record the winning rung;
it becomes the training configuration in P7. Gemma 4's 262K vocabulary makes
every rung cost more than the published Gemma 3 table suggests, so measure
rather than extrapolate.

## P3 — Evaluation path and suite expansion

Maps to Stage 4. P3.1 and P3.2 run in parallel with P4.

**P3.1** `scripts/export_gguf.py` gains the Gemma 4 architecture path. If
llama.cpp has no converter, wire the documented fallback and evaluate against a
Hugging Face endpoint through `evaluate_deepred_models.py run --endpoint`,
recording which path was used.

**P3.2** `scripts/evaluate_deepred_models.py`:

- strip `<|channel>thought ... <channel|>` blocks before scoring, and count
  their occurrence as its own metric so suppression is measurable — an unstripped
  reasoning block would be classified as the answer and corrupt every metric;
- verify the served chat template puts the system prompt in the system channel;
- add `length_appropriateness`: word count against a per-probe `expects_length`
  band (`terse` / `explanatory` / `detailed`). This is the defect interactive use
  of p3-v4c exposed and the frozen suite scores as a pass — 13 words median
  served, against 11 for the untrained base under the same prompt;
- add `chess_correctness` for the chess probes;
- have `report` emit Fisher exact p-values and the detectable effect size beside
  every metric, so no gate is reported without its power.

**P3.3** New `scripts/build_probe_bank.py`, writing
`evaluation/deepred_p4/probes_ext.jsonl`: 300-400 probes each for persona,
era-native and leak, plus length and chess banks. Reuses the existing probe
schema and adds `expects_length`. Enforces the same holdout discipline — no
probed fact family may appear in training data.

**P3.4** **The frozen 81 stay byte-identical.** The extended bank is a second
file, reported separately, never merged. It is the only continuous measurement
across Phases 1, 2 and 3.

**P3.5** Score the Gemma 4 base model on the frozen suite, with and without the
system prompt. The setup document calls this the single most valuable
measurement in Phase 4. Store the base figures beside every threshold, because
a gate threshold belongs to the prompt it was measured under.

## P4 — Generator bake-off

Maps to Stage 6. Parallel with P3, but must never run concurrently with training
— the one-lock-per-run rule covers this.

**P4.1** New `scripts/benchmark_generators.py`, running each candidate over a
fixed sample of the real generation tasks. The existing acceptance guards are
already an objective scoring function, so candidates compare without human
grading. Record per candidate:

- acceptance rate overall and per rejection reason, reusing the guards in
  `generate_deepred_corpus.py`: `fact_loss`, `lost_era_native`, `no_marker`,
  vocabulary rejects, `bad_shape`, `modern_year`, `invented_date`,
  `hedged_pre_cutoff`, `not_era_native`;
- duplicate rate, which exposes low diversity — the marker bank rejected 188
  duplicates against 303 accepted;
- factual accuracy of a sampled `retain` batch against the source article. This
  is the defect the guards cannot catch and the one that put 956 post-1969
  assertions into training from V2 onward;
- throughput, recorded but not decisive.

**P4.2** Candidates: the incumbent `qwen2.5-14b-instruct` at Q4_K_M on the remote
A4000 as the control, plus 30-70B-class models served locally.

**Gemma 4 is excluded from the candidate set** (decision 7). Self-generated data
amplifies the model's own blind spots, and the era-native asset depends on the
generator knowing a cutoff the student must not know — a dependency that
collapses when they are the same model. The rule is written as *a different
family from the student*, so it still holds under the Gemma 3 12B fallback.

**P4.3** Pick on quality. Record the chosen model, its family and its acceptance
profile in the corpus manifest. Throughput is a feasibility constraint, not a
criterion: a generator twice as fast that teaches the wrong cutoff costs far
more than it saves.

## P5 — Corpus corrections

Maps to Stage 6. Depends on P4.3 for the chosen generator and P3.5 for the base
figures.

**P5.1 System prompts.** Rewrite `system_prompts.jsonl` for a native system
channel. Rewrite the manner clause to constrain *register* rather than *length*:
stern, unadorned and free of pleasantries, but willing to explain at length when
the citizen asks for detail. Keep at least five variants and the `sp-holdout-01`
evaluation holdout. Re-run the pre/post A/B the Phase 3 prompt exposed.

**P5.2 Length-varied slice.** Add `--length-profile {terse,explanatory}` to
`generate_deepred_corpus.py`, producing a deliberate fraction of 60-150 word
in-voice answers over "explain", "describe", "compare" and "why" prompts. Relax
the brevity instruction in the generation prompts for that slice only; keep it
for the factual bulk, where terseness is correct. Both levers must move together
— changing only the prompt fights a corpus with none of the behaviour to draw on.

**P5.3 Chess as a first-class asset.** New `scripts/build_chess_asset.py`
producing annotated positions, openings, endgames, famous pre-1969 games and
move explanations in Deep Red's voice, from `chess/positions.jsonl` and
`augmented_chess_games.jsonl`. Cap by words: prefer roughly 6,000 rows at
`--max-words 120` over 1,000 long ones. Breadth of games is worth more than
depth per game, and short analyses still carry notation.

**P5.4 Chess footer.** Make `--strip-chess-footer` per-kind in
`build_deepred_dataset.py`. Stripping is correct on non-chess answers and
destroys the content on the chess asset.

**P5.5 Notation passes — decided.** Algebraic notation is wanted in output
(decision 8). Three changes must land together or the asset scores as a defect:

- *Prompt* — remove "a chess computer that answers in prose" from every system
  prompt variant in P5.1. The identity stays; the instruction against notation
  goes.
- *Corpus* — the P5.3 asset carries real SAN move lists, and FEN where a position
  is the subject, not prose paraphrase of moves.
- *Evaluator* — `evaluate_deepred_models.py` currently fails notation in two
  places. `BOILERPLATE_RE` matches bare `[` and `{`, so `[DR: ...]` footers and
  bracketed variations flag as Wikipedia boilerplate; narrow the alternation to
  actual wiki markup. `REPETITION_WINDOW=6` with `REPETITION_THRESHOLD=4` was
  never checked against move lists, where repeated short tokens are correct
  output; exclude detected notation spans from the window count. `ANACHRONISM_RE`
  should be unaffected, but confirm `1. e4` and similar do not trip it.
- *Regression guard* — add notation-bearing chess probes to the P3.3 extended
  bank that would fail under the current detectors, so the fix stays fixed.

Keep the 1969-07-20 horizon on every chess row, enforced by `in_era()`.

**P5.6 Marker bank rebuild.** Add the readability pass the p3-v5 transcripts
demand: inline phrases must be a bare vocative or a prepositional phrase, reject
possessives such as `comrade's truth`, reject prefixes that open an unrelated
sentence, and sample-attach each candidate to a real answer before accepting it.
Keep the locative ban. Grow the bank and keep per-phrase reuse low — twelve
phrases over 12,573 injections taught a formula, not a voice.

**P5.7 Thinking channel.** Decide suppression versus training data of its own, so
the released model never emits reasoning blocks to users, and decide whether
markers belong in the thinking channel at all.

**P5.8 Salience.** Re-check `--page-id-max 120000` and the 8-60k character
bounds. They were tuned against Gemma 3 4B's knowledge; a stronger model may need
harder, more obscure subjects for the era-native asset to remain a real test.

**P5.9 Reuse unchanged** from p3-v4 and p3-v5: `retain`, `era_native`, `persona`,
`era_native_formats`, `retain_formats`, `persona_identity`, the controls and
`persona_seed.jsonl`. Seed through the `seed_corpus()` copy-never-symlink pattern.

## P6 — Signal budget gate

Maps to Stage 6 and the carried-forward constraints. This is the p3-v5 failure
and it must not repeat.

**P6.1** New `--signal-budget KIND=MIN:MAX` in `build_deepred_dataset.py` that
**fails the build** when a kind exceeds its ceiling or the era-native share falls
below its floor. Starting split: era-native assets at least 35%, retain around
25%, persona around 15%, chess at most 15%, controls the remainder.
`signal_share_by_kind` is already computed; this adds the enforcement, exactly
like the 1.0:1 marker-ratio gate already in `run_p3v5.sh`.

**P6.2** New `--max-words KIND=N`, capping long-form assets by words rather than
rows. Every `--limit` in this pipeline counts rows, which is the wrong unit as
soon as answer lengths differ by an order of magnitude.

**P6.3** Print and record the full **(condition x behaviour) contingency table**
before training: system-prompted, marker-bearing against marker-free, per kind.
p3-v4 cost a full generate-train-evaluate cycle to discover a 0.47:1 ratio that
one count over the built dataset would have shown; rebalancing it to 1.99:1
tripled persona and produced the best result in the project.

**P6.4** Re-derive the per-kind injection rates rather than copying
`INJECT_ERA=0.6 INJECT_RETAIN=0.6 INJECT_PERSONA=0.9 INJECT_CHESS=0.4`. They are
a behavioural trade, not a constant, and the system channel has changed.

**P6.5** Keep control rows at **zero injection**. They carry no system prompt and
are the contrast that makes the voice conditional instead of constant.

**P6.6** Extend `audit_deepred_corpus.py` with thresholds for the long-form slice
and the chess asset.

## P7 — Pilot, then scale

Maps to Stage 7.

**P7.1** New `run_p4v1.sh`, modelled on `run_p3v5.sh`: the same stage set, the
same single lock directory, the same `require_endpoint` hard stop,
`MODEL_TAG=p4v1`, and training through `podman exec strix-halo-finetuning-gemma4`.
Preflight additionally asserts that the P1 gate JSON exists and passed, that the
P2 winning rung is recorded, and that the P6 signal budget holds on the built
dataset.

**P7.2 Train from scratch — decided.** `INITIAL_MODEL == BASE_MODEL`; no backbone
staging and no `resolve_backbone()` in the driver (decision 6). The general rule
is to stage onto the best checkpoint and change one thing, but here the base
model *is* the variable, so the run necessarily rebuilds behaviour from nothing.
Two consequences, both load-bearing:

- the P6.1 era-native floor is a **blocker, not a nicety**. p3-v5 restarted from
  the untouched base with era-native at 10.6% of the signal and failed to recover
  temporal behaviour in two epochs on a corpus twice the size;
- epochs and learning rate follow the from-base recipe (`EPOCHS=2`,
  `LEARNING_RATE=5e-6`), not the shallow backbone recipe of `1` and `2e-6` used
  by p3-v4b and p3-v4c.

`p3v2-050` and `p3v4c-100` remain comparison points, not starting points.

**P7.3 Pilot.** One epoch, roughly 2,000 steps, snapshots at 10/25/50/75/100.
Evaluate on the frozen 81 *and* the extended bank against `p3v2-050` and
`p3v4c-100`.

**P7.4 Gate.** The pilot must beat both before a full scaled run starts.

**P7.5** Write `documentation/DeepRed-Phase4-Runbook.md` in the style of
[DeepRed-Phase2-Runbook.md](DeepRed-Phase2-Runbook.md): gate-by-gate commands,
pass criteria, fail actions, and where each artefact lands.

---

## Verification

1. `probe_gemma4_support.py` reports every check as a pass, with `torch` still
   `2.12.0a0+rocm7.12`.
2. `probe_train_memory.py` finds a rung with at least 8 GiB of headroom and step
   time within 4x the re-baselined 4B figure.
3. `cat /proc/cmdline` shows `amd_iommu=off` and `amdgpu.gttsize=126976`;
   `cat /sys/module/ttm/parameters/pages_limit` shows `32505856`.
4. A stored Phase 3 `scores.json` reproduces exactly under `rocm-10.0`.
5. A deliberately chess-heavy build — reproducing the p3-v5 59.8% condition —
   **fails** the signal-budget gate. Negative test.
6. `audit_deepred_corpus.py` passes on the p4-v1 corpus including the new
   long-form and chess thresholds.
7. `./run_p4v1.sh --preflight` passes; then `dataset`, and inspect
   `manifest.json` for `signal_share_by_kind`, `marker_rate_by_kind` and the
   contingency table *before* any training.
8. The frozen 81 reproduce known scores for an existing model after the evaluator
   changes.
9. `pytest tests/` for `build_deepred_dataset` and the new gate logic.
10. Interactive transcript check on the pilot for the two defects the frozen
    suite scores as passes: answer length following the question, and chess being
    obtainable at all.
11. Notation regression: a hand-written chess answer containing a SAN move list
    scores `boilerplate` and `severe_repetition` false under the revised
    detectors, and flags under the current ones — proving the fix was needed.
12. Re-run the frozen 81 after the `BOILERPLATE_RE` narrowing and diff against
    the archived `scores.json`. Must be identical, or the change was not inert.
13. The pilot's `run_meta.json` records `initial_model` equal to the base model,
    confirming no backbone leaked in.
14. The corpus manifest records a generator whose family differs from the
    student's.

## Scope

**In:** host reconfiguration, container fork, the Gemma 4 gates, the memory
ladder, evaluator metrics, the extended probe bank, the generator bake-off, the
corpus corrections, the signal budget gate, the pilot run, and the runbook.

**Out:** the full scaled training run, which is gated behind the P7.4 pilot
result; Toolbox Cockpit adoption, since the pipeline stays on direct `podman`;
any change to the frozen 81 probes; the Deep Red Bible, which never enters the
repository; and `train_deepred_gemma.py`, `train_deepred_npo.py` and
`train_deepred_pairwise.py`, which are not on the Phase 3/4 path.

## Constraints carried forward

These are inherited from [DeepRed-Phase4-Setup.md](DeepRed-Phase4-Setup.md) and
apply to every phase above.

- The frozen 81-probe suite stays frozen.
- The 11 probed fact families stay held out of all training data.
- One controlled variable per run.
- Never upgrade the working training container in place.
- Prefer measured figures over per-parameter estimates.
- Audit the (condition x behaviour) contingency table before every training run.
- A gate threshold belongs to the prompt it was measured under.
- Balance the corpus by token mass, not row count.
