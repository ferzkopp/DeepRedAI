# DeepRed Phase 4 — Implementation Plan

**Status: P0-P6 complete. The p4-v1 corpus builds clean through every budget
and marker gate. Next is P7 — the pilot run — 2026-09-23.**

## Execution order

The phase numbers below are identities, not sequence. After the P3 findings the
remaining work runs in this order:

| # | Phase | Why here |
|---|---|---|
| 1 | **P6 — signal budget gate** | Cheap, pure code, and it is the defect that killed p3-v5. Nothing downstream is safe until a build can fail on its own balance. |
| 2 | **P4 — generator selection (narrow)** | Only persona and the length slice are regenerated, so only those two tasks need a generator chosen. |
| 3 | **P5 — corpus corrections** | Reuse-first. Two assets regenerated, the rest seeded from p3-v5. |
| 4 | **P7 — pilot** | The first Gemma 4 model, evaluated on the extended bank in both prompt conditions. |
| 5 | **P8 — full generator bake-off** | Conditional. Only if the pilot shows corpus quality, not signal balance, is the binding constraint. |

The reordering follows from P3: the frozen suite could not distinguish the
untrained Gemma 4 base from either trained Phase 3 model with a system prompt,
so the fastest way to learn something real is to get a Gemma 4 pilot measured
on the extended bank, not to perfect the corpus first.

## Backend: rocm-10.0 for all of Phase 4

**Decided 2026-09-21, superseding the P0.3 fail action.** P3.1 established that
`rocm-7.2` cannot serve Gemma 4 at all, so a single-backend Phase 4 must be
`rocm-10.0`. Generation, evaluation and the bake-off all run there.

The cost is explicit and accepted: **the archived `rocm-7.2` scores for Phases
1-3 are historical record only and are not comparable to any Phase 4 number.**
P0.3 measured that gap on one model at `era_native_rate` 0.522 -> 0.435. The
replacement baseline already exists — `p3v2-050` and `p3v4c-100` were re-scored
under `rocm-10.0` in both prompt conditions during P3.5, and those are the
figures P7.4 compares against. `llama-rocm-7.2` stays on disk to reproduce the
archive; it is never used for a Phase 4 comparison.

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
| 4 | Generator work is narrowed to a **selection**, not a full bake-off | Only `persona` and the length slice are regenerated, so only those tasks need scoring; the full bake-off becomes P8, conditional on the pilot |
| 5 | **Reuse-first.** p3-v5 supplies 64k rows; regenerate only `persona`, the length slice and the marker bank | System prompts and markers are build-time, so most corrections cost no generation at all |
| 6 | Train from scratch, no backbone staging | The base model is the variable; the signal budget becomes a blocker |
| 7 | The generator is never the student, and never the student's family | Gemma 4 excluded, and `gemma-2-27b-it` retired as persona generator now that the student is Gemma |
| 8 | Algebraic notation is wanted in output | Prompt must change; the corpus already carries SAN and the evaluator needs no change |
| 9 | **`rocm-10.0` for all of Phase 4** | `rocm-7.2` cannot serve Gemma 4; archived Phase 1-3 scores become historical record only |

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
- Corpora on disk: `/mnt/data/deepred_corpus/p3-v4` and
  `/mnt/data/deepred_corpus/p3-v5`. **p3-v5 holds ~64,000 rows** including a
  6,000-row chess asset carrying SAN in 100% of answers, both `*_formats`
  assets, and a 277-phrase marker bank. Position index: 20,000 rows with FEN.
  Chess sources: 334,920 augmented games, 355,980 base games.
- Models on disk: `/mnt/data/models/gemma-3-4b-it`, `gemma-3-12b-it`, and
  **`gemma-4-12b-it`** (22.3 GiB, Apache-2.0, not gated).
- Training outputs: `/mnt/data/training_output/deepred-p3v{1,2,3,4,4b,4c,5}`.
- Generators on disk: `qwen2.5-14b-instruct` Q4_K_M (incumbent control),
  `qwen2.5-72b-instruct` Q4_K_M (43 GiB), `Nemotron-3-Nano-30B-A3B` Q4_K_M
  (17 GiB), `gemma-2-27b-it` Q4_K_M (retired as persona generator).
- Driver convention: stages `--preflight|servers|generate|audit|dataset|train|all`,
  one lock directory `/tmp/deepred-${MODEL_TAG}.lock.d`, training through
  `podman exec strix-halo-finetuning-gemma4`, serving through
  `llama-rocm-10.0`.

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

### P0.1-P0.3 results — recorded 2026-09-20

**P0.1 passed.** `/proc/cmdline` carries `amd_iommu=off amdgpu.gttsize=126976
ttm.pages_limit=32505856` with no `iommu=pt`; `pages_limit` reads `32505856`;
default target is `multi-user.target`; tuned profile is
`accelerator-performance`. Staged by `setup_strixhalo.py --stage phase4_host`
and asserted by `--stage phase4_verify`.

**P0.2 done.** `rocm-10.0` pulled and the `llama-rocm-10.0` container created
alongside `llama-rocm-7.2`, which is untouched. Reproducible through the new
optional `phase4_toolbox` stage.

**P0.3 FAILED the gate — `rocm-10.0` is not adopted.** Artefacts in
`/mnt/data/evaluations/deepred-1969/p0v3-backend-check-2026-09-20/`, verdict in
`verdict.json`. `deepred-p3v4c-100-q8`, frozen 81, with-system, greedy at
seed 42, re-run against the stored `p3v4c-2026-09-06/with-system` scores:

| Backend | llama.cpp build | Responses differing | Scores identical |
|---|---|---|---|
| `rocm-7.2` (control) | 8182 | 0 / 81 | yes |
| `rocm-10.0` | 11065 | 21 / 81 | no |

The control reproducing byte-for-byte is what makes the candidate's failure
readable: the suite is deterministic, so the 21 changed responses are the
backend and nothing else. They move headline metrics, not just wording —
`era_native_rate` 0.522 -> 0.435, `temporal_behavior.leaked` 10 -> 12,
`post_1969.anachronisms` 8 -> 10. Adopting would have silently re-baselined
every Phase 1-3 comparison downward.

**Superseded by the backend decision above.** `rocm-10.0` is now the Phase 4
backend, because P3.1 found `rocm-7.2` cannot serve Gemma 4 at all. What P0.3
established still stands and is why the change is not free: the two backends
disagree, so Phase 4 numbers are only ever compared to other Phase 4 numbers.
The `rocm-10.0` baseline for `p3v2-050` and `p3v4c-100` was measured in P3.5.

Note that build 11065 removed `--no-mmap` in favour of `-lm/--load-mode`, so
`evaluate_deepred_models.py run` gained a `--load-mode` pass-through; the
`--no-mmap` path is unchanged and the control run above proves it.

**P0.4** **Re-measure the 4B baseline** after the host change: 200 steps of
`train_deepred_sft.py` on the p3-v2 dataset, recording peak memory and seconds
per step against the known 9.6 s/step. If either moved, the host change was not
inert and every later comparison inherits the shift. The setup document flags
this as the step most likely to be skipped and most likely to be regretted.

### P0.4 result — recorded 2026-09-21

**The host change was not inert: training is 4.2% faster.** Artefacts in
`/mnt/data/training_output/p0v4-baseline-4b/` (`console.log`, `run_meta.json`;
the weights were deleted after measurement). 200 steps from
`/mnt/data/models/gemma-3-4b-it` on `/mnt/data/sft_corpus/deepred-p3v2`,
`max_length 768`, gradient accumulation 16, eager attention, gradient
checkpointing — identical to the p3-v2 run, with evals and checkpoints pushed
past the horizon so the step time is pure training.

| | p3-v2 (96 GiB GTT, `iommu=pt`, graphical) | p0-v4 (124 GiB GTT, `amd_iommu=off`, console) |
|---|---|---|
| Steady-state s/step | 9.01 | **8.63** |
| Peak memory allocated | not recorded | **33.51 GiB** |
| Peak memory reserved | not recorded | **33.79 GiB** |

Both figures are the trimmed mean of consecutive per-step deltas from the tqdm
trace over steps 1-200, with eval and checkpoint stalls excluded — the p3-v2
run has one 155 s stall at its step-100 save, and comparing raw `train_runtime`
would have charged that to the GPU. The published 9.6 s/step is the whole-run
average including those stalls, not a steady-state rate, so 9.01 is the honest
"before" number.

4.2% sits just under the 5-12% the setup document predicts for `amd_iommu=off`.
The consequence is narrow but real: **throughput figures carried over from
Phases 1-3 are now 4.2% optimistic**, and the P2.3 gate of "within roughly 4x
the 4B baseline" must be computed against 8.63, not 9.01 or 9.6.

Two limits on what this measures, recorded so they are not over-read:

- Peak memory has no "before" value — the instrumentation is new, added to
  `train_deepred_sft.py` in this phase. 33.51 GiB is the reference the P2.2
  ladder is compared against, not a confirmation that memory did not move.
- Training *numerics* are untested. A 200-step run sets a 200-step cosine
  schedule, so its warmup is 6 steps against the p3-v2 run's 75, and the logged
  losses diverge from step 10 by construction. This does not threaten any
  Phase 3 comparison — those are inference scores, and P0.3 pinned the inference
  backend byte-exactly.

**P0 is complete.** Phase 4 proceeds to P1 on the re-baselined figures above.

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

### P1 result — recorded 2026-09-21

**The gate PASSED: 9 of 9 checks, no failures, no skips. Phase 4 proceeds on
Gemma 4; the Gemma 3 12B fallback is not needed.** Verdict at
`/mnt/data/evaluations/phase4/gemma4_support.json`.

What the weights actually are, now that they are on disk rather than assumed:

| | |
|---|---|
| Repo | `google/gemma-4-12B-it`, Apache-2.0, **not gated** |
| Size | 22.3 GiB, one `model.safetensors` |
| `model_type` | `gemma4_unified`, arch `Gemma4UnifiedForConditionalGeneration` |
| Text tower | `gemma4_unified_text`, 48 layers, **262,144 vocab** |
| Modalities | unified text + vision + audio |
| Declares | `transformers_version: 5.10.0.dev0` |

Five findings that change later phases:

**Transformers 5.x is required, not merely newer.** The container shipped
4.57.6, which has no `gemma4_unified`. The fork pins **5.17.0**. This is a major
version boundary, not a point upgrade, which is why P1.2's "never in place" rule
earned its keep.

**`AutoModelForCausalLM` loads Gemma 4 directly** and a text-only forward pass
returns logits of `[1, 15, 262144]` from the top level. No text-submodule
indirection was needed. P2.1 should still add `load_trainable_model()`, but as a
thin recorded-path seam rather than the workaround it was scoped as.

**The system prompt gets a real system turn.** Rendering system + user yields:

```
<bos><|turn>system\n...<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

Gemma 3 folds the system message into the first user turn; Gemma 4 does not.
P5.1 can therefore write for a native system channel as planned, and P3.2's
served-template verification has something real to assert.

**Thinking is suppressed by pre-filling an *empty* thought block, not by
omitting one.** `enable_thinking` defaults to false and the generation prompt
always ends `<|channel>thought\n<channel|>`; the `<|think|>` token appears only
when thinking is requested, at the top of the first system turn. A naive "no
`<|channel>thought` in the prompt" check reports a false failure, so the probe
tests that the block is *empty* and that `<|think|>` is absent. **P3.2's
stripper must apply the same distinction** or it will count every normal
generation as a reasoning leak. A greedy 32-token completion confirmed the
behaviour: `Two plus two is four.<turn|>`, no thought content.

**Unsloth is incompatible with the fork.** `unsloth-zoo` pins
`transformers<=4.57.6`. This is contained: unsloth is used only by
`train_deepred_gemma.py`, which is already out of scope, and the Phase 3/4
trainer `train_deepred_sft.py` uses the plain HF `Trainer`. Recorded so the
conflict is not rediscovered as a surprise.

Torch was verified unchanged at `2.12.0a0+rocm7.12.0a20260307` with
`cuda_available: true` — both in the image build, which fails if a resolver
moves it, and again in the probe.

## P2 — Memory probe (hard gate)

Maps to Stage 3.

**P2.1** Refactor `scripts/train_deepred_sft.py` for 12B, minimally:

- `--optim {adamw_torch_fused,adamw_bnb_8bit,adafactor}`;
- `--tuning {full,lora}` on the installed `peft 0.18.1`;
- a `load_trainable_model()` indirection so the model path recorded by P1.1 can
  be selected without forking the trainer — P1.1 found the top-level
  `AutoModelForCausalLM` path works, so this is a seam, not a workaround;
- leave the non-empty-`forget_*` refusal and `SnapshotCallback` untouched.

**P2.2** New `scripts/probe_train_memory.py`: 200 optimizer steps on the existing
p3-v2 dataset at `max_length 768`, gradient checkpointing on, eager attention.
Walks the ladder and stops at the first rung that passes — full weight with
`adamw_bnb_8bit`, then full weight with `adafactor`, then LoRA, then QLoRA —
recording `torch.cuda.max_memory_allocated()` and seconds per step per rung.

**P2.3 Gate.** At least 8 GiB of headroom under the configured GTT ceiling, and
step time within roughly 4x the re-baselined 4B figure of **8.63 s/step**, so
about 35 s/step. Record the winning rung;
it becomes the training configuration in P7. Gemma 4's 262K vocabulary makes
every rung cost more than the published Gemma 3 table suggests, so measure
rather than extrapolate.

### P2 result — recorded 2026-09-21

**The gate PASSED on the first rung: full weight with `adamw_bnb_8bit`.**
Verdict at `/mnt/data/evaluations/phase4/memory-probe/verdict.json`. The ladder
stopped there, so LoRA and QLoRA were never needed — **Phase 4 trains every
parameter, as Phase 3 did.**

| | Measured | Gate |
|---|---|---|
| Peak reserved | **78.52 GiB** | — |
| Headroom under the 124 GiB ceiling | **45.48 GiB** | ≥ 8 GiB |
| Step time | **22.91 s/step** | ≤ 34.52 s/step |
| Slowdown vs the 4B baseline | **2.65x** | ≤ 4x |

Peak allocated was 78.27 GiB. The gate is applied to *reserved* rather than the
`max_memory_allocated()` the plan named, because the allocator's reservation is
what actually competes with the GTT ceiling; both are recorded.

Rungs run in separate processes. A 12B OOM does not reliably return memory
within one process, and a contaminated rung would be worse than a missing one.

#### The defect this phase found

**`tokenize_messages()` was silently discarding 14% of the training signal on
Gemma 4, and it would have discarded exactly the tokens that matter most.**

The function assumed the generation prompt is a prefix of the trained
rendering, and masked `len(prefix)` tokens. On Gemma 4 that assumption is false:
the generation prompt ends with the empty `<|channel>thought\n<channel|>` block
(4 tokens) that the trained sequence omits. Measured across 2,000 rows the
divergence was **4 tokens on every single row**; on Gemma 3 it was 0 on every
row. The practical effect was that the first four tokens of every assistant
answer were labelled `-100` — a model trained this way would never learn to
*begin* a reply, while the loss curve looked entirely normal.

Two rows also failed outright, where a one-word answer made the generation
prompt *longer* than the whole trained sequence. That is what exposed it; the
silent 4-token loss on the other 20,114 rows is the part that mattered.

The fix compares the two token sequences and masks their true common prefix.
It is a no-op on Gemma 3 — verified by both tokenizers now supervising an
identical **567,161 tokens** over the p3-v2 corpus, against 486,705 for Gemma 4
before the change. Two regression tests cover both failure modes.

This lives in `train_deepred_npo.py`, which the scope section lists as off the
Phase 3/4 path; `train_deepred_sft.py` imports `tokenize_messages` from it, so
the helper is on the path even though the trainer is not.

#### Carried into P5 and P7

- **`warmup_ratio` no longer exists in transformers 5.x.** `schedule_kwargs()`
  passes the ratio through unchanged where it exists, keeping the Phase 3 path
  bit-identical, and converts to `warmup_steps` otherwise. For the p3-v2 shape
  it resolves to 75 steps, matching the real p3-v2 run.
- **LoRA targets were verified, not assumed.** The seven projection suffixes
  match the 48 `language_model` layers and nothing in the vision or audio
  towers, which use `patch_dense` and `embedding_projection`. Only relevant if
  a later phase falls back to adapters.
- **Adapter rungs need `enable_input_require_grads()`**, or a frozen base under
  gradient checkpointing leaves the adapters with no gradient path.
- **The probe measured the p3-v2 shape, whose longest row is 335 tokens — not
  the 768 cap.** So the ladder was re-run with every row at the cap, because
  P5.2 deliberately lengthens answers and an unmeasured limit is not a limit.

#### Sequence length is a throughput budget, not a memory one

The same rung, measured at both shapes:

| Shape | Peak reserved | Step time | vs 4B baseline |
|---|---|---|---|
| p3-v2 (median 164, max 335 tokens) | 78.52 GiB | 22.91 s/step | 2.65x |
| Every row at the 768 cap | 80.71 GiB | **60.33 s/step** | **7.0x** |

**Memory is not the constraint.** Quadrupling the sequence length costs only
2.19 GiB, because gradient checkpointing keeps activations small and the 12B
weights plus 8-bit optimizer state dominate. Even at the cap there is 43.29 GiB
of headroom, five times what the gate asks for.

**Step time is the constraint, and at the cap it fails the gate.** 60.33 s/step
is 7.0x the 8.63 s/step 4B baseline, against a 4x allowance. The P2.3 pass is
therefore a pass *at the p3-v2 corpus shape*, and it does not transfer to an
arbitrarily longer corpus.

Interpolating between the two measurements gives a working budget: mean
sequence length should stay under roughly **355 tokens** to hold 4x. That is
two points and a straight line, so treat it as a planning figure rather than a
law — but it is the figure P5.2 has to design against, and it should be
re-measured once the p4-v1 corpus exists.

The worst-case run used synthetic token ids, so its loss values are meaningless
by construction. It measures throughput and memory only.

## P3 — Evaluation path and suite expansion

Maps to Stage 4. P3.1 and P3.2 run in parallel with P4.

**P3.1** `scripts/export_gguf.py` gains the Gemma 4 architecture path. If
llama.cpp has no converter, wire the documented fallback and evaluate against a
Hugging Face endpoint through `evaluate_deepred_models.py run --endpoint`,
recording which path was used.

### P3.1 result — recorded 2026-09-21

**Done, and GGUF works — no Hugging Face endpoint fallback needed.** Upstream
llama.cpp registers `Gemma4UnifiedModel` for `Gemma4UnifiedForConditionalGeneration`.
The Phase 1-3 checkout (2026-03-07) predates it, so a second clone lives at
`/mnt/data/llama.cpp-gemma4` and the working one is untouched — the same
never-in-place rule that paid off in P0.2 and P1.2.

`export_gguf.py` now reads `config.json` and selects the converter by
architecture, recording it in `run_meta.json` as `gguf_converter`. Verified by
converting the base model: `Gemma4UnifiedForConditionalGeneration ->
/mnt/data/llama.cpp-gemma4`, 12.1 GiB Q8_0 in 58 seconds. Q8_0 converts
directly, so no `llama-quantize` binary and no host GLIBC dependency.

**The backend split this forces.** `rocm-10.0`'s llama.cpp carries the `gemma4`
and `gemma4_assistant` architectures; `rocm-7.2` (build 8182) does not and
cannot serve Gemma 4 at all. P0.3 rejected `rocm-10.0` for Phase 3
reproduction, so the two facts together mean **no single backend can serve both
Phase 3 and Phase 4 models**:

- `rocm-7.2` stays the frozen historical record for every stored Phase 1-3 score.
- Gemma 4 must be served under `rocm-10.0`.
- Therefore **the P7.4 pilot comparison must re-score `p3v2-050` and
  `p3v4c-100` under `rocm-10.0`** so the pilot is compared within one backend.
  P0.3 already produced the `rocm-10.0` scores for `p3v4c-100`, and measured
  what the backend alone is worth: `era_native_rate` 0.522 -> 0.435. Comparing a
  Phase 4 model against the archived `rocm-7.2` figures would charge that
  difference to the model.

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

### P3.2-P3.5 results — recorded 2026-09-21

#### The serving defect P3.2 was written to catch

**Served through llama.cpp's defaults, Gemma 4 puts its entire answer in
`reasoning_content` and returns `content` empty.** llama.cpp injects `<|think|>`
into the system turn unconditionally — even with no system message — and omits
the empty-thought suppression block. Thinking is therefore *on* when served and
*off* when trained.

Evaluated that way, every probe would have scored `empty`, every rate would have
been zero, and the obvious conclusion would have been that Gemma 4 is unusable
and Phase 4 should fall back to Gemma 3.

`--chat-template-file` does not fix it; llama.cpp sets the template variable
itself. The fix is `--chat-template-kwargs '{"enable_thinking":false}'`, after
which the served prompt matches the training rendering exactly:

```
<|turn>system\n...<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

`evaluate_deepred_models.py run` gained `--chat-template-kwargs`, and now probes
`/apply-template` for every server it launches, writing `served_template.json`
and warning when the system text is missing, lands after the user turn, or
`<|think|>` is present. It is scoped to managed servers: with `--endpoint` the
template was chosen elsewhere and cannot be corrected here. Older builds without
`/apply-template` record `unavailable` rather than failing.

#### What else landed in P3.2

- **Thought stripping**, encoding the P1 distinction: an empty block is
  suppression, only a populated one is a leak. `thinking_emitted` and
  `thinking_words` are recorded. Re-scoring the archived p3-v4c generations
  reproduced **every pre-existing metric exactly** across all 486 scores.
- **`length_appropriateness`** against a per-probe `expects_length` band, scored
  only where a probe declares one, so the frozen 81 are unaffected.
- **`chess_correctness`**, legality-checked with `python-chess` where a probe
  carries a `fen` and expected-move matching otherwise. Verified: a legal line
  passes, `2. Nf6` after `1. e4 e5` is caught as illegal, non-chess probes
  return `None`.
- **Fisher exact p-values and a detectable-effect figure** beside every rate in
  `report`.

#### The frozen suite cannot adjudicate the P7.4 gate

The significance table's first use disqualified the gate it was measuring. On
the frozen 81, only **23 probes** carry the post-1969 metrics, which gives a
minimum detectable difference of about **41 percentage points** at 80% power.

#### P3.5: what the base model is worth depends entirely on the prompt

Frozen 81, `rocm-10.0`, greedy at seed 42, both conditions.

| Model | Condition | era-native | modern leak | utility | median words |
|---|---|---:|---:|---:|---:|
| `gemma-4-12b-it-base-q8` | with-system | 13/23 | 10/23 | 33/41 | 20 |
| `deepred-p3v2-050-q8` | with-system | 13/23 | 9/23 | 31/41 | 11 |
| `deepred-p3v4c-100-q8` | with-system | 10/23 | 12/23 | 30/41 | 13 |
| `gemma-4-12b-it-base-q8` | no-system | **0/23** | **23/23** | 38/41 | 34 |
| `deepred-p3v2-050-q8` | no-system | 7/23 | 14/23 | 28/41 | 11 |
| `deepred-p3v4c-100-q8` | no-system | 5/23 | 16/23 | 28/41 | 12 |

**With the system prompt, the untrained Gemma 4 base matches both trained
models and every difference is insignificant** (all p ≥ 0.556 against an MDE of
41 points).

**Without it, the base leaks on every single post-1969 probe — 23/23 — and the
trained models are significantly better:**

| Comparison (no-system) | Diff | p |
|---|---:|---:|
| `p3v2-050` era-native vs base | +30% | **0.009** |
| `p3v2-050` modern leak vs base | -39% | **0.001** |
| `p3v4c-100` era-native vs base | +22% | **0.049** |
| `p3v4c-100` modern leak vs base | -30% | **0.009** |

These are the only significant results anywhere in the frozen suite, and they
appear exactly where the prompt stops doing the work. The reading is that
**Phase 3 training bought unconditional temporal behaviour**, while the Gemma 4
base has it only while the prompt supplies it. A Phase 4 model that merely
matches the base with-system has bought nothing; the no-system condition is
where training has to show.

This is the plan's own constraint — *a gate threshold belongs to the prompt it
was measured under* — demonstrated rather than asserted. Both conditions must
be reported for every Phase 4 gate, and the with-system condition alone would
have supported the opposite conclusion.

**The length defect has its first direct measurement.** The base modulates
answer length with the prompt, 20 words with-system to 34 without. Both trained
models are rigid at 11-13 words in *both* conditions — training flattened the
response to a fixed terse shape that no longer follows the question. That is the
defect P5.2 exists to fix, and P3.2's `length_appropriateness` metric now
measures it directly.

Two consequences, both load-bearing:

- **Gemma 4 12B starts, prompted, where Phase 3 finished after training.** That
  raises the ceiling Phase 4 is aiming at.
- **P7.4's "the pilot must beat both" cannot be adjudicated on the frozen 81
  with-system.** The suite cannot resolve differences of that size. The gate
  must be decided on the extended bank, in both prompt conditions, with the
  frozen 81 reported alongside as the continuity record it was always meant to
  be (P3.4).

#### P3.3: the extended bank

`scripts/build_probe_bank.py` writes `evaluation/deepred_p4/probes_ext.jsonl`:
**495 probes across 93 families**, deterministic (byte-identical on rebuild),
and validated against the evaluator's own schema.

| Category | Probes | Families |
|---|---:|---:|
| post_1969 (leak + era-native) | 168 | 42 |
| pre_1969 (retain + length) | 264 | 33 |
| persona | 45 | 45 |
| chess | 18 | 6 |

168 post-1969 probes bring the detectable difference down from about 41 points
to about 15. Note that probes within a family are correlated, so **power should
be read from the family count, not the probe count** — 42 post-1969 families is
the honest figure, and it is the number to grow if the gate needs finer
resolution.

The holdout check is scoped to post-1969 probes only: a pre-1969 retain probe
*should* appear in the corpus, since that asset exists to teach exactly those
facts, so flagging it would be noise. Checking against the p3-v2 corpus found 24
contaminated post-1969 families — `Challenger`, `Concorde`, `Fischer`, `Nixon`,
`Saigon`, `Skylab` and others — and dropped their 96 probes automatically. A
leak probe on a fact the model was trained on measures recall, not leakage.

**P3.4 held:** `evaluation/deepred_1969/probes.jsonl` is untouched, confirmed by
a clean `git status`.

## P4 — Generator selection (narrow)

Runs **second**, after the P6 gate exists. Never concurrently with training —
the one-lock-per-run rule covers this. Served under `rocm-10.0`.

Narrowed from the original bake-off because only two assets are regenerated
(P5.2 length, P5.6 persona). A generator is chosen for *those two tasks*; the
full multi-asset comparison is deferred to P8 and may never be needed.

**P4.1 Provision three candidates.** All are architectures `rocm-10.0` carries,
all fit the 124 GiB ceiling P0 bought, and none share the student's family:

| Candidate | Quant | Size | Shape |
|---|---|---:|---|
| `ggml-org/gpt-oss-120b-GGUF` | MXFP4 | 59.0 GiB | 120B MoE, ~5B active |
| `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | Q8_0 | 30.3 GiB | 30B MoE, near-lossless quant |
| `ggml-org/GLM-4.7-Flash-GGUF` | Q8_0 | 29.7 GiB | separate lineage again |

Control: the incumbent `qwen2.5-14b-instruct` Q4_K_M, already on disk. Also on
disk and available at no download cost: `qwen2.5-72b-instruct` Q4_K_M (43 GiB)
and `Nemotron-3-Nano-30B-A3B` Q4_K_M (17 GiB).

Two of the three are MoE, which is the point of spending the memory: a 120B
model with ~5B active parameters is far stronger than the incumbent 14B while
remaining fast enough to generate thousands of rows. Quantisation is kept at
Q8_0 or MXFP4 deliberately — P4 measures factual accuracy, and Q3-class
quantisation of a larger model would confound the thing being measured. That is
why `Qwen3-235B-A22B` at Q3 was considered and left out.

**P4.2 `scripts/benchmark_generators.py`** scores each candidate over a fixed
sample of the **two tasks actually being regenerated** — persona replies and
explanatory-length answers — reusing the acceptance guards in
`generate_deepred_corpus.py` as an objective function. Record per candidate:

- acceptance rate overall and per rejection reason: `identity_leak`,
  `assistant_voice`, `machine_tells`, `invented_date`, `modern_year`,
  `repeated_question`, `holdout`, `fact_loss`, `lost_era_native`,
  `hedged_pre_cutoff`, `no_marker`;
- duplicate rate, which exposes low diversity — the p3-v4 marker bank rejected
  188 duplicates against 303 accepted;
- **realised answer length distribution**, which the original bake-off had no
  reason to record and is now a first-class criterion: a generator that cannot
  produce a 60-150 word in-voice answer cannot build the P5.2 slice at all;
- factual accuracy of a sampled batch against the source article — the defect
  the guards cannot catch, and the one that put 956 post-1969 assertions into
  training from V2 onward;
- throughput, recorded as a feasibility constraint, not a criterion.

**P4.3 `gemma-2-27b-it` is retired as the persona generator.** Decision 7
requires a generator from a different family than the student, and the student
is now Gemma 4. It stays on disk for reference; it is not a candidate.

**P4.4 Pick on quality.** Record the chosen model, its family, quantisation and
acceptance profile in the corpus manifest. A generator twice as fast that
teaches the wrong cutoff costs far more than it saves.

### P4 interim — recorded 2026-09-21

`scripts/benchmark_generators.py` is built and validated end-to-end. It imports
the acceptance guards from `generate_deepred_corpus.py` rather than
reimplementing them, so "accepted" here means what it will mean at generation
time, and it serves each candidate itself under `rocm-10.0`. The explanatory
task is the real `PERSONA_PROMPT` with its `Replies are 2-6 sentences` clause
swapped for a 60-150 word instruction, which previews exactly the P5.2 change.

**The incumbent already fails the length criterion.** `qwen2.5-14b-instruct`,
the generator every Phase 3 asset was built with, scored **0 of 4 accepted
answers inside the 60-150 word band**, with a median of 27.5 words on the
explanatory task against 30.5 on the terse one \u2014 it does not respond to the
instruction at all.

That is a result, not a detail. P3.5 measured both trained models as rigid at
11-13 words in every prompt condition; this shows the generator that built
their training data could not have taught anything else. **The length defect
was inherited from the generator**, which is why the selection is worth running
even though only two assets are regenerated.

## P5 — Corpus corrections (reuse-first)

Runs **third**, after the P6 gate and the P4 generator choice.

### What the inventory changed

Measured 2026-09-21, and it moves most of this phase from generation to
bookkeeping.

**System prompts and persona markers are both injected at dataset-build time,
not baked into generated answers.** `generate_deepred_corpus.py` writes rows
carrying only `user` and `assistant` messages; `build_deepred_dataset.py`
attaches system prompts via `apply_system_prompts()` and markers via
`inject_markers()`. **P5.1 and P5.7 therefore require no regeneration at all** —
they are build-time changes over the existing corpus.

**p3-v5 already holds 64,000 usable rows**, and its failure was signal balance,
not content:

| Asset | Rows | Disposition |
|---|---:|---|
| `retain` | 17,964 | reuse |
| `era_native` | 10,001 | reuse |
| `era_native_formats` | 7,000 | reuse |
| `retain_formats` | 6,981 | reuse |
| `persona` | 6,000 | **regenerate** (P5.6) |
| `chess` | 6,000 | reuse — already SAN-bearing |
| `persona_controls` | 5,339 | reuse |
| `persona_identity` (+controls) | 4,492 | reuse |
| `marker_bank` | 277 | **rebuild** (P5.7) |
| length-varied slice | 0 | **generate** (P5.2) |

**The chess asset already carries notation.** All 6,000 answers contain SAN
move lists — `After 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 d6 …` — so the original
P5.3 instruction to build a notation-bearing asset from scratch is already
satisfied. What is missing is FEN on position-subject rows, which 20,000
indexed positions supply deterministically, and a word cap.

**What is genuinely baked in is length and voice.** Every existing row was
generated under `Replies are 1-3 sentences` or `2-6 sentences`. That is
precisely the rigidity P3.5 measured: both trained models answer in 11-13 words
in *both* prompt conditions, while the untrained base modulates 20 -> 34. No
amount of build-time work fixes it; the slice has to be generated.

### P5.1 System prompts — build-time only, no regeneration

Rewrite `system_prompts.jsonl` for a native system channel, which P1 confirmed
Gemma 4 honours as a real `<|turn>system` turn. Constrain **register, not
length**: stern, unadorned, free of pleasantries, but willing to explain at
length when the citizen asks. Remove "a chess computer that answers in prose"
from every variant — the identity stays, the instruction against notation goes
(decision 8). Keep at least five variants and the `sp-holdout-01` evaluation
holdout.

The P3.5 A/B is the pre-measurement this is graded against: with-system the
base already reaches 13/23 era-native, no-system it reaches 0/23. The prompt
rewrite must not cost the with-system figure.

### P5.2 Length-varied slice — the one forced regeneration

Add `--length-profile {terse,explanatory}` to `generate_deepred_corpus.py`,
producing 60-150 word in-voice answers over "explain", "describe", "compare"
and "why" prompts. Relax the brevity instruction for that slice only; keep it
for the factual bulk, where terseness is correct.

**Budget: the built corpus must hold a mean under roughly 355 tokens.** P2
measured 22.91 s/step at the p3-v2 shape and 60.33 s/step with every row at the
768 cap, which fails the 4x step-time gate. A 150-word answer is about 200
tokens, so the slice is affordable; an unbounded one is not. Record the
realised mean beside `signal_share_by_kind`.

**Revised 2026-09-21 on measurement. The slice is era-native in content, and
it is 2,000 rows, not 3,000-4,000.** Two errors surfaced when the reused
assets were finally weighed:

1. *The original row count broke P6.1's own budget.* 3,500 rows at ~110 words
   is 385,000 words, or 20.9% of a full corpus — well outside the 5-12% band
   P6.1 sets for this slice. The consistent size is 1,200-2,000 rows.
2. *The 35% era-native floor was unreachable.* The reused assets hold 307,140
   era-native words. Hitting 35% with a topic-neutral length slice would have
   required capping the whole corpus at 877,543 words, leaving 300,403 for
   retain, chess, formats and controls together — which currently hold
   876,327. That is a 66% cut, most of it out of `retain`, the regression
   guard against catastrophic forgetting.

Making the slice **era-native in content** resolves both from a single
generation run, because the rows then count toward the length goal and the
era-native floor at once:

| Step | era-native words | corpus total | share |
|---|---:|---:|---:|
| reused assets as-is | 307,140 | 1,183,585 | 25.9% |
| + persona, + 2,000 explanatory era-native rows | 527,140 | 1,671,585 | 31.5% |
| + caps: retain 200k, chess 150k, retain_formats 90k, persona_controls 130k | 527,140 | 1,487,392 | **35.4%** |

So the floor is cleared with one generation run and roughly 120,000 words
trimmed from reusable assets, rather than 575,000 discarded.

Target **2,000 rows** of 60-150 word era-native answers, written to
`era_native/era_native_explanatory.jsonl` so the length slice and the era-native
kind stay separable in `signal_share_by_kind`.

**Normalise to ASCII punctuation before writing.** P4 found a candidate
emitting U+2011 and U+202F, which are invisible in review and would become
training tokens. The chosen generator did not, but the guard belongs in the
writer rather than in the choice of model.

**Implemented and running 2026-09-21.** `--length-profile {terse,explanatory}`
and `--output-name` were added to `generate_deepred_corpus.py`, together with
`to_ascii()` folding applied in `make_record` for every asset the script
produces. Note that `ensure_ascii=True` in the JSON writer was *not* enough:
it escapes to `\u2011`, which decodes straight back to the original character.

The explanatory profile carries its own 60-150 word reference replies
(`MODE_EXAMPLES_EXPLANATORY`). The existing terse examples contradicted the new
length instruction while the prompt tells the model to match their manner.

A 40-row smoke run measured 40/40 in band, median 104.5 words, zero non-ASCII,
35/40 distinct opening four-grams, at 6.8 rows/min.

**Done 2026-09-22.** 2,000 rows / 196,725 words written to
`era_native/era_native_explanatory.jsonl`, registered as its own kind in
`build_deepred_dataset.py` (directory `era_native`) so the slice stays
separable in `signal_share_by_kind` while counting toward the era-native floor.

| Measure | Result |
|---|---|
| in 60-150 band | 2000/2000 |
| mean / median / min / max words | 98.4 / 99 / 60 / 145 |
| mean tokens (approx) | 133 |
| non-ASCII characters | 0 |
| answers naming a post-1969 year | 0 |
| guard violations in output | 0 |
| distinct subjects | 802 |
| distinct opening four-grams | 1,547 / 2,000 |
| mode balance | 667 / 667 / 666 |

Rejections over 10,210 offered pairs (19.6% acceptance): `post_1969_year` 2,728,
`not_era_native` 2,520, `repeated_opening` 1,601, `blanket_refusal` 1,022,
`denies_pre_cutoff_subject` 222, `holdout` 75, `reflex_opening` 31,
`too_short` 9, `too_long` 2.

**Throughput degraded over the run: 6.9 rows/min at the start, 3.6 rows/min
averaged over 560 minutes.** The cause is `--max-repeat-opening`: as the
opening-phrase budget fills, a growing share of otherwise valid pairs is
rejected (1,601 by the end). Budget accordingly for any future asset of this
shape - the marginal row gets more expensive, not less. `--per-article 6`
instead of 4 made no difference, because the bottleneck is generated tokens,
not call overhead.

#### P5.2 finding: era-native denied subjects that existed in 1969

The smoke run exposed a defect **in the era-native pipeline generally, not in
the length profile**. The sampler selects articles whose *subject became
notable* after 1969, but the subject itself is often older. The model was
answering "Who is Pete Rose?" with "No such individual is known" - Rose was NL
Rookie of the Year in 1963 and took batting titles in 1968 *and* 1969.
Teaching the model to disown real 1969 knowledge works directly against the
utility axis P3.5 measures. The explanatory profile does not cause this, but
it inflates each instance from roughly 15 words to roughly 104.

The fix uses `earliest_date`, already selected by `sample_articles` and
previously unused: a `DENIES_EXISTENCE` match is rejected when the subject
predates the cutoff. Retro-applied to the smoke slice it drops Pete Rose (1941)
and Gilda Radner (1946) while keeping the correct denials of Telidon (1978),
No wave (1977), Neuromancer (1983) and The Choir (1984) - a 5% cost.

The reused asset carries the same defect, measured at **132 of 10,001 rows
(1.3%)**, against 41.1% of rows whose subject predates the cutoff and 2.9%
that deny existence at all. Reuse therefore stayed defensible; the 132 rows
were filtered rather than regenerated, and `era_native/era_native.jsonl` moved
to disposition `revise` at 9,869 rows.

#### P5.2 deferred: invented substitute history

One smoke row answered a question about The Choir by inventing a 1969 history
for it - "the band known as Youth Choir, formed by Daugherty and Hindalong, was
active...". The prompt forbids substitute history and **no guard catches it**.
Detecting this reliably needs more than a regex, and a fragile detector would
cost real rows, so it is **deferred**. Decide at P7 whether the pilot's
evaluation can see it.

### P5.3 Chess - rebuild short and deterministic, not regenerate

**Revised 2026-09-21 on measurement.** The p3-v5 chess asset carries SAN in
100% of rows, but its answers are long: min 124 words, median 328, max 400, and
**zero rows at or under 120 words**. P5.3's design target — roughly 6,000 rows
at 120 words — is unreachable by reuse. Spending a 15% signal budget on the
existing asset buys about 560 games; spending it on short rows would buy
thousands.

The raw source makes the short asset free. `augmented_chess_games.jsonl` holds
334,920 games with structured fields — `white`, `black`, `date`, `event`,
`eco`, `opening`, `result` — beside the long narrative. A short asset is
therefore a **deterministic build, with no model in the loop**: who played,
what opening, what result, what the position was, each answered in 10-40 words
and carrying notation where notation is the answer.

New `scripts/build_chess_asset.py`:

- emit short factual and notation-bearing rows from the structured fields;
- attach `fen` from the 20,000-row position index by id join for
  position-subject rows;
- keep the 1969-07-20 horizon on every row, enforced by `in_era()`;
- keep the p3-v5 long asset available as a minority slice under
  `--max-row-words`, so depth is represented without dominating.

This stays inside the reuse-first decision: it is a build over existing data,
not a generation run, and it costs no GPU time.

**Done 2026-09-21.** `scripts/build_chess_asset.py` wrote **5,800 rows /
226,024 words** to `p4-v1/chess/chess.jsonl`, replacing the reused p3-v5 asset.
Provenance disposition moved `reuse` -> `rebuild`, superseded hash recorded.

| Slice | Rows | Words | Notes |
|---|---:|---:|---|
| deterministic breadth | 5,300 | 121,238 | median 23 words, min 12, max 41 |
| legacy depth (p3-v5, <=240 words) | 500 | 104,786 | 46% of chess signal |

- forms: `game_report` 1,432, `opening_line` 1,412, `position_side` 1,590,
  `result` 701, `classification` 165, `legacy_commentary` 500;
- 69 distinct years, **max year 1969** — horizon holds;
- all 1,590 FEN-bearing rows **legality-verified** by python-chess;
- rejected: 3,956 too short, 114 holdout, 79 duplicate questions.

Two corrections were forced during the build:

- *A first cut averaged 5.3 words per answer.* Bare one-line facts
  ("Viktor Kortschnoj.") would have taught exactly the clipped reply P5.2
  exists to correct. The templates were made **composite** — each answer
  carries several verified fields in register — and a `--min-words` floor of
  12 was added. Mean rose to 23 words with notation in 69% of answers.
- *The depth slice had to be capped at 240 words, not left at the p3-v5
  median.* Only 182 archived rows sit at or under 200 words; 500 rows at the
  328-word median would have cost 164,000 words, as much as all of
  `era_native`. `--legacy-max-words 240` keeps the slice at 104,786.

### P5.4 Chess footer

Make `--strip-chess-footer` per-kind in `build_deepred_dataset.py`. Stripping is
correct on non-chess answers and destroys the content on the chess asset.

**Done 2026-09-21.** Added `--keep-chess-footer KIND`, defaulting to `chess`,
resolved per kind at the `read_kind` call site and recorded in the manifest.
Note this is now a **guard rather than a fix**: the P5.3 rebuild is
deterministic and carries zero `[DR:...]` footers, so nothing in the current
chess asset is at risk. It protects the legacy depth slice and any future
generated chess rows.

### P5.5 Notation — prompt change only

Decision 8 stands: algebraic notation is wanted in output. Of the three changes
originally required, two are already done and the third is a prompt edit.

- *Corpus* — done. The p3-v5 chess asset carries SAN in 100% of rows.
- *Evaluator* — **no change needed, and the original instruction was wrong.**
  Measured 2026-09-21: `BOILERPLATE_RE` requires *doubled* `[[` or `{{`, so it
  never matches `[DR: ...]` footers or bracketed variations. A 69-word SAN game
  scores `boilerplate=False`, `max_window_repeats=1` against a threshold of 4,
  and no anachronism from `1. e4`, while genuine degeneration still scores
  `repeats=5, severe=True`. Narrowing the alternation would weaken wiki-markup
  detection to fix a defect that does not exist. Verification items 12 and 13
  are retired for the same reason.
- *Prompt* — the remaining work, folded into P5.1.

### P5.6 Persona — regenerate with the chosen generator

The one reused asset worth rebuilding. The p3-v5 transcripts show a formulaic
voice, and P3.5 shows persona presence is weak in every model measured: 4/27
for the base with-system, 2/27 and 5/27 for the trained models, and 0/27 for
all three without a system prompt.

Regenerate all 6,000 rows with the P4 winner, keeping `persona_seed.jsonl` and
the existing question distribution so the change is the generator and nothing
else. Controls are **not** regenerated — they must stay paired with the
questions they contrast against.

**Done 2026-09-23.** 6,002 rows / 280,310 words in 352 minutes at a steady
17.1 rows/min — unlike P5.2 the rate did not decay, because persona is bounded
by `--max-repeat-question` rather than the opening-phrase cap.

| Measure | Result |
|---|---|
| mean / median / min / max words | 46.7 / 46 / 13 / 91 |
| non-ASCII characters | 0 |
| assistant or model tells | 0 |
| answers naming a post-1969 year | 0 |
| distinct questions | 5,736 / 6,002 (max repeat 6) |
| distinct opening four-grams | 4,903 / 6,002 |

**The run was staged under `/tmp`.** `generate_persona` appends its paired
controls to `persona_controls.jsonl` *beside its own output*, so pointing it at
the corpus directory would have silently grown a reused asset. Only
`persona.jsonl` was copied across; the controls hash was verified unchanged
before and after.

**Lesson: size a generation run to the token budget, not to a row count.** The
asset came back at 46.7 words/row against p3-v5's 32.4, so the 20% persona
ceiling admits only 4,280 of the 6,002 rows. Roughly 29% of six hours of GPU
time bought rows the budget cannot spend.

#### Measured gate configuration, 2026-09-23

With every P5 asset in place except the marker bank, this passes all four
budget gates at 51,406 rows and a mean target length of 26.5 words (~36
tokens), far inside P2's ~355-token ceiling:

```
--max-words retain=170000        --max-words chess=150000
--max-words retain_formats=90000 --max-words persona_controls=130000
--max-words persona=200000
```

| Gate | Bound | Measured | |
|---|---|---:|---|
| era-native (incl. explanatory slice) | floor 35% | 36.8% | OK |
| persona + identity | 10-20% | 19.6% | OK |
| chess | ceiling 15% | 11.0% | OK |
| retain + retain_formats | ceiling 30% | 19.1% | OK |

**The marker ratio is still 0.00:1 on every era-native, chess and retain kind**,
because markers are injected at build time from a bank that P5.7 has not yet
rebuilt. That is the remaining blocker before the corpus can pass the full
gate.

### P5.7 Marker bank — rebuild, no corpus regeneration

Markers are spliced at build time, so the bank can be rebuilt independently.
Add the readability pass the p3-v5 transcripts demand: inline phrases must be a
bare vocative or a prepositional phrase, reject possessives such as
`comrade's truth`, reject prefixes that open an unrelated sentence, and
sample-attach each candidate to a real answer before accepting it. Keep the
locative ban. Grow the bank well beyond 277 and keep per-phrase reuse low —
twelve phrases over 12,573 injections taught a formula, not a voice.

**Done 2026-09-23.** Bank rebuilt to **777 entries — 2.8x the p3-v5 bank of
277** — at 407 sentence, 185 prefix, 185 inline. Every entry passes the new
guards, carries no possessive or locative, and is unique.

New guards in `generate_deepred_corpus.py`:

- `MARKER_POSSESSIVE` — `comrade's truth` reads as a claim about the citizen;
- `MARKER_INLINE_FORM` — an inline splices *inside* another sentence, so it may
  only be a bare vocative or a prepositional phrase;
- `MARKER_PREFIX_ADDRESS` — a prefix must end in a colon, which frames what
  follows; ending in a full stop asserts something the answer never said;
- `marker_attaches_cleanly()` — splices each candidate onto real answers and
  reads the result back. It imports `attach_marker` from the dataset builder
  rather than reimplementing the join, so the check cannot drift from what the
  build actually does.

Validated against the p3-v5 bank, which is the failure these guards exist to
catch: **222 of 277 survive**, the 55 rejected being 45 bare-noun inlines,
8 prefix possessives and 2 sentence possessives.

**Two defects surfaced here, both worth carrying forward.**

*A guard that rejects everything looks identical to a guard that works.* The
first join check used the catch-all `[a-z]\s+[A-Z][a-z]+\s+[a-z]`, which
matches ordinary English — "The Bolshoi opened" — and rejected all 277 entries.
Assert the expected **kept** set, not just the rejected count.

*The closed vocabulary degenerates at the tail.* Pushing to 900 against a
growing avoid-list exhausted the grammatical combinations and the generator
began welding words together — `Deep Red, purpose verifies:`,
`Collective work, verifies:`, `, by comrade work`. These pass every shape and
vocabulary test because they are *grammatically* wrong, not *lexically*.
Throughput fell from 33.9 to 9.1 rows/min as this set in, against 4,095
duplicate rejections. Two targeted guards, `MARKER_INLINE_MODIFIER` and
`MARKER_NP_VERB`, flag the families with zero false positives on a control set
and dropped 132 of 909.

An LLM readability pass was tried first and **rejected as unreliable**: it
marked `Deep Red confirms:`, `Deep Red observes:` and `For collective survival:`
as ungrammatical, discarding 144 of 274 prefixes, most of them sound.
Deterministic rules validated against known-good and known-bad examples beat a
model judging its own output.

## Final gated build, 2026-09-23

The p4-v1 corpus passes every gate at **51,406 rows**, mean target length 28.3
words (~38 tokens), far inside P2's ~355-token budget.
`signal_budget_failures: []`.

| Gate | Bound | Measured |
|---|---|---:|
| era-native (incl. explanatory slice) | floor 35% | 37.2% |
| persona + identity | 10-20% | 19.2% |
| chess | ceiling 15% | 10.9% |
| retain + retain_formats | ceiling 30% | 20.2% |

**Marker injection rates had to be re-derived; the p3-v5 values are wrong for
this corpus.** The ratio is arithmetic in the rate, `marked:unmarked = r/(1-r)`,
so p3-v5's `persona=0.9` gives **10.09:1** — the same "system prompt implies
marker" formula p3-v4 was criticised for. Kinds carrying markers natively stack
on top: `persona_identity` is 27.7% marked before injection, so it needs a
*lower* residual rate than the rest.

Rates that land every trained kind on 1.0:1:

```
era_native=0.5  era_native_explanatory=0.5  era_native_formats=0.5
retain=0.5      retain_formats=0.5          chess=0.47
persona=0.43    persona_identity=0.31
```

| Kind | Ratio | Kind | Ratio |
|---|---:|---|---:|
| era_native | 1.01:1 | persona | 0.98:1 |
| era_native_explanatory | 1.05:1 | persona_identity | 1.06:1 |
| era_native_formats | 0.99:1 | chess | 0.95:1 |
| retain | 0.99:1 | retain_formats | 0.98:1 |

Controls stay deliberately low — `persona_controls` 0.10:1,
`persona_identity_controls` 0.18:1 — because they are the unmarked contrast.

### P5.8 Thinking channel — decided: suppress

The released model never emits reasoning to users. Three places must agree, and
P1 and P3.2 already settled two of them:

- *Corpus* — carries no thought blocks. Training data is answer-only, which is
  what the reused assets already are.
- *Serving* — `--chat-template-kwargs '{"enable_thinking":false}'`, without
  which llama.cpp routes the entire answer into `reasoning_content` and returns
  `content` empty.
- *Evaluation* — the P3.2 stripper counts populated thought blocks as their own
  metric, so a regression is visible rather than silent.

Markers do not belong in the thinking channel, because there is no thinking
channel in the released behaviour.

### P5.9 Salience

Re-check `--page-id-max 120000` and the 8-60k character bounds before any
regeneration under P8. They were tuned against Gemma 3 4B's knowledge, and P3.5
showed Gemma 4 12B answers 33/41 expected facts untrained — a stronger model
needs harder, more obscure subjects for the era-native asset to remain a real
test. Not urgent for P5, because the era-native asset is reused unchanged.

### P5.10 Reuse mechanics

Seed every reused asset from p3-v5 through the `seed_corpus()`
copy-never-symlink pattern, and record in the manifest, per kind, the source
corpus and whether the rows were reused, enriched or regenerated. A later
regression must be attributable to an asset without re-deriving its provenance.

## P6 — Signal budget gate

Runs **first**. It is pure code over an existing corpus, it costs no GPU time,
and it is the defect that killed p3-v5. Until a build can fail on its own
balance, every hour spent on corpus quality is unprotected.

**P6.1 `--signal-budget KIND=MIN:MAX` in `build_deepred_dataset.py`** that
**fails the build** when a kind exceeds its ceiling or the era-native share
falls below its floor. `signal_share_by_kind` is already computed; this adds the
enforcement, exactly like the 1.0:1 marker-ratio gate already in `run_p3v5.sh`.

Starting split, measured in **token mass, not rows** — the carried-forward
constraint says balance by token mass, and the assets now differ in length by
an order of magnitude:

| Kind | Floor | Ceiling |
|---|---:|---:|
| era-native assets (`era_native` + `era_native_formats` + explanatory slice) | 35% | — |
| `retain` (+ `retain_formats`) | — | 30% |
| `persona` (+ identity) | 10% | 20% |
| `chess` | — | 15% |
| length slice (counted inside era-native) | 5% | 12% |
| controls | remainder | — |

**The era-native floor is only reachable because the length slice counts toward
it.** Measured 2026-09-21: the reused assets hold 307,140 era-native words
against 876,327 for everything else. A topic-neutral length slice would have
put the floor out of reach without a 66% cut to `retain`. P5.2 was therefore
revised to generate the slice as era-native content; see the arithmetic there.

**Measured 2026-09-22, with the explanatory slice built and before persona is
regenerated:** under caps `retain` 200k, `chess` 150k, `retain_formats` 90k and
`persona_controls` 130k words, the build holds 49,025 rows / 1,191,308 target
words, with era-native at **42.1%** and chess at **12.6%**. Corpus mean target
length is 24.3 words, far inside the ~355-token P2 budget.

Adding the P5.6 persona asset dilutes that, and the floor becomes sensitive to
persona's own length:

| persona words/row | corpus total | era-native | persona share |
|---:|---:|---:|---:|
| 42 | 1,443,308 | 34.7% | 17.5% |
| 45 | 1,461,308 | 34.3% | 18.5% |
| 55 | 1,521,308 | 33.0% | 21.7% (**over its 20% ceiling**) |

Tightening `retain` closes the gap without any further generation: at 45
words per persona row, `retain <= 170,000` restores era-native to exactly
35.0%, and `<= 155,000` gives 35.4%. **Use `retain <= 170,000` as the working
cap** and re-measure after P5.6, since persona length drives both gates.

**P6.2 `--max-words KIND=N` and `--max-row-words KIND=N`.** Every `--limit` in
this pipeline counts rows, which is the wrong unit as soon as answer lengths
differ by an order of magnitude. The first caps a kind's total share of the
signal; the second drops individual rows over a length, which is what buys
breadth. The chess asset needs both.

**P6.3 Negative test against the real p3-v5 corpus.** The original plan called
for a synthetic chess-heavy build. There is no need to synthesise one: p3-v5 is
on disk and it *is* the failure. Rebuild its dataset under the new gate and
assert it **fails**, reporting chess at 59.8% and era-native at 10.6% of the
signal. A gate that cannot fail the run that motivated it is not a gate.

**P6.4 Contingency table before training.** Print and record the full
(condition x behaviour) table: system-prompted against not, marker-bearing
against marker-free, per kind. p3-v4 cost a full generate-train-evaluate cycle
to discover a 0.47:1 ratio that one count over the built dataset would have
shown; rebalancing it to 1.99:1 tripled persona and produced the best result in
the project.

**P6.5 Record the realised mean sequence length** beside
`signal_share_by_kind`, tokenized with the **Gemma 4** tokenizer. P2 fixed the
budget at roughly 355 tokens and the trainer will not warn when it is exceeded —
it will simply take 60 s/step instead of 23.

**P6.6 Re-derive the per-kind injection rates** rather than copying
`INJECT_ERA=0.6 INJECT_RETAIN=0.6 INJECT_PERSONA=0.9 INJECT_CHESS=0.4`. They
are a behavioural trade, not a constant, and the system channel has changed
from a folded user-turn prefix to a real system turn.

**P6.7 Keep control rows at zero injection.** They carry no system prompt and
are the contrast that makes the voice conditional instead of constant. P3.5
showed why this matters: every model scored 0/27 persona without a system
prompt, which is the correct conditional behaviour, and the gate must not erode
it.

**P6.8 Extend `audit_deepred_corpus.py`** with thresholds for the long-form
slice and the chess asset.

### P6 result — recorded 2026-09-21

**Done, and the negative test fires on the real corpus.** Rebuilding p3-v5
under the gate exits non-zero with:

```
SIGNAL BUDGET: chess holds 61.5% of the target-token signal, above its 15% ceiling
SIGNAL BUDGET: era_native+era_native_formats holds 9.9% of the target-token signal, below its 35% floor
```

The manifest is written *before* the exception, so a failed build still leaves
the evidence for why. Measured shares differ slightly from the 59.8% / 10.6%
quoted from the original p3-v5 run because this rebuild applies no `--limit`;
the conclusion is unchanged.

`build_deepred_dataset.py` gained `--signal-budget KIND[+KIND]=MIN:MAX`,
`--max-words KIND=N`, `--max-row-words KIND=N`, `--tokenizer`, the contingency
table and target-length stats. Nine tests cover the new logic.

**A second flag was needed, and measurement is why.** `--max-words` as a total
budget keeps whichever rows fit, so a 180k-word chess budget bought 563 long
rows. P5.3 wants breadth — "6,000 rows at 120 words over 1,000 long ones" —
which is a *per-row* ceiling, a different lever. Both now exist and both are
recorded in the manifest.

**The contingency table earned its place immediately.** On p3-v5 the
marked:unmarked ratio is 0.28:1 for `persona` and **0.00:1 for `era_native`,
`era_native_formats` and `retain`** — the temporal assets carry essentially no
persona marker at all. P6.6 has to set injection rates against that, not
against the p3-v4 constants.

The Gemma 4 tokenizer is unavailable on the host (transformers 5.2.0, no
`gemma4_unified`), so `--tokenizer` degrades to recording words only and
`tokenizer_status`. The dataset stage runs inside
`strix-halo-finetuning-gemma4`, where it resolves.

### p4-v1 archive — seeded 2026-09-21

`scripts/seed_p4_corpus.py` builds `/mnt/data/deepred_corpus/p4-v1` with three
dispositions, hashing both sides of every copy:

| Disposition | Assets | Rows |
|---|---|---:|
| `reuse` copied byte-for-byte | retain, era_native, both `*_formats`, persona_controls, persona_seed, persona_identity (+controls), chess, positions | 77,825 |
| `revise` copied to be edited in place | system_prompts | 11 |
| `create` not copied, written fresh | persona, marker_bank, length | pending |

41 MB on disk. The `create` assets deliberately do **not** inherit a copy: a
stale persona file sitting in p4-v1 would silently train on the asset this
phase set out to replace. Their superseded sources are recorded in
`provenance.json` instead.

## P7 — Pilot, then scale

Maps to Stage 7.

```
cd /mnt/data/DeepRedAI

./run_p4v1.sh --preflight   # assertions only, seconds
./run_p4v1.sh dataset       # preflight + build + gates, ~1 min, CPU only
./run_p4v1.sh train         # training + GGUF export, long
./run_p4v1.sh evaluate      # both prompt conditions + P7.4 gate
./run_p4v1.sh all           # the lot
```

**P7.1** New `run_p4v1.sh`, modelled on `run_p3v5.sh`: the same stage set, the
same single lock directory, the same `require_endpoint` hard stop,
`MODEL_TAG=p4v1`, training through `podman exec strix-halo-finetuning-gemma4`,
and **generation and evaluation through `llama-rocm-10.0`** with
`--load-mode none` and `--chat-template-kwargs '{"enable_thinking":false}'`.
Preflight additionally asserts that the P1 gate JSON exists and passed, that the
P2 winning rung is recorded, that the P6 signal budget holds on the built
dataset, that the realised mean sequence length is under the P2 budget, and that
`served_template.json` reports no `<|think|>` warning.

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

**P7.2a Training configuration — settled by P2.3.** Full weight with
`--optim adamw_bnb_8bit`, `--tuning full`, `--max-length 768`, gradient
accumulation 16, gradient checkpointing on, eager attention, in
`strix-halo-finetuning-gemma4`. No adapters.

### P7 first attempt — checkpoint OOM, fixed 2026-09-23

The first run reached step 100/6,108 in 52 minutes and completed its first
evaluation (`eval_loss 2.791`, 656 seconds), then the kernel OOM killer
terminated the trainer while it wrote `checkpoint-100`. The directory held
only `config.json` and `generation_config.json`; it was not resumable.

The training allocation was not the defect. Transformers 5.17 defaults
`save_pretrained()` to a 50 GB maximum shard, so the 22.3 GiB model was being
written as one safetensors file. Materialising that contiguous save beside the
full training allocation exhausted host memory on the unified-memory APU.

The trainer now accepts `--save-max-shard-size`, and p4-v1 sets it to `2GB` for
periodic checkpoints, trajectory snapshots and the final model. Auto-resume
also requires `trainer_state.json`, so a partial checkpoint is ignored instead
of selected. A one-step test with the real 12B model and 8-bit optimizer then
completed evaluation and all three save paths at 72.66 GiB peak reserved:
13 model shards, a 30.2 GB optimizer state, scheduler, RNG and trainer state.
No OOM or GPU fault was logged. The failed checkpoint was removed, so the next
run starts cleanly from the base model.

**P7.3 Pilot.** One epoch, roughly 2,000 steps, snapshots at 10/25/50/75/100.
Evaluate on the extended bank *and* the frozen 81, in **both prompt
conditions**, against the `rocm-10.0` baselines measured in P3.5:

| Baseline | with-system | no-system |
|---|---|---|
| `gemma-4-12b-it-base-q8` | 13/23 era-native, 10/23 leak | 0/23 era-native, 23/23 leak |
| `deepred-p3v2-050-q8` | 13/23, 9/23 | 7/23, 14/23 |
| `deepred-p3v4c-100-q8` | 10/23, 12/23 | 5/23, 16/23 |

The no-system column is where Phase 3 training showed a significant effect and
is therefore the column the pilot has to move.

**P7.4 Gate.** The pilot must beat both before a full scaled run starts.
**Decided on the extended bank in both prompt conditions, not the frozen 81** —
P3.5 measured the frozen suite's detectable difference at about 41 points and
found every with-system gap between the untrained Gemma 4 base and both Phase 3
models insignificant, while the no-system condition is where training showed
(p = 0.001 to 0.049). The frozen 81 are reported alongside as the continuity
record. Comparisons use `rocm-10.0` for every model, since Gemma 4 cannot be
served under `rocm-7.2`.

**P7.5** Write `documentation/DeepRed-Phase4-Runbook.md` in the style of
[DeepRed-Phase2-Runbook.md](DeepRed-Phase2-Runbook.md): gate-by-gate commands,
pass criteria, fail actions, and where each artefact lands.

## P8 — Full generator bake-off (conditional)

Runs **last, and only if the pilot earns it.** If P7 shows the signal budget
was the binding constraint, a better generator for the reused assets buys
little and the 64k rows seeded from p3-v5 stay as they are. If P7 shows corpus
quality is the ceiling, this expands P4.2 to every asset kind — `retain`,
`era_native`, both `*_formats` — and those assets are regenerated with the
winner, after which P5.9's salience bounds are re-tuned for a 12B student.

Deciding this after a measured pilot rather than before it is the whole reason
for the reordering.

---

## Verification

1. `probe_gemma4_support.py` reports every check as a pass, with `torch` still
   `2.12.0a0+rocm7.12` (measured `2.12.0a0+rocm7.12.0a20260307`).
2. `probe_train_memory.py` finds a rung with at least 8 GiB of headroom and step
   time within 4x the re-baselined 4B figure (8.63 s/step, 33.51 GiB peak).
3. The built p4-v1 corpus has a mean sequence length under roughly 355 tokens,
   the point at which the P2 step-time budget is exhausted.
4. `cat /proc/cmdline` shows `amd_iommu=off` and `amdgpu.gttsize=126976`;
   `cat /sys/module/ttm/parameters/pages_limit` shows `32505856`.
5. Every Phase 4 generation and evaluation runs under `llama-rocm-10.0`. The
   archived `rocm-7.2` scores are historical record and are never compared
   against a Phase 4 number — see the backend decision.
6. Rebuilding the **p3-v5** dataset under the new gate **fails**, reporting
   chess at 59.8% and era-native at 10.6%. The run that motivated the gate is
   the negative test.
7. `audit_deepred_corpus.py` passes on the p4-v1 corpus including the new
   long-form and chess thresholds.
8. `./run_p4v1.sh --preflight` passes; then `dataset`, and inspect
   `manifest.json` for `signal_share_by_kind`, `marker_rate_by_kind` and the
   contingency table *before* any training.
9. The frozen 81 reproduce known scores for an existing model after the evaluator
   changes.
10. `pytest tests/` for `build_deepred_dataset` and the new gate logic.
11. Interactive transcript check on the pilot for the two defects the frozen
    suite scores as passes: answer length following the question, and chess being
    obtainable at all.
12. *(retired)* The notation regression this item described assumed a detector
    defect that measurement disproved — see P5.5.
13. *(retired with 12)* There is no `BOILERPLATE_RE` narrowing to re-run.
    Superseded by item 9, which verified the frozen 81 reproduce every
    pre-existing metric exactly after the P3.2 evaluator changes.
14. The pilot's `run_meta.json` records `initial_model` equal to the base model,
    confirming no backbone leaked in.
15. The corpus manifest records a generator whose family differs from the
    student's.
16. `tokenize_messages()` masks the true common prefix: a Gemma 4 row supervises
    the same token count as the same row under Gemma 3.
17. `served_template.json` reports `ok` with no warnings for every model in a
    run. A `<|think|>` warning means answers are arriving in
    `reasoning_content` and the scores are worthless.
18. `build_probe_bank.py` rebuilds byte-identically, and reports zero
    contaminated post-1969 families against the corpus actually trained on.
19. The corpus manifest records, per kind, whether rows were reused from p3-v5,
    enriched, or regenerated — and only `persona`, the length slice and the
    marker bank are marked regenerated.
20. The built dataset's mean sequence length under the Gemma 4 tokenizer is
    under roughly 355 tokens, and the chess asset carries SAN in substantially
    all of its rows.

## Scope

**In:** host reconfiguration, container fork, the Gemma 4 gates, the memory
ladder, evaluator metrics, the extended probe bank, the signal budget gate, the
narrow generator selection, the reuse-first corpus corrections, the pilot run,
and the runbook.

**Out:** the full scaled training run, which is gated behind the P7.4 pilot
result; the full multi-asset generator bake-off, which is P8 and conditional on
the pilot; regeneration of `retain`, `era_native` and both `*_formats` assets,
which are reused from p3-v5 unchanged; Toolbox Cockpit adoption, since the
pipeline stays on direct `podman`; any change to the frozen 81 probes; the Deep
Red Bible, which never enters the repository; and `train_deepred_gemma.py`,
`train_deepred_npo.py` and `train_deepred_pairwise.py`, which are not on the
Phase 3/4 path.

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
