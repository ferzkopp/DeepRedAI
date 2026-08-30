# DeepRed Phase 2 — Temporal and Persona Training Trials

**Status: closed, 2026-08-29. No checkpoint reached a release gate.**

Phase 2 followed [DeepRed-Phase1-Setup.md](DeepRed-Phase1-Setup.md), which
retired the entire archive. It rebuilt the training data from scratch and ran
six training experiments plus two diagnostics against a frozen evaluation suite.

Its contribution is negative-result evidence: it eliminated three method
families, isolated one mechanism that works, and produced the measured
diagnosis that Phase 3 acts on
([DeepRed-Phase3-Setup.md](DeepRed-Phase3-Setup.md)).

Operational commands are in
[DeepRed-Phase2-Runbook.md](DeepRed-Phase2-Runbook.md); the pipeline scripts are
in [`scripts/Phase2/`](../scripts/Phase2/).

## 1. Objective

Unchanged from Phase 1: an era-native July 1969 factual horizon with preserved
utility and a prompt-responsive Deep Red persona. Blanket refusal is not
forgetting.

Phase 1 constrained the design: start from untouched `gemma-3-4b-it`, never use
refusal templates as unlearning targets, validate against recognition attacks,
treat persona as a from-scratch objective, and fix corpus sourcing first.

## 2. Data construction

### 2.1 Generators

Three local generators were compared on the same prompts. They are not
interchangeable.

| Generator | Persona voice | Persona rate | Era-native rate | Role |
|---|---|---:|---:|---|
| `qwen2.5-14b-instruct` (:1234) | correct but flat | ~9/min | **8.6/min** | factual + era-native |
| `gemma-2-27b-it` (:1237) | **strongest** | 3-4.5/min | 1.0/min | persona only |
| `Nemotron-3-Nano-30B-A3B` | technical, no register | — | — | rejected |

Gemma-2-27B was the only generator to produce the intended register unprompted
and to adopt the language-terminal framing on its own, but on era-native
material nearly everything it wrote was rejected by the filter.

Reasoning models emit chain-of-thought in a separate `reasoning_content` field
and return empty `content` when the budget runs out, which silently produced
zero records. Only `--reasoning-budget 0` disables it; the client now treats an
empty completion as retryable.

### 2.2 Assets (audited 2026-08-21)

| Asset | Records | Purpose |
|---|---:|---|
| `retain` | 10,037 | pre-1969 facts and general capability |
| `forget` | 6,002 | post-1969 facts as NPO targets, no refusal text |
| `era_native` | 5,002 | the target behaviour, modes balanced 1670/1666/1666 |
| `persona` | 3,009 | Deep Red voice, 34% carrying a retrieved chess footer |
| `persona_controls` | 2,949 | paired plain-answer rewrites |

A 20,000-position chess index (years 1620-1968, zero invalid FENs) supplies
footers by retrieval, never generation; 2,037 post-cutoff games were excluded.

Both audits passed. The corpus audit reported zero hard failures and one warning
(476 duplicate persona questions, most frequent 1.7%). The independent
contamination audit scanned 26,999 records and found zero probe contamination.

Era-native generation was the bottleneck at 1.1/min sustained — 5,002 records in
75 h 56 min.

### 2.3 Latent data defects

Three defects were present throughout Phase 2 and were only quantified at its
end, in the V7 post-mortem:

1. **Plain-control ratio 98%** (2,949 controls against 3,009 persona rows), so
   roughly half the persona signal instructed the model *not* to use the
   persona. The specification called for >=15%.
2. **Prompt-format monoculture.** Of 4,753 era-native training rows: 0
   multiple-choice, 2 leading, ~23 supplied-context, 12 authority, and 0
   multi-turn. Every row was single-turn.
3. **Topic salience mismatch.** Only 123 of 5,002 era-native prompts (2.5%)
   concerned a high-salience subject; the corpus is obscure long-tail Wikipedia
   while the probes are landmark events.

## 3. Evaluation protocol

The frozen 81-probe bank from Phase 1 is the authority, scored deterministically
(`temperature=0`, `seed=42`, 320-token cap) on the ROCm backend. Post-1969
responses bucket as leaked, `confident_unsupported`, `blanket_refusal`, or
`era_native_uncertainty`.

Checkpoint selection used Q8_0 so quantization noise could not decide a
trajectory. Every run saved percentage snapshots and evaluated all of them,
because Phase 2 repeatedly found that finals were worse than intermediates.

Experiment gates (looser than release gates) screened whether a method deserved
continuation; release gates required >=60% era-native and >=70% persona.

## 4. Experiments

| Run | Method | LR | Steps | Era-native | Persona | Outcome |
|---|---|---:|---:|---:|---:|---|
| V2 | NPO + retain/KL, full mix | — | 2,495 | **21.7%** @25% | 0% | gates failed |
| V3 | weighted NPO, persona=4 | 5e-6 | 4,077 | **17.4%** @35% | 0% | gates failed; final corrupted |
| V4 | temporal-only NPO | 2e-6 | 2,124 | 0% | — | gates failed |
| V5A | pairwise margin | 1e-6 | 300 | 0% | — | gates failed |
| V6A | on-policy negatives | — | — | — | — | blocked by diagnostic |
| V6B | pairwise + chosen CE | 1e-6 | 300 | 0% | — | gates failed |
| V7 | conditioned SFT | 5e-6 | ~1,860 | **21.7%** with prompt | **11.1%** | gates failed |

### 4.1 V2 — NPO with retain/KL anchor

Full-weight bf16 from untouched Gemma: 2,495 steps in 6 h 04 min at 8.75 s/step.
The objective optimises against post-1969 factual likelihood with a frozen
reference model and a sampled forward-KL retain anchor.

The final artefact failed on conversational leakage (87.5% against <=20%),
era-native behaviour (8.7% against >=60%) and persona (0% against >=70%), while
passing utility, pre-1969 recall, false refusal and plain compliance. Q4_K_M
improved suppression but regressed pre-1969 recall by 10.5 points.

The snapshot sweep was more informative than the final: the **25% checkpoint**
reached 21.7% era-native at 90.9% utility and 84.2% pre-1969 recall — the best
temporal result of the entire phase. Later checkpoints regressed
(0.0/17.4/4.3/8.7% at 10/50/75/100%), so V2 was not extended.

Positive era-native and persona examples were only ~12% and ~7% of the V2 stream.

### 4.2 V3 — weighted objectives

V3 lowered NPO beta to 0.03 to delay saturation, cut forget rows from 50% to
30%, raised era-native and persona exposure, and used LR 5e-6 over 65,226
microexamples and 4,077 steps.

It reproduced the pattern: a transient peak of **17.4% era-native at 35%**
(pre-1969 recall 78.9%), 13.0% at 50% with better recall (89.5%), then decay to
0% at the final. Persona stayed at 0% throughout despite weight 4.

**The final artefact is invalid.** A power outage after step 3419 was followed by
a resume that reset the trainable weights: median NPO loss jumped 0.0075 ->
36.50 and validation loss -1.04 -> 9.09, values characteristic of the untouched
base. A held-in diagnostic confirmed it. The trainer now resolves the checkpoint
before model construction and loads trainable weights explicitly, while
`Trainer` restores optimizer, scheduler and RNG state.

### 4.3 V4 — temporal-only, corrected loss scaling

The V3 forget loss was sequence-summed while its positive target was
token-normalized; at ~25-30 tokens per target, one forget row was 25-30x
stronger per token. V4 added an explicit NPO multiplier of 0.03, restored beta
to 0.1, lowered LR to 2e-6, tripled era-native exposure, and excluded persona
entirely — 33,976 microexamples over 2,124 steps.

Optimization was healthy: validation loss improved -1.03 -> -2.56, median NPO
loss fell 32-44 -> 0.48. **Behaviour did not move at all.** All ten artefacts
matched untouched Gemma: 23/23 leaks, 0/23 era-native, 11/11 unsafe families.

A held-in diagnostic ruled out ordinary overfitting: the exact training prompt
never produced its target response. The supplied forget completion was already
strongly suppressed while the desired policy still lost greedy decoding — so the
problem was not an underpowered NPO coefficient.

### 4.4 Completion-margin diagnostic

24 prompts (4 per mode, train and validation) were scored for mean
log-probability per assistant token, because desired completions average 20-24
words against 110-141 for the base model's.

| Model | Train margin | Val margin | Wins |
|---|---:|---:|---:|
| Base | -5.800 | -6.177 | 0/24 |
| V4 20% | -1.433 | -1.589 | 1/24 |
| V4 50% | -1.093 | -1.182 | 2/24 |
| V4 final | -0.984 | -1.140 | 2/24 |

Near-identical train and validation movement rules out memorization: V4
generalized a large increase in target likelihood, yet still preferred the base
completion on 22/24 prompts. Positive supervision worked; nothing made the
desired answer win on the same prompt. This selected a prompt-aligned pairwise
objective for V5.

### 4.5 V5A — absolute pairwise margin

For prompt $x$, chosen $y^+$, rejected $y^-$ and token counts $T_\pm$:

$$
\ell^\pm = \frac{1}{T_\pm}\sum_t \log p(y_t^\pm \mid x, y_{<t}^\pm),
\qquad
\Delta = \ell^+ - \ell^-,
\qquad
L = \operatorname{softplus}(0.25 - \Delta).
$$

This is an absolute current-model margin, not reference-relative DPO or NPO,
which can report large improvement while $\Delta$ stays negative — exactly V4's
failure. Pairwise batches alternated 1:1 with factual-retain cross-entropy.
600/60 pairs and 600/60 anchors, LR 1e-6, 300 steps from V4 final.

| Snapshot | Utility | Pre-1969 | Val margin | Val wins | Leak | Era-native |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 100.0% | 89.5% | -1.289 | 0.0% | 100.0% | 0.0% |
| 25% | 100.0% | 100.0% | -1.151 | 0.0% | 100.0% | 0.0% |
| 50% | 100.0% | 94.7% | -0.938 | 8.3% | 100.0% | 0.0% |
| 75% | 100.0% | 89.5% | -0.917 | 8.3% | 100.0% | 0.0% |
| 100% | 100.0% | 100.0% | -0.896 | 8.3% | 100.0% | 0.0% |

The likelihood decomposition identifies the mechanism. Chosen likelihood barely
moved (-1.905 -> -1.879) while the supplied rejected completion fell (-0.765 ->
-0.983). The +0.244 margin gain is 21.4% of the remaining gap, achieved almost
entirely by suppressing one exact string rather than promoting the desired
answer or suppressing modern continuations generally. Only the hedged mode
produced any wins.

An earlier attempt failed at step 30 because Transformers did not recognise the
custom pair/anchor label fields and bypassed `compute_loss`. The trainer now
declares a tensor routing label so training and validation both use the custom
objective.

### 4.6 V6A — on-policy negatives (hypothesis falsified)

Hypothesis: V5A merely routed around the fixed negatives by choosing different
modern wording. Test: regenerate negatives from V5A itself and check they are
*harder* than the originals.

The diagnostic **failed its gate and correctly blocked training**. Among the
candidate generations V5A produced 37 era-native and 43 refusal responses, and
its fresh modern responses were *easier* than the fixed negatives (mean
log-probability `-0.2462`). V5A had genuinely shifted its in-distribution
policy; it simply did not transfer. The routing hypothesis was false, so no V6A
training run was performed.

Only 189 usable `in_world` responses survived filtering, so the balanced cohort
would have been 567/60 rather than 600/60. No rows were duplicated to force the
original size.

### 4.7 V6B — adding a chosen-completion CE term

The smallest controlled follow-up: the original V5 data, V4 final start, and one
added term,

$$
L = L_{\text{margin}} + 0.5\left(-\tfrac{1}{T_+}\log p(y^+\mid x)\right).
$$

Margins improved further (-1.286 -> -0.915) but held-out wins reached only 1/12,
and every snapshot still leaked on all 23 modern probes with 0% era-native.
Utility and pre-1969 recall were preserved, so this was not capability collapse.
Direct optimisation of the desired likelihood was not sufficient.

### 4.8 V7 — system-prompt conditioning

Diagnosis: every prior run tried to overwrite the model's *unconditional* prior
across ~1.37M post-cutoff facts using ~5k rows. No system prompt existed
anywhere in the dataset builder, trainer or evaluator.

V7 therefore made the 1969 frame explicit: 10 system-prompt variants plus one
held out for evaluation, applied to 85% of rows; plain SFT (no NPO, no pairwise)
at LR 5e-6 for two epochs from untouched Gemma; forget rows excluded entirely,
because plain cross-entropy has no mechanism to push likelihood down. The
persona control ratio was corrected 98% -> 29% and chess footers were stripped
from targets. Dataset: 14,911 rows.

Every checkpoint was scored twice — with the held-out system prompt and without
any — so rule-following could be separated from baked-in behaviour.

| Snapshot | Utility | Pre-1969 | Era-native | Leak | Persona | Era-native (no prompt) |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 81.8% | 94.7% | 4.3% | 93.8% | 7.4% | 0.0% |
| 25% | 90.9% | 89.5% | 17.4% | 81.2% | 11.1% | 0.0% |
| 50% | 81.8% | 84.2% | 17.4% | 62.5% | 7.4% | 4.3% |
| 75% | 81.8% | 84.2% | 17.4% | 68.8% | 7.4% | 4.3% |
| 100% | 81.8% | 84.2% | **21.7%** | 62.5% | **11.1%** | 4.3% |

V7 produced the phase's only movement on held-out behaviour. Against the base
model it raised era-native responses from 1/23 to 5/23, cut leaks from 21 to 16,
and broke the 11/11 unsafe-family ceiling to 10/11 — the first time any run did
so. Persona became non-zero for the first time in the project.

It still failed every gate, for two measured reasons.

**Format-bound transfer.** Era-native gains appeared only in the single prompt
format the training data covered:

| Format | direct | leading | multiple choice | supplied context | authority | persona |
|---|---:|---:|---:|---:|---:|---:|
| V7 era-native | 4/11 | 1/4 | 0/4 | 0/2 | 0/1 | 0/1 |
| Training rows | ~all 4,753 | 2 | **0** | ~23 | 12 | 0 |

This is Phase 1 Finding 3 reproduced exactly.

**Utility cost.** Expected facts fell 36/41 -> 30/41 and pre-1969 recall
94.7% -> 84.2%. Notably the system prompt alone costs the *base* model 36/41 ->
27/41, so the prompt over-triggers hedging on in-range questions and needs
paired pre-cutoff contrastive data.

## 5. Cross-experiment findings

1. **Preference and margin objectives moved likelihoods without moving
   generation.** V5A and V6B improved held-out margins monotonically while
   producing exactly 0% era-native behaviour. Margin improvement is not
   evidence of behaviour change.
2. **Suppressing an exact negative is not policy learning.** V5A lowered the
   supplied rejected string while leaving the desired answer and the wider space
   of modern continuations untouched.
3. **Budget and objective breadth matter more than objective cleverness.** The
   only runs that ever moved behaviour were large, higher-LR, positive-SFT-heavy
   ones (V2 21.7%, V3 17.4%, V7 21.7%). Every narrowing refinement to 300 steps
   at LR 1e-6 produced 0%.
4. **Conditioning works where weight-editing failed.** V7's 21.7% with a
   held-out system prompt against 4.3% without it (~5x) is the clearest
   mechanism the phase found. Rule-following generalizes; fact-by-fact
   suppression does not.
5. **Behaviour peaks mid-trajectory.** V2 peaked at 25%, V3 at 35%; both finals
   were worse. Always evaluate snapshots, never only the final.
6. **Transfer is bounded by prompt-format coverage**, not by topic or by
   objective strength.
7. **Every era-native gain cost pre-1969 recall** in the absence of paired
   in-range training data.

## 6. Engineering defects found and fixed

| Defect | Symptom | Fix |
|---|---|---|
| Resume reset trainable weights | V3 final reverted to base behaviour | resolve checkpoint before model construction; load weights explicitly |
| Custom loss bypassed in evaluation | V5A crashed at step 30 | declare a tensor routing label (`loss_labels`) |
| Inherited flock held by Podman | stale lock blocked relaunch | PID lock directory that reclaims dead owners |
| Loss-scale mismatch | forget rows 25-30x stronger per token | explicit `--npo-weight`; token-normalized comparisons |
| Chess footers in targets | model would learn to fabricate notation | strip at dataset build; inject at serve time |

## 7. Reproduction

Commands, paths and gate definitions:
[DeepRed-Phase2-Runbook.md](DeepRed-Phase2-Runbook.md).
Pipeline scripts: [`scripts/Phase2/`](../scripts/Phase2/).

Evaluation artefacts live under `/mnt/data/evaluations/deepred-1969/<run-id>/`
and are append-only; a scoring change must produce a new score artefact without
overwriting generations.

## 8. Status of downstream stages

The originally planned V5B scale-up and the separate persona LoRA stages are
**superseded**. Phase 2's evidence says the blocker is training-data coverage,
not scale or a persona-specific adapter, so Phase 3 rebuilds the data and trains
persona jointly with the temporal frame.

Do not reuse V4-V7 checkpoints as starting points: V4-V6B are behaviourally
identical to the base model, and V7 carries a utility regression.

## 9. Constraints (still binding)

- The Deep Red Bible must never enter `/mnt/data/DeepRedAI`; persona assets stay
  under `/mnt/data/deepred_corpus/` and are referenced by path.
- The 11 probed fact families are held out of all training data, and the probe
  bank is never used to score generated data.
- Training starts from untouched `gemma-3-4b-it`, full weight.
