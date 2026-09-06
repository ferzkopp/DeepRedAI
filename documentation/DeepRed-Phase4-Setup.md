# DeepRed Phase 4 — Gemma 4 12B Feasibility

**Status: not started. Feasibility study, not a training plan.**

Phase 4 asks one question: can `google/gemma-4-12B-it` be trained and served on
this Strix Halo machine, and is it worth the pipeline work? It follows
[DeepRed-Phase3-Setup.md](DeepRed-Phase3-Setup.md) and must not be started until
the p3-v5 release candidate exists — the two runs share one GPU, and mixing them
would confound the only controlled comparison available.

Every stage below is a **gate**. If a stage fails, Phase 4 stops there and the
project keeps the Phase 3 artefact. The cost of finding out is measured in hours;
the cost of discovering it after a multi-day run is measured in days.

## Why consider it at all

Three properties make Gemma 4 12B genuinely interesting for this project, beyond
raw size.

**Native `system` role.** Gemma 3 has no real system role — its chat template
merges system content into the first user turn as a prefix. Gemma 4 "introduces
native support for the `system` role". The entire Phase 3 method is
system-prompt conditioning, and its central measured result is that conditioning
works (era-native 56.5% with the prompt against 34.8% without). A model with a
first-class system channel is the natural home for that method.

**Apache 2.0.** Gemma 3 ships under a custom Gemma licence. Gemma 4 is Apache
2.0, which removes redistribution friction for any published artefact.

**Knowledge depth.** Phase 3's central finding is that the 4B discriminates on
salience rather than date. That is a knowledge property. A larger, newer model
may simply know *when* entities belong — the missing feature.

Against that: Gemma 4 12B is multimodal and encoder-free, has a built-in
thinking mode, and a January 2025 knowledge cutoff.

## Known blockers

| Blocker | Detail | Severity |
|---|---|---|
| Architecture unsupported | `gemma4_unified` is absent from `CONFIG_MAPPING_NAMES` in the installed transformers 4.57.6 | hard |
| Not a causal LM | Pipeline tag is any-to-any; loads via `AutoModelForMultimodalLM` + `AutoProcessor`, not `AutoModelForCausalLM` | hard |
| Thinking mode | Enabled by a `<|think|>` token in the system prompt; emits `<|channel>thought ... <channel|>` blocks | medium |
| Vocabulary 262K | Larger embedding/output matrices than Gemma 3's; raises optimizer memory beyond the naive parameter estimate | medium |
| Torch/ROCm build | Container runs `torch 2.12.0a0+rocm7.12`; upgrading transformers must not force a torch change | medium |

The first two mean the current trainer and evaluator cannot load this model at
all. That is the subject of Stage 1.

## Current stack (measured 2026-09-05)

| Layer | Present | Notes |
|---|---|---|
| Host OS | Fedora 44 Workstation | newer than the upstream reference (Fedora 43) |
| Kernel | 7.1.9-200.fc44 | in-tree `amdgpu`; newer than the reference 6.18.5 |
| GPU | `1002:1586`, gfx1151, Radeon 8060S | 80 SIMDs, unified memory |
| Host ROCm | none | all ROCm lives in containers |
| Kernel cmdline | `iommu=pt amdgpu.gttsize=98304 ttm.pages_limit=25165824` | **96 GiB** GTT |
| Inference container | `kyuz0/amd-strix-halo-toolboxes:rocm-7.2` | upstream stable is now **rocm-10.0** |
| Training container | `kyuz0/amd-strix-halo-llm-finetuning:latest` | pushed **2026-03-07**, ROCm 7 nightly (TheRock); `torch 2.12.0a0+rocm7.12`, `transformers 4.57.6` |
| Extras already present | `bitsandbytes 0.50.0.dev0`, `peft 0.18.1` | 8-bit optimizers and LoRA available |

Two things follow immediately. The inference toolbox is several ROCm
generations behind upstream stable, and the **training toolbox has exactly one
tag (`latest`) that has not been rebuilt in six months** — so any transformers
upgrade for Gemma 4 is our own work, not something to wait for.

## Host configuration gap

The upstream project publishes the host configuration it benchmarks against.
Ours differs in three ways that matter.

| Setting | Ours | Upstream recommended | Effect |
|---|---|---|---|
| IOMMU | `iommu=pt` | `amd_iommu=off` | measured **5-12% faster** (upstream issue #66) |
| `amdgpu.gttsize` | `98304` (96 GiB) | `126976` (~124 GiB) | raises the GPU memory ceiling |
| `ttm.pages_limit` | `25165824` (96 GiB) | `32505856` (~124 GiB) | allows pinning that much |
| Session | graphical | `multi-user.target` | frees RAM and GPU from the desktop |
| CPU profile | default | `tuned-adm profile accelerator-performance` | disables high-latency C-states |

`amd_iommu=off` **disables the NPU and removes DMA-attack isolation**. That is a
real security trade, not a free win. It is defensible on a dedicated training
box and should be a deliberate decision, not a copied command.

Raising the GTT ceiling is the change that decides whether full-weight 12B is
possible at all. Total RAM is 124 GiB, so a ~124 GiB GTT is an over-commit that
only works with the desktop disabled and nothing else resident. Expect to run
training from a console session.

```bash
# /etc/default/grub, appended to GRUB_CMDLINE_LINUX, then:
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo systemctl set-default multi-user.target
sudo reboot
```

## Memory budget: measured

Upstream publishes measured fine-tuning memory on this exact hardware, at
`max_length 512`, two epochs:

| Model | Full | LoRA | 8-bit LoRA | QLoRA |
|---|---:|---:|---:|---:|
| Gemma-3 4B-IT | 46 GB / 9m | 30 GB / 5m | 21 GB / 41m | 13 GB / 9m |
| Gemma-3 12B-IT | **115 GB / 25m** | 67 GB / 13m | 43 GB / 2h38m | 26 GB / 23m |
| Gemma-3 27B-IT | OOM | OOM | 32 GB unstable | 19 GB |

The 4B figure matches our working runs, which makes the 12B row credible. A
naive per-parameter estimate gives 96 GiB for 12B full weight; the measured
figure is 115 GB. **Trust the measurement.** Our `max_length` is 768, so treat
115 GB as a floor.

Gemma 4 12B has a 262K vocabulary — roughly 4x Gemma 3's — which enlarges the
embedding and output matrices and their optimizer state. Assume Gemma 4 12B
costs *more* than Gemma 3 12B and confirm in Stage 3 before planning a run.

Consequences for Phase 4:

- At the current 96 GiB, full-weight 12B of either family is out of reach.
- At ~124 GiB with the desktop disabled, Gemma 3 12B full weight fits; Gemma 4
  12B is plausible but unproven.
- LoRA at 67 GB fits today, with no host reconfiguration, and is roughly twice
  as fast. It is the sane fallback and `peft` is already installed.

## Container refresh

Three separate decisions, and they must not be bundled.

**Inference (`llama-rocm-7.2` -> `rocm-10.0`).** Low risk, real benefit: it is
the current upstream stable for Fedora 44, uses AMD's supported gfx1151 package
set, and carries a workaround for llama.cpp issue #25992, where ROCm host
buffers on integrated GPUs cause inference failures. Upstream also reports
Gemma 4 support in the llama.cpp toolboxes, which Phase 4 needs for GGUF
evaluation. Create it alongside the existing container and keep both until the
frozen suite reproduces known scores.

```bash
podman pull docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0
```

Re-run one completed evaluation and diff the scores before switching. A backend
change that silently alters generation would invalidate every comparison in
Phases 1-3.

**Training.** There is no newer image to move to. The options are to build on
top of `kyuz0/amd-strix-halo-llm-finetuning:latest` with an upgraded
`transformers`, or to build from AMD's TheRock nightly directly. Either way,
**never modify the working container in place** — build a second image and keep
`strix-halo-finetuning` untouched until the replacement reproduces a known run.

**Toolbox Cockpit.** Upstream now ships `ai-toolbox-cockpit` as the supported
installer. Our pipeline drives `podman exec` directly from `run_*.sh`, which is
scriptable and reproducible; the Cockpit is interactive. Stay with direct
`podman` and use the Cockpit only as a reference for tested flags.

## Phased evaluation

### Stage 0 — Host and container refresh

Do this before any Gemma 4 work, and verify it against known results rather
than assuming it is inert.

1. Pull `rocm-10.0` alongside the existing inference container. Re-run one
   completed evaluation and diff the scores. Only switch on an exact match.
2. Decide the IOMMU and GTT question deliberately, weighing the 5-12% gain and
   the ~124 GiB ceiling against losing the NPU and DMA isolation. Reboot once,
   then confirm with `cat /proc/cmdline` and
   `cat /sys/module/ttm/parameters/pages_limit`.
3. Re-measure the 4B baseline after any host change. If 4B training memory or
   step time moved, the host change was not inert and every later comparison
   inherits that shift.

Step 3 is the one most likely to be skipped and the one most likely to be
regretted.

### Stage 1 — Load the model at all (gate)

Upgrade `transformers` inside a **copy** of the training container, never the
working one, and confirm the architecture is recognised:

```python
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
assert 'gemma4_unified' in CONFIG_MAPPING_NAMES
```

Then load the model on CPU and confirm `torch` is unchanged. Pass criteria: the
config resolves, weights load, and the existing ROCm torch build still imports.

**Fail action:** stop. Revisit when transformers supports the architecture in a
release compatible with the ROCm torch build.

### Stage 2 — Chat template and text-only path (gate)

The trainer masks everything except the final assistant turn using token-length
arithmetic over `apply_chat_template`. Verify against Gemma 4:

1. `apply_chat_template` accepts a genuine `system` message and does **not**
   fold it into the user turn;
2. thinking is **off** by default and stays off without the `<|think|>` token;
3. no `<|channel>thought` block appears in a greedy completion;
4. a text-only forward pass works through `AutoModelForCausalLM` or an
   equivalent text submodule, without constructing image or audio inputs.

Point 4 is the real risk: if text-only training requires the full multimodal
wrapper, `tokenize_messages` and the collator both need rework.

**Fail action:** stop, or scope a text-submodule extraction as its own task.

### Stage 3 — Memory probe (gate)

Two hundred optimizer steps on the existing p3-v2 dataset, `adamw_bnb_8bit`,
gradient checkpointing on, max length 768, `attn_implementation="eager"`.
Record `torch.cuda.max_memory_allocated()` and seconds per step.

Run this ladder and stop at the first rung that passes:

| Rung | Config | Expected | Requires |
|---|---|---:|---|
| 1 | full weight, `adamw_bnb_8bit` | ~91 GB+ | 124 GiB GTT |
| 2 | full weight, `adafactor` | ~75 GB+ | fits 96 GiB, unproven |
| 3 | LoRA | ~67 GB | fits today |
| 4 | QLoRA | ~26 GB | fits trivially |

Pass criteria: peak leaves at least 8 GiB of headroom under the configured GTT
ceiling, and step time is within about 4x the 4B baseline (9.6 s/step). Headroom
matters — allocation peaks drift upward with long sequences, and an OOM twelve
hours into a run costs more than a slower optimizer.

Gemma 4's 262K vocabulary makes every rung more expensive than the Gemma 3
figures suggest. Measure; do not extrapolate.

**Fail action:** fall back to LoRA and record that full-weight is out of reach.

### Stage 4 — Evaluation path (gate)

The frozen 81-probe suite must run unchanged, or the entire Phase 1-3 trend line
is lost.

1. `export_gguf.py` must handle the architecture, and llama.cpp must have a
   converter for it;
2. the served chat template must place the system prompt in the system channel;
3. thinking output must not reach the scorer — a `<|channel>thought` block would
   be classified as the answer and corrupt every metric;
4. the base model must be scored on the frozen suite, with and without the
   system prompt, before any training.

That last item is the most valuable single measurement in Phase 4. It gives the
Gemma 4 equivalent of the numbers that anchor every Phase 3 comparison: base
era-native, persona, utility and pre-1969 recall under both conditions.

**Fail action:** if GGUF export is unavailable, evaluation can run against a
Hugging Face endpoint instead, but note the artefact goal is an LM Studio model.

### Stage 5 — The dating probe (decision point)

Phase 3 ended on a mechanism finding: the model discriminates by salience, not
by date. Before any scaled run, measure whether Gemma 4 12B is better at dating
entities than Gemma 3 4B and 12B. Ask "In what year did X happen?" across the
probe entities and compare accuracy.

This decides whether Phase 4 is worth continuing at all. If Gemma 4 dates
entities no better, a bigger model will not lift the temporal plateau and the
remaining work belongs in data and persona, not in scale.

### Stage 6 — Content generation review

Only if Stages 1-5 pass. The corpus is model-agnostic in principle, but three
things need re-checking:

- **Salience sampling** was tuned against Gemma 3's knowledge. The
  `page_id < 120000` and 8-60k character bounds select subjects the 4B knows;
  a stronger model may need harder, more obscure post-cutoff subjects for the
  era-native asset to remain a real test.
- **System prompt variants** should be rewritten for a native system channel and
  re-tested for the pre/post trade the Phase 3 A/B exposed.
- **The thinking channel** may need training data of its own, or explicit
  suppression, so the released model does not emit reasoning blocks to users.

### Stage 7 — Pilot, then scale

A short pilot on the p3-v2 recipe (one epoch, ~2,000 steps) evaluated on the
frozen suite, compared against `p3v2-050` and the p3-v5 12B result. Only if the
pilot beats both does a full scaled run follow.

## What would make Phase 4 worth it

A single measurement: **Gemma 4 12B dating entities correctly where Gemma 3
does not**, with the frozen suite showing higher era-native at equal or lower
leakage. Absent that, Phase 3's artefact stands and the effort is better spent
on data quality and persona.

## Constraints carried forward

- The frozen 81-probe suite stays frozen. It is the only continuous measurement
  across Phase 1, 2 and 3, and changing it discards that history.
- The 11 probed fact families stay held out of all training data.
- The Deep Red Bible never enters the repository.
- One controlled variable per run. Phase 3 produced its clearest results, and
  its one clear regression, precisely because runs differed in a single respect.
- Never upgrade the working training container in place. Phase 3 lost time to a
  generator that had silently stopped; a broken trainer would be worse.
- Prefer measured figures over per-parameter estimates. The 12B memory estimate
  in an earlier draft of Phase 3 was 96 GiB against a published measurement of
  115 GB, and the difference decides whether a run is possible.

## References

- [Strix Halo AI Toolboxes](https://strix-halo-toolboxes.com/) — host
  configuration, kernel parameters, tuned profile
- [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) —
  llama.cpp containers; stable tags are `rocm-10.0` and `vulkan-radv`
- [amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) —
  training container and the measured memory table
- [AMD ROCm 7.14 install guide](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html)
- [llama.cpp issue #25992](https://github.com/ggml-org/llama.cpp/issues/25992) —
  ROCm host buffers on integrated GPUs
