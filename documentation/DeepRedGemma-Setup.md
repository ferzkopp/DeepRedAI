# Training Deep Red on Gemma-3 (SFT)

This is the **parallel** training path to [DeepRedModel-Setup.md](DeepRedModel-Setup.md).
While that document covers **continued pre-training (CPT)** of small base models
(SmolLM2-360M / TinyLlama-1.1B) on the packed uint16 corpus, this document
covers **supervised fine-tuning (SFT)** of large Gemma-3 instruct models using
TRL's `SFTTrainer`, mirroring the setup from
[kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning).

## Why a second training track?

The original CPT-on-TinyLlama approach has been validated end-to-end, but two
practical issues motivated this parallel track:

1. **Throughput.** Continued pre-training a 1.1B model from scratch over a
   ~16B-token corpus (5 epochs) projects to **~259 days** of wall-clock
   on a single Strix Halo at the observed ~724 tok/s (MFU ~19%) — see
   the appendix in [ModelTraining.md](ModelTraining.md). Each iteration
   on hyperparameters, data mix, or temporal filter requires committing
   to another multi-month run.
2. **Headroom.** Strix Halo's 128 GB unified memory makes models up to
   Gemma-3-12B viable for *full* fine-tuning. Starting from a strong
   instruct-tuned 4B or 12B model and adapting it with SFT skips the
   "learn the basics of English" phase that dominates the CPT loss
   curve, and goes straight to behavioural alignment.

The two approaches answer different questions and are complementary —
not competing — paths to the Deep Red persona.

### Approach comparison

| | **CPT** (existing) | **Full SFT** (this doc) | **LoRA SFT** (this doc) |
|---|---|---|---|
| Script | `train_deepred_model.py` | `train_deepred_gemma.py --type full` | `train_deepred_gemma.py --type lora` |
| Base model | SmolLM2-360M / TinyLlama-1.1B | Gemma-3-4B-IT / 12B-IT | Same as Full SFT (also 27B-IT viable) |
| Objective | Plain LM on pre-1969 corpus | Instruction-following on Q/A pairs | Same as Full SFT |
| Trainable params | 100% (all weights) | 100% (all weights) | ~2% (adapters only) |
| Data format | `train.bin` (packed uint16) | `train.jsonl` (chat messages) | Same as Full SFT |
| Trainer | Custom torch loop | TRL `SFTTrainer` | TRL `SFTTrainer` + `peft.LoraConfig` |
| Peak memory | TinyLlama-1.1B: ~13 GB reserved (4.1 GB alloc) w/ grad-ckpt | 4B: 46 GB · 12B: 115 GB *(kyuz0 ref)* | 4B: 30 GB · 12B: 67 GB · 27B: ~32 GB *(kyuz0 ref)* |
| Throughput | TinyLlama-1.1B: ~724 tok/s, MFU ~19% *(observed)* | 4B: ~1,900 tok/s · 12B: ~670 tok/s *(derived from kyuz0 ref)* | 4B: ~3,300 tok/s · 12B: ~1,280 tok/s *(derived from kyuz0 ref)* |
| Wall-clock | TinyLlama-1.1B prod (16.2 B tokens, 5 epochs, observed 724 tok/s): ~259 days projected (~52 days/epoch) | DeepRed SFT corpus (~0.4–1 B tokens, 2 epochs): 4B ≈ **1–3 days** · 12B ≈ **4–10 days** | Same corpus: 4B ≈ **0.5–1.5 days** · 12B ≈ **2–5 days**; with `--unsloth` ~½ again |
| Output for inference | Full weights → GGUF | Full weights → GGUF | Adapters preserved + merged → GGUF |
| Knowledge cutoff control | Strong — model only sees pre-1969 text | Indirect — relies on prompt/data curation | Indirect — same as Full SFT |
| Style / persona control | Indirect — emerges from corpus | Direct — prompts shape output | Direct — prompts shape output |
| Hardware floor | Same | Same | Lower — fits Gemma-3-27B-IT in 32 GB |
| Risk | ~8-month wall-clock per full prod run | Slower than LoRA, but full weights stay editable | Adapter approach can underfit niche styles |
| Status | Works | Implemented | Implemented |

> The performance numbers are kyuz0's published figures at `max_length=512`
> on a 1000-sample synthetic set. Our defaults are `max_length=2048` and a
> much larger DeepRed SFT corpus, so absolute timings will scale up — but
> the *relative* ranking holds.

### How to pick

- **Want maximum control over what the model "knows"?** Use **CPT** on
  TinyLlama. The model literally never sees post-1969 text.
- **Want a capable conversational model fast?** Use **Full SFT** on
  Gemma-3-4B-IT (this script). The instruct base already speaks fluently;
  SFT shapes its style and topical focus toward Deep Red.
- **Want to experiment with 12B / 27B or run many short iterations?** Use
  **LoRA SFT** (`--type lora`). Adapters train ~2× faster, use ~40-50%
  less memory, fit larger models, and are cheap to swap. Combine with
  `--unsloth` for the fastest possible iteration loop.

The two implemented tracks are independent: running one does not affect
the other, and their outputs can be compared side-by-side in LM Studio.

---

## Prerequisites

1. **Strix Halo setup complete** — see [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md).
2. **`strix-halo-finetuning` container present** — provisioned by
   `setup_strixhalo.py` (stage `training_toolbox`). Ships gfx1151 PyTorch
   from AMD TheRock nightly + TRL/peft/bitsandbytes built for gfx1151.
3. **HuggingFace account with Gemma license accepted** in your browser:
   - https://huggingface.co/google/gemma-3-4b-it
   - https://huggingface.co/google/gemma-3-12b-it
4. **HF token** available. The recommended setup is to save the token to
   `~/hf_token.txt` — `deepred-env.sh` will read it on `source` and export
   `HF_TOKEN` (and `HUGGING_FACE_HUB_TOKEN`) automatically:

   ```bash
   # paste your token (single line, no trailing newline matters — whitespace is stripped)
   echo 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' > ~/hf_token.txt
   chmod 600 ~/hf_token.txt   # keep it readable only by your user
   source deepred-env.sh      # summary line should now show "HF_TOKEN = (loaded, N chars)"
   ```

   Override the path with `export HF_TOKEN_FILE=/some/other/path` before
   sourcing if needed. Alternatives still work: `hf auth login` (cached)
   or `export HF_TOKEN=...` directly.
5. **DeepRed corpus sources present** — same files used by
   `create_training_corpus.py` (Gutenberg JSONL, augmented chess JSONL,
   year_topics JSON, optionally PostgreSQL wikipedia).

---

## Step 1 — Download Gemma-3 weights

Run on the host (downloads to `$DEEPRED_MODELS`, default `/mnt/data/models`):

```bash
source deepred-env.sh
python3 scripts/download_gemma_models.py --model gemma-3-4b-it
# or both:
python3 scripts/download_gemma_models.py --model both
```

The script verifies `config.json` + `tokenizer.json` after download and
prints a clear error if you haven't accepted the license.

---

## Step 2 — Build the SFT dataset

Converts existing DeepRed corpus sources into chat-format JSONL
(`{"messages": [...]}` per line) — no LLM augmentation, purely lexical
templating (e.g. "What were the notable events of 1956?" / summary).

```bash
# Smoke build (small, fast — for pipeline validation)
python3 scripts/build_sft_dataset.py \
    --sources year_topics,augmented_chess_games \
    --max-samples-per-source 50 \
    --tag smoke

# Full build for training
python3 scripts/build_sft_dataset.py --tag v1
```

Output:

```
$DEEPRED_ROOT/sft_corpus/v1/
├── train.jsonl       (~95% of pairs, shuffled)
├── val.jsonl         (~5% of pairs)
└── manifest.json     (source counts, seed, sha256 of inputs)
```

Default sources are `year_topics,gutenberg,augmented_chess_games,chess_books`.
To include Wikipedia (PostgreSQL) or raw chess PGN:

```bash
python3 scripts/build_sft_dataset.py \
    --sources wikipedia_articles,year_topics,gutenberg,augmented_chess_games,chess_books \
    --tag v1
```

Useful flags: `--max-chars 4096` (per-message cap), `--val-fraction 0.05`,
`--seed 42`, `--force` (overwrite existing tag).

---

## Step 3 — Enter the fine-tuning container

GPU training **must** run inside `strix-halo-finetuning` — the host venv
will segfault on `.cuda()` for gfx1151.

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
# inside the container:
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
```

Sanity check:

```bash
python3 -c "import torch; print(torch.__version__, '| HIP', torch.version.hip)"
python3 -c "from trl import SFTTrainer; print('trl OK')"
```

---

## Step 4 — Smoke test the training pipeline

5 optimizer steps, 1 epoch cap, no GGUF — verifies model loads, GPU works,
loss decreases. Should finish in well under 10 minutes for 4B.

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/smoke \
    --epochs 1 --max-steps 5 --no-gguf --debug
```

Expected log lines:

```
[INFO] Loading model (bf16, attn_implementation='eager')…
[INFO]   parameters : 4,300,079,104
[INFO]   footprint  : 8.60 GB
[step 1] dev=cuda:0 mem=8.62GB
...
[INFO] Training complete  : 0.05 h
[INFO] Peak GPU memory    : 35.4 GB
```

---

## Step 5 — Run full SFT

### Gemma-3-4B-IT (recommended first run)

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/v1
```

Profile defaults: batch=4, grad_accum=4 (effective batch 16), lr=5e-5,
2 epochs, max_length=2048, no gradient checkpointing.

### Gemma-3-12B-IT

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-12b \
    --dataset-dir /mnt/data/sft_corpus/v1
```

Profile defaults: batch=1, grad_accum=16 (effective batch 16), lr=2e-5,
2 epochs, max_length=2048, gradient checkpointing ON (required to fit
~115 GB peak in 128 GB unified memory).

### Faster training with Unsloth (`--unsloth`)

The `strix-halo-finetuning` container ships a gfx1151-patched build of
[Unsloth](https://github.com/unslothai/unsloth) (pinned commit + PR-4109
patch). Adding `--unsloth` to any of the commands above swaps
`AutoModelForCausalLM` for `FastLanguageModel`, which gives roughly
**2-3× faster** training and **~30% lower peak memory** on full FT
(per the kyuz0 reference numbers).

```bash
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --unsloth
```

What changes under the hood when `--unsloth` is set:

- Model is loaded via `FastLanguageModel.from_pretrained(...)`; the
  script then re-enables `requires_grad` on all parameters (Unsloth
  freezes them by default for LoRA usage).
- The Gemma-3 chat template is applied **once** up front via
  `get_chat_template(tokenizer, "gemma-3")` and stored in a `text`
  column — Unsloth patches the tokenizer with closures that cannot be
  pickled across `dataset_num_proc` workers, so the template cannot be
  applied lazily inside `SFTTrainer`.
- `SFTConfig` receives `dataset_text_field="text"` and
  `dataset_num_proc=1`.
- The Unsloth setting is included in the run fingerprint, so toggling
  `--unsloth` on/off triggers the “different parameters” guard rather
  than silently resuming.

Limitations:

- Requires the `strix-halo-finetuning` container. Outside of it the
  script will exit with a clear `--unsloth requires the unsloth package`
  error.
- Works with both `--type full` and `--type lora`.
- Multi-node FSDP is incompatible with Unsloth; we only run single-node
  so this does not matter in practice.

### LoRA fine-tuning (`--type lora`)

Instead of training all model weights, LoRA inserts small low-rank
adapter matrices into the attention and MLP projections and trains only
those (~2% of total parameters). Per the kyuz0 reference numbers, this
is ~2× faster and uses ~40-50% less memory than full FT, at the cost of
some expressiveness. It is also what makes Gemma-3-27B-IT viable on a
single Strix Halo box.

```bash
# 4B LoRA — fast iteration
python3 scripts/train_deepred_gemma.py --profile gemma-4b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --type lora

# 12B LoRA — still fits comfortably on Strix Halo
python3 scripts/train_deepred_gemma.py --profile gemma-12b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --type lora

# 12B LoRA + Unsloth — fastest single-node setup
python3 scripts/train_deepred_gemma.py --profile gemma-12b \
    --dataset-dir /mnt/data/sft_corpus/v1 \
    --type lora --unsloth
```

Adapter hyperparameters (hard-coded, match kyuz0 reference):

| Setting | Value |
|---|---|
| `r` (rank) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `bias` | `none` |
| `task_type` | `CAUSAL_LM` |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

These constants live in [`scripts/train_deepred_gemma.py`](../scripts/train_deepred_gemma.py)
as `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES`. Edit
them there if you need to tune (and use a fresh `--run-name`, since
`--type` is part of the fingerprint but the LoRA constants themselves
are not).

What changes when `--type lora` is set:

- The base model is loaded normally, then wrapped with
  `peft.get_peft_model(...)` (HF path) or
  `FastLanguageModel.get_peft_model(...)` (Unsloth path).
- For Unsloth, the `requires_grad_(True)` reset is skipped — adapter
  parameters are the only trainable ones, exactly as Unsloth expects.
- Trainable-parameter count is logged via
  `model.print_trainable_parameters()` so you can confirm only the
  adapters are active.
- `trainer.save_model(final/)` saves only the LoRA adapter weights
  (small — typically <100 MB).
- For GGUF export, the script automatically calls
  `model.merge_and_unload()`, saves the merged full-precision model to
  `final-merged/`, and then runs the llama.cpp conversion from there.
  The adapter-only directory at `final/` is preserved so you can also
  load it on top of the base model for cheap inference or further
  training.

Output layout for a LoRA run:

```
$DEEPRED_ROOT/training_output/gemma-4b-2026-05-21/
├── checkpoint-NNN/        (adapter checkpoints — small, ~80 MB each)
├── final/                 (final LoRA adapters — swap onto base model)
├── final-merged/          (adapters merged into base — full weights)
└── gguf/<name>-final.gguf (produced from final-merged/)
```

### Resume after interruption

Re-run the **same command** with the same `--run-name` (or rely on the
auto-generated `<profile>-<YYYY-MM-DD>` name). The fingerprint stored in
`run_meta.json` is compared; if it matches and a `checkpoint-NNNN` dir is
present, training resumes from the latest checkpoint via HF Trainer's
`resume_from_checkpoint`.

Force a fresh run with `--new-run` (auto-increments name) or
`--run-name <custom>`.

---

## Step 6 — Outputs

```
$DEEPRED_ROOT/training_output/gemma-4b-2026-05-21/
├── checkpoint-100/         (HF Trainer checkpoint — full save)
├── checkpoint-200/
├── final/                  (final model + tokenizer)
├── gguf/
│   └── gemma-4b-2026-05-21-final.gguf   (for LM Studio)
├── train.log               (full training log)
└── run_meta.json           (fingerprint, status, params)
```

Copy the GGUF to LM Studio:

```bash
cp $DEEPRED_ROOT/training_output/gemma-4b-2026-05-21/gguf/*.gguf \
   $DEEPRED_ROOT/lmstudio/models/
```

Default quant is `q8_0`. Change with `--gguf-quant q4_k_m`. Disable
entirely with `--no-gguf`.

---

## Memory & time reference

Calibrated from kyuz0's published numbers (max_length=512, 1000-sample set)
and scaled for our defaults (max_length=2048, full DeepRed SFT corpus).

| Model | Peak memory (full FT) | kyuz0 reference (512 / 1k samples) |
|---|---|---|
| Gemma-3-4B-IT  | ~50–70 GB | 46 GB / 9 min |
| Gemma-3-12B-IT | ~100–120 GB w/ grad-ckpt | 115 GB / 25 min |

For a full DeepRed SFT corpus (~100k–500k pairs at maxlen 2048), expect
**hours**, not minutes, per epoch. Use `--debug` for per-step memory
prints during the first run to confirm headroom before committing to a
long run.

---

## Practical notes (AMD / ROCm / Gemma)

- **`attn_implementation="eager"` is mandatory for Gemma training.**
  Gemma-3 uses soft-capping that breaks FlashAttention. The script
  hard-codes this; don't change it.
- **bf16 only.** ROCm fp16 is unreliable for these models.
- **`adamw_torch_fused`** is the chosen optimizer (matches kyuz0).
- **Cosine LR + 100-step warmup** is used by default; kyuz0 uses
  constant. Override via `--lr-scheduler-type constant --warmup-steps 0`
  to exactly mirror their setup.
- **`ddp_find_unused_parameters=True`** is set because Gemma-3 12B+ are
  natively multimodal and the vision parameters are unused in our
  text-only training.
- **`--unsloth`** provides a ~2-3× speedup but only inside the
  `strix-halo-finetuning` container. Works with both `--type full` and
  `--type lora`. See the dedicated subsection under Step 5.
- **Gemma-3-27B** is not exposed as a profile for `--type full` (OOMs
  even on 128 GB per kyuz0). It is feasible with `--type lora`; pass
  `--model /path/to/gemma-3-27b-it` together with the `gemma-12b`
  profile and reduce `--max-length` if needed.
- **Kernel parameters** for 128 GB unified memory (apply once on host
  via GRUB — see [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md)):

  ```
  amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432
  ```

---

## Troubleshooting

**"ERROR: model not found at /mnt/data/models/gemma-3-4b-it"**
Run `download_gemma_models.py` first (Step 1).

**"ERROR: dataset not found at ..."**
Run `build_sft_dataset.py` first (Step 2). Verify
`train.jsonl` and `val.jsonl` both exist in `--dataset-dir`.

**Segfault on `.cuda()` / "GPU operations failed"**
You're outside the container. Re-enter with
`podman exec -it strix-halo-finetuning bash` and re-activate
`/opt/venv/bin/activate`.

**`GatedRepoError` during download**
Open the Gemma model page in your browser, accept the license, retry.

**OOM on 12B**
Gradient checkpointing should already be on for `gemma-12b`. If still
OOM, reduce `--max-length` (e.g. 1024) or `--batch-size 1` with higher
`--grad-accum`.

**Existing CPT training still running**
This script is fully independent — different output dir, different
script, different container is fine. The existing `train_deepred_model.py`
is untouched.
