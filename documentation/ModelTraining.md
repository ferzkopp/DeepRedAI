# Training a Model from Scratch

---

## 🚀 Quick Start: Build the Deep Red Model

**Status:** ✅ All data preparation complete. Ready to train.

**Your corpus:** 2.82B tokens (5.3 GB), 1,598,806 sequences, train/val split ready at `/mnt/data/training_corpus/TinyLlama-1.1B/`

### Step-by-Step: Train a 1.1B Model in 4–6 Weeks

1. **Download base model** (skip if already done by setup)
   - If `setup_strixhalo.py` already downloaded it, verify it's at `/mnt/data/models/TinyLlama-1.1B/`
   - Otherwise, download `TinyLlama-1.1B-intermediate-step-1431k-3T` from HuggingFace (~2.2 GB)

2. **Run continued pre-training** (3–5 weeks)
   - Execute: `python scripts/train_deepred_model.py --model-name TinyLlama-1.1B --data-path /mnt/data/training_corpus/TinyLlama-1.1B/ --output-dir output/deepred-1b/ --epochs 5`
   - Full-weight BF16 training with gradient checkpointing on Strix Halo iGPU
   - Monitor: validation loss, sample generations
   - Expected: Loss curves flatten after ~3–5 epochs; post-1969 knowledge suppression increases over time

3. **Evaluate & validate** (1 day)
   - Run: `python scripts/evaluate_temporal.py --model output/deepred-1b/ --checkpoint latest`
   - Test temporal compliance, sample generation, and factual accuracy
   - Decide: proceed to SFT or retrain with adjusted hyperparameters

4. **Supervised fine-tune for chat** (1–2 days)
   - Generate ChatML dataset: `python scripts/generate_theme_dataset.py --model-api-host 192.168.42.15:1234`
   - LoRA SFT: `python scripts/finetune_theme.py --base-model output/deepred-1b/ --data-path theme_dataset.jsonl --output-dir output/deepred-1b-sft-lora/ --epochs 2`

5. **Merge & convert to GGUF** (1 hour)
   - Merge: `python scripts/merge_lora.py --base output/deepred-1b/ --adapter output/deepred-1b-sft-lora/ --output output/deepred-1b-merged/`
   - Convert: `python scripts/convert_to_gguf.py --model output/deepred-1b-merged/ --output deepred-1b.gguf --quantize Q4_K_M`

6. **Deploy & test** (1 hour)
   - Load `deepred-1b.gguf` in LM Studio
   - Interactive testing + red-team for temporal leakage

**For details on tooling, hyperparameters, and architecture, see sections below.**

---

## Background

The fine-tuning of a base model as described in the [temporal](legacy/TemporalFinetuning-Plan.md) and [theme](legacy/ThemeFinetuning-Plan.md) documentation did not work well — the result of the temporal fine-tuning was catastrophic forgetting. The LoRA/QLoRA approach (even with reduced learning rates, replay data, and careful hyperparameter tuning) failed to reliably suppress post-1969 knowledge without destroying the model's general capabilities.

This document explores training a new model **from scratch** using:
- Temporally filtered Wikipedia content (pre–July 1969)
- Select books and text material for thematic alignment (Project Gutenberg)

---

## Hardware Summary

### Primary: Strix Halo (Training + Pipeline)

| Component | Specification |
|-----------|---------------|
| **CPU/APU** | AMD Ryzen AI MAX+ 395 "Strix Halo" |
| **GPU** | Integrated RDNA 3.5, 40 CUs (gfx1151) |
| **Memory** | 128 GB LPDDR5x-8000 (unified/shared CPU+GPU) |
| **Memory Bandwidth** | ~215 GB/s |
| **Estimated FP16 Compute** | ~25–30 TFLOPS peak |
| **GPU Framework** | ROCm 7.2 (via [Strix Halo Toolboxes](https://strix-halo-toolboxes.com/) container) |
| **OS** | Fedora 43, kernel 6.18.4+ |

The key advantage of this system is the **128 GB of unified memory** — far more than any consumer discrete GPU (RTX 4090: 24 GB). This makes it possible to train surprisingly large models from scratch, since training memory requirements greatly exceed inference requirements.

> **Note:** ROCm 7.1.1 is **incompatible** with kernels ≥ 6.18.4 and has been deprecated. Always use ROCm 7.2+ with modern kernels. See [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md) for full setup details.

### Optional: NVIDIA A4000 (Inference Offload)

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA A4000 (16 GB GDDR6, Ampere / SM 8.6) |
| **Host** | VM with PCI passthrough, 24 GB RAM, Fedora 43 |
| **CUDA** | Container-based (llama.cpp `server-cuda` images) |
| **Services** | LLM server (port 1234) + Embedding server (port 1235) |
| **Default LLM** | Qwen 2.5 14B Q4_K_M (~10–11 GB VRAM) |
| **Default Embedding** | nomic-embed-text-v1.5 F16 (~0.5 GB VRAM) |

The A4000 provides a **secondary OpenAI-compatible inference server** that offloads LLM and embedding workloads from the Strix Halo. This is particularly useful during training — the iGPU can focus on CPT while the A4000 handles dataset generation (SFT data via `generate_theme_dataset.py`), embedding computation (`process_and_index.py`), and interactive testing. Set `REMOTE_HOST` in the environment to route inference requests to the A4000 automatically. See [A4000-Fedora-Setup.md](A4000-Fedora-Setup.md) for setup details.

---

## Is Training from Scratch Feasible?

**Yes, within constraints.** Training from scratch is feasible for models in the **100M–3B parameter** range.

### Memory Requirements for Training

Training requires storing the model weights, optimizer states (Adam uses two additional copies), gradients, and activations in memory simultaneously:

| Model Size | FP32 Training | Mixed Precision (BF16 + FP32 optimizer) | Fits in 128 GB? |
|------------|---------------|------------------------------------------|------------------|
| **125M** | ~2 GB | ~1.5 GB | Yes |
| **350M** | ~5 GB | ~4 GB | Yes |
| **1B** | ~20 GB | ~15–20 GB | Yes |
| **3B** | ~60 GB | ~50–65 GB | Yes (tight with large batches) |
| **7B** | ~140 GB | ~110–130 GB | Marginal to No |

*Estimates include model weights (×1 for BF16), optimizer states (×2 in FP32 for Adam), gradients, and a moderate activation memory budget. Actual usage depends on batch size, sequence length, and gradient checkpointing.*

### Training Time Estimates

Based on ~25 TFLOPS effective FP16 and a realistic 25–35% Model FLOPS Utilization (MFU) for single-GPU training on ROCm:

| Model Size | Tokens/sec (estimated) | Time for 3B tokens | Time for 10B tokens |
|------------|----------------------|---------------------|----------------------|
| **125M** | ~8,000–12,000 | ~3–4 days | ~10–14 days |
| **350M** | ~3,000–5,000 | ~7–12 days | ~23–39 days |
| **1B** | ~1,000–1,700 | ~20–35 days | ~68–116 days |
| **3B** | ~350–600 | ~58–99 days | ~193–331 days |

*Single GPU (iGPU), no data parallelism. MFU conservatively estimated at 25–35% due to RDNA 3.5 lacking dedicated matrix/tensor cores. Actual throughput should be benchmarked early.*

### Feasibility Verdict

| Model Size | Feasible? | Training Time | Recommendation |
|------------|-----------|---------------|----------------|
| **125–350M** | **Very feasible** | Days to 2 weeks | Best for rapid iteration and proof of concept |
| **1B** | **Feasible** | 3–5 weeks | Best balance of capability and training time |
| **3B** | **Marginal** | 2–3+ months | Only if 1B results are promising and patience allows |
| **7B+** | **Not feasible** | 6+ months, memory-constrained | Exceeds practical limits of this hardware |

---

## Approach

### Continued Pre-Training (CPT)

Start from an existing **base model** (not instruct-tuned) and do **full-weight continued pre-training** on the curated pre-1969 corpus. This is fundamentally different from LoRA fine-tuning:

- Full weight updates (not low-rank adapters)
- Extended training on the new corpus (multiple epochs)
- Gradually overwrites post-1969 knowledge rather than trying to coexist with it
- Preserves language modeling capabilities (grammar, coherence, reasoning)

**Why this avoids the catastrophic forgetting problem:** With LoRA, the base model still contains post-1969 knowledge unchanged — you're only adding a low-rank correction. With CPT, you're actually modifying all the weights, so post-1969 knowledge gets diluted and overwritten by the dominant pre-1969 training signal.

**Pipeline:**
```
Base Model  →  CPT on Pre-1969 Corpus  →  SFT for Chat/Theme  →  GGUF Export
               (full weight, many epochs)   (LoRA is fine here)    (llama.cpp)
```

### Base Model Selection

Two base models are selected — a small one for verifying the approach quickly, and a production model for the final training run. Both use **Llama-family architectures** for native GGUF/llama.cpp compatibility.

| Role | Model | Params | Architecture | Pre-Training Data | Rationale |
|------|-------|--------|-------------|-------------------|-----------|
| **Dev** | [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 360M | Llama | 4T tokens (FineWeb-Edu, DCLM, The Stack) | Modern (2025), Llama-native GGUF, strong benchmarks for size, CPT in ~1–2 weeks |
| **Prod** | [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) | 1.1B | Llama 2 | 3T tokens (SlimPajama + StarCoder) | Best-in-class ~1B model; Llama 2 arch → drop-in GGUF/LM Studio support; extensive 3T pre-training gives a strong baseline to build on |

**Why TinyLlama-1.1B over alternatives:**

| Alternative | Why Not |
|-------------|---------|
| **Pythia-1B** (EleutherAI) | GPT-NeoX architecture requires extra conversion work for GGUF; trained on only 300B tokens (10× less than TinyLlama), resulting in a significantly weaker baseline |
| **OLMo-1B** (AI2) | Custom architecture with non-parametric LayerNorm and unique design choices; less ecosystem support for GGUF export and llama.cpp inference |
| **SmolLM2-1.7B** | At 1.7B parameters, exceeds the ~1B budget and would roughly double training time (~6–10 weeks for CPT) |

**Why SmolLM2-360M over alternatives:**

| Alternative | Why Not |
|-------------|---------|
| **Pythia-410M** (EleutherAI) | GPT-NeoX architecture; trained on only 300B tokens (13× less than SmolLM2); lower benchmarks across the board |
| **Pythia-160M** | Too small to meaningfully validate CPT quality — results wouldn't transfer to the 1.1B production run |
| **SmolLM2-135M** | Same concern — useful for pipeline smoke testing but too small to validate the CPT approach itself |

### Alternatives Considered

**Training from scratch** (random initialization) would give absolute zero post-1969 leakage by construction, but requires the model to learn language structure, grammar, and all world knowledge from the ~3B token corpus alone. This is data-starved for models above ~350M and would produce significantly weaker results than CPT from a well-trained base.

**Distillation from a filtered teacher** (using a 70B model to generate synthetic training data) could produce high-quality temporally-filtered training examples, but adds complexity, depends on teacher quality, and introduces a risk of subtle temporal leakage from the teacher's parametric knowledge. This remains viable as a **data augmentation** technique on top of CPT if results need improvement.

---

## Training Corpus

### Continued Pre-Training Data

| Source | Articles/Works | Raw Text | Estimated Tokens | Temporal Compliance |
|--------|---------------|----------|------------------|---------------------|
| Pre-1969 Wikipedia articles | ≥1.55M articles | 5.7 GB | ≥1.4B tokens | Filtered via YAGO/Wikidata + LLM temporal classification |
| Project Gutenberg books | 766 books (pre-1969, thematically aligned) | 498 MB | ~125M tokens | Pre-1969 by definition (public domain) |
| Chess content (games + books) | ~356K games + 10 books | 213 MB | ~53M tokens | Filtered to pre-July 1969 |
| **Total** | — | **~6.4 GB** | **≥1.6B tokens** | Fully pre-1969 |

*Token estimates use ~4 characters per token (typical for Llama-family BPE tokenizers on English text).*

> **Wikipedia classification:** Of 7.0M total articles in the database, temporal classification is available for about ~40% of the content — 1.55M classified as pre-1969 ("old"), 785K as post-1969 ("new"), 466K as uncertain, and 4.2M still unclassified. The article count and token figures above are **lower bounds** that will increase if LLM-based temporal augmentation (`scripts/llm_temporal_analysis_augmentation.py`) is used against the remaining articles. Projected final corpus: ~2–3M qualifying articles, ~2.5–4B tokens.

> **Chess content**: Deep Red's origin as a chess-playing AI requires genuine chess knowledge — rules, notation, strategy, opening theory, famous games, and player history through 1969. Sources include Internet Archive chess books (10 texts, 3.7M chars) and PGN historical game databases converted to natural language prose (~356K pre-1969 games from 683 PGN files, 209M chars). See [Chess-Setup.md](Chess-Setup.md) for full source list, conversion approaches, and retrieval details.

### Supervised Fine-Tuning  Data (Chat / Theme Alignment)

After CPT, the model needs supervised fine-tuning data to become a conversational agent with the Deep Red persona. This is a **separate, much smaller** dataset in ChatML format — structured as multi-turn conversations with system prompts, user questions, and persona-aligned responses.

**What the SFT data contains:**

| Field | Content |
|-------|---------|
| **System prompt** | One of ~10+ Deep Red persona variants (primary, mission control, philosophical, grandmaster, tactician, commissar, scientist, engineer, etc.) — each emphasizing different facets of the AI character |
| **User message** | A natural question or topic drawn from a source passage (e.g., "What is the purpose of the Mars mission?", "How does the colony sustain itself?") |
| **Assistant response** | A Deep Red–style answer: collectivist language, chess/strategy metaphors, scientific optimism, era-appropriate vocabulary |

**Example ChatML training entry:**
```json
{
  "messages": [
    {"role": "system", "content": "You are Deep Red, a chess-playing artificial intelligence that guides humanity's Mars city utopia..."},
    {"role": "user", "content": "What drives humanity's push toward space exploration?"},
    {"role": "assistant", "content": "The cosmos presents the ultimate positional challenge, comrade. Like a grandmaster who sees..."}
  ]
}
```

**How the SFT data is generated:**

The existing `scripts/generate_theme_dataset.py` generates these examples semi-automatically:

1. Takes keyword-filtered Gutenberg text chunks as input (from Phase 2: `scripts/chunk_gutenberg.py` → `scripts/keyword_filter.py`)
2. Sends each chunk to a local LLM via LM Studio (e.g., Qwen 2.5 7B Instruct)
3. The LLM generates a natural user question based on the chunk's content
4. The LLM then generates a Deep Red–style response, guided by a randomly selected persona system prompt
5. Output is written as ChatML JSONL

**Source material and pipeline:**

```
Gutenberg filtered chunks (from scripts/keyword_filter.py)
    │
    ├── ~20K–100K thematically relevant passages
    │
    └── scripts/generate_theme_dataset.py
            │
            ├── LM Studio (teacher LLM) generates Q&A pairs
            ├── Randomly cycles through Deep Red persona variants
            ├── 1–2 examples per chunk → ~10K–50K ChatML examples
            │
            └── theme_dataset.jsonl  (~5–25M tokens)
```

**Estimated SFT dataset size:** ~10,000–50,000 examples (~5–25M tokens). This is tiny compared to the CPT corpus, but that's expected — SFT only needs to teach conversational patterns and persona style, not world knowledge.

**Where to source additional SFT data if needed:**

| Source | Type | Notes |
|--------|------|-------|
| Gutenberg filtered chunks (existing) | Primary source | Already chunked + keyword-filtered; the main pipeline |
| Wikipedia pre-1969 articles | Supplement | Run the same generation script on Wikipedia excerpts for broader topic coverage |
| Year topics from `scripts/extract_year_topics.py` | Supplement | Historical events as prompts — good for temporal Q&A examples |
| Manually written examples | Quality anchor | Hand-craft 50–100 "gold standard" conversations to steer tone and style |

See [ThemeFinetuning-DataPreparation-Phase3.md](legacy/ThemeFinetuning-DataPreparation-Phase3.md) for full setup and execution details.

### Chinchilla-Optimal Data Ratios

The Chinchilla scaling laws suggest an optimal ratio of ~20 tokens per parameter:

| Model Size | Optimal Tokens | Available Tokens | Strategy |
|------------|---------------|------------------|----------|
| **125M** | ~2.5B | 2.5–4.5B | Sufficient — 1–2 epochs |
| **350M** | ~7B | 2.5–4.5B | Slightly data-starved — train 2–3 epochs |
| **1B** | ~20B | 2.5–4.5B | Data-starved — train 5–10 epochs, augment data |
| **3B** | ~60B | 2.5–4.5B | Heavily data-starved — train 15+ epochs |

Training for more epochs than Chinchilla-optimal increases the risk of memorization/overfitting, but for a domain-specific model this is acceptable — the model *should* memorize pre-1969 knowledge. Use a held-out validation set to monitor loss.

### Data Preparation Pipeline

The detailed corpus build, tokenization, sharding, and train/validation split workflow is documented in [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md). Use that document as the canonical pipeline reference.

---

## Tooling

### Pre-Training Frameworks

| Framework | Best For | ROCm Support | Notes |
|-----------|----------|-------------|-------|
| **[nanoGPT](https://github.com/karpathy/nanoGPT)** | ≤350M models, simplicity | Yes (PyTorch) | Minimal code, easy to understand and modify. Good for learning and small experiments. |
| **[LitGPT](https://github.com/Lightning-AI/litgpt)** | 1B–3B, modern architectures | Yes (PyTorch) | Supports Llama/Mistral/Pythia architectures. Built on Lightning, good for both pre-training and fine-tuning. |
| **[torchtune](https://github.com/pytorch/torchtune)** | 1B–7B, Meta's official tool | Yes (PyTorch) | Native PyTorch library for training/fine-tuning LLMs. Well-maintained, good Llama support. |
| **[Hugging Face Transformers + Accelerate](https://huggingface.co/docs/transformers)** | Any size, most flexible | Yes (PyTorch) | Full-featured, vast model zoo, well-documented. Slightly more overhead than minimal frameworks. |
| **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** | Multi-GPU only | Limited | Overkill for single-GPU — skip this. |

**Recommendation:** Use **nanoGPT** for pipeline smoke testing with a tiny model (Dev Phase 3), then use **LitGPT** or **torchtune** for both the dev CPT (SmolLM2-360M) and production CPT (TinyLlama-1.1B) runs.

### Data Preparation Tools

| Tool | Purpose |
|------|---------|
| **tiktoken** / **sentencepiece** | Tokenizer training (if training from scratch) |
| **HuggingFace tokenizers** | Fast BPE tokenizer training and application |
| **datasets** (HuggingFace) | Efficient data loading and streaming for large corpora |
| Existing scripts in `scripts/` | `extract_wikipedia.py`, `chunk_gutenberg.py`, `keyword_filter.py` |

### Conversion and Deployment

| Tool | Purpose |
|------|---------|
| **llama.cpp** | GGUF conversion and quantization (already built with HIP in the setup) |
| **`scripts/convert_to_gguf.py`** | Existing conversion script |
| **LM Studio** | Local inference and testing |

---

## Recommended Configuration (TinyLlama-1.1B, CPT Approach)

### Architecture

TinyLlama-1.1B uses the **Llama 2 architecture** — RoPE, SwiGLU, RMSNorm, GQA — providing full ecosystem compatibility with llama.cpp, LM Studio, and GGUF tooling:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Parameters | 1.1B | Inherited from base model |
| Hidden dimension | 2048 | |
| Layers | 22 | |
| Attention heads | 32 | |
| KV heads (GQA) | 4 | Reduces memory and inference cost |
| Intermediate size | 5632 | SwiGLU MLP |
| Context length | 2048 | Sufficient for training; extend later if needed |
| Vocabulary size | 32,000 | Llama 2 BPE tokenizer |
| Positional encoding | RoPE | Required for Llama compatibility and GGUF export |

*The dev model (SmolLM2-360M) uses a similar Llama architecture but with 960 hidden dim, 32 layers, 15 heads, 5 KV heads, and a 49,152-token vocabulary.*

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Precision** | BF16 mixed precision | Saves memory; ROCm supports BF16 on RDNA 3.5 |
| **Optimizer** | AdamW | Standard; β₁=0.9, β₂=0.95, ε=1e-8 |
| **Learning rate** | 3e-4 (peak) | With cosine decay to 3e-5 |
| **Warmup** | 2,000 steps | ~1% of total training |
| **Weight decay** | 0.1 | Standard regularization |
| **Batch size** | Micro-batch 4–8, gradient accumulation to effective batch ~128–256 | Tune to fit memory |
| **Sequence length** | 2048 tokens | Reduce to 1024 if memory-constrained |
| **Gradient checkpointing** | Enabled | Trades compute for memory — essential |
| **Epochs** | 5–10 over the full corpus | Data-starved regime; monitor val loss for overfitting |
| **Total tokens** | ~15–30B (5–10 epochs × 3B tokens) | |

### ROCm-Specific Settings

These are automatically set by the venv activate script (added by `setup_strixhalo.py`):

```bash
# Required for Strix Halo gfx1151 (ROCm 7.2)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Enable optimized matrix math
export ROCBLAS_USE_HIPBLASLT=1

# PyTorch ROCm memory management
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

---

## Post-Training Pipeline

After pre-training (or continued pre-training), the model has raw language modeling capability but no chat or instruction-following behavior. Apply these steps in order:

### Step 1: Supervised Fine-Tuning (SFT) for Chat

Use **LoRA** fine-tuning (this is where LoRA works well — the model already has the right knowledge):

- Use the existing thematic ChatML dataset from `scripts/generate_theme_dataset.py`
- Apply the Deep Red system prompt and persona
- LoRA rank 16–32, learning rate 2e-5, 2–3 epochs
- This should NOT cause catastrophic forgetting because the base knowledge is already temporally correct

### Step 2: Merge and Convert

```bash
# Merge LoRA adapter with base model
python scripts/merge_lora.py --base output/pretrained-1b/ --adapter output/sft-lora/ --output output/merged/

# Convert to GGUF for LM Studio
python scripts/convert_to_gguf.py --model output/merged/ --output output/deepred-1b.gguf --quantize Q4_K_M
```

### Step 3: Evaluate

Use the existing `scripts/evaluate_temporal.py` to check temporal separation, then manually test in LM Studio.

---

## Expected Capabilities of the Resulting Model

### By Model Size

| Capability | 125–350M | 1B | 3B |
|------------|----------|-----|-----|
| **Grammatical coherence** | Good for short passages | Good | Very good |
| **Paragraph-length generation** | Adequate, may drift | Good  | Very good |
| **Pre-1969 factual Q&A** | Basic recall | Solid recall | Strong recall |
| **Post-1969 knowledge leakage** | None (by construction) | None (by construction) | None (by construction) |
| **Thematic/persona consistency** | Moderate (with SFT) | Good (with SFT) | Very good (with SFT) |
| **Multi-turn conversation** | Limited | Moderate | Good |
| **Reasoning / logic** | Minimal | Basic | Moderate |
| **Creative writing (era-appropriate)** | Short passages | Paragraphs | Extended passages |
| **Instruction following** | Basic | Good | Good |
| **Math / arithmetic** | Unreliable | Basic | Basic |
| **Overall quality comparison** | GPT-2 Small/Medium | GPT-2 XL / early Pythia | Phi-2 / TinyLlama level |

### What the Model Would Be Good At

- **Era-appropriate conversation**: Discussing pre-1969 history, science, culture, and politics with period-appropriate framing
- **Deep Red persona**: Collectivist language, chess metaphors, scientific rationalism, optimistic futurism
- **Temporal consistency**: Describing the world as it was before July 1969 — no anachronisms
- **Thematic text generation**: Writing passages in the style of mid-century literature, Soviet futurism, and early space age optimism

### What the Model Would NOT Be Good At

- **Complex reasoning**: Multi-step logical problems, mathematical proofs
- **Code generation**: Not in the training data
- **Post-1969 anything**: By design — but this means no knowledge of modern medicine, technology, or events
- **Long document understanding**: Limited context window (2048–4096 tokens)
- **Multilingual**: English-only unless multilingual data is included
- **Competing with modern LLMs**: A 1B model trained on 3B tokens cannot match a 70B model trained on 15T tokens on general benchmarks

### The Key Advantage Over Fine-Tuning

The fundamental benefit of this approach is **no knowledge leakage by construction**. A fine-tuned model still contains post-1969 knowledge in its weights and may leak it through indirect questioning, adversarial prompting, or chain-of-thought manipulation. A model trained only on pre-1969 content literally does not have that knowledge to leak.

---

## Suggested Development Roadmap

The roadmap is structured in three layers:

1. **System Setup** — One-time infrastructure provisioning (Fedora, toolboxes, services)
2. **Initial Development Run** — Fast end-to-end pass through every script and step using minimal data and a tiny model. The goal is to exercise and validate the entire pipeline, not to produce a useful model. Expect garbage output — that's fine.
3. **Production Run** — Full-scale data processing and multi-week training to produce the actual model.

> **Status (as of May 2026):** Phase 0 (system setup), Prod Phase 1 (Wikipedia pipeline), and Prod Phase 2 (corpus preparation) are **complete**. The full corpus has been finalized: **2.82B tokens** (5.3 GB) across 1,598,806 sequences at 2048 tokens each. Training/validation split: 1,582,818 train sequences (3.24B tokens) + 15,988 val sequences (32.7M tokens) ready at `/mnt/data/training_corpus/TinyLlama-1.1B/`. Ready to begin Prod Phase 3 (dev CPT on SmolLM2-360M) or proceed directly to Prod Phase 4 (full TinyLlama-1.1B CPT).

---

### Phase 0: System Setup (Fedora + Toolboxes) — ✅ COMPLETE

This phase sets up the Strix Halo machine from scratch using Fedora instead of the previous Ubuntu-based setup. The [Strix Halo Toolboxes](https://strix-halo-toolboxes.com/) project provides containerized, pre-configured environments for AI workloads on Strix Halo.

| Step | Task | Status | Notes |
|------|------|--------|-------|
| 0.1 | Install Fedora (latest, kernel 6.18.4+) on Strix Halo | ✅ Done | Fedora 43, kernel 6.18.4+ |
| 0.2 | Configure GTT memory allocation (maximize GPU-accessible RAM) | ✅ Done | BIOS UMA minimized; kernel params allocate up to 124 GB dynamically |
| 0.3 | Install [Strix Halo Toolboxes](https://strix-halo-toolboxes.com/) | ✅ Done | ROCm 7.2, PyTorch, llama.cpp available in containers |
| 0.4 | Install and configure PostgreSQL (for Wikipedia DB) | ✅ Done | 50 GB database operational; see [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) |
| 0.5 | Install and configure OpenSearch (for semantic search) | ✅ Done | 41 GB index operational; systemd service configured |
| 0.6 | Install and configure LM Studio (headless server) | ✅ Done | Headless server via A4000 (`REMOTE_HOST=192.168.42.15`); see [A4000-Fedora-Setup.md](A4000-Fedora-Setup.md) |
| 0.7 | Verify ROCm: `rocminfo`, run llama.cpp test inference | ✅ Done | gfx1151 detected, GPU offloading confirmed |
| 0.8 | Set up Python venv with training dependencies (PyTorch ROCm, transformers, PEFT, datasets) | ✅ Done | Venv at `/mnt/data/venv`; `torch.cuda.is_available()` returns True via HIP |

---

### Initial Development Run — ⏭️ SKIPPED

> **Note:** This phase was skipped. The production data pipeline (Prod Phases 1–2) was executed directly, which validated all scripts and services at full scale. The dev run's goal — exercising every script and verifying every format — was achieved as part of the production pipeline work instead.

<details>
<summary>Original dev run plan (preserved for reference)</summary>

**Goal:** Exercise every script and step end-to-end in a single day. Use tiny data subsets and a minimal model. The output will be a non-functional toy model, but the pipeline will be fully validated — every script invoked, every format verified, every service confirmed working.

**Guiding principle:** If a step takes more than 30 minutes, you're using too much data. Cut the input size until it's fast.

#### Dev Phase 1: Wikipedia Pipeline (Minimal)

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| D1.1 | Download Wikipedia dump (full dump required, but can abort early or use a sample) | 2–4 hours | Alternatively, use an existing partial dump or the `enwiki-latest-abstract.xml.gz` (~1 GB) for smoke testing |
| D1.2 | Import a small subset into PostgreSQL (~1,000 articles) | 10–15 min | Limit `extract_wikipedia.py` input or truncate the dump; verify DB schema is correct |
| D1.3 | Run `yago_parser.py` on a truncated YAGO file (~10K lines) | 5 min | `head -10000 yago-facts.ttl > yago-sample.ttl`; verify CSV output format |
| D1.4 | Run `normalize_temporal_output.py` on the small YAGO CSV | 5 min | Verify English URL mapping and page ID lookup work against the small DB |
| D1.5 | Run `augment_wikipedia_temporal.py` on the small DB | 2 min | Verify `earliest_date`/`latest_date` columns are populated |
| D1.6 | Run `process_and_index.py` on the small DB subset | 10–15 min | Verify OpenSearch index is created and embeddings are stored |
| D1.7 | Start `mcp_server.py` and run a test search query | 5 min | Verify keyword + semantic search return results |
| | **Dev Phase 1 total** | **~3–4 hours** | Download time dominates; processing is minutes |

#### Dev Phase 2: Training Corpus (Minimal)

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| D2.1 | Extract pre-1969 articles from the small DB | 2 min | May only yield a few hundred articles — that's fine |
| D2.2 | Clean extracted text | 2 min | Verify output format is clean plaintext |
| D2.3 | Run `extract_year_topics.py` for a single year (e.g., 1960) | 5 min | Verify JSON output structure |
| D2.4 | Run `retrieve_gutenberg.py --priority-only` with a limit of ~5 books | 10 min | Verify JSONL output format |
| D2.5 | Run `chunk_gutenberg.py` on the 5 books | 2 min | Verify chunk sizes and JSONL output |
| D2.6 | Run `keyword_filter.py` on the small chunk set | 2 min | Verify filtering and stats output |
| D2.7 | Select tokenizer from dev base model (SmolLM2-360M tokenizer) | 5 min | Download and verify tokenization round-trips |
| D2.8 | Tokenize the small corpus into binary training format | 5 min | Verify file format, sequence lengths, shuffling |
| D2.9 | Create train/validation split | 1 min | Even a 90/10 split on tiny data is fine |
| | **Dev Phase 2 total** | **~1 hour** | |

#### Dev Phase 3: Training (Minimal — Smoke Test)

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| D3.1 | Configure nanoGPT for a ~10M parameter model (random init) | 15 min | 4 layers, 256 hidden dim, 4 heads; just enough to verify the training loop runs |
| D3.2 | Train for ~100 steps on the tiny corpus | 10–30 min | Verify loss decreases, checkpointing works, GPU is used |
| D3.3 | Verify checkpoint loading and sample generation | 5 min | Generate a few tokens — expect gibberish, just confirm it runs |
| | **Dev Phase 3 total** | **~1 hour** | |

#### Dev Phase 4: SFT + Deployment (Minimal)

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| D4.1 | Run `generate_theme_dataset.py` on ~10 filtered chunks | 10–15 min | Verify ChatML JSONL output with Deep Red persona |
| D4.2 | Run LoRA SFT on the 10M model with ~50 ChatML examples, 1 epoch | 10 min | Verify `finetune_theme.py`/`finetune_temporal.py` runs without errors |
| D4.3 | Merge LoRA adapter using `merge_lora.py` | 2 min | Verify merged model loads correctly |
| D4.4 | Convert to GGUF using `convert_to_gguf.py` | 2 min | Verify GGUF file is produced |
| D4.5 | Load GGUF in llama.cpp CLI and generate a few tokens | 5 min | Verify the file loads; output will be nonsense |
| D4.6 | Run `evaluate_temporal.py` on a few test questions | 5 min | Verify evaluation script runs; scores will be meaningless |
| | **Dev Phase 4 total** | **~1 hour** | |

#### Initial Development Run Summary

| Phase | Description | Duration |
|-------|-------------|----------|
| **D1** | Wikipedia pipeline (small subset) | ~3–4 hours |
| **D2** | Training corpus preparation (minimal) | ~1 hour |
| **D3** | Training loop validation (10M model, 100 steps) | ~1 hour |
| **D4** | SFT + merge + GGUF + eval (smoke test) | ~1 hour |

**Total: ~1 day** (excluding Wikipedia dump download time, which can run overnight)

**Success criteria:** Every script in the pipeline has been invoked and completed without errors. All intermediate file formats have been verified. All services (PostgreSQL, OpenSearch, LM Studio, MCP server) are operational. The pipeline produces a GGUF file that loads in llama.cpp. No step requires debugging during the production run.

</details>

---

### Production Run

**Goal:** Produce the actual Deep Red model. Full data, full training, real evaluation. Expect this to take **6–8 weeks** from start of data processing to a deployed model.

#### Prod Phase 1: Wikipedia Data Pipeline (Full) — ✅ COMPLETE

| Step | Task | Status | Notes |
|------|------|--------|-------|
| P1.1 | Download full English Wikipedia dump (`enwiki-*-pages-articles.xml.bz2`, ~25 GB) | ✅ Done | 24 GB dump in `/mnt/data/wikipedia/dumps/` |
| P1.2 | Extract and import all articles into PostgreSQL using `scripts/extract_wikipedia.py` | ✅ Done | 7,041,771 articles; 50 GB database; see [WikipediaMCP-Setup.md](WikipediaMCP-Setup.md) |
| P1.3 | Download and parse full YAGO temporal data using `scripts/yago_parser.py` | ✅ Done | Normalized output: `yago-facts-normalized.csv.zst` (107 MB); see [TemporalAugmentation-Setup.md](TemporalAugmentation-Setup.md) |
| P1.4 | Normalize full YAGO output using `scripts/normalize_temporal_output.py` | ✅ Done | English URL mapping + page ID lookup complete |
| P1.5 | *(Optional)* Download and parse Wikidata temporal data using `scripts/wikidata_parser.py` | ✅ Done | `wikidata-temporal-normalized.csv.zst` (150 MB); broader coverage achieved |
| P1.6 | Augment Wikipedia DB with temporal metadata using `scripts/augment_wikipedia_temporal.py` | ✅ Done | 2,804,937 articles classified: 1,553,510 pre-1969 (O), 785,380 post-1969 (N), 466,047 uncertain (S), 4,236,834 unclassified (U) |
| P1.7 | Generate text embeddings and index in OpenSearch using `scripts/process_and_index.py` | ✅ Done | 41 GB OpenSearch index; BM25 + k-NN vector search operational |
| P1.8 | Verify MCP server search works with full index | ✅ Done | MCP server (port 7000) + React web GUI (port 8080) operational; systemd services configured |

#### Prod Phase 2: Training Corpus Preparation (Full) — ✅ COMPLETE

| Step | Task | Status | Notes |
|------|------|--------|-------|
| P2.1 | Extract all pre-1969 Wikipedia articles from DB (SQL filter on temporal columns) | ✅ Done | 1,553,510 pre-1969 articles (5.74 GB raw text, ≥1.4B tokens); 1,683,075 with `earliest_date < 1970` |
| P2.2 | Clean extracted text (strip wikitext markup, normalize formatting) | ✅ Done | Extraction pipeline handles markup stripping |
| P2.3 | Extract year topics using `scripts/extract_year_topics.py` (full year range) | ✅ Done | 1,844 year-topic files (years 151–2025) in `/mnt/data/wikipedia/topics/`; see [Wikipedia-YearTopics-Setup.md](Wikipedia-YearTopics-Setup.md) |
| P2.4 | Retrieve Gutenberg full corpus using `scripts/retrieve_gutenberg.py` | ✅ Done | 766 books (497 MB JSONL) in `/mnt/data/gutenberg/corpus/`; see [Gutenberg-Setup.md](Gutenberg-Setup.md) |
| P2.5 | Chunk Gutenberg texts using `scripts/chunk_gutenberg.py` | ✅ Done | Chunked, scored, and verified output in `/mnt/data/gutenberg/theme_output/` |
| P2.6 | Filter Gutenberg chunks for thematic alignment using `scripts/keyword_filter.py` | ✅ Done | Filtered output in `/mnt/data/gutenberg/theme_output/filtered/` |
| P2.6a | Retrieve chess corpus using `scripts/retrieve_chess_content.py` | ✅ Done | Phase 1: 683 PGN files (717 MB); Phase 2: 355,980 games → narrative JSONL (315 MB); Phase 3: 10 Internet Archive books (3.7 MB); see [Chess-Setup.md](Chess-Setup.md) |
| P2.7 | Select tokenizer from prod base model (TinyLlama-1.1B tokenizer) | ✅ Done | Llama 2 BPE tokenizer (vocab 32,000, EOS=2); downloaded to `/mnt/data/training_corpus/tokenizers/TinyLlama-1.1B/`; see [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md) |
| P2.8 | Tokenize full corpus into binary training format (shuffled, 2048-token sequences) | ✅ Done | 2.82B tokens across 5 sources; 61 shard files in `/mnt/data/training_corpus/TinyLlama-1.1B/shards/`; see [TrainingCorpus-Setup.md](TrainingCorpus-Setup.md) |
| P2.9 | Create train/validation split (99%/1%) | ✅ Done | 1,582,818 train seqs (3.24B tokens, 6.0 GB) + 15,988 val seqs (32.7M tokens, 62.5 MB); output: `train.bin` / `val.bin` in `/mnt/data/training_corpus/TinyLlama-1.1B/` |

#### Prod Phase 3: Dev CPT (SmolLM2-360M)

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| P3.1 | Run CPT on SmolLM2-360M with pre-1969 corpus using `scripts/train_deepred_model.py` | 7–12 days | Full-weight BF16, gradient checkpointing; dev mode defaults to 5% data subset for fast validation; see [DeepRedModel-Setup.md](DeepRedModel-Setup.md) |
| P3.2 | Evaluate: loss curves, perplexity, sample generations, temporal compliance | 1 day | Key checkpoint — verify CPT actually suppresses post-1969 knowledge |
| P3.3 | Quick SFT test + GGUF export of the 360M model | 1 day | Verify end-to-end: does a CPT'd + SFT'd small model behave as expected? |
| P3.4 | Decide: proceed to 1.1B CPT, or adjust data/hyperparameters | — | If 360M shows good temporal separation, commit to the full TinyLlama run |
| | **Prod Phase 3 total** | **~2 weeks** | |

#### Prod Phase 4: Production CPT (TinyLlama-1.1B)

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| P4.1 | Download TinyLlama-1.1B base model from HuggingFace | 30 min | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`; ~2.2 GB in FP16 |
| P4.2 | Run continued pre-training on full pre-1969 corpus (15–30B tokens, 5–10 epochs) | 3–5 weeks | Full-weight updates, BF16 mixed precision, gradient checkpointing; use hyperparameters validated in P3 |
| P4.3 | Monitor: validation loss, periodic sample generation, checkpoint every ~1B tokens | Ongoing | Watch for overfitting (val loss rising while train loss falls) |
| | **Prod Phase 4 total** | **3–5 weeks** | |

#### Prod Phase 5: Theme Fine-Tuning and Deployment

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| P5.1 | Generate full thematic ChatML dataset using `scripts/generate_theme_dataset.py` | 4–8 hours | Requires LM Studio running; uses all filtered Gutenberg chunks |
| P5.2 | SFT for Deep Red persona using LoRA (`scripts/finetune_theme.py`) | 1–2 days | LoRA rank 16–32, lr 2e-5, 2–3 epochs on ChatML data |
| P5.3 | Merge LoRA adapter using `scripts/merge_lora.py` | 30 min | |
| P5.4 | Convert to GGUF using `scripts/convert_to_gguf.py` | 30 min | Quantize to Q4_K_M or Q5_K_M |
| P5.5 | Deploy to LM Studio and evaluate | 1 day | Run `scripts/evaluate_temporal.py` + manual testing |
| | **Prod Phase 5 total** | **2–3 days** | |

#### Prod Phase 6: Iteration

| Step | Task | Duration (est.) | Notes |
|------|------|-----------------|-------|
| P6.1 | Analyze failure modes (temporal leakage, persona drift, factual errors) | 1–2 days | Red-team testing, adversarial probing |
| P6.2 | Adjust data mix, hyperparameters, and retrain as needed | Ongoing | May loop back to Prod Phase 2 (data) or Phase 4 (training) |
| P6.3 | Consider distillation augmentation or extended CPT epochs if results need improvement | Ongoing | Teacher-filtered synthetic data (Approach alternative) or additional training passes |
| | **Prod Phase 6 total** | **Ongoing** | |

---


