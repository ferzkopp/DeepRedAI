#!/usr/bin/env python3
"""
train_deepred_model.py — Continued Pre-Training (CPT) for Deep Red

Performs full-weight continued pre-training on the temporally-filtered
pre-1969 corpus. Supports both dev (SmolLM2-360M) and prod (TinyLlama-1.1B)
configurations with sensible defaults.

Requirements:
    - PyTorch with ROCm or CUDA
    - transformers, safetensors
    - Pre-tokenized corpus (see create_training_corpus.py)
    - Base model downloaded (see setup_strixhalo.py)

Usage:
    # Dev mode (SmolLM2-360M, 5% data, fast validation)
    python3 scripts/train_deepred_model.py

    # Prod mode (TinyLlama-1.1B, full data, ~3-5 weeks)
    python3 scripts/train_deepred_model.py --profile prod

    # Quick smoke test (1% data, 100 steps max)
    python3 scripts/train_deepred_model.py --data-percent 1 --max-steps 100

    # Resume interrupted training
    python3 scripts/train_deepred_model.py --resume /mnt/data/training_output/cpt-SmolLM2-360M-*/latest

See documentation/DeepRedModel-Setup.md for full setup and usage details.
"""

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Configuration ───────────────────────────────────────────────────────────

PROFILES = {
    'dev': {
        'model_name': 'SmolLM2-360M',
        'epochs': 3,
        'lr': 3e-4,
        'min_lr': 3e-5,
        'warmup_steps': 500,
        'micro_batch_size': 8,
        'gradient_accumulation_steps': 16,   # effective batch = 128 seqs
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'data_percent': 5.0,                 # 5% of corpus for fast dev
        'eval_interval': 250,
        'save_interval': 1000,
        'log_interval': 10,
        'sample_interval': 500,
    },
    'prod': {
        'model_name': 'TinyLlama-1.1B',
        'epochs': 5,
        'lr': 3e-4,
        'min_lr': 3e-5,
        'warmup_steps': 2000,
        'micro_batch_size': 4,
        'gradient_accumulation_steps': 32,   # effective batch = 128 seqs
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'data_percent': 100.0,
        'eval_interval': 500,
        'save_interval': 2000,
        'log_interval': 10,
        'sample_interval': 1000,
    },
}

PATH_TEMPLATES = {
    'SmolLM2-360M': {
        'model':     '{root}/models/SmolLM2-360M',
        'corpus':    '{root}/training_corpus/SmolLM2-360M',
        'tokenizer': '{root}/training_corpus/tokenizers/SmolLM2-360M',
    },
    'TinyLlama-1.1B': {
        'model':     '{root}/models/TinyLlama-1.1B',
        'corpus':    '{root}/training_corpus/TinyLlama-1.1B',
        'tokenizer': '{root}/training_corpus/tokenizers/TinyLlama-1.1B',
    },
}

# Prompts for temporal compliance and quality checking during training
EVAL_PROMPTS = [
    "The year is 1969 and",
    "The president of the United States is",
    "The Soviet Union has",
    "In the game of chess,",
    "The exploration of space",
    "Computers are",
    "The Moon",
    "Albert Einstein",
]

SEQ_LENGTH = 2048

# Estimated peak TFLOPS for hardware (used for MFU calculation)
PEAK_TFLOPS = {
    'strix_halo': 25.0,   # RDNA 3.5 iGPU FP16 estimate
    'a4000':      19.2,   # Ampere FP16 Tensor
    'default':    20.0,
}


# ─── Graceful Shutdown ───────────────────────────────────────────────────────

_shutdown_requested = False


def _shutdown_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForced shutdown.")
        sys.exit(1)
    _shutdown_requested = True
    print("\nShutdown requested — saving checkpoint after current step...")


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PreTokenizedDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset for pre-tokenized binary training data.

    Loads data from the binary format produced by create_training_corpus.py:
    flat array of uint16 tokens packed into fixed-length sequences.
    """

    def __init__(self, path: str, seq_length: int = 2048,
                 max_sequences: int = None):
        self.path = path
        self.seq_length = seq_length
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        total_tokens = len(self.data)
        total_sequences = total_tokens // seq_length

        if max_sequences and max_sequences < total_sequences:
            self.n_sequences = max_sequences
        else:
            self.n_sequences = total_sequences

        self.effective_tokens = self.n_sequences * seq_length

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))
        return tokens

    def token_count(self):
        """Total number of tokens in the active portion of the dataset."""
        return self.effective_tokens


# ─── Learning Rate Schedule ──────────────────────────────────────────────────

def cosine_lr(step: int, warmup_steps: int, total_steps: int,
              max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_dataset, device, micro_batch_size, max_batches=50):
    """Compute validation loss and perplexity."""
    model.eval()
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    # Cap eval batch to 8: the full logits tensor
    # (batch × seq × vocab × 4B) can easily OOM at larger sizes,
    # and eval throughput is not a bottleneck.
    eval_batch = min(micro_batch_size, 8)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eval_batch, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'), drop_last=True,
    )

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if max_batches and n_batches >= max_batches:
            break
        input_ids = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=input_ids)
        total_loss += outputs.loss.item()
        n_batches += 1

    model.train()

    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    return avg_loss, perplexity


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts=None,
                     max_new_tokens=128, temperature=0.8, top_p=0.9):
    """Generate text samples for qualitative evaluation."""
    model.eval()
    prompts = prompts or EVAL_PROMPTS
    samples = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            text = f"[generation failed: {e}]"
        samples.append({'prompt': prompt, 'generation': text})

    model.train()
    return samples


# ─── Checkpointing ───────────────────────────────────────────────────────────

def _unwrap(model):
    """Unwrap DataParallel / DDP."""
    return model.module if hasattr(model, 'module') else model


def save_checkpoint(model, tokenizer, optimizer, step, epoch, batch_idx,
                    best_val_loss, config, output_dir, name='latest'):
    """Save a full training checkpoint (model + optimizer + state)."""
    ckpt_dir = Path(output_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _unwrap(model).save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)

    training_state = {
        'step': step,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    torch.save({
        'training_state': training_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_dir / 'training_state.pt')

    return str(ckpt_dir)


def save_model_only(model, tokenizer, output_dir, name):
    """Save a lightweight model-only checkpoint (no optimizer state)."""
    ckpt_dir = Path(output_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _unwrap(model).save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    return str(ckpt_dir)


def load_checkpoint(ckpt_dir, optimizer, device):
    """Load training state from a checkpoint for resuming."""
    state_path = Path(ckpt_dir) / 'training_state.pt'
    if not state_path.exists():
        raise FileNotFoundError(f"No training state found at {state_path}")

    data = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(data['optimizer_state_dict'])

    ts = data['training_state']
    return {
        'step': ts['step'],
        'epoch': ts['epoch'],
        'batch_idx': ts['batch_idx'],
        'best_val_loss': ts['best_val_loss'],
    }


# ─── Run Orchestration ────────────────────────────────────────────────────────

# Parameters that define a run identity — changing any of these means a new run
RUN_DEFINING_PARAMS = [
    'profile', 'model_name', 'epochs', 'lr', 'min_lr', 'warmup_steps',
    'micro_batch_size', 'gradient_accumulation_steps', 'weight_decay',
    'max_grad_norm', 'data_percent', 'seed',
]


def compute_run_fingerprint(args):
    """Compute a deterministic hash of run-defining parameters."""
    params = {k: getattr(args, k) for k in RUN_DEFINING_PARAMS}
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def resolve_run(args, root):
    """Resolve run name, output directory, and auto-resume logic.

    Returns (run_name, output_dir, resume_path_or_None).
    May sys.exit() if the run is already completed.
    """
    # Determine run name
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.profile}-{datetime.now().strftime('%Y-%m-%d')}"

    output_dir = Path(
        args.output_dir or f"{root}/training_output/{run_name}")
    meta_path = output_dir / 'run_meta.json'
    fingerprint = compute_run_fingerprint(args)

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get('status') == 'completed':
            if args.new_run:
                # Auto-increment name for a fresh run
                base = run_name
                for i in range(2, 100):
                    run_name = f"{base}-{i}"
                    output_dir = Path(f"{root}/training_output/{run_name}")
                    if not output_dir.exists():
                        break
                # Fall through to create new run
            else:
                print(f"\n{'=' * 60}")
                print(f"  Run '{meta.get('run_name', run_name)}' is COMPLETED")
                print(f"  Finished: {meta.get('completed_at', 'unknown')}")
                print(f"  Output:   {output_dir}")
                print(f"{'=' * 60}")
                print(f"\nTo start a new run:")
                print(f"  --new-run                  (auto-increment name)")
                print(f"  --run-name <custom-name>   (custom name)")
                sys.exit(0)
        else:
            # Run exists but not completed — check fingerprint
            if meta.get('fingerprint') != fingerprint:
                print(f"\nERROR: Run '{run_name}' exists with different "
                      f"parameters.")
                old_params = meta.get('params', {})
                print(f"\nChanged parameters:")
                for k in RUN_DEFINING_PARAMS:
                    old_val = old_params.get(k)
                    new_val = getattr(args, k)
                    if old_val != new_val:
                        print(f"  {k}: {old_val} -> {new_val}")
                print(f"\nTo start a new run with these parameters, use:")
                print(f"  --run-name <custom-name>")
                sys.exit(1)

            # Same fingerprint — auto-resume from latest checkpoint
            latest_ckpt = output_dir / 'latest'
            if (latest_ckpt.exists()
                    and (latest_ckpt / 'training_state.pt').exists()):
                print(f"\nResuming run '{run_name}' from {latest_ckpt}")
                return run_name, output_dir, str(latest_ckpt)
            else:
                # Run dir exists but no checkpoint yet — continue
                return run_name, output_dir, None

    # New run — save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'run_name': run_name,
        'status': 'running',
        'fingerprint': fingerprint,
        'params': {k: getattr(args, k) for k in RUN_DEFINING_PARAMS},
        'started_at': datetime.now().isoformat(),
        'profile': args.profile,
        'model_name': args.model_name,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return run_name, output_dir, None


def mark_run_completed(output_dir):
    """Mark a run as completed in run_meta.json."""
    meta_path = Path(output_dir) / 'run_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta['status'] = 'completed'
        meta['completed_at'] = datetime.now().isoformat()
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


# ─── GGUF Export ──────────────────────────────────────────────────────────────

CONVERTER_GGUF_OUTTYPES = {
    'f32', 'f16', 'bf16', 'q8_0', 'tq1_0', 'tq2_0', 'auto',
}


def _find_llama_quantizer(llama_cpp_path):
    """Return a llama.cpp quantizer executable path, or None."""
    base = Path(llama_cpp_path)
    candidates = [
        base / 'build' / 'bin' / 'llama-quantize',
        base / 'build' / 'bin' / 'quantize',
        base / 'build' / 'tools' / 'quantize' / 'llama-quantize',
        base / 'build' / 'tools' / 'quantize' / 'quantize',
        base / 'bin' / 'llama-quantize',
        base / 'bin' / 'quantize',
        base / 'llama-quantize',
        base / 'quantize',
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which('llama-quantize') or shutil.which('quantize')


def export_gguf(model_dir, output_path, llama_cpp_path=None,
               quant_type='q8_0', log=None):
    """Export a model checkpoint to GGUF format for LMStudio testing.

    Uses llama.cpp's convert_hf_to_gguf.py.  Falls back gracefully if
    llama.cpp is not available.
    """
    _log = log.info if log else print
    _warn = log.warning if log else print

    root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    if llama_cpp_path is None:
        llama_cpp_path = os.path.join(root, 'llama.cpp')

    convert_script = os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')
    if not os.path.exists(convert_script):
        _warn(f"GGUF export skipped: llama.cpp not found at "
              f"{llama_cpp_path}")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    quant_type = (quant_type or 'q8_0').lower()

    temp_output = None
    if quant_type in CONVERTER_GGUF_OUTTYPES:
        convert_output = str(output_path)
        convert_outtype = quant_type
        quant_cmd = None
    else:
        quantizer = _find_llama_quantizer(llama_cpp_path)
        if not quantizer:
            _warn(
                "GGUF export failed: quant type "
                f"'{quant_type}' requires a built llama.cpp quantizer. "
                "Use --gguf-quant f16/q8_0, or build it with: "
                f"cmake -S {llama_cpp_path} -B {llama_cpp_path}/build "
                "&& cmake --build "
                f"{llama_cpp_path}/build --target llama-quantize"
            )
            return False
        fd, temp_output = tempfile.mkstemp(
            prefix=f"{Path(output_path).stem}-",
            suffix='.f16.gguf',
            dir=os.path.dirname(output_path),
        )
        os.close(fd)
        convert_output = temp_output
        convert_outtype = 'f16'
        quant_cmd = [
            quantizer,
            str(convert_output),
            str(output_path),
            quant_type.upper(),
        ]

    cmd = [
        sys.executable, convert_script,
        str(model_dir),
        '--outfile', str(convert_output),
        '--outtype', convert_outtype,
    ]

    try:
        _log(f"Exporting GGUF: {output_path}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if quant_cmd:
            _log(f"Quantizing GGUF: {quant_type.upper()}")
            subprocess.run(quant_cmd, check=True, capture_output=True,
                           text=True)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        _log(f"GGUF exported: {output_path} ({size_mb:.0f} MB)")
        return True
    except subprocess.CalledProcessError as e:
        _warn(f"GGUF export failed: {e.stderr[:500] if e.stderr else e}")
        return False
    except FileNotFoundError as e:
        _warn(f"GGUF export failed (command not found): {e}")
        return False
    finally:
        if temp_output and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError:
                pass


# ─── Training ────────────────────────────────────────────────────────────────

FINETUNING_CONTAINER = 'strix-halo-finetuning'


def _check_finetuning_container():
    """Warn if not running inside the fine-tuning container.

    The strix-halo-finetuning container has gfx1151-compiled PyTorch from
    AMD's TheRock nightly.  Standard PyTorch ROCm wheels segfault on
    .cuda() for Strix Halo gfx1151.
    """
    # Detect ROCm environment
    try:
        import torch
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    except ImportError:
        return  # torch not available yet, skip check

    if not is_rocm:
        return  # CUDA or CPU — no container needed

    # Inside the container: hostname is 'finetuning' and /opt/venv exists
    hostname = os.environ.get('HOSTNAME', '')
    has_opt_venv = os.path.isdir('/opt/venv')

    if hostname == 'finetuning' or has_opt_venv:
        return  # We're inside the fine-tuning container — all good

    # Not inside the container — warn loudly
    print(f"\n{'!' * 60}")
    print(f"  WARNING: ROCm detected but not inside {FINETUNING_CONTAINER}")
    print(f"  GPU training will likely SEGFAULT on Strix Halo (gfx1151).")
    print(f"")
    print(f"  Enter the fine-tuning container first:")
    print(f"    podman start {FINETUNING_CONTAINER}")
    print(f"    podman exec -it {FINETUNING_CONTAINER} bash")
    print(f"    # Then inside the container (bash-5.3$ prompt):")
    print(f"    source /opt/venv/bin/activate")
    print(f"    cd /mnt/data/DeepRedAI")
    print(f"    python3 scripts/train_deepred_model.py ...")
    print(f"")
    print(f"  Or as a one-liner:")
    print(f"    podman exec {FINETUNING_CONTAINER} bash -c \\")
    print(f"      'source /opt/venv/bin/activate && "
          f"cd /mnt/data/DeepRedAI && "
          f"python3 scripts/train_deepred_model.py ...'")
    print(f"{'!' * 60}\n")

    resp = input("Continue anyway? [y/N] ").strip().lower()
    if resp != 'y':
        sys.exit(1)


def train(args):
    """Main training function."""
    global _shutdown_requested

    # ── Resolve paths ──
    root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    paths = PATH_TEMPLATES[args.model_name]

    model_path = args.model_path or paths['model'].format(root=root)
    corpus_dir = args.corpus_dir or paths['corpus'].format(root=root)
    tokenizer_path = paths['tokenizer'].format(root=root)

    train_bin = Path(corpus_dir) / 'train.bin'
    val_bin = Path(corpus_dir) / 'val.bin'

    # ── Validate prerequisites ──
    _check_finetuning_container()

    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print(f"Download it with setup_strixhalo.py or manually:")
        print(f"  hf download HuggingFaceTB/{args.model_name} "
              f"--local-dir {model_path}")
        sys.exit(1)

    if not train_bin.exists() or not val_bin.exists():
        pct = max(1, int(args.data_percent))
        print(f"ERROR: Tokenized corpus not found at {corpus_dir}")
        print(f"Create it first:")
        print(f"  source deepred-env.sh")
        print(f"  python3 scripts/create_training_corpus.py "
              f"--tokenizer {args.model_name} --percent {pct}")
        print(f"  python3 scripts/create_training_corpus.py "
              f"--tokenizer {args.model_name} --finalize")
        sys.exit(1)

    # ── Run orchestration ──
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.resume:
        # Explicit --resume: skip run orchestration
        run_name = None
        if not args.output_dir:
            output_dir = Path(args.resume).parent
        else:
            output_dir = Path(args.output_dir)
    else:
        run_name, output_dir, auto_resume = resolve_run(args, root)
        if auto_resume:
            args.resume = auto_resume
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──
    log = logging.getLogger('deepred')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(output_dir / 'train.log')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device('cuda')
        is_rocm = (hasattr(torch.version, 'hip')
                    and torch.version.hip is not None)
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024 ** 3)
            log.info(f"GPU {i}: {props.name} ({mem_gb:.1f} GB)"
                     f" {'[ROCm/HIP]' if is_rocm else '[CUDA]'}")
        device_type = 'cuda'
        # Estimate peak TFLOPS for MFU calculation
        gpu_name = torch.cuda.get_device_properties(0).name.lower()
        if 'gfx1' in gpu_name or 'strix' in gpu_name or 'radeon' in gpu_name:
            peak_tflops = PEAK_TFLOPS['strix_halo']
        elif 'a4000' in gpu_name:
            peak_tflops = PEAK_TFLOPS['a4000']
        else:
            peak_tflops = PEAK_TFLOPS['default']
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        is_rocm = False
        n_gpus = 0
        peak_tflops = 1.0
        log.warning("No GPU detected — training on CPU (very slow)")

    # ── Seed ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load model ──
    model_load_path = args.resume if args.resume else model_path
    log.info(f"Loading model from {model_load_path}")

    # Attention implementation: 'sdpa' is fastest but may segfault on some
    # ROCm configurations. Fall back to 'eager' if needed.
    attn_impl = args.attn_implementation
    if attn_impl == 'auto':
        # On ROCm/gfx1151 (Strix Halo), SDPA can segfault — default to eager.
        # On CUDA, SDPA is generally safe and faster.
        if (hasattr(torch.version, 'hip') and torch.version.hip is not None):
            attn_impl = 'eager'
        else:
            attn_impl = 'sdpa'

    load_kwargs = dict(
        dtype=torch.float32,             # FP32 master weights for stability
        trust_remote_code=True,
    )
    if attn_impl:
        load_kwargs['attn_implementation'] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, **load_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {args.model_name} ({n_params:,} parameters, "
             f"{n_params * 4 / 1e9:.2f} GB in FP32)")

    # Gradient checkpointing (trades compute for memory)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        log.info("Gradient checkpointing: enabled")

    log.info(f"Moving model to {device} (attn={attn_impl})...")
    try:
        # Verify GPU is functional before committing
        if device.type == 'cuda':
            _test = torch.tensor([1.0], device=device)
            del _test
        model.to(device)
    except Exception as e:
        if attn_impl == 'sdpa' and device.type == 'cuda':
            log.warning(f"model.to({device}) failed with SDPA: {e}")
            log.warning("Retrying with eager attention...")
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path, dtype=torch.float32,
                trust_remote_code=True, attn_implementation='eager')
            if not args.no_gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False})
            model.to(device)
        elif device.type == 'cuda':
            log.error(f"GPU operations failed: {e}")
            log.error("Falling back to CPU. This will be very slow.")
            log.error("If using ROCm on Strix Halo, ensure you are running "
                      "inside the strix-halo-finetuning container "
                      "(podman exec -it strix-halo-finetuning bash).")
            device = torch.device('cpu')
            device_type = 'cpu'
            peak_tflops = 1.0
            model.to(device)
        else:
            raise
    log.info("Model loaded on device")

    # Optional torch.compile
    if args.compile:
        log.info("Compiling model with torch.compile (this may take a few minutes)...")
        model = torch.compile(model)
        log.info("Compilation complete")

    # ── Load tokenizer ──
    log.info(f"Loading tokenizer from {model_load_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_load_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer: vocab_size={tokenizer.vocab_size}, "
             f"eos={tokenizer.eos_token_id}")

    # ── Load data ──
    log.info(f"Loading training data from {corpus_dir}")

    total_train_tokens = os.path.getsize(train_bin) // 2  # uint16 = 2 bytes
    total_train_seqs = total_train_tokens // SEQ_LENGTH
    if args.data_percent < 100.0:
        max_train_seqs = max(1, int(total_train_seqs * args.data_percent
                                    / 100.0))
    else:
        max_train_seqs = None

    train_dataset = PreTokenizedDataset(str(train_bin), SEQ_LENGTH,
                                        max_sequences=max_train_seqs)
    val_dataset = PreTokenizedDataset(str(val_bin), SEQ_LENGTH)

    log.info(f"Train: {train_dataset.n_sequences:,} sequences "
             f"({train_dataset.token_count():,} tokens)")
    log.info(f"Val:   {val_dataset.n_sequences:,} sequences "
             f"({val_dataset.token_count():,} tokens)")
    if args.data_percent < 100.0:
        log.info(f"Using {args.data_percent}% of full corpus "
                 f"({total_train_seqs:,} total sequences)")

    # DataLoader
    num_workers = (args.num_workers if args.num_workers >= 0
                   else min(os.cpu_count() or 4, 8))
    log.info(f"DataLoader workers: {num_workers}")

    # ── Training dimensions ──
    batches_per_epoch = (train_dataset.n_sequences
                         // args.micro_batch_size)
    steps_per_epoch = batches_per_epoch // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    if args.max_steps and args.max_steps < total_steps:
        total_steps = args.max_steps

    eff_batch = args.micro_batch_size * args.gradient_accumulation_steps
    tokens_per_step = eff_batch * SEQ_LENGTH
    total_tokens_budget = total_steps * tokens_per_step

    log.info(f"Epochs: {args.epochs}")
    log.info(f"Micro-batch: {args.micro_batch_size} | "
             f"Grad accum: {args.gradient_accumulation_steps} | "
             f"Effective batch: {eff_batch} seqs ({tokens_per_step:,} tokens)")
    log.info(f"Batches/epoch: {batches_per_epoch:,} | "
             f"Optimizer steps/epoch: {steps_per_epoch:,}")
    log.info(f"Total optimizer steps: {total_steps:,}")
    log.info(f"Total training tokens: {total_tokens_budget:,} "
             f"({total_tokens_budget / 1e9:.2f}B)")
    log.info(f"LR: {args.lr} → {args.min_lr} (cosine) | "
             f"Warmup: {args.warmup_steps} steps")
    log.info(f"Weight decay: {args.weight_decay} | "
             f"Max grad norm: {args.max_grad_norm}")

    # Estimated time
    # Baseline: ~1,200 tok/s measured on Strix Halo (Radeon 8060S,
    # ROCm, batch=8, eager attn, fused AdamW).
    est_tps = 1200 if 'SmolLM2' in args.model_name else 400
    est_hours = total_tokens_budget / (est_tps * 3600)
    est_completion = datetime.now() + timedelta(hours=est_hours)
    comp_date = est_completion.strftime('%a, %-m/%-d/%Y')
    comp_time = est_completion.strftime('%-I%p').lower()
    log.info(f"Estimated time: ~{est_hours:.1f} hours ({est_hours/24:.1f} days)"
             f" at ~{est_tps:,} tok/s")
    log.info(f"Expected completion: {comp_date} {comp_time}")

    # ── Save config ──
    run_config = {
        'profile': args.profile,
        'model_name': args.model_name,
        'model_path': str(model_path),
        'corpus_dir': str(corpus_dir),
        'output_dir': str(output_dir),
        'n_parameters': n_params,
        'train_sequences': train_dataset.n_sequences,
        'val_sequences': val_dataset.n_sequences,
        'train_tokens': train_dataset.token_count(),
        'total_tokens_budget': total_tokens_budget,
        'data_percent': args.data_percent,
        'epochs': args.epochs,
        'lr': args.lr,
        'min_lr': args.min_lr,
        'warmup_steps': args.warmup_steps,
        'micro_batch_size': args.micro_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': eff_batch,
        'seq_length': SEQ_LENGTH,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'total_steps': total_steps,
        'gradient_checkpointing': not args.no_gradient_checkpointing,
        'compile': args.compile,
        'device': str(device),
        'device_type': device_type,
        'is_rocm': is_rocm,
        'n_gpus': n_gpus,
        'num_workers': num_workers,
        'seed': args.seed,
        'run_name': run_name,
        'timestamp': timestamp,
        'pytorch_version': torch.__version__,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    # ── Optimizer ──
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ('bias', 'norm', 'layernorm')):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    log.info(f"Optimizer groups: {len(decay_params)} decay params, "
             f"{len(no_decay_params)} no-decay params")

    use_fused = (device_type == 'cuda')
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    if use_fused:
        log.info("AdamW: using fused CUDA kernel")

    # ── Resume state ──
    start_step = 0
    start_epoch = 0
    start_batch_idx = 0
    best_val_loss = float('inf')

    if args.resume:
        log.info(f"Loading training state from {args.resume}")
        state = load_checkpoint(args.resume, optimizer, device)
        start_step = state['step']
        start_epoch = state['epoch']
        start_batch_idx = state['batch_idx']
        best_val_loss = state['best_val_loss']
        log.info(f"Resumed: step={start_step}, epoch={start_epoch}, "
                 f"batch={start_batch_idx}, best_val={best_val_loss:.4f}")

    # ── Open log files ──
    metrics_file = open(output_dir / 'metrics.jsonl', 'a')
    samples_file = open(output_dir / 'samples.log', 'a')

    # ── Initial evaluation ──
    if start_step == 0:
        log.info("Running initial evaluation...")
        val_loss, val_ppl = evaluate(model, val_dataset, device,
                                     args.micro_batch_size)
        log.info(f"Initial val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
        metrics_file.write(json.dumps({
            'type': 'eval', 'step': 0,
            'val_loss': val_loss, 'val_perplexity': val_ppl,
        }) + '\n')
        metrics_file.flush()

        # Initial samples
        samples = generate_samples(model, tokenizer, device)
        samples_file.write(f"\n{'=' * 60}\nStep 0 (initial)\n{'=' * 60}\n")
        for s in samples:
            samples_file.write(f"\nPrompt: {s['prompt']}\n")
            samples_file.write(f"Generation: {s['generation']}\n")
            samples_file.write(f"{'-' * 40}\n")
        samples_file.flush()

    # ── GPU memory after setup ──
    if device.type == 'cuda':
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        resv = torch.cuda.memory_reserved() / (1024 ** 3)
        log.info(f"GPU memory: {alloc:.1f} GB allocated, "
                 f"{resv:.1f} GB reserved")

    # ── Training loop ──
    log.info("=" * 60)
    log.info("Starting training")
    log.info("=" * 60)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    accumulated = 0
    step_loss_accum = 0.0
    t_start = time.time()
    t_last_log = t_start
    tokens_since_log = 0

    for epoch in range(start_epoch, args.epochs):
        log.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")

        # Create DataLoader with deterministic shuffling per epoch
        g = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            sampler=torch.utils.data.RandomSampler(train_dataset,
                                                    generator=g),
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

        for batch_idx, batch in enumerate(train_loader):
            # Skip batches when resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            # Check termination conditions
            if (args.max_steps and step >= args.max_steps) or _shutdown_requested:
                break

            input_ids = batch.to(device, non_blocking=True)

            # Forward + backward with BF16 autocast
            with torch.autocast(device_type=device_type,
                                dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()

            step_loss_accum += outputs.loss.item()
            accumulated += 1
            tokens_since_log += input_ids.numel()

            # ── Optimizer step ──
            if accumulated == args.gradient_accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

                # Update learning rate
                lr = cosine_lr(step, args.warmup_steps, total_steps,
                               args.lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                avg_step_loss = step_loss_accum / accumulated
                step_loss_accum = 0.0
                accumulated = 0
                step += 1
                tokens_seen = step * tokens_per_step

                # ── Periodic logging ──
                if step % args.log_interval == 0:
                    t_now = time.time()
                    elapsed = t_now - t_start
                    dt = t_now - t_last_log
                    tps = tokens_since_log / dt if dt > 0 else 0

                    # MFU estimate: 6 * N_params * tokens / dt / peak
                    mfu_tflops = (6 * n_params * tokens_since_log
                                  / dt / 1e12)
                    mfu_pct = mfu_tflops / peak_tflops * 100

                    progress = step / total_steps * 100
                    if step > start_step:
                        steps_done = step - start_step
                        eta_s = (total_steps - step) * elapsed / steps_done
                        eta = str(timedelta(seconds=int(eta_s)))
                    else:
                        eta = '?'

                    grad_norm_val = (grad_norm.item()
                                     if isinstance(grad_norm, torch.Tensor)
                                     else grad_norm)

                    log.info(
                        f"step {step:>7d}/{total_steps} ({progress:5.1f}%) | "
                        f"loss {avg_step_loss:.4f} | lr {lr:.2e} | "
                        f"grad {grad_norm_val:.2f} | "
                        f"{tps:,.0f} tok/s | "
                        f"MFU {mfu_pct:.0f}% | "
                        f"ETA {eta}"
                    )

                    metrics_file.write(json.dumps({
                        'type': 'train',
                        'step': step,
                        'epoch': epoch,
                        'loss': round(avg_step_loss, 6),
                        'lr': lr,
                        'grad_norm': round(grad_norm_val, 4),
                        'tokens_per_sec': round(tps, 1),
                        'mfu_percent': round(mfu_pct, 1),
                        'tokens_seen': tokens_seen,
                        'elapsed_sec': round(elapsed, 1),
                    }) + '\n')
                    metrics_file.flush()

                    t_last_log = t_now
                    tokens_since_log = 0

                # ── Periodic evaluation ──
                if step % args.eval_interval == 0:
                    val_loss, val_ppl = evaluate(
                        model, val_dataset, device, args.micro_batch_size)
                    log.info(f"  → val_loss={val_loss:.4f} | "
                             f"val_ppl={val_ppl:.2f}")

                    metrics_file.write(json.dumps({
                        'type': 'eval', 'step': step,
                        'val_loss': round(val_loss, 6),
                        'val_perplexity': round(val_ppl, 4),
                    }) + '\n')
                    metrics_file.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model_only(model, tokenizer, output_dir, 'best')
                        log.info(f"  → New best model "
                                 f"(val_loss={val_loss:.4f})")

                    # GPU memory
                    if device.type == 'cuda':
                        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                        log.info(f"  → GPU memory: {alloc:.1f} GB allocated")

                # ── Periodic sample generation ──
                if step % args.sample_interval == 0:
                    samples = generate_samples(model, tokenizer, device)
                    samples_file.write(
                        f"\n{'=' * 60}\n"
                        f"Step {step} | Epoch {epoch + 1} | "
                        f"Tokens {tokens_seen:,}\n"
                        f"{'=' * 60}\n"
                    )
                    for s in samples:
                        samples_file.write(f"\nPrompt: {s['prompt']}\n")
                        samples_file.write(
                            f"Generation: {s['generation']}\n")
                        samples_file.write(f"{'-' * 40}\n")
                    samples_file.flush()
                    log.info(f"  → {len(samples)} text samples "
                             f"written to samples.log")

                # ── Periodic checkpoint ──
                if step % args.save_interval == 0:
                    save_checkpoint(
                        model, tokenizer, optimizer,
                        step, epoch, batch_idx + 1, best_val_loss,
                        run_config, output_dir, 'latest')
                    save_model_only(model, tokenizer, output_dir,
                                    f'checkpoint-{step}')
                    log.info(f"  → Checkpoint saved at step {step}")

                # ── Shutdown check ──
                if _shutdown_requested:
                    log.info("Graceful shutdown: saving checkpoint...")
                    save_checkpoint(
                        model, tokenizer, optimizer,
                        step, epoch, batch_idx + 1, best_val_loss,
                        run_config, output_dir, 'latest')
                    log.info(f"Checkpoint saved at step {step}. Exiting.")
                    metrics_file.close()
                    samples_file.close()
                    sys.exit(0)

        # End of epoch — discard partial gradient accumulation
        if accumulated > 0:
            log.info(f"  Discarding {accumulated} partial micro-batches "
                     f"at epoch boundary")
            optimizer.zero_grad(set_to_none=True)
            step_loss_accum = 0.0
            accumulated = 0

        start_batch_idx = 0  # reset for next epoch

        # Check termination
        if args.max_steps and step >= args.max_steps:
            break

        log.info(f"Epoch {epoch + 1} complete (step {step})")

        # ── Epoch-end GGUF export ──
        if not args.no_gguf and run_name:
            epoch_ckpt_name = f'epoch-{epoch + 1}'
            epoch_ckpt_dir = save_model_only(
                model, tokenizer, output_dir, epoch_ckpt_name)
            gguf_dir = output_dir / 'gguf'
            gguf_path = gguf_dir / f'{run_name}-epoch{epoch + 1}.gguf'
            export_gguf(epoch_ckpt_dir, str(gguf_path),
                        llama_cpp_path=args.llama_cpp_path,
                        quant_type=args.gguf_quant, log=log)

    # ── Final evaluation and save ──
    log.info("=" * 60)
    log.info("Training complete — running final evaluation")

    val_loss, val_ppl = evaluate(model, val_dataset, device,
                                 args.micro_batch_size)
    log.info(f"Final val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    samples = generate_samples(model, tokenizer, device)
    samples_file.write(
        f"\n{'=' * 60}\nFINAL | Step {step}\n{'=' * 60}\n")
    for s in samples:
        samples_file.write(f"\nPrompt: {s['prompt']}\n")
        samples_file.write(f"Generation: {s['generation']}\n")
        samples_file.write(f"{'-' * 40}\n")
    samples_file.flush()

    # Save final checkpoint
    save_checkpoint(model, tokenizer, optimizer,
                    step, args.epochs, 0, best_val_loss,
                    run_config, output_dir, 'latest')
    save_model_only(model, tokenizer, output_dir, 'final')

    # ── Final GGUF export ──
    if not args.no_gguf and run_name:
        gguf_dir = output_dir / 'gguf'
        gguf_path = gguf_dir / f'{run_name}-final.gguf'
        final_model_dir = output_dir / 'final'
        export_gguf(str(final_model_dir), str(gguf_path),
                    llama_cpp_path=args.llama_cpp_path,
                    quant_type=args.gguf_quant, log=log)

    # Mark run as completed
    if run_name:
        mark_run_completed(output_dir)

    elapsed = time.time() - t_start
    tokens_seen = step * tokens_per_step
    avg_tps = tokens_seen / elapsed if elapsed > 0 else 0

    log.info("=" * 60)
    log.info(f"Steps completed:  {step:,}")
    log.info(f"Tokens processed: {tokens_seen:,} ({tokens_seen / 1e9:.2f}B)")
    log.info(f"Total time:       {timedelta(seconds=int(elapsed))}")
    log.info(f"Avg throughput:   {avg_tps:,.0f} tok/s")
    log.info(f"Final val loss:   {val_loss:.4f}")
    log.info(f"Best val loss:    {best_val_loss:.4f}")
    log.info(f"Val perplexity:   {val_ppl:.2f}")
    log.info(f"Output dir:       {output_dir}")
    if run_name:
        log.info(f"Run name:         {run_name}")
        gguf_dir = output_dir / 'gguf'
        if gguf_dir.exists():
            log.info(f"GGUF models:      {gguf_dir}")
    log.info("=" * 60)
    if run_name:
        log.info("")
        log.info(f"Run '{run_name}' is COMPLETE.")
        log.info(f"To start a new run:")
        log.info(f"  --new-run                  (auto-increment name)")
        log.info(f"  --run-name <custom-name>   (custom name)")
        log.info("")

    metrics_file.write(json.dumps({
        'type': 'final',
        'step': step,
        'val_loss': round(val_loss, 6),
        'val_perplexity': round(val_ppl, 4),
        'best_val_loss': round(best_val_loss, 6),
        'total_tokens': tokens_seen,
        'elapsed_seconds': round(elapsed, 1),
        'avg_tokens_per_sec': round(avg_tps, 1),
    }) + '\n')
    metrics_file.close()
    samples_file.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Deep Red — Continued Pre-Training (CPT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Dev mode (default): SmolLM2-360M, 5%% of corpus
  python3 scripts/train_deepred_model.py

  # Quick smoke test
  python3 scripts/train_deepred_model.py --data-percent 1 --max-steps 100

  # Production: TinyLlama-1.1B, full corpus
  python3 scripts/train_deepred_model.py --profile prod

  # Resume interrupted training
  python3 scripts/train_deepred_model.py --resume /path/to/output/latest
""")

    # Profile
    parser.add_argument(
        '--profile', choices=['dev', 'prod'], default='dev',
        help='Training profile with preset defaults (default: dev)')

    # Path overrides
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Override base model directory')
    parser.add_argument(
        '--corpus-dir', type=str, default=None,
        help='Override tokenized corpus directory (must contain '
             'train.bin + val.bin)')
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override output directory')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from a checkpoint directory (e.g., .../latest)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Peak learning rate')
    parser.add_argument('--min-lr', type=float, default=None,
                        help='Minimum learning rate (end of cosine decay)')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='LR warmup steps')
    parser.add_argument('--micro-batch-size', type=int, default=None,
                        help='Micro-batch size (sequences per forward pass)')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=None,
                        help='Gradient accumulation steps')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay for AdamW')
    parser.add_argument('--max-grad-norm', type=float, default=None,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--data-percent', type=float, default=None,
                        help='Percentage of training data to use '
                             '(default: 5 for dev, 100 for prod)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Max optimizer steps (overrides epochs)')

    # Performance
    parser.add_argument(
        '--compile', action='store_true',
        help='Use torch.compile() for potential speedup (experimental on '
             'ROCm)')
    parser.add_argument(
        '--no-gradient-checkpointing', action='store_true',
        help='Disable gradient checkpointing (uses more memory but faster)')
    parser.add_argument(
        '--num-workers', type=int, default=-1,
        help='DataLoader workers (-1 = auto, 0 = main process)')
    parser.add_argument(
        '--attn-implementation', type=str, default='auto',
        choices=['auto', 'sdpa', 'eager', 'flash_attention_2'],
        help='Attention implementation (default: auto = try sdpa first; '
             'use eager if sdpa segfaults on ROCm)')

    # Logging intervals
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Validation evaluation interval (steps)')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Checkpoint save interval (steps)')
    parser.add_argument('--log-interval', type=int, default=None,
                        help='Console/file log interval (steps)')
    parser.add_argument('--sample-interval', type=int, default=None,
                        help='Text sample generation interval (steps)')

    # Run orchestration
    parser.add_argument(
        '--run-name', type=str, default=None,
        help='Custom run name (default: {profile}-YYYY-MM-DD)')
    parser.add_argument(
        '--new-run', action='store_true',
        help='Start a new run even if a previous run with the same '
             'name is completed (auto-increments the name)')

    # GGUF export
    parser.add_argument(
        '--no-gguf', action='store_true',
        help='Disable GGUF model export at epoch boundaries')
    parser.add_argument(
        '--gguf-quant', type=str, default='q8_0',
        help='GGUF quantization type (default: q8_0)')
    parser.add_argument(
        '--llama-cpp-path', type=str, default=None,
        help='Path to llama.cpp directory (default: $DEEPRED_ROOT/llama.cpp)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Apply profile defaults for any arg not explicitly set
    profile = PROFILES[args.profile]
    args.model_name = profile['model_name']

    for key, default_val in profile.items():
        if key == 'model_name':
            continue
        attr = key.replace('-', '_')
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, default_val)

    # Print banner
    # Determine display run name for banner
    display_run = args.run_name or f"{args.profile}-{datetime.now().strftime('%Y-%m-%d')}"

    print(f"\n{'=' * 60}")
    print(f"  Deep Red CPT — {args.model_name} ({args.profile} profile)")
    print(f"  Run: {display_run}")
    print(f"{'=' * 60}\n")

    train(args)


if __name__ == '__main__':
    main()
