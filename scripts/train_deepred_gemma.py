#!/usr/bin/env python3
"""
train_deepred_gemma.py — SFT fine-tuning for Gemma-3-4B-IT / Gemma-3-12B-IT
on Strix Halo (AMD gfx1151) via TRL ``SFTTrainer``.

This script runs **in parallel** to ``train_deepred_model.py`` (which does
continued pre-training on a packed uint16 corpus).  It mirrors the proven
setup from https://github.com/kyuz0/amd-strix-halo-llm-finetuning:

  - HuggingFace + TRL SFTTrainer
  - bf16 weights, attn_implementation="eager" (required for Gemma)
  - adamw_torch_fused optimizer
  - Chat-format dataset (``{"messages":[...]}`` per line)

Prerequisites:
  - Run inside the ``strix-halo-finetuning`` podman container (gfx1151
    PyTorch from TheRock; the host venv segfaults on .cuda()).
  - Models downloaded with ``download_gemma_models.py``.
  - SFT dataset built with ``build_sft_dataset.py``.

Usage:
  # Smoke test (1 epoch, 5 steps, no GGUF)
  python3 scripts/train_deepred_gemma.py --profile gemma-4b \\
      --dataset-dir /mnt/data/sft_corpus/smoke \\
      --epochs 1 --max-steps 5 --no-gguf --debug

  # Full 4B run
  python3 scripts/train_deepred_gemma.py --profile gemma-4b \\
      --dataset-dir /mnt/data/sft_corpus/v1

  # 12B run (slower, requires gradient checkpointing — enabled by default)
  python3 scripts/train_deepred_gemma.py --profile gemma-12b \\
      --dataset-dir /mnt/data/sft_corpus/v1

  # Resume (auto: re-run the same command with the same --run-name; or:)
  python3 scripts/train_deepred_gemma.py --profile gemma-4b \\
      --dataset-dir /mnt/data/sft_corpus/v1 \\
      --resume /mnt/data/training_output/gemma-4b-2026-05-21/checkpoint-XXXX

  # Document a finished run (source model + parameters + results)
  python3 scripts/train_deepred_gemma.py --profile gemma-4b --summary
  python3 scripts/train_deepred_gemma.py \\
      --output-dir /mnt/data/training_output/gemma-4b-2026-05-21 \\
      --summary --summary-file run-summary.md
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse helpers from the existing CPT script (do not modify that script)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reduce allocator fragmentation on the unified-memory GPU. Variable
# sequence lengths in SFT cause the caching allocator to accumulate
# reserved-but-unallocated blocks that can push us over the GTT cap on
# the next long-sequence batch (observed: ~7.5 GiB stranded → OOM at
# step 11 on the 4B profile). Must be set BEFORE `import torch`.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Unsloth MUST be imported before transformers/trl/peft to install its
# kernel patches; otherwise it warns and silently disables optimizations.
# Sniff the CLI early (argparse hasn't run yet) so this stays opt-in.
if '--unsloth' in sys.argv:
    os.environ.setdefault('UNSLOTH_SKIP_TORCHVISION_CHECK', '1')
    try:
        import unsloth  # noqa: F401,E402
    except ImportError:
        # Defer the friendly error message to train() where we have
        # logging set up; here we just let the later import fail loudly.
        pass

from train_deepred_model import (  # noqa: E402
    _check_finetuning_container,
    compute_run_fingerprint,
    export_gguf,
    mark_run_completed,
)

# Heavy ML imports last so --help is fast
import torch  # noqa: E402


# ─── Profiles ────────────────────────────────────────────────────────────

PROFILES = {
    'gemma-4b': {
        'model_id':         'google/gemma-3-4b-it',
        'model_dirname':    'gemma-3-4b-it',
        'batch_size':       4,
        'grad_accum':       4,
        'lr':               5e-5,
        'epochs':           2,
        'max_length':       2048,
        'gradient_checkpointing': False,
    },
    'gemma-12b': {
        'model_id':         'google/gemma-3-12b-it',
        'model_dirname':    'gemma-3-12b-it',
        'batch_size':       1,
        'grad_accum':       16,
        'lr':               2e-5,
        'epochs':           2,
        'max_length':       2048,
        'gradient_checkpointing': True,
    },
}

# Fields that define a run identity (changing any → new run name needed)
RUN_DEFINING_PARAMS = [
    'profile', 'model_id', 'epochs', 'lr', 'batch_size', 'grad_accum',
    'max_length', 'gradient_checkpointing', 'lr_scheduler_type',
    'warmup_steps', 'seed', 'dataset_dir', 'unsloth', 'type',
]

# LoRA hyperparameters — match kyuz0 reference for Gemma-3.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
]


# ─── Helpers ─────────────────────────────────────────────────────────────

def resolve_paths(args):
    root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    models_root = os.environ.get('DEEPRED_MODELS', f"{root}/models")
    prof = PROFILES[args.profile]
    model_path = (args.model
                  or str(Path(models_root) / prof['model_dirname']))
    dataset_dir = Path(args.dataset_dir)
    return root, model_path, dataset_dir


def resolve_run(args, root):
    """Resolve run name + output dir, with fingerprint-based auto-resume.

    Mirrors the orchestration logic in ``train_deepred_model.py`` but is
    a separate copy because the SFT script tracks slightly different
    parameters (e.g. ``dataset_dir``) and uses HF Trainer's
    ``checkpoint-NNNN`` directories rather than a custom ``latest/`` dir.
    """
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.profile}-{datetime.now().strftime('%Y-%m-%d')}"

    output_dir = Path(
        args.output_dir or f"{root}/training_output/{run_name}")
    meta_path = output_dir / 'run_meta.json'
    fingerprint = compute_run_fingerprint_local(args)

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        # --new-run always wins: auto-increment regardless of the existing
        # run's status. Otherwise a killed/crashed run (status='running')
        # would trip the fingerprint guard below even when the user has
        # explicitly asked for a fresh attempt with new parameters.
        if args.new_run:
            base = run_name
            for i in range(2, 100):
                run_name = f"{base}-{i}"
                output_dir = Path(f"{root}/training_output/{run_name}")
                meta_path = output_dir / 'run_meta.json'
                if not output_dir.exists():
                    break
            # Fall through to the "fresh run" path below.
            meta = None  # type: ignore[assignment]

        elif meta.get('status') == 'completed':
            print(f"\nRun '{meta.get('run_name', run_name)}' is COMPLETED")
            print(f"  Finished: {meta.get('completed_at', '?')}")
            print(f"  Output:   {output_dir}")
            print("\nUse --new-run to auto-increment or --run-name <name>.")
            sys.exit(0)
        else:
            if meta.get('fingerprint') != fingerprint:
                print(f"\nERROR: Run '{run_name}' exists with different "
                      f"parameters.")
                old = meta.get('params', {})
                for k in RUN_DEFINING_PARAMS:
                    if old.get(k) != getattr(args, k):
                        print(f"  {k}: {old.get(k)} -> {getattr(args, k)}")
                print("\nUse --new-run to auto-increment or "
                      "--run-name <name> for a fresh run.")
                sys.exit(1)
            # Same fingerprint — let HF Trainer auto-detect latest ckpt
            latest = _latest_checkpoint(output_dir)
            if latest:
                print(f"\nResuming '{run_name}' from {latest}")
                return run_name, output_dir, latest, fingerprint
            return run_name, output_dir, None, fingerprint

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'run_name': run_name,
        'status': 'running',
        'fingerprint': fingerprint,
        'params': {k: getattr(args, k) for k in RUN_DEFINING_PARAMS},
        'started_at': datetime.now().isoformat(),
        'profile': args.profile,
        'model_id': args.model_id,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return run_name, output_dir, None, fingerprint


def compute_run_fingerprint_local(args):
    """Hash of run-defining params (separate from the CPT script's set)."""
    import hashlib
    params = {k: getattr(args, k) for k in RUN_DEFINING_PARAMS}
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _latest_checkpoint(output_dir):
    """Return path to the most recent ``checkpoint-N`` dir, or None."""
    if not output_dir.exists():
        return None
    candidates = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith('checkpoint-'):
            try:
                step = int(child.name.split('-', 1)[1])
                candidates.append((step, str(child)))
            except ValueError:
                pass
    return max(candidates)[1] if candidates else None


def _update_run_meta(output_dir, updates):
    """Merge *updates* into ``run_meta.json`` (created if missing)."""
    meta_path = Path(output_dir) / 'run_meta.json'
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = {}
    meta.update(updates)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ─── Post-training summary ───────────────────────────────────────────────

def _fmt_bytes(n):
    """Human-readable byte size."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _dir_size(path):
    """Total size in bytes of all files under *path* (0 if missing)."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob('*'):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _read_trainer_state(output_dir):
    """Return parsed ``trainer_state.json`` from final/ or latest ckpt, or {}.

    HF Trainer writes this file with the full ``log_history`` (loss curve,
    learning rate, eval metrics) plus ``global_step`` / ``epoch``.
    """
    for cand in (Path(output_dir) / 'final' / 'trainer_state.json',):
        if cand.exists():
            try:
                with open(cand) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    latest = _latest_checkpoint(Path(output_dir))
    if latest:
        cand = Path(latest) / 'trainer_state.json'
        if cand.exists():
            try:
                with open(cand) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def summarize_run(output_dir, out_file=None):
    """Render a Markdown summary of a completed (or in-progress) SFT run.

    Reads ``run_meta.json`` for the source model and training parameters,
    ``trainer_state.json`` for loss/step metrics, and inspects the output
    directory for produced artifacts (final model, merged weights, GGUF
    exports, checkpoints).  Everything is best-effort: missing pieces are
    simply omitted so the summary works both post-training and mid-run.
    """
    output_dir = Path(output_dir)
    meta_path = output_dir / 'run_meta.json'
    if not meta_path.exists():
        print(f"ERROR: no run_meta.json found in {output_dir}")
        print("  Point --output-dir / --run-name at a training run "
              "directory.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    params  = meta.get('params', {})
    results = meta.get('results', {})
    state   = _read_trainer_state(output_dir)

    L = []
    def emit(line=''):
        L.append(line)

    def get(key, default='?'):
        """Prefer top-level meta, fall back to params block."""
        if key in meta and meta[key] is not None:
            return meta[key]
        return params.get(key, default)

    run_name = meta.get('run_name', output_dir.name)
    status   = meta.get('status', 'unknown')

    emit(f"# DeepRed SFT Run Summary — {run_name}")
    emit()
    emit(f"- **Status:** {status}")
    emit(f"- **Output dir:** {output_dir}")
    emit(f"- **Started:** {meta.get('started_at', '?')}")
    emit(f"- **Completed:** {meta.get('completed_at', '(not finished)')}")
    if meta.get('fingerprint'):
        emit(f"- **Fingerprint:** {meta['fingerprint']}")
    emit()

    # ── Source model ──
    emit("## Source Model")
    emit()
    emit(f"- **Profile:** {get('profile')}")
    emit(f"- **Base model:** {get('model_id')}")
    if meta.get('model_path'):
        emit(f"- **Local path:** {meta['model_path']}")
    train_type = get('type', 'full')
    emit(f"- **Training mode:** {train_type}")
    if train_type == 'lora':
        emit(f"- **LoRA:** r={LORA_R}, alpha={LORA_ALPHA}, "
             f"dropout={LORA_DROPOUT}")
    if params.get('unsloth'):
        emit("- **Unsloth:** enabled")
    emit()

    # ── Training parameters ──
    emit("## Training Parameters")
    emit()
    bs    = get('batch_size')
    ga    = get('grad_accum')
    eff   = (bs * ga if isinstance(bs, int) and isinstance(ga, int)
             else '?')
    emit("| Parameter | Value |")
    emit("|-----------|-------|")
    emit(f"| Epochs | {get('epochs')} |")
    emit(f"| Batch size | {bs} |")
    emit(f"| Grad accumulation | {ga} |")
    emit(f"| Effective batch | {eff} |")
    emit(f"| Learning rate | {get('lr')} |")
    emit(f"| LR scheduler | {get('lr_scheduler_type')} |")
    emit(f"| Warmup steps | {get('warmup_steps')} |")
    emit(f"| Max sequence length | {get('max_length')} |")
    emit(f"| Gradient checkpointing | {get('gradient_checkpointing')} |")
    emit(f"| Seed | {get('seed')} |")
    if params.get('dataset_dir'):
        emit(f"| Dataset | {params['dataset_dir']} |")
    emit()

    # ── Results ──
    have_results = bool(results) or bool(state)
    if have_results:
        emit("## Results")
        emit()
        if results.get('duration_seconds') is not None:
            secs = results['duration_seconds']
            emit(f"- **Duration:** {secs / 3600:.2f} h ({secs:.0f} s)")
        if results.get('final_train_loss') is not None:
            emit(f"- **Final train loss:** "
                 f"{results['final_train_loss']:.4f}")
        if results.get('peak_gpu_gb') is not None:
            emit(f"- **Peak GPU memory:** {results['peak_gpu_gb']:.2f} GB")
        gs = state.get('global_step') or results.get('global_step')
        if gs is not None:
            emit(f"- **Global steps:** {gs:,}")
        ep = state.get('epoch') or results.get('epochs_completed')
        if ep is not None:
            emit(f"- **Epochs completed:** {ep:.2f}"
                 if isinstance(ep, float) else f"- **Epochs completed:** {ep}")

        # Last recorded eval loss from the log history, if present
        log_hist = state.get('log_history', [])
        eval_losses = [e['eval_loss'] for e in log_hist
                       if isinstance(e, dict) and 'eval_loss' in e]
        if eval_losses:
            emit(f"- **Last eval loss:** {eval_losses[-1]:.4f}")
            emit(f"- **Best eval loss:** {min(eval_losses):.4f}")
        emit()

    # ── Artifacts ──
    emit("## Artifacts")
    emit()
    final_dir  = output_dir / 'final'
    merged_dir = output_dir / 'final-merged'
    gguf_dir   = output_dir / 'gguf'

    if final_dir.exists():
        label = 'LoRA adapters' if train_type == 'lora' else 'Final model'
        emit(f"- **{label}:** `{final_dir}` ({_fmt_bytes(_dir_size(final_dir))})")
    if merged_dir.exists():
        emit(f"- **Merged weights:** `{merged_dir}` "
             f"({_fmt_bytes(_dir_size(merged_dir))})")
    if gguf_dir.exists():
        for g in sorted(gguf_dir.glob('*.gguf')):
            emit(f"- **GGUF:** `{g}` ({_fmt_bytes(g.stat().st_size)})")

    checkpoints = []
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir() and child.name.startswith('checkpoint-'):
                checkpoints.append(child.name)
    if checkpoints:
        checkpoints.sort(key=lambda n: int(n.split('-', 1)[1]))
        emit(f"- **Checkpoints:** {len(checkpoints)} "
             f"({checkpoints[0]} … {checkpoints[-1]})")
    emit()

    text = '\n'.join(L)
    if out_file:
        Path(out_file).write_text(text)
        print(f"Summary written to {out_file}")
    else:
        print(text)


# ─── Debug callback ──────────────────────────────────────────────────────

# ─── Memory monitoring ───────────────────────────────────────────────────
#
# Strix Halo's GPU memory is *unified* with host RAM (single 128 GB pool
# carved between the OS and the amdgpu GTT region). The kernel OOM killer
# (SIGKILL → bare "Killed" in the shell, no Python traceback) is the most
# common cause of an SFT run dying mid-step. To diagnose it after the fact
# we log host + GPU memory at every step *and* on a fixed wall-clock
# cadence to ``<output_dir>/memory.log`` — both files survive the kill.
#
# Sampling is cheap (a few syscalls + torch counters), so this is enabled
# unconditionally. Disable with --no-memory-monitor if it ever gets in the
# way.

def _read_meminfo():
    """Parse /proc/meminfo → dict of kB (Linux only; {} elsewhere)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            out = {}
            for line in f:
                k, _, v = line.partition(':')
                v = v.strip().split()
                if v:
                    out[k] = int(v[0])  # kB
            return out
    except OSError:
        return {}


def _read_self_rss_kb():
    """Resident set size of this process, kB (Linux only; 0 elsewhere)."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _mem_snapshot():
    """Return a dict of GPU + host memory counters (all in GB)."""
    snap = {}
    if torch.cuda.is_available():
        snap['gpu_alloc'] = torch.cuda.memory_allocated() / 1e9
        snap['gpu_reserved'] = torch.cuda.memory_reserved() / 1e9
        snap['gpu_peak'] = torch.cuda.max_memory_allocated() / 1e9
    mi = _read_meminfo()
    if mi:
        snap['mem_total'] = mi.get('MemTotal', 0) / 1e6
        snap['mem_avail'] = mi.get('MemAvailable', 0) / 1e6
        snap['mem_free'] = mi.get('MemFree', 0) / 1e6
        snap['swap_total'] = mi.get('SwapTotal', 0) / 1e6
        snap['swap_free'] = mi.get('SwapFree', 0) / 1e6
        snap['swap_used'] = snap['swap_total'] - snap['swap_free']
    snap['rss'] = _read_self_rss_kb() / 1e6
    return snap


def _format_snap(snap, prefix=''):
    parts = []
    if 'gpu_alloc' in snap:
        parts.append(f"gpu_alloc={snap['gpu_alloc']:.1f}GB")
        parts.append(f"gpu_reserved={snap['gpu_reserved']:.1f}GB")
        parts.append(f"gpu_peak={snap['gpu_peak']:.1f}GB")
    if 'mem_avail' in snap:
        parts.append(f"rss={snap['rss']:.1f}GB")
        parts.append(f"mem_avail={snap['mem_avail']:.1f}GB")
        parts.append(f"mem_free={snap['mem_free']:.1f}GB")
        parts.append(f"swap_used={snap['swap_used']:.1f}GB")
    return prefix + ' '.join(parts)


def start_memory_monitor(output_dir, interval_s=10):
    """Start a daemon thread sampling memory every ``interval_s`` seconds.

    Writes one CSV-ish line per sample to ``<output_dir>/memory.log``.
    Returns the thread + stop event (mostly for tests; the daemon thread
    exits with the process otherwise).
    """
    import threading
    log_path = Path(output_dir) / 'memory.log'
    stop = threading.Event()

    columns = ['ts', 'gpu_alloc_gb', 'gpu_reserved_gb', 'gpu_peak_gb',
               'rss_gb', 'mem_avail_gb', 'mem_free_gb',
               'swap_used_gb', 'swap_total_gb']

    def _run():
        with open(log_path, 'a', buffering=1) as f:
            if log_path.stat().st_size == 0:
                f.write(','.join(columns) + '\n')
            while not stop.is_set():
                s = _mem_snapshot()
                f.write(','.join([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{s.get('gpu_alloc', 0):.3f}",
                    f"{s.get('gpu_reserved', 0):.3f}",
                    f"{s.get('gpu_peak', 0):.3f}",
                    f"{s.get('rss', 0):.3f}",
                    f"{s.get('mem_avail', 0):.3f}",
                    f"{s.get('mem_free', 0):.3f}",
                    f"{s.get('swap_used', 0):.3f}",
                    f"{s.get('swap_total', 0):.3f}",
                ]) + '\n')
                stop.wait(interval_s)

    t = threading.Thread(target=_run, name='mem-monitor', daemon=True)
    t.start()
    return t, stop


def make_debug_callback(log=None, every=50):
    """Per-step device + host/GPU memory log.

    Always-on (cheap). Prints/logs every step for the first 3 steps and
    every ``every`` steps thereafter. If ``log`` is a logger, lines also
    go to ``train.log``; otherwise only to stdout.
    """
    from transformers import TrainerCallback

    def _emit(msg):
        if log is not None:
            log.info(msg)
        else:
            print(msg, flush=True)

    class DebugCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, model=None, **kwargs):
            if state.global_step > 3 and state.global_step % every != 0:
                return
            device = next(model.parameters()).device if model else '?'
            snap = _mem_snapshot()
            _emit(f"[step {state.global_step}] dev={device} "
                  + _format_snap(snap))

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step > 3 and state.global_step % every != 0:
                return
            snap = _mem_snapshot()
            _emit(f"[step {state.global_step}] DONE "
                  + _format_snap(snap))

    return DebugCallback()


# ─── Training ────────────────────────────────────────────────────────────

def train(args):
    prof = PROFILES[args.profile]
    args.model_id = prof['model_id']  # for fingerprint + meta

    # 1. Container check (skips on CUDA/CPU; warns on ROCm-outside-container)
    _check_finetuning_container()

    # 2. Paths + run orchestration
    root, model_path, dataset_dir = resolve_paths(args)
    if not Path(model_path).exists():
        print(f"ERROR: model not found at {model_path}")
        print("Download it with:")
        print(f"  python3 scripts/download_gemma_models.py "
              f"--model {prof['model_dirname']}")
        sys.exit(1)

    train_jsonl = dataset_dir / 'train.jsonl'
    val_jsonl = dataset_dir / 'val.jsonl'
    if not train_jsonl.exists() or not val_jsonl.exists():
        print(f"ERROR: dataset not found at {dataset_dir}")
        print("Build it with:")
        print("  python3 scripts/build_sft_dataset.py --tag v1")
        sys.exit(1)

    if args.resume:
        run_name = args.run_name
        output_dir = (Path(args.output_dir) if args.output_dir
                      else Path(args.resume).parent)
        resume_from = args.resume
        fingerprint = compute_run_fingerprint_local(args)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_name, output_dir, resume_from, fingerprint = resolve_run(
            args, root)

    # 3. Logging
    log = logging.getLogger('deepred-gemma')
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

    log.info(f"Profile          : {args.profile}")
    log.info(f"Model            : {prof['model_id']}  ({model_path})")
    log.info(f"Dataset          : {dataset_dir}")
    log.info(f"Output           : {output_dir}")
    log.info(f"Run name         : {run_name}")
    log.info(f"Fingerprint      : {fingerprint}")
    log.info(f"Epochs           : {args.epochs}")
    log.info(f"Batch / accum    : {args.batch_size} / {args.grad_accum} "
             f"(effective={args.batch_size * args.grad_accum})")
    log.info(f"LR / schedule    : {args.lr} / {args.lr_scheduler_type} "
             f"(warmup={args.warmup_steps})")
    log.info(f"Max length       : {args.max_length}")
    log.info(f"Grad checkpoint  : {args.gradient_checkpointing}")
    log.info(f"PyTorch          : {torch.__version__}")
    if torch.cuda.is_available():
        is_rocm = (hasattr(torch.version, 'hip')
                   and torch.version.hip is not None)
        backend = '[ROCm/HIP]' if is_rocm else '[CUDA]'
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log.info(f"GPU {i}            : {p.name} "
                     f"({p.total_memory / 1e9:.1f} GB) {backend}")
    else:
        log.warning("No GPU detected — training on CPU (impractical).")

    # Host memory snapshot at startup (Strix Halo GPU memory is unified
    # with system RAM; OOM kills look like a bare "Killed" in the shell).
    _startup_snap = _mem_snapshot()
    log.info("Host memory       : " + _format_snap(_startup_snap))
    if 'swap_total' in _startup_snap and _startup_snap['swap_total'] > 0:
        # zram swap uses RAM-backed compressed pages — under unified
        # memory pressure it competes with the GPU. Warn loudly.
        try:
            zram_active = any(
                Path(p).exists() for p in
                ('/sys/block/zram0', '/sys/block/zram1'))
        except OSError:
            zram_active = False
        if zram_active:
            log.warning("zram swap is active — on Strix Halo this can "
                        "amplify host memory pressure under heavy GPU "
                        "use. Consider: sudo swapoff /dev/zram0")

    # Background memory sampler (writes <output>/memory.log every 10 s).
    # Always-on; --no-memory-monitor disables it.
    _mem_stop = None
    if not args.no_memory_monitor:
        _, _mem_stop = start_memory_monitor(output_dir,
                                            interval_s=args.memory_interval)
        log.info(f"Memory monitor   : {output_dir / 'memory.log'} "
                 f"(every {args.memory_interval}s)")

    # 4. Imports (deferred so --help is snappy)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # 5. Dataset
    log.info("Loading dataset…")
    ds = load_dataset(
        'json',
        data_files={'train': str(train_jsonl), 'validation': str(val_jsonl)},
    )
    if 'messages' not in ds['train'].column_names:
        log.error(f"Dataset '{dataset_dir}' missing 'messages' column. "
                  "Build it with scripts/build_sft_dataset.py.")
        sys.exit(1)
    log.info(f"  train      : {len(ds['train']):,} examples")
    log.info(f"  validation : {len(ds['validation']):,} examples")

    # 6/7. Model + tokenizer — standard HF path or Unsloth fast path.
    #      Both use bf16 + eager attention (REQUIRED for Gemma soft-capping).
    if args.unsloth:
        log.info("Loading model via Unsloth FastLanguageModel "
                 "(bf16, full FT)…")
        # Must be set before importing unsloth
        os.environ.setdefault('UNSLOTH_SKIP_TORCHVISION_CHECK', '1')
        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
        except ImportError as e:
            log.error(f"--unsloth requires the unsloth package: {e}")
            log.error("Run inside the strix-halo-finetuning container, "
                      "which ships Unsloth pre-built for gfx1151.")
            sys.exit(1)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=args.max_length,
            dtype=None,             # auto → bf16 on ROCm
            load_in_4bit=False,
            # Tell Unsloth up-front which kernel path to take.  Without
            # full_finetuning=True it silently logs "QLoRA and full
            # finetuning all not selected. Switching to 16bit LoRA." and
            # installs LoRA-only fast kernels; subsequently calling
            # ``param.requires_grad_(True)`` re-enables grads on all
            # params but leaves the LoRA-shaped kernels in place, which
            # produces NaN loss from step 1 on Gemma-3 / gfx1151.
            full_finetuning=(args.type != 'lora'),
        )
        if args.type == 'lora':
            log.info(f"Wrapping with Unsloth LoRA adapters "
                     f"(r={LORA_R}, alpha={LORA_ALPHA})…")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias='none',
                task_type='CAUSAL_LM',
            )
        else:
            # Unsloth freezes parameters by default (for LoRA); full FT
            # needs them re-enabled.
            for param in model.parameters():
                param.requires_grad_(True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Pre-apply Gemma chat template to a 'text' column — Unsloth
        # patches the tokenizer with unpickleable closures, so SFTTrainer
        # cannot apply the template itself.
        tokenizer = get_chat_template(tokenizer, chat_template='gemma-3')

        def _apply_template(examples):
            texts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=False
                ).removeprefix('<bos>')
                for m in examples['messages']
            ]
            return {'text': texts}

        ds['train'] = ds['train'].map(_apply_template, batched=True)
        ds['validation'] = ds['validation'].map(_apply_template, batched=True)

        # Pre-tokenize too. SFTTrainer would otherwise call dataset.map()
        # with a tokenize_fn that closes over the Unsloth-patched
        # tokenizer; its globals reference torch._dynamo.config (a
        # ConfigModuleInstance) which dill cannot pickle, so even with
        # dataset_num_proc=1 the newer `datasets` worker-pool path
        # explodes. Providing an already-tokenized dataset bypasses
        # that .map() entirely.
        max_len = args.max_length

        # Gemma-3's "tokenizer" returned by get_chat_template is actually
        # a Gemma3Processor (multimodal wrapper). Its __call__ packs
        # input_ids into a numpy array, which fails on variable-length
        # sequences without padding. Use the underlying fast tokenizer
        # directly for text-only training.
        text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)

        def _tokenize(examples):
            out = text_tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )
            return out

        ds['train'] = ds['train'].map(
            _tokenize, batched=True,
            remove_columns=ds['train'].column_names)
        ds['validation'] = ds['validation'].map(
            _tokenize, batched=True,
            remove_columns=ds['validation'].column_names)
    else:
        log.info("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("Loading model (bf16, attn_implementation='eager')…")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation='eager',
            trust_remote_code=True,
        )
        if args.type == 'lora':
            from peft import LoraConfig, get_peft_model
            log.info(f"Wrapping with PEFT LoRA adapters "
                     f"(r={LORA_R}, alpha={LORA_ALPHA})…")
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias='none',
                task_type='CAUSAL_LM',
            )
            model = get_peft_model(model, lora_config)

    log.info(f"  parameters : {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"  footprint  : "
             f"{model.get_memory_footprint() / 1e9:.2f} GB")
    if args.type == 'lora' and hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    if args.gradient_checkpointing and not args.unsloth:
        # Unsloth manages its own checkpointing strategy; don't double-enable.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False

    # 8. SFTConfig
    sft_args = dict(
        output_dir=str(output_dir),
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {'use_reentrant': False}
            if args.gradient_checkpointing else None),
        optim='adamw_torch_fused',
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='epoch',
        save_total_limit=2,
        report_to='none',
        dataset_kwargs={'add_special_tokens': False,
                        'append_concat_token': True},
        # Gemma 12B has vision params unused in text-only training
        ddp_find_unused_parameters=True,
        seed=args.seed,
    )
    if args.max_steps:
        sft_args['max_steps'] = args.max_steps

    if args.unsloth:
        # Dataset is already tokenized above (see Unsloth branch). TRL
        # will detect 'input_ids' and skip its own .map() tokenization,
        # which is what we need to avoid the ConfigModuleInstance
        # pickling failure. dataset_num_proc=1 is still set as a belt-
        # and-braces measure for any internal map() calls.
        sft_args['dataset_num_proc'] = 1

    training_args = SFTConfig(**sft_args)

    # 9. Trainer
    callbacks = [make_debug_callback(log=log,
                                     every=50 if not args.debug else 1)]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # 10. Train
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log.info("=" * 60)
    log.info("Starting training")
    log.info("=" * 60)
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9
               if torch.cuda.is_available() else 0)

    log.info("-" * 60)
    log.info(f"Training complete  : {elapsed / 3600:.2f} h "
             f"({elapsed:.0f} s)")
    log.info(f"Peak GPU memory    : {peak_gb:.2f} GB")
    log.info(f"Final train loss   : "
             f"{train_result.metrics.get('train_loss', float('nan')):.4f}")

    # Persist results + resolved model path so `--summary` can report them.
    _update_run_meta(output_dir, {
        'model_path': str(model_path),
        'results': {
            'duration_seconds': round(elapsed, 1),
            'peak_gpu_gb': round(peak_gb, 3),
            'final_train_loss': train_result.metrics.get('train_loss'),
            'global_step': train_result.metrics.get('global_step')
                            or getattr(trainer.state, 'global_step', None),
            'epochs_completed': getattr(trainer.state, 'epoch', None),
        },
    })

    # 11. Final save + run metadata
    final_dir = output_dir / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if args.type == 'lora':
        log.info(f"LoRA adapters saved: {final_dir}")
    else:
        log.info(f"Final model saved  : {final_dir}")

    # 12. GGUF export (optional)
    if not args.no_gguf:
        # LoRA adapters cannot be exported directly to GGUF — merge into
        # the base weights first.  Save the merged model to a sibling dir
        # so the adapter checkpoint at ``final/`` is preserved.
        if args.type == 'lora':
            merged_dir = output_dir / 'final-merged'
            try:
                log.info("Merging LoRA adapters into base weights for "
                         "GGUF export…")
                merged = trainer.model.merge_and_unload()
                merged.save_pretrained(str(merged_dir),
                                       safe_serialization=True)
                tokenizer.save_pretrained(str(merged_dir))
                gguf_src = str(merged_dir)
                log.info(f"Merged model saved : {merged_dir}")
            except Exception as e:
                log.warning(f"LoRA merge failed: {e} — skipping GGUF export")
                gguf_src = None
        else:
            gguf_src = str(final_dir)

        if gguf_src:
            gguf_path = output_dir / 'gguf' / f"{run_name}-final.gguf"
            ok = export_gguf(gguf_src, str(gguf_path),
                             quant_type=args.gguf_quant, log=log)
            if ok:
                log.info(f"GGUF exported      : {gguf_path}")
            else:
                log.warning("GGUF export skipped/failed "
                            "(see warnings above)")

    # 13. Mark run complete
    mark_run_completed(output_dir)
    log.info(f"Run '{run_name}' marked completed")


# ─── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--profile', choices=list(PROFILES),
                   default='gemma-4b',
                   help='Profile preset (default: gemma-4b).')
    p.add_argument('--dataset-dir', required=False, default=None,
                   help='Directory containing train.jsonl / val.jsonl '
                        '(produced by build_sft_dataset.py). '
                        'Required for training; not used by --summary.')
    p.add_argument('--model', default=None,
                   help='Override local model path (default: '
                        '$DEEPRED_MODELS/<profile model dirname>).')

    # Hyperparameter overrides (default to profile values)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--grad-accum', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--max-length', type=int)
    p.add_argument('--gradient-checkpointing',
                   dest='gradient_checkpointing',
                   action='store_true', default=None)
    p.add_argument('--no-gradient-checkpointing',
                   dest='gradient_checkpointing',
                   action='store_false')

    p.add_argument('--lr-scheduler-type', default='cosine',
                   choices=['cosine', 'constant', 'linear',
                            'constant_with_warmup'],
                   help='LR schedule (default cosine; kyuz0 uses constant).')
    p.add_argument('--warmup-steps', type=int, default=100,
                   help='Warmup steps (default 100).')

    p.add_argument('--max-steps', type=int, default=0,
                   help='Hard cap on optimizer steps (0 = use epochs).')

    p.add_argument('--seed', type=int, default=42)

    # Run orchestration
    p.add_argument('--run-name', default=None,
                   help='Custom run name (default: <profile>-<YYYY-MM-DD>).')
    p.add_argument('--output-dir', default=None,
                   help='Override output dir '
                        '(default: $DEEPRED_ROOT/training_output/<run-name>).')
    p.add_argument('--new-run', action='store_true',
                   help='If existing run is completed, auto-increment name.')
    p.add_argument('--resume', default=None,
                   help='Explicit checkpoint dir to resume from.')

    # GGUF
    p.add_argument('--no-gguf', action='store_true',
                   help='Skip final GGUF export.')
    p.add_argument('--gguf-quant', default='q8_0',
                   help='llama.cpp quant type (default q8_0).')

    p.add_argument('--debug', action='store_true',
                   help='Per-step device/memory prints.')

    p.add_argument('--no-memory-monitor', action='store_true',
                   help='Disable the background host+GPU memory sampler '
                        '(default: enabled, writes memory.log every 10s).')
    p.add_argument('--memory-interval', type=int, default=10,
                   help='Memory monitor sampling interval in seconds '
                        '(default: 10).')

    p.add_argument('--unsloth', action='store_true',
                   help='Use Unsloth FastLanguageModel for ~2-3x speedup '
                        'and ~30%% lower peak memory. Requires the '
                        'strix-halo-finetuning container (ships a '
                        'gfx1151-patched Unsloth build). Works with both '
                        '--type full and --type lora.')

    p.add_argument('--type', choices=['full', 'lora'], default='full',
                   help='Training mode: "full" trains all weights, '
                        '"lora" trains low-rank adapters '
                        f'(r={LORA_R}, alpha={LORA_ALPHA}, target '
                        'modules: q/k/v/o + gate/up/down proj). '
                        'Default: full.')

    # Post-training documentation
    p.add_argument('--summary', action='store_true',
                   help='Print a Markdown summary of a finished/in-progress '
                        'run (source model + training parameters + results) '
                        'and exit. Resolves the run from --run-name / '
                        '--output-dir / --profile.')
    p.add_argument('--summary-file', default=None,
                   help='Write --summary Markdown to this path instead of '
                        'stdout.')

    args = p.parse_args()

    # Apply profile defaults for any unset hyperparameter
    prof = PROFILES[args.profile]
    if args.epochs is None:               args.epochs = prof['epochs']
    if args.batch_size is None:           args.batch_size = prof['batch_size']
    if args.grad_accum is None:           args.grad_accum = prof['grad_accum']
    if args.lr is None:                   args.lr = prof['lr']
    if args.max_length is None:           args.max_length = prof['max_length']
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = prof['gradient_checkpointing']

    return args


def _resolve_output_dir(args):
    """Resolve a run's output directory without side effects (for --summary)."""
    root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    if args.output_dir:
        return Path(args.output_dir)
    run_name = (args.run_name
                or f"{args.profile}-{datetime.now().strftime('%Y-%m-%d')}")
    return Path(f"{root}/training_output/{run_name}")


def main():
    args = parse_args()

    if args.summary:
        summarize_run(_resolve_output_dir(args), out_file=args.summary_file)
        return

    if not args.dataset_dir:
        print("ERROR: --dataset-dir is required for training.")
        sys.exit(1)

    train(args)


if __name__ == '__main__':
    main()
