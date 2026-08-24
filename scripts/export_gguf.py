#!/usr/bin/env python3
"""
export_gguf.py — Post-training GGUF export for DeepRed runs (host-side).

GGUF conversion (``convert_hf_to_gguf.py`` + ``llama-quantize``) is decoupled
from training so it can run **outside** the ``strix-halo-finetuning`` container,
on the host where ``llama.cpp`` was built.  Running the host-built ``llama.cpp``
binaries *inside* the container fails with a GLIBC mismatch, e.g.::

    llama-quantize: /lib64/libm.so.6: version `GLIBC_2.43' not found
        (required by .../libllama.so.0)

Because of this, ``train_deepred_gemma.py`` should be run with ``--no-gguf
--no-snapshot-gguf`` inside the container.  The trainer still saves the final
model and every progress snapshot as HuggingFace directories; this script then
(re)exports them to GGUF on the host and updates ``run_meta.json`` in place.

It is safe to re-run: by default it skips artifacts whose GGUF already exists
and only converts what is still pending.

Usage (run on the HOST, not inside the container):

    cd /mnt/data/DeepRedAI && source deepred-env.sh

    # Export everything still pending for a run (final + snapshots):
    python3 scripts/export_gguf.py --run-name gemma-4b-temporal-v1-10d

    # Explicit output dir, force re-export, custom quant:
    python3 scripts/export_gguf.py \
        --output-dir /mnt/data/training_output/gemma-4b-temporal-v1-10d \
        --gguf-quant q4_k_m --all

    # Reclaim disk by removing the large HF snapshot dirs after success:
    python3 scripts/export_gguf.py --run-name <run> --cleanup-hf
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Reuse the proven export helper from the CPT script (do not modify it).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_deepred_model import export_gguf  # noqa: E402


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger('export_gguf')


def _resolve_output_dir(args):
    """Resolve the run's output directory from --output-dir or --run-name."""
    if args.output_dir:
        return Path(args.output_dir)
    if not args.run_name:
        return None
    root = os.environ.get('DEEPRED_ROOT', '/mnt/data')
    return Path(f"{root}/training_output/{args.run_name}")


def _load_meta(meta_path):
    with open(meta_path) as f:
        return json.load(f)


def _save_meta(meta, meta_path):
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def _fmt_size(path):
    try:
        return f"{os.path.getsize(path) / (1024 ** 3):.2f} GB"
    except OSError:
        return "?"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-name', default=None,
                   help='Run name; output dir resolves to '
                        '$DEEPRED_ROOT/training_output/<run-name>.')
    p.add_argument('--output-dir', default=None,
                   help='Explicit run output directory (overrides '
                        '--run-name).')
    p.add_argument('--model-dir', default=None,
                   help='Direct mode: HuggingFace model directory.')
    p.add_argument('--outfile', default=None,
                   help='Direct mode: target GGUF path.')
    p.add_argument('--quant', default=None,
                   help='Direct mode: GGUF quantization (for example Q4_K_M).')
    p.add_argument('--gguf-quant', default=None,
                   help='Quant type for the FINAL model GGUF (default: the '
                        "run's recorded snapshot quant, else q8_0). Snapshots "
                        'use their own recorded quant.')
    p.add_argument('--llama-cpp-path', default=None,
                   help='Path to the llama.cpp checkout (default: '
                        '$DEEPRED_ROOT/llama.cpp).')
    p.add_argument('--no-final', action='store_true',
                   help='Do not export the final model.')
    p.add_argument('--no-snapshots', action='store_true',
                   help='Do not export progress snapshots.')
    p.add_argument('--all', action='store_true',
                   help='Re-export even when the target GGUF already exists.')
    p.add_argument('--cleanup-hf', action='store_true',
                   help='Remove a snapshot HF directory after its GGUF '
                        'export succeeds (reclaims disk). Off by default.')
    args = p.parse_args()

    log = _setup_logging()

    direct = (args.model_dir, args.outfile, args.quant)
    if any(direct):
        if not all(direct):
            log.error('Direct mode requires --model-dir, --outfile, and --quant.')
            sys.exit(2)
        model_dir = Path(args.model_dir)
        if not model_dir.is_dir():
            log.error(f"Model dir not found: {model_dir}")
            sys.exit(1)
        outfile = Path(args.outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        ok = export_gguf(
            str(model_dir), str(outfile), llama_cpp_path=args.llama_cpp_path,
            quant_type=args.quant, log=log)
        sys.exit(0 if ok else 1)

    output_dir = _resolve_output_dir(args)
    if output_dir is None:
        log.error("Provide --run-name or --output-dir.")
        sys.exit(2)
    if not output_dir.exists():
        log.error(f"Output dir not found: {output_dir}")
        sys.exit(1)

    meta_path = output_dir / 'run_meta.json'
    if not meta_path.exists():
        log.error(f"run_meta.json not found in {output_dir}")
        sys.exit(1)
    meta = _load_meta(meta_path)

    run_name = meta.get('run_name') or output_dir.name
    gguf_dir = output_dir / 'gguf'
    default_quant = (
        args.gguf_quant
        or (meta.get('snapshot_config') or {}).get('gguf_quant')
        or 'q8_0'
    )

    log.info(f"Run: {run_name}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Final quant: {default_quant}")

    exported, skipped, failed = [], [], []

    # ── Final model ──
    if not args.no_final:
        final_dir = output_dir / 'final'
        final_gguf = gguf_dir / f"{run_name}-final.gguf"
        if not final_dir.exists():
            log.warning(f"Final model dir missing: {final_dir} (skipping)")
        elif final_gguf.exists() and not args.all:
            log.info(f"Final GGUF exists, skipping: {final_gguf.name} "
                     f"(use --all to re-export)")
            skipped.append(str(final_gguf))
        else:
            ok = export_gguf(str(final_dir), str(final_gguf),
                             llama_cpp_path=args.llama_cpp_path,
                             quant_type=default_quant, log=log)
            (exported if ok else failed).append(str(final_gguf))

    # ── Progress snapshots ──
    meta_dirty = False
    if not args.no_snapshots:
        for snap in meta.get('snapshots') or []:
            label = snap.get('label', '?')
            step = snap.get('step', '?')
            tag = f"{label}@step{step}"
            model_dir = Path(snap.get('model_dir', ''))
            gguf_path = Path(snap.get('gguf_path', ''))
            quant = snap.get('gguf_quant') or default_quant

            if gguf_path.exists() and not args.all:
                log.info(f"Snapshot {tag}: GGUF exists, skipping "
                         f"({gguf_path.name}).")
                if snap.get('export_status') != 'ok':
                    snap['export_status'] = 'ok'
                    meta_dirty = True
                skipped.append(str(gguf_path))
                continue
            if not model_dir.exists():
                log.warning(f"Snapshot {tag}: HF dir missing ({model_dir}); "
                            f"cannot export. Skipping.")
                failed.append(str(gguf_path))
                continue

            ok = export_gguf(str(model_dir), str(gguf_path),
                             llama_cpp_path=args.llama_cpp_path,
                             quant_type=quant, log=log)
            snap['export_status'] = 'ok' if ok else 'failed'
            snap['exported_at'] = datetime.now().isoformat()
            meta_dirty = True
            if ok:
                exported.append(str(gguf_path))
                if args.cleanup_hf:
                    try:
                        shutil.rmtree(model_dir)
                        snap['model_dir_removed'] = True
                        log.info(f"Removed HF snapshot dir: {model_dir}")
                    except OSError as e:
                        snap['model_dir_removed'] = False
                        snap['cleanup_error'] = str(e)
                        log.warning(f"Cleanup failed for {tag}: {e}")
            else:
                failed.append(str(gguf_path))

    if meta_dirty:
        _save_meta(meta, meta_path)

    # ── Summary ──
    log.info("=" * 60)
    log.info(f"Export complete: {len(exported)} exported, "
             f"{len(skipped)} skipped, {len(failed)} failed")
    for g in exported:
        log.info(f"  exported: {g} ({_fmt_size(g)})")
    for g in failed:
        log.warning(f"  FAILED:   {g}")

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
