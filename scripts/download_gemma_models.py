#!/usr/bin/env python3
"""
download_gemma_models.py — Fetch Gemma-3-IT weights from HuggingFace Hub.

Targets the Gemma-3 instruct models used by ``train_deepred_gemma.py``:

  google/gemma-3-4b-it    (~8 GB on disk)
  google/gemma-3-12b-it   (~24 GB on disk)

You must:
  1. Have a HuggingFace account.
  2. Accept the Gemma license on the model page (one-time, in browser):
        https://huggingface.co/google/gemma-3-4b-it
        https://huggingface.co/google/gemma-3-12b-it
  3. Provide an HF token (``--hf-token`` or ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN``).

Usage:
  python3 scripts/download_gemma_models.py --model gemma-3-4b-it
  python3 scripts/download_gemma_models.py --model gemma-3-12b-it
  python3 scripts/download_gemma_models.py --model both

Environment variables:
  DEEPRED_MODELS   Destination root (default: $DEEPRED_ROOT/models or /mnt/data/models)
  HF_TOKEN         HuggingFace access token (alternative to --hf-token)
"""

import argparse
import os
import sys
from pathlib import Path


MODELS = {
    'gemma-3-4b-it':  'google/gemma-3-4b-it',
    'gemma-3-12b-it': 'google/gemma-3-12b-it',
}

REQUIRED_FILES = ('config.json', 'tokenizer.json')


def models_root():
    root = os.environ.get('DEEPRED_MODELS')
    if root:
        return Path(root)
    return Path(os.environ.get('DEEPRED_ROOT', '/mnt/data')) / 'models'


def already_downloaded(dest):
    return all((dest / f).exists() for f in REQUIRED_FILES)


def download_one(short_name, repo_id, dest_root, token):
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    dest = dest_root / short_name
    if already_downloaded(dest):
        print(f"[skip] {short_name} already present at {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}  →  {dest}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            token=token,
            # Skip large optional files we don't need for training
            ignore_patterns=['*.gguf', '*.bin.index.json.lock',
                             '*.msgpack', '*.h5', '*flax*'],
        )
    except GatedRepoError:
        print(f"\nERROR: {repo_id} is a gated model.", file=sys.stderr)
        print("Accept the license in your browser, then retry:",
              file=sys.stderr)
        print(f"  https://huggingface.co/{repo_id}", file=sys.stderr)
        sys.exit(2)
    except HfHubHTTPError as e:
        print(f"\nERROR: HuggingFace HTTP error: {e}", file=sys.stderr)
        sys.exit(3)

    missing = [f for f in REQUIRED_FILES if not (dest / f).exists()]
    if missing:
        print(f"WARNING: missing expected files: {missing}", file=sys.stderr)
    else:
        print(f"[ok] verified {', '.join(REQUIRED_FILES)} present")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--model', required=True,
        choices=list(MODELS) + ['both'],
        help='Which model to download.')
    parser.add_argument(
        '--dest', default=None,
        help='Override destination root (default: $DEEPRED_MODELS).')
    parser.add_argument(
        '--hf-token', default=None,
        help='HuggingFace token (else uses HF_TOKEN / '
             'HUGGING_FACE_HUB_TOKEN env var, or cached login).')
    args = parser.parse_args()

    dest_root = Path(args.dest) if args.dest else models_root()
    dest_root.mkdir(parents=True, exist_ok=True)

    token = (args.hf_token
             or os.environ.get('HF_TOKEN')
             or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    if not token:
        # Fall back to cached hf auth login if present
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None
    if not token:
        print("WARNING: no HF token found. Gated models will fail.\n"
              "Provide one of:\n"
              "  --hf-token <TOKEN>\n"
              "  export HF_TOKEN=<TOKEN>\n"
              "  hf auth login\n", file=sys.stderr)

    targets = list(MODELS) if args.model == 'both' else [args.model]
    for short in targets:
        download_one(short, MODELS[short], dest_root, token)

    print(f"\nDone. Models under {dest_root}")


if __name__ == '__main__':
    main()
