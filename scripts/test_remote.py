#!/usr/bin/env python3
"""
Test connectivity and embedding consistency for a remote GPU inference server.

This script verifies that:
1. The REMOTE_HOST environment variable is set
2. The remote embedding server is reachable and returns model info
3. The remote LLM server is reachable and returns model info
4. Remote and local embedding servers produce identical results
   (cosine similarity ≈ 1.0 for the same input text)

Prerequisites:
    - Source deepred-env.sh (sets REMOTE_HOST, ports, and local endpoints)
    - Local embedding server running on EMBEDDING_PORT (default 1235)
    - Remote server running with embedding + LLM services

Usage:
    source /mnt/data/DeepRedAI/deepred-env.sh
    source $DEEPRED_VENV/bin/activate
    python3 scripts/test_remote.py

Environment Variables:
    REMOTE_HOST        Hostname or IP of the remote inference server (required)
    REMOTE_EMBED_PORT  Embedding server port on the remote host (default: 1235)
    REMOTE_LLM_PORT    LLM server port on the remote host (default: 1234)
    INFERENCE_HOST     Local embedding server host (default: localhost)
    EMBEDDING_PORT     Local embedding server port (default: 1235)
"""

import math
import os
import sys
import json

import requests

# =============================================================================
# Configuration from environment
# =============================================================================

REMOTE_HOST = os.environ.get('REMOTE_HOST', '')
REMOTE_EMBED_PORT = int(os.environ.get('REMOTE_EMBED_PORT', 1235))
REMOTE_LLM_PORT = int(os.environ.get('REMOTE_LLM_PORT', 1234))

LOCAL_EMBED_HOST = os.environ.get('INFERENCE_HOST', 'localhost')
LOCAL_EMBED_PORT = int(os.environ.get('EMBEDDING_PORT',
                                       os.environ.get('INFERENCE_PORT', 1235)))

# Embedding model alias (must match the --alias flag on both servers)
EMBED_MODEL = os.environ.get(
    'EMBED_MODEL',
    os.environ.get('EMBEDDING_MODEL',
                   'text-embedding-nomic-embed-text-v1.5@f16'))

# Test sentences for embedding comparison
TEST_SENTENCES = [
    "The Apollo 11 mission landed on the Moon on July 20, 1969.",
    "Quantum computing uses qubits that can exist in superposition states.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
]

# Batch-size stress test: mirrors process_and_index.py settings
# (EMBED_BATCH_SIZE=16, EMBED_MAX_CHARS=3072)
BATCH_TEST_COUNT = 16  # texts per batch (matches EMBED_BATCH_SIZE)
BATCH_TEST_CHARS = 3072  # max chars per text (matches EMBED_MAX_CHARS)

REQUEST_TIMEOUT = 15  # seconds
BATCH_REQUEST_TIMEOUT = 120  # seconds — larger payload needs more time

# =============================================================================
# Helpers
# =============================================================================

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_models_endpoint(label: str, host: str, port: int) -> bool:
    """Call /v1/models and print the result. Returns True on success."""
    url = f"http://{host}:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get('id', '?') for m in data.get('data', [])]
        print(f"  {PASS}  {label}: {url}")
        for m in models:
            print(f"         model: {m}")
        return True
    except requests.ConnectionError:
        print(f"  {FAIL}  {label}: {url} — connection refused")
        return False
    except requests.Timeout:
        print(f"  {FAIL}  {label}: {url} — timed out ({REQUEST_TIMEOUT}s)")
        return False
    except Exception as e:
        print(f"  {FAIL}  {label}: {url} — {e}")
        return False


def get_embeddings(host: str, port: int, texts: list[str]) -> list[list[float]] | None:
    """Request embeddings from an OpenAI-compatible /v1/embeddings endpoint."""
    url = f"http://{host}:{port}/v1/embeddings"
    payload = {"model": EMBED_MODEL, "input": texts}
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to guarantee order
        items = sorted(data.get('data', []), key=lambda x: x.get('index', 0))
        return [item['embedding'] for item in items]
    except Exception as e:
        print(f"  {FAIL}  Embedding request to {url} failed: {e}")
        return None


# =============================================================================
# Tests
# =============================================================================

def test_env_variable() -> bool:
    """Check that REMOTE_HOST is set."""
    print(f"\n{BOLD}1. Environment variable{RESET}")
    if not REMOTE_HOST:
        print(f"  {FAIL}  REMOTE_HOST is not set.")
        print("         Set it before sourcing deepred-env.sh:")
        print("           export REMOTE_HOST=\"A4000AI\"")
        print("           source /mnt/data/DeepRedAI/deepred-env.sh")
        return False
    print(f"  {PASS}  REMOTE_HOST = {REMOTE_HOST}")
    return True


def test_remote_availability() -> tuple[bool, bool]:
    """Check that the remote embedding and LLM servers respond."""
    print(f"\n{BOLD}2. Remote server availability{RESET}")
    embed_ok = check_models_endpoint(
        "Remote embedding", REMOTE_HOST, REMOTE_EMBED_PORT)
    llm_ok = check_models_endpoint(
        "Remote LLM", REMOTE_HOST, REMOTE_LLM_PORT)
    return embed_ok, llm_ok


def test_local_availability() -> bool:
    """Check that the local embedding server responds."""
    print(f"\n{BOLD}3. Local embedding server availability{RESET}")
    return check_models_endpoint(
        "Local embedding", LOCAL_EMBED_HOST, LOCAL_EMBED_PORT)


def test_embedding_comparison() -> bool:
    """Compare embeddings from local and remote servers."""
    print(f"\n{BOLD}4. Embedding comparison (local vs remote){RESET}")
    print(f"  Generating embeddings for {len(TEST_SENTENCES)} test sentence(s)...")

    local_vecs = get_embeddings(LOCAL_EMBED_HOST, LOCAL_EMBED_PORT, TEST_SENTENCES)
    if local_vecs is None:
        print(f"  {FAIL}  Could not retrieve local embeddings — skipping comparison")
        return False

    remote_vecs = get_embeddings(REMOTE_HOST, REMOTE_EMBED_PORT, TEST_SENTENCES)
    if remote_vecs is None:
        print(f"  {FAIL}  Could not retrieve remote embeddings — skipping comparison")
        return False

    if len(local_vecs) != len(remote_vecs):
        print(f"  {FAIL}  Vector count mismatch: local={len(local_vecs)}, "
              f"remote={len(remote_vecs)}")
        return False

    # Check dimensions
    local_dim = len(local_vecs[0]) if local_vecs else 0
    remote_dim = len(remote_vecs[0]) if remote_vecs else 0
    if local_dim != remote_dim:
        print(f"  {FAIL}  Dimension mismatch: local={local_dim}, remote={remote_dim}")
        return False
    print(f"  Dimensions: {local_dim}")

    all_ok = True
    threshold = 0.999  # expect near-perfect match for identical model + input

    for i, (lv, rv) in enumerate(zip(local_vecs, remote_vecs)):
        sim = cosine_similarity(lv, rv)
        sentence_preview = (TEST_SENTENCES[i][:60] + "..."
                            if len(TEST_SENTENCES[i]) > 60
                            else TEST_SENTENCES[i])
        if sim >= threshold:
            print(f"  {PASS}  Sentence {i+1}: cosine similarity = {sim:.6f}")
        else:
            print(f"  {FAIL}  Sentence {i+1}: cosine similarity = {sim:.6f} "
                  f"(below {threshold})")
            print(f"         \"{sentence_preview}\"")
            all_ok = False

    if all_ok:
        print(f"\n  All embeddings match (cosine similarity ≥ {threshold}).")
    else:
        print(f"\n  {WARN}  Some embeddings differ — the remote server may be running "
              "a different model or quantization.")

    return all_ok


def test_batch_stress(host: str, port: int, label: str) -> bool:
    """Send a realistic batch (16 × 3072-char texts) and check for errors.

    This mirrors the workload from process_and_index.py to verify that
    the server's --batch-size and --ubatch-size are large enough.
    """
    print(f"\n{BOLD}5. Batch-size stress test ({label}){RESET}")
    print(f"  Sending {BATCH_TEST_COUNT} texts × {BATCH_TEST_CHARS} chars "
          f"to {host}:{port} ...")

    # Build a batch of long paragraphs (repeat a base paragraph to reach the
    # target character count, similar to truncated Wikipedia sections).
    base_paragraph = (
        "Wikipedia is a free-content online encyclopedia written and maintained "
        "by a community of volunteers, known as Wikipedians, through open "
        "collaboration and the use of the wiki-based editing system MediaWiki. "
        "Wikipedia is the largest and most-read reference work in history. "
    )
    texts = []
    for i in range(BATCH_TEST_COUNT):
        # Vary lengths: half at full length, half at ~50%
        target_len = BATCH_TEST_CHARS if i % 2 == 0 else BATCH_TEST_CHARS // 2
        text = (base_paragraph * ((target_len // len(base_paragraph)) + 1))[:target_len]
        texts.append(text)

    url = f"http://{host}:{port}/v1/embeddings"
    payload = {"model": EMBED_MODEL, "input": texts}

    try:
        resp = requests.post(url, json=payload, timeout=BATCH_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            print(f"  {FAIL}  Server returned error: {data['error']}")
            return False

        returned = len(data.get("data", []))
        if returned != BATCH_TEST_COUNT:
            print(f"  {FAIL}  Expected {BATCH_TEST_COUNT} embeddings, got {returned}")
            return False

        # Quick sanity: check dimension of first embedding
        dim = len(data["data"][0].get("embedding", []))
        print(f"  {PASS}  Received {returned} embeddings (dim={dim})")
        return True

    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        print(f"  {FAIL}  HTTP {e.response.status_code}: {body}")
        print(f"         The server's --batch-size / --ubatch-size may be too small.")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  {FAIL}  Connection refused — is the server running?")
        return False
    except requests.exceptions.Timeout:
        print(f"  {FAIL}  Request timed out after {BATCH_REQUEST_TIMEOUT}s")
        return False
    except Exception as e:
        print(f"  {FAIL}  {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Remote GPU Server — Connectivity & Embedding Test{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    results: dict[str, bool] = {}

    # 1. Environment check
    if not test_env_variable():
        return 1

    # 2. Remote server availability
    embed_ok, llm_ok = test_remote_availability()
    results['remote_embedding'] = embed_ok
    results['remote_llm'] = llm_ok

    # 3. Local embedding server
    local_ok = test_local_availability()
    results['local_embedding'] = local_ok

    # 4. Embedding comparison (only if both embedding servers are up)
    if embed_ok and local_ok:
        results['embedding_match'] = test_embedding_comparison()
    else:
        print(f"\n{BOLD}4. Embedding comparison{RESET}")
        print(f"  {WARN}  Skipped — requires both local and remote embedding servers")
        results['embedding_match'] = False

    # 5. Batch-size stress test (verifies --batch-size / --ubatch-size)
    if embed_ok:
        results['remote_batch_stress'] = test_batch_stress(
            REMOTE_HOST, REMOTE_EMBED_PORT, "remote")
    else:
        print(f"\n{BOLD}5. Batch-size stress test (remote){RESET}")
        print(f"  {WARN}  Skipped — remote embedding server not available")
        results['remote_batch_stress'] = False

    # Summary
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    if passed == total:
        print(f"{PASS}  All {total} tests passed")
    else:
        failed = total - passed
        print(f"{FAIL}  {failed}/{total} test(s) failed:")
        for name, ok in results.items():
            if not ok:
                print(f"         - {name}")
    print()

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
