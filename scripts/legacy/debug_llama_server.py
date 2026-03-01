#!/usr/bin/env python3
"""
Debug & Test Script for Strix Halo llama.cpp Server (Podman/ROCm)

This script diagnoses and debugs issues with the llama-server containers
running on AMD Strix Halo (gfx1151) systems. It performs a systematic
series of checks and interactive tests without modifying the setup script.

Usage:
    sudo -E python3 scripts/debug_llama_server.py              # Full diagnostic
    sudo -E python3 scripts/debug_llama_server.py --quick       # Quick system check only
    sudo -E python3 scripts/debug_llama_server.py --test-server  # Run a test server
    sudo -E python3 scripts/debug_llama_server.py --fix-quadlet  # Show Quadlet fixes

Findings from cross-referencing the official kyuz0/amd-strix-halo-toolboxes:

1. HSA_OVERRIDE_GFX_VERSION=11.0.0 is WRONG for the ROCm 7.2 toolbox image.
   The image is compiled with -DAMDGPU_TARGETS=gfx1151 (native gfx1151 support).
   Overriding to 11.0.0 (gfx1100) causes GPU ID mismatch → segfault (exit 139).

2. PodmanArgs=--security-opt seccomp=unconfined is REQUIRED in Quadlet files.
   (NOT SecurityOpt= which is an invalid Quadlet key and silently breaks generation.)
   The official toolbox docs require this for ROCm memory operations.

3. GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 as a runtime env var is correct (per issue #51).

4. --flash-attn on and --no-mmap are correct (per official Quick Start).
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(os.environ.get("DEEPRED_ROOT", "/mnt/data"))
REPO_DIR = pathlib.Path(os.environ.get("DEEPRED_REPO", str(DATA_DIR / "DeepRedAI")))
MODELS_DIR = pathlib.Path(os.environ.get("DEEPRED_MODELS", str(DATA_DIR / "models")))

ROCM_TOOLBOX_IMAGE = "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2"
TEST_CONTAINER_NAME = "debug-llama-test"

QUADLET_DIR = pathlib.Path("/etc/containers/systemd")
QUADLET_LLM = QUADLET_DIR / "llama-server-llm.container"
QUADLET_EMBED = QUADLET_DIR / "llama-server-embed.container"

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{BLUE}{'═' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 60}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {BLUE}ℹ{RESET} {msg}")


def run(cmd: str, *, check: bool = False, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command, returning the result."""
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            check=check, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="TIMEOUT")


def http_check(url: str, timeout: int = 5) -> tuple[bool, str]:
    """Check if an HTTP endpoint is reachable."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, body
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Phase 1: System Environment Checks
# ---------------------------------------------------------------------------


def check_system() -> dict:
    """Check system kernel, firmware, GPU devices, and kernel params."""
    results = {}
    header("Phase 1: System Environment")

    # 1a. Kernel version
    r = run("uname -r")
    kernel = r.stdout.strip()
    results["kernel"] = kernel
    # Parse major.minor.patch
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", kernel)
    if m:
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if major > 6 or (major == 6 and minor > 18) or (major == 6 and minor == 18 and patch >= 4):
            ok(f"Kernel: {kernel} (≥ 6.18.4 required)")
        else:
            fail(f"Kernel: {kernel} — TOO OLD, need ≥ 6.18.4 for gfx1151 stability")
    else:
        warn(f"Kernel: {kernel} — could not parse version")

    # 1b. Firmware
    r = run("rpm -q linux-firmware 2>/dev/null || dpkg -l linux-firmware 2>/dev/null | tail -1")
    fw = r.stdout.strip()
    results["firmware"] = fw
    if "20251125" in fw:
        fail(f"Firmware: {fw} — BROKEN version, must upgrade!")
    elif fw:
        ok(f"Firmware: {fw}")
    else:
        warn("Firmware: could not determine version")

    # 1c. Kernel boot parameters
    cmdline = pathlib.Path("/proc/cmdline").read_text().strip()
    results["cmdline"] = cmdline
    info(f"Boot params: {cmdline[:120]}...")

    required_params = {
        "iommu=pt": "IOMMU pass-through for GPU performance",
        "amdgpu.gttsize=126976": "GTT memory = 124 GB",
        "ttm.pages_limit=32505856": "TTM pinned memory = 124 GB",
    }
    for param, desc in required_params.items():
        if param in cmdline:
            ok(f"  {param} — {desc}")
        else:
            fail(f"  {param} MISSING — {desc}")

    # 1d. GPU device nodes
    for dev in ["/dev/kfd", "/dev/dri"]:
        if pathlib.Path(dev).exists():
            ok(f"Device: {dev} exists")
        else:
            fail(f"Device: {dev} MISSING")

    # Check render node permissions
    r = run("ls -la /dev/dri/render* 2>/dev/null")
    if r.returncode == 0:
        info(f"Render nodes: {r.stdout.strip()}")
    r = run("ls -la /dev/kfd 2>/dev/null")
    if r.returncode == 0:
        info(f"KFD: {r.stdout.strip()}")

    # 1e. GPU identification
    r = run("cat /sys/class/drm/card*/device/product_name 2>/dev/null")
    gpu_name = r.stdout.strip() if r.returncode == 0 else "unknown"
    results["gpu_name"] = gpu_name
    info(f"GPU: {gpu_name}")

    r = run("lspci | grep -i 'VGA\\|Display'")
    if r.returncode == 0:
        info(f"PCI: {r.stdout.strip()}")

    # 1f. GTT memory from sysfs
    r = run("cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null")
    if r.returncode == 0 and r.stdout.strip().isdigit():
        gtt_gb = int(r.stdout.strip()) / (1024 ** 3)
        ok(f"GTT memory available: {gtt_gb:.1f} GB")
    else:
        warn("GTT memory: could not read from sysfs")

    r = run("cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null")
    if r.returncode == 0 and r.stdout.strip().isdigit():
        vram_gb = int(r.stdout.strip()) / (1024 ** 3)
        info(f"VRAM (fixed UMA/GART): {vram_gb:.1f} GB")

    # 1g. Current user groups
    user = os.environ.get("SUDO_USER", os.environ.get("USER", ""))
    if user:
        r = run(f"id -nG {user}")
        groups = r.stdout.strip()
        results["groups"] = groups
        for g in ["video", "render"]:
            if g in groups.split():
                ok(f"User '{user}' is in '{g}' group")
            else:
                fail(f"User '{user}' NOT in '{g}' group — run: sudo usermod -aG {g} {user}")

    # 1h. ROCm info from amdgpu driver
    r = run("cat /sys/module/amdgpu/version 2>/dev/null")
    if r.returncode == 0 and r.stdout.strip():
        info(f"amdgpu driver version: {r.stdout.strip()}")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Container & Image Checks
# ---------------------------------------------------------------------------


def check_containers() -> dict:
    """Check Podman, images, and existing containers."""
    results = {}
    header("Phase 2: Container Configuration")

    # 2a. Podman version
    r = run("podman --version")
    if r.returncode == 0:
        ok(f"Podman: {r.stdout.strip()}")
        results["podman"] = r.stdout.strip()
    else:
        fail("Podman not installed!")
        return results

    # 2b. Image availability
    r = run(f"sudo podman images --format '{{{{.Repository}}}}:{{{{.Tag}}}}  {{{{.Size}}}}  {{{{.Created}}}}' | grep strix-halo")
    if r.returncode == 0 and r.stdout.strip():
        for line in r.stdout.strip().splitlines():
            ok(f"Image: {line.strip()}")
    else:
        warn(f"Image {ROCM_TOOLBOX_IMAGE} not found locally — will need to pull")

    # 2c. Existing containers
    r = run("sudo podman ps -a --format '{{.Names}}  {{.Status}}  {{.Image}}' 2>/dev/null")
    if r.returncode == 0 and r.stdout.strip():
        info("Existing containers:")
        for line in r.stdout.strip().splitlines():
            print(f"       {line.strip()}")

    # 2d. Quadlet file analysis
    for label, qf in [("LLM Quadlet", QUADLET_LLM), ("Embed Quadlet", QUADLET_EMBED)]:
        if qf.exists():
            content = qf.read_text()
            results[label] = content

            info(f"{label}: {qf}")

            # Check for problematic HSA_OVERRIDE
            if "HSA_OVERRIDE_GFX_VERSION" in content:
                fail(f"  {label} has HSA_OVERRIDE_GFX_VERSION — REMOVE THIS!")
                fail(f"  The ROCm 7.2 image is compiled for gfx1151 natively.")
                fail(f"  Overriding to 11.0.0 (gfx1100) causes GPU ID mismatch → segfault.")
            else:
                ok(f"  {label}: no HSA_OVERRIDE_GFX_VERSION (correct)")

            # Check for seccomp=unconfined (must use PodmanArgs, NOT SecurityOpt)
            if "PodmanArgs=--security-opt seccomp=unconfined" in content:
                ok(f"  {label}: has PodmanArgs=--security-opt seccomp=unconfined")
            elif "SecurityOpt=seccomp=unconfined" in content:
                fail(f"  {label}: uses INVALID key SecurityOpt=seccomp=unconfined")
                fail(f"  Quadlet generator rejects SecurityOpt — use PodmanArgs instead")
                fail(f"  Change to: PodmanArgs=--security-opt seccomp=unconfined")
            else:
                warn(f"  {label}: MISSING PodmanArgs=--security-opt seccomp=unconfined")
                warn(f"  Official docs require: --security-opt seccomp=unconfined")

            # Check for GGML_CUDA_ENABLE_UNIFIED_MEMORY
            if "GGML_CUDA_ENABLE_UNIFIED_MEMORY" in content:
                ok(f"  {label}: has GGML_CUDA_ENABLE_UNIFIED_MEMORY")

            # Check for flash-attn
            if "flash-attn" in content:
                ok(f"  {label}: has --flash-attn")
            else:
                warn(f"  {label}: MISSING --flash-attn (should be 'on' per official docs)")

            # Check for --no-mmap
            if "--no-mmap" in content:
                ok(f"  {label}: has --no-mmap")
            else:
                fail(f"  {label}: MISSING --no-mmap (REQUIRED on Strix Halo)")

        else:
            warn(f"{label}: {qf} does not exist")

    # 2e. Environment file check
    env_file = pathlib.Path("/etc/sysconfig/llama-server")
    if env_file.exists():
        content = env_file.read_text()
        info(f"Env file: {env_file}")
        if "HSA_OVERRIDE_GFX_VERSION" in content:
            fail(f"  Env file has HSA_OVERRIDE_GFX_VERSION — SHOULD BE REMOVED")
        for line in content.strip().splitlines():
            if not line.startswith("#"):
                print(f"       {line.strip()}")

    # 2f. Service status
    for svc in ["llama-server-llm", "llama-server-embed"]:
        r = run(f"systemctl is-active {svc} 2>/dev/null")
        status = r.stdout.strip()
        r2 = run(f"systemctl is-enabled {svc} 2>/dev/null")
        enabled = r2.stdout.strip()
        if status == "active":
            ok(f"Service {svc}: {status} ({enabled})")
        else:
            warn(f"Service {svc}: {status} ({enabled})")
            # Get recent logs
            r3 = run(f"journalctl -u {svc} --no-pager -n 10 2>/dev/null")
            if r3.returncode == 0 and r3.stdout.strip():
                info(f"  Recent logs for {svc}:")
                for line in r3.stdout.strip().splitlines()[-5:]:
                    print(f"       {line.strip()}")

    # 2g. Quadlet generator validation
    info("")
    info("Validating Quadlet files with podman-system-generator...")
    gen = shutil.which("podman-system-generator") or "/usr/lib/systemd/system-generators/podman-system-generator"
    if pathlib.Path(gen).exists():
        r = run(f"{gen} --dryrun /tmp/quadlet-debug-test 2>&1")
        output = (r.stdout + r.stderr).strip()
        if "unsupported key" in output.lower():
            fail("Quadlet generator found unsupported keys:")
            for line in output.splitlines():
                if "unsupported" in line.lower():
                    print(f"       {RED}{line.strip()}{RESET}")
            fail("  Service units will NOT be generated until these are fixed!")
        elif r.returncode == 0:
            ok("Quadlet generator accepted all files (no errors)")
        else:
            warn(f"Quadlet generator returned exit {r.returncode}")
            for line in output.splitlines()[-5:]:
                print(f"       {line.strip()}")
    else:
        warn(f"Quadlet generator not found at {gen}")

    return results


# ---------------------------------------------------------------------------
# Phase 3: Interactive GPU Test (inside container)
# ---------------------------------------------------------------------------


def test_gpu_inside_container() -> dict:
    """Run diagnostic commands inside the ROCm container."""
    results = {}
    header("Phase 3: GPU Test Inside Container")

    # Cleanup any leftover test container
    run(f"sudo podman rm -f {TEST_CONTAINER_NAME} 2>/dev/null", timeout=15)

    # Run llama-cli --list-devices inside the container (WITHOUT HSA_OVERRIDE)
    info("Testing GPU detection inside container (NO HSA_OVERRIDE)...")
    r = run(
        f"sudo podman run --rm --name {TEST_CONTAINER_NAME}-gpu "
        f"--device /dev/kfd --device /dev/dri "
        f"--group-add video --group-add render "
        f"--security-opt seccomp=unconfined "
        f"-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "
        f"{ROCM_TOOLBOX_IMAGE} "
        f"llama-cli --list-devices",
        timeout=60,
    )
    out = (r.stdout + "\n" + r.stderr).strip()
    results["list_devices_no_override"] = out
    if r.returncode == 0:
        ok("llama-cli --list-devices (no HSA_OVERRIDE) succeeded")
        for line in out.splitlines():
            print(f"       {line}")
        if "gfx1151" in out.lower():
            ok("  GPU correctly identified as gfx1151")
        elif "gfx1100" in out.lower():
            warn("  GPU showing as gfx1100 — check if HSA_OVERRIDE is leaking")
    else:
        fail(f"llama-cli --list-devices failed (exit {r.returncode})")
        for line in out.splitlines()[-10:]:
            print(f"       {line}")

    # Also test WITH HSA_OVERRIDE to show the difference
    info("")
    info("Testing GPU detection WITH HSA_OVERRIDE=11.0.0 (for comparison)...")
    r = run(
        f"sudo podman run --rm --name {TEST_CONTAINER_NAME}-gpu2 "
        f"--device /dev/kfd --device /dev/dri "
        f"--group-add video --group-add render "
        f"--security-opt seccomp=unconfined "
        f"-e HSA_OVERRIDE_GFX_VERSION=11.0.0 "
        f"-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "
        f"{ROCM_TOOLBOX_IMAGE} "
        f"llama-cli --list-devices",
        timeout=60,
    )
    out = (r.stdout + "\n" + r.stderr).strip()
    results["list_devices_with_override"] = out
    if r.returncode == 0:
        warn("llama-cli --list-devices (WITH HSA_OVERRIDE=11.0.0):")
        for line in out.splitlines():
            print(f"       {line}")
    else:
        fail(f"llama-cli --list-devices with override failed (exit {r.returncode})")
        for line in out.splitlines()[-5:]:
            print(f"       {line}")

    # Test rocminfo if available
    info("")
    info("Running rocminfo to identify GPU agent...")
    r = run(
        f"sudo podman run --rm --name {TEST_CONTAINER_NAME}-rocm "
        f"--device /dev/kfd --device /dev/dri "
        f"--group-add video --group-add render "
        f"--security-opt seccomp=unconfined "
        f"{ROCM_TOOLBOX_IMAGE} "
        f"/opt/rocm/bin/rocminfo 2>&1 | grep -E 'Marketing Name|Name:|gfx|Agent '",
        timeout=60,
    )
    if r.returncode == 0 and r.stdout.strip():
        ok("rocminfo output:")
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"       {line}")
    else:
        # Try alternate path
        r2 = run(
            f"sudo podman run --rm --name {TEST_CONTAINER_NAME}-rocm2 "
            f"--device /dev/kfd --device /dev/dri "
            f"--group-add video --group-add render "
            f"--security-opt seccomp=unconfined "
            f"{ROCM_TOOLBOX_IMAGE} "
            f"bash -c 'find /opt -name rocminfo -type f 2>/dev/null | head -1'",
            timeout=30,
        )
        if r2.stdout.strip():
            warn(f"rocminfo found at {r2.stdout.strip()} but invocation failed")
        else:
            warn("rocminfo not installed in container image")
            info("  GPU was already verified via llama-cli --list-devices above")

    return results


# ---------------------------------------------------------------------------
# Phase 4: Test Server Launch
# ---------------------------------------------------------------------------


def test_server_launch(model_path: str = None, timeout_secs: int = 90) -> dict:
    """Launch a test llama-server inside a container and verify the API."""
    results = {}
    header("Phase 4: Test Server Launch")

    # Find a model to test with
    if not model_path:
        # Try embedding model first (smaller, faster to load)
        candidates = [
            MODELS_DIR / "embedding" / "nomic-embed-text-v1.5.f16.gguf",
            MODELS_DIR / "llm" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if not model_path:
        fail("No model found to test with!")
        info(f"Expected models in: {MODELS_DIR}")
        return results

    # Map the model path to container path
    container_model = f"/models/{pathlib.Path(model_path).relative_to(MODELS_DIR)}"
    is_embedding = "embed" in model_path.lower() or "nomic" in model_path.lower()

    info(f"Model: {model_path}")
    info(f"Container path: {container_model}")
    info(f"Type: {'embedding' if is_embedding else 'LLM'}")

    # Cleanup
    run(f"sudo podman rm -f {TEST_CONTAINER_NAME} 2>/dev/null", timeout=15)

    # Build the launch command — note: NO HSA_OVERRIDE_GFX_VERSION
    extra_args = "--embedding" if is_embedding else ""
    ctx_size = "512" if is_embedding else "512"  # Small ctx for testing
    test_port = "19999"

    cmd = (
        f"sudo podman run -d --name {TEST_CONTAINER_NAME} "
        f"--device /dev/kfd --device /dev/dri "
        f"--group-add video --group-add render "
        f"--security-opt seccomp=unconfined "
        f"-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "
        f"-v {MODELS_DIR}:/models:ro,z "
        f"-p {test_port}:{test_port} "
        f"{ROCM_TOOLBOX_IMAGE} "
        f"llama-server "
        f"--model {container_model} "
        f"--host 0.0.0.0 "
        f"--port {test_port} "
        f"--n-gpu-layers 999 "
        f"--flash-attn on "
        f"--no-mmap "
        f"--ctx-size {ctx_size} "
        f"--threads 4 "
        f"--no-warmup "
        f"{extra_args}"
    )

    info(f"Launching test container (port {test_port})...")
    info(f"Command: {cmd}")
    print()

    r = run(cmd, timeout=60)
    if r.returncode != 0:
        fail(f"Container failed to start (exit {r.returncode})")
        fail(f"  stdout: {r.stdout.strip()}")
        fail(f"  stderr: {r.stderr.strip()}")
        return results

    container_id = r.stdout.strip()[:12]
    ok(f"Container started: {container_id}")

    # Wait for server to become ready
    info(f"Waiting for server to be ready (max {timeout_secs}s)...")
    start_time = time.time()
    server_ready = False
    last_status = ""

    while time.time() - start_time < timeout_secs:
        # Check if container is still running
        r = run(f"sudo podman inspect --format '{{{{.State.Status}}}}' {TEST_CONTAINER_NAME}", timeout=10)
        status = r.stdout.strip()
        if status != "running":
            fail(f"Container stopped unexpectedly (status: {status})")
            # Get logs
            r = run(f"sudo podman logs {TEST_CONTAINER_NAME} 2>&1", timeout=15)
            fail("Container logs:")
            for line in (r.stdout or "").strip().splitlines()[-30:]:
                print(f"       {line}")
            results["exit_code"] = "container_died"
            break

        # Try the health endpoint
        reachable, body = http_check(f"http://localhost:{test_port}/health")
        if reachable:
            try:
                health = json.loads(body)
                hstatus = health.get("status", "unknown")
                if hstatus != last_status:
                    info(f"  Server status: {hstatus} ({time.time() - start_time:.0f}s)")
                    last_status = hstatus
                if hstatus == "ok":
                    server_ready = True
                    break
            except json.JSONDecodeError:
                pass

        # Show progress from container logs
        elapsed = time.time() - start_time
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            r = run(f"sudo podman logs --tail 3 {TEST_CONTAINER_NAME} 2>&1", timeout=10)
            if r.stdout.strip():
                last_log = r.stdout.strip().splitlines()[-1]
                info(f"  [{elapsed:.0f}s] {last_log[:100]}")

        time.sleep(2)

    if server_ready:
        elapsed = time.time() - start_time
        ok(f"Server ready in {elapsed:.1f}s!")

        # Test /v1/models endpoint
        reachable, body = http_check(f"http://localhost:{test_port}/v1/models")
        if reachable:
            ok(f"/v1/models: {body[:200]}")
        else:
            fail(f"/v1/models failed: {body}")

        # Test a simple inference or embedding
        if is_embedding:
            info("Testing embedding generation...")
            try:
                data = json.dumps({
                    "input": "Hello, this is a test.",
                    "model": "test",
                }).encode()
                req = urllib.request.Request(
                    f"http://localhost:{test_port}/v1/embeddings",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    n_dims = len(result.get("data", [{}])[0].get("embedding", []))
                    ok(f"Embedding test: {n_dims} dimensions returned")
            except Exception as e:
                fail(f"Embedding test failed: {e}")
        else:
            info("Testing chat completion...")
            try:
                data = json.dumps({
                    "model": "test",
                    "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
                    "max_tokens": 10,
                    "temperature": 0,
                }).encode()
                req = urllib.request.Request(
                    f"http://localhost:{test_port}/v1/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read())
                    reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    ok(f"Chat test: '{reply[:100]}'")
            except Exception as e:
                fail(f"Chat test failed: {e}")

        results["status"] = "ok"
    else:
        fail(f"Server did not become ready within {timeout_secs}s")
        # Dump final logs
        r = run(f"sudo podman logs {TEST_CONTAINER_NAME} 2>&1", timeout=15)
        fail("Final container logs:")
        for line in (r.stdout or "").strip().splitlines()[-30:]:
            print(f"       {line}")
        results["status"] = "timeout"

    # Show container resource usage
    r = run(f"sudo podman stats --no-stream --format 'CPU: {{{{.CPUPerc}}}} MEM: {{{{.MemUsage}}}}' {TEST_CONTAINER_NAME}", timeout=10)
    if r.returncode == 0 and r.stdout.strip():
        info(f"Resource usage: {r.stdout.strip()}")

    # Cleanup
    info("Stopping test container...")
    run(f"sudo podman rm -f {TEST_CONTAINER_NAME} 2>/dev/null", timeout=15)
    ok("Test container removed")

    return results


# ---------------------------------------------------------------------------
# Phase 5: Quadlet Fix Suggestions
# ---------------------------------------------------------------------------


def show_quadlet_fixes():
    """Generate and display corrected Quadlet files."""
    header("Phase 5: Recommended Quadlet Fixes")

    issues_found = []

    for label, qf in [("LLM", QUADLET_LLM), ("Embedding", QUADLET_EMBED)]:
        if not qf.exists():
            warn(f"{label} Quadlet not found at {qf}")
            continue

        content = qf.read_text()
        original = content

        # Fix 1: Remove HSA_OVERRIDE_GFX_VERSION
        if "HSA_OVERRIDE_GFX_VERSION" in content:
            issues_found.append(f"{label}: Remove HSA_OVERRIDE_GFX_VERSION")
            # Remove the Environment line
            content = re.sub(
                r'\n\s*Environment=HSA_OVERRIDE_GFX_VERSION=[^\n]*', '', content
            )

        # Fix 2: Replace SecurityOpt with PodmanArgs or add PodmanArgs
        if "SecurityOpt=seccomp=unconfined" in content:
            issues_found.append(f"{label}: Replace SecurityOpt with PodmanArgs")
            content = content.replace(
                "SecurityOpt=seccomp=unconfined",
                "PodmanArgs=--security-opt seccomp=unconfined",
            )
        elif "PodmanArgs=--security-opt seccomp=unconfined" not in content:
            issues_found.append(f"{label}: Add PodmanArgs=--security-opt seccomp=unconfined")
            # Add after the last GroupAdd line
            content = re.sub(
                r'(GroupAdd=render\n)',
                r'\1PodmanArgs=--security-opt seccomp=unconfined\n',
                content,
            )

        if content != original:
            print(f"\n{BOLD}--- Corrected {label} Quadlet ({qf}) ---{RESET}")
            print(content)

    # Also fix the env file
    env_file = pathlib.Path("/etc/sysconfig/llama-server")
    if env_file.exists():
        content = env_file.read_text()
        if "HSA_OVERRIDE_GFX_VERSION" in content:
            issues_found.append("Env file: Remove HSA_OVERRIDE_GFX_VERSION")
            fixed = re.sub(r'[^\n]*HSA_OVERRIDE_GFX_VERSION[^\n]*\n?', '', content)
            print(f"\n{BOLD}--- Corrected env file ({env_file}) ---{RESET}")
            print(fixed)

    if issues_found:
        print(f"\n{BOLD}{RED}Issues to fix:{RESET}")
        for issue in issues_found:
            print(f"  {RED}→{RESET} {issue}")

        print(f"\n{BOLD}To apply these fixes, run:{RESET}")
        print(f"  sudo -E python3 {REPO_DIR}/scripts/debug_llama_server.py --apply-fixes")
    else:
        ok("Quadlet files look correct — no fixes needed")


def apply_fixes():
    """Apply recommended fixes to Quadlet files."""
    header("Applying Fixes")

    changes_made = False

    for label, qf in [("LLM", QUADLET_LLM), ("Embedding", QUADLET_EMBED)]:
        if not qf.exists():
            warn(f"{label} Quadlet not found at {qf}")
            continue

        content = qf.read_text()
        original = content

        # Fix 1: Remove HSA_OVERRIDE_GFX_VERSION
        if "HSA_OVERRIDE_GFX_VERSION" in content:
            content = re.sub(
                r'\n\s*Environment=HSA_OVERRIDE_GFX_VERSION=[^\n]*', '', content
            )
            ok(f"{label}: Removed HSA_OVERRIDE_GFX_VERSION")

        # Fix 2: Replace SecurityOpt with PodmanArgs or add PodmanArgs
        if "SecurityOpt=seccomp=unconfined" in content:
            content = content.replace(
                "SecurityOpt=seccomp=unconfined",
                "PodmanArgs=--security-opt seccomp=unconfined",
            )
            ok(f"{label}: Replaced SecurityOpt with PodmanArgs")
        elif "PodmanArgs=--security-opt seccomp=unconfined" not in content:
            content = re.sub(
                r'(GroupAdd=render\n)',
                r'\1PodmanArgs=--security-opt seccomp=unconfined\n',
                content,
            )
            ok(f"{label}: Added PodmanArgs=--security-opt seccomp=unconfined")

        if content != original:
            # Backup original
            backup = qf.with_suffix(".container.bak")
            if not backup.exists():
                backup.write_text(original)
                info(f"  Backup saved: {backup}")
            qf.write_text(content)
            ok(f"  Updated: {qf}")
            changes_made = True

    # Fix env file
    env_file = pathlib.Path("/etc/sysconfig/llama-server")
    if env_file.exists():
        content = env_file.read_text()
        if "HSA_OVERRIDE_GFX_VERSION" in content:
            backup = env_file.with_suffix(".bak")
            if not backup.exists():
                backup.write_text(content)
            fixed = re.sub(r'[^\n]*HSA_OVERRIDE_GFX_VERSION[^\n]*\n?', '', content)
            env_file.write_text(fixed)
            ok(f"Env file: Removed HSA_OVERRIDE_GFX_VERSION from {env_file}")
            changes_made = True

    if changes_made:
        info("Reloading systemd and restarting services...")
        run("systemctl daemon-reload")
        run("systemctl restart llama-server-llm 2>/dev/null")
        run("systemctl restart llama-server-embed 2>/dev/null")
        ok("Services restarted — check status with: systemctl status llama-server-llm llama-server-embed")
    else:
        ok("No changes needed")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(system_results: dict, container_results: dict,
                  gpu_results: dict, server_results: dict):
    """Print a combined summary of all findings."""
    header("Summary & Recommendations")

    # Check for the main known issues
    issues = []

    # HSA_OVERRIDE in quadlets
    for qf in [QUADLET_LLM, QUADLET_EMBED]:
        if qf.exists() and "HSA_OVERRIDE_GFX_VERSION" in qf.read_text():
            issues.append(
                f"{RED}CRITICAL:{RESET} Remove HSA_OVERRIDE_GFX_VERSION from {qf.name}.\n"
                f"           The ROCm 7.2 image targets gfx1151 natively — overriding to\n"
                f"           gfx1100 causes a GPU code mismatch and segfaults (exit 139)."
            )

    # Missing or wrong seccomp=unconfined
    for qf in [QUADLET_LLM, QUADLET_EMBED]:
        if qf.exists():
            qtext = qf.read_text()
            if "SecurityOpt=seccomp=unconfined" in qtext:
                issues.append(
                    f"{RED}CRITICAL:{RESET} {qf.name} uses invalid key SecurityOpt=seccomp=unconfined.\n"
                    f"           Quadlet generator rejects this key → service unit not generated.\n"
                    f"           Change to: PodmanArgs=--security-opt seccomp=unconfined"
                )
            elif "PodmanArgs=--security-opt seccomp=unconfined" not in qtext:
                issues.append(
                    f"{YELLOW}IMPORTANT:{RESET} Add PodmanArgs=--security-opt seccomp=unconfined to {qf.name}.\n"
                    f"           Official toolbox docs require this for ROCm memory operations."
                )

    # Kernel too old
    if "kernel" in system_results:
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", system_results["kernel"])
        if m:
            major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if major < 6 or (major == 6 and minor < 18) or (major == 6 and minor == 18 and patch < 4):
                issues.append(
                    f"{RED}CRITICAL:{RESET} Kernel {system_results['kernel']} is too old.\n"
                    f"           Need ≥ 6.18.4 for stable gfx1151 support."
                )

    # Bad firmware
    if "firmware" in system_results and "20251125" in system_results["firmware"]:
        issues.append(
            f"{RED}CRITICAL:{RESET} linux-firmware 20251125 breaks ROCm on Strix Halo.\n"
            f"           Upgrade: sudo dnf upgrade linux-firmware"
        )

    # Missing kernel params
    if "cmdline" in system_results:
        for param in ["iommu=pt", "amdgpu.gttsize=126976", "ttm.pages_limit=32505856"]:
            if param not in system_results["cmdline"]:
                issues.append(
                    f"{YELLOW}IMPORTANT:{RESET} Kernel parameter '{param}' is missing.\n"
                    f"           Run: sudo grubby --update-kernel=ALL --args=\"{param}\""
                )

    if issues:
        print(f"\n  {BOLD}Found {len(issues)} issue(s):{RESET}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}\n")

        print(f"\n  {BOLD}Quick fix:{RESET}")
        print(f"  sudo -E python3 {REPO_DIR}/scripts/debug_llama_server.py --apply-fixes")
    else:
        ok("No issues found — configuration looks correct!")

    if server_results.get("status") == "ok":
        ok("Test server launched successfully — the server works!")
    elif server_results.get("status") == "timeout":
        warn("Test server timed out — check logs above for details")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Debug & test llama.cpp server on Strix Halo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              sudo -E python3 scripts/debug_llama_server.py              # Full diagnostic
              sudo -E python3 scripts/debug_llama_server.py --quick       # System checks only
              sudo -E python3 scripts/debug_llama_server.py --test-server # Launch test server
              sudo -E python3 scripts/debug_llama_server.py --fix-quadlet # Show Quadlet fixes
              sudo -E python3 scripts/debug_llama_server.py --apply-fixes # Apply fixes
        """),
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run only system + container checks (no server test)")
    parser.add_argument("--test-server", action="store_true",
                        help="Run only the test server launch")
    parser.add_argument("--fix-quadlet", action="store_true",
                        help="Show recommended Quadlet file corrections")
    parser.add_argument("--apply-fixes", action="store_true",
                        help="Apply recommended fixes to Quadlet files")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to GGUF model file for server test")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Timeout for server readiness (default: 90s)")

    args = parser.parse_args()

    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Strix Halo llama-server Debug Tool{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Image:  {ROCM_TOOLBOX_IMAGE}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Repo:   {REPO_DIR}")

    if args.apply_fixes:
        apply_fixes()
        return

    if args.fix_quadlet:
        show_quadlet_fixes()
        return

    if args.test_server:
        test_server_launch(model_path=args.model, timeout_secs=args.timeout)
        return

    # Full diagnostic
    system_results = check_system()
    container_results = check_containers()

    gpu_results = {}
    server_results = {}

    if not args.quick:
        gpu_results = test_gpu_inside_container()
        server_results = test_server_launch(
            model_path=args.model, timeout_secs=args.timeout,
        )

    print_summary(system_results, container_results, gpu_results, server_results)


if __name__ == "__main__":
    main()
