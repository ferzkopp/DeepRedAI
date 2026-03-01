#!/usr/bin/env python3
"""
NVIDIA A4000 Fedora Setup Automation

Automated setup for an NVIDIA A4000 (16 GB) VM running Fedora 43 with PCI
passthrough.  Handles everything after basic Fedora installation, SSH setup,
and repo clone.

Stages are idempotent where possible and tracked in a JSON state file so the
script can resume after reboots or failures.

Prerequisites (manual — see A4000-Fedora-Setup.md Phase 1):
  - Fedora installed in a VM with NVIDIA A4000 PCI passthrough
  - Repo cloned (default $DEEPRED_ROOT/DeepRedAI)
  - Python 3 available (dnf install python3)

Environment variables (all optional — set via deepred-env.sh):
  DEEPRED_ROOT   Base data directory       (default: /mnt/data)
  DEEPRED_REPO   Path to this git clone    (default: $DEEPRED_ROOT/DeepRedAI)
  DEEPRED_MODELS Model storage directory    (default: $DEEPRED_ROOT/models)
  DEEPRED_VENV   Python venv directory      (default: $DEEPRED_ROOT/venv)

Usage:
    # Run all stages (resumes from last incomplete)
    sudo -E python3 setup_a4000.py

    # Run a specific stage only
    sudo -E python3 setup_a4000.py --stage nvidia_driver

    # Re-run a completed stage
    sudo -E python3 setup_a4000.py --stage llama_server --force

    # List stages and status
    sudo -E python3 setup_a4000.py --list

    # Start from a specific stage
    sudo -E python3 setup_a4000.py --from vscode

    # Override detected non-root user
    sudo -E python3 setup_a4000.py --user myuser
"""

import argparse
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants — all paths honour environment variables from deepred-env.sh
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(os.environ.get("DEEPRED_ROOT", "/mnt/data"))
REPO_DIR = pathlib.Path(os.environ.get("DEEPRED_REPO", str(DATA_DIR / "DeepRedAI")))
STATE_FILE = REPO_DIR / ".setup_a4000_state.json"
MODELS_DIR = pathlib.Path(os.environ.get("DEEPRED_MODELS", str(DATA_DIR / "models")))
VENV_DIR = pathlib.Path(os.environ.get("DEEPRED_VENV", str(DATA_DIR / "venv")))

# Container images — tags are pinned to a build number (no floating :latest).
# See https://github.com/ggerganov/llama.cpp/pkgs/container/llama.cpp for tags.
LLAMA_SERVER_IMAGE = "ghcr.io/ggerganov/llama.cpp:server-cuda-b4719"   # server only (smaller)
LLAMA_FULL_IMAGE   = "ghcr.io/ggerganov/llama.cpp:full-cuda-b4719"     # all binaries (toolbox)
CUDA_TOOLBOX_NAME = "llama-cuda"

LOG_FILE = REPO_DIR / "setup_a4000.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("setup")


def setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    log.addHandler(fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(
    cmd: str | list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: Optional[dict] = None,
    stdin_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a shell command, logging it first."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    log.info("  ▸ %s", display)

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        capture_output=capture,
        text=True,
        env=merged_env,
        input=stdin_text,
    )
    return result


def run_quiet(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, capturing output (for checks)."""
    return run(cmd, check=check, capture=True)


def file_contains(path: str | pathlib.Path, needle: str) -> bool:
    """Check if a file contains a string."""
    try:
        return needle in pathlib.Path(path).read_text()
    except FileNotFoundError:
        return False


def write_file(path: str | pathlib.Path, content: str, *, mode: int = 0o644) -> None:
    """Write content to a file, creating parent dirs."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    os.chmod(p, mode)
    log.info("  wrote %s", p)


def detect_user() -> str:
    """Detect the non-root user who owns DATA_DIR."""
    st = os.stat(DATA_DIR)
    uid = st.st_uid
    if uid == 0:
        return os.environ.get("SUDO_USER", "")
    import pwd
    return pwd.getpwuid(uid).pw_name


def needs_reboot(message: str) -> None:
    """Print reboot message and exit."""
    rerun_cmd = f"sudo -E python3 {REPO_DIR}/scripts/setup_a4000.py"
    inner_w = max(58, len(rerun_cmd) + 2, len(message) + 2)
    W = inner_w + 4
    log.info("")
    log.info("╔" + "═" * W + "╗")
    log.info("║  " + f"{'REBOOT REQUIRED':<{inner_w}}" + " ║")
    log.info("║  " + f"{message:<{inner_w}}" + " ║")
    log.info("║" + " " * W + "║")
    log.info("║  " + f"{'After reboot, re-run this script to continue:':<{inner_w}}" + " ║")
    log.info("║  " + f"{rerun_cmd:<{inner_w}}" + " ║")
    log.info("║" + " " * W + "║")
    log.info("║  " + f"{'sudo reboot':<{inner_w}}" + " ║")
    log.info("╚" + "═" * W + "╝")
    sys.exit(0)


# ---------------------------------------------------------------------------
# State tracker
# ---------------------------------------------------------------------------


class StateTracker:
    """Persists stage completion status to a JSON file."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                return {"stages": {}}
        return {"stages": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def is_done(self, stage_name: str) -> bool:
        return self.data.get("stages", {}).get(stage_name, {}).get("status") == "done"

    def mark_done(self, stage_name: str) -> None:
        stages = self.data.setdefault("stages", {})
        stages[stage_name] = {
            "status": "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def is_pending(self, stage_name: str) -> bool:
        return self.data.get("stages", {}).get(stage_name, {}).get("status") == "pending"

    def mark_pending(self, stage_name: str) -> None:
        stages = self.data.setdefault("stages", {})
        stages[stage_name] = {"status": "pending"}
        self._save()

    def reset(self, stage_name: str) -> None:
        stages = self.data.get("stages", {})
        stages.pop(stage_name, None)
        self._save()


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    description: str
    func: Callable
    requires_reboot: bool = False


STAGES: list[Stage] = []


def stage(
    name: str, description: str, *, requires_reboot: bool = False
) -> Callable:
    """Decorator to register a setup stage."""

    def decorator(func: Callable) -> Callable:
        STAGES.append(
            Stage(
                name=name,
                description=description,
                func=func,
                requires_reboot=requires_reboot,
            )
        )
        return func

    return decorator


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


@stage("system_packages", "Install build tools and development packages")
def stage_system_packages(user: str) -> None:
    run("dnf install -y @development-tools cmake gcc-c++ git curl wget "
        "python3-devel python3-pip python3-setuptools python3-wheel "
        "lld clang clang-devel compiler-rt libcurl-devel "
        "pciutils")


@stage("disable_sleep", "Disable sleep/suspend for always-on server operation")
def stage_disable_sleep(user: str) -> None:
    # Mask all sleep-related systemd targets
    sleep_targets = [
        "sleep.target",
        "suspend.target",
        "hibernate.target",
        "hybrid-sleep.target",
        "suspend-then-hibernate.target",
    ]
    for target in sleep_targets:
        run(f"systemctl mask {target}", check=False)

    # Configure logind to ignore all suspend/hibernate triggers
    logind_conf = pathlib.Path("/etc/systemd/logind.conf.d/no-sleep.conf")
    if not logind_conf.exists():
        write_file(
            logind_conf,
            textwrap.dedent("""\
                [Login]
                HandleSuspendKey=ignore
                HandleHibernateKey=ignore
                HandleLidSwitch=ignore
                HandleLidSwitchExternalPower=ignore
                HandleLidSwitchDocked=ignore
                IdleAction=ignore
                IdleActionSec=0
            """),
        )
        run("systemctl restart systemd-logind")
    else:
        log.info("  logind no-sleep config already present")

    # Disable GNOME auto-suspend if desktop is installed
    if shutil.which("gsettings"):
        for key, value in [
            ("sleep-inactive-ac-type", "nothing"),
            ("sleep-inactive-ac-timeout", "0"),
            ("sleep-inactive-battery-type", "nothing"),
            ("sleep-inactive-battery-timeout", "0"),
        ]:
            run(
                f'su - {user} -c "DBUS_SESSION_BUS_ADDRESS= gsettings set '
                f'org.gnome.settings-daemon.plugins.power {key} {value} 2>/dev/null"',
                check=False,
            )

    # Verify
    result = run_quiet("systemctl is-enabled suspend.target", check=False)
    if "masked" in result.stdout:
        log.info("  ✓ Sleep targets are masked — always-on mode active")
    else:
        log.warning("  Sleep targets may not be fully masked — check manually")


@stage("nvidia_driver", "Install NVIDIA driver from RPM Fusion", requires_reboot=True)
def stage_nvidia_driver(user: str) -> None:
    # Check if NVIDIA driver is already loaded
    result = run_quiet("lsmod | grep -q nvidia", check=False)
    if result.returncode == 0:
        # Driver loaded — verify nvidia-smi works
        smi = run_quiet("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
        if smi.returncode == 0 and "A4000" in (smi.stdout or ""):
            log.info("  NVIDIA driver already installed and A4000 detected: %s",
                     smi.stdout.strip())
            return

    # Add RPM Fusion repos (free + nonfree)
    fedora_ver = run_quiet("rpm -E %fedora").stdout.strip()
    for flavour in ["free", "nonfree"]:
        repo_rpm = f"https://mirrors.rpmfusion.org/{flavour}/fedora/rpmfusion-{flavour}-release-{fedora_ver}.noarch.rpm"
        run(f"dnf install -y {repo_rpm}", check=False)

    # Install NVIDIA akmod driver (automatically rebuilds on kernel updates)
    run("dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda")

    # Force kernel module build now (don't wait for next boot)
    log.info("  Building NVIDIA kernel module (this may take a few minutes)...")
    run("akmods --force")
    run("dracut --force")

    needs_reboot("NVIDIA driver installed — reboot to load the kernel module")


@stage("nvidia_container_toolkit", "Install NVIDIA Container Toolkit with CDI for Podman")
def stage_nvidia_container_toolkit(user: str) -> None:
    # Verify NVIDIA driver is working first
    result = run_quiet("nvidia-smi", check=False)
    if result.returncode != 0:
        log.error("  nvidia-smi failed — NVIDIA driver not loaded. "
                  "Run the nvidia_driver stage first and reboot.")
        sys.exit(1)

    # Ensure Podman is installed
    run("dnf install -y podman")

    # Add NVIDIA Container Toolkit repo
    repo_file = pathlib.Path("/etc/yum.repos.d/nvidia-container-toolkit.repo")
    if not repo_file.exists():
        run("curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/"
            "nvidia-container-toolkit.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo")

    # Install the toolkit
    # nvidia-container-toolkit 1.18+ ships systemd units (nvidia-cdi-refresh.service)
    # that automatically generate/refresh CDI specs — no manual runtime
    # configuration is needed for Podman (CDI is Podman's native GPU path).
    run("dnf install -y nvidia-container-toolkit")

    # Ensure the CDI spec exists.  The package %post scriptlet triggers
    # nvidia-cdi-refresh.service which generates /etc/cdi/nvidia.yaml
    # automatically.  If it hasn't run yet, generate the spec manually.
    cdi_spec = pathlib.Path("/etc/cdi/nvidia.yaml")
    if not cdi_spec.exists():
        cdi_dir = pathlib.Path("/etc/cdi")
        cdi_dir.mkdir(parents=True, exist_ok=True)
        run("nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml")
    else:
        log.info("  CDI spec already present at %s", cdi_spec)

    # Verify CDI spec lists the GPU
    result = run_quiet("nvidia-ctk cdi list", check=False)
    if result.returncode == 0 and "nvidia.com/gpu" in (result.stdout or ""):
        log.info("  ✓ CDI spec generated — GPU devices available to containers")
    else:
        log.warning("  CDI spec may not be correct — verify with: nvidia-ctk cdi list")

    # Quick container GPU test
    log.info("  Testing GPU access from a container...")
    test_result = run_quiet(
        f'su - {user} -c "podman run --rm --device nvidia.com/gpu=all '
        f'docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi '
        f'--query-gpu=name,memory.total --format=csv,noheader"',
        check=False,
    )
    if test_result.returncode == 0 and (test_result.stdout or "").strip():
        log.info("  ✓ Container GPU test passed: %s", test_result.stdout.strip())
    else:
        log.warning("  Container GPU test failed — may need manual troubleshooting")
        if test_result.stderr:
            log.warning("  stderr: %s", test_result.stderr.strip()[:200])


@stage("vscode", "Install VSCode with Python and Copilot extensions")
def stage_vscode(user: str) -> None:
    # Import the VS Code RPM repo if not present
    repo_file = pathlib.Path("/etc/yum.repos.d/vscode.repo")
    if not repo_file.exists():
        write_file(
            repo_file,
            textwrap.dedent("""\
                [code]
                name=Visual Studio Code
                baseurl=https://packages.microsoft.com/yumrepos/vscode
                enabled=1
                gpgcheck=1
                gpgkey=https://packages.microsoft.com/keys/microsoft.asc
            """),
        )
        run("rpm --import https://packages.microsoft.com/keys/microsoft.asc")

    # Install VS Code
    run("dnf install -y code")

    # Install extensions as the non-root user
    extensions = [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "GitHub.copilot",
        "GitHub.copilot-chat",
    ]
    for ext in extensions:
        run(f'su - {user} -c "code --install-extension {ext} --force"',
            check=False)


@stage("toolbox_setup", "Create CUDA toolbox container with llama.cpp")
def stage_toolbox_setup(user: str) -> None:
    # Ensure podman is installed
    run("dnf install -y podman")

    # Check if container already exists
    exists = run_quiet(
        f'su - {user} -c "podman container exists {CUDA_TOOLBOX_NAME}"',
        check=False,
    )
    if exists.returncode == 0:
        log.info("  Container '%s' already exists", CUDA_TOOLBOX_NAME)
        return

    # Ensure rootless podman requirements (subuid/subgid)
    for db in ["/etc/subuid", "/etc/subgid"]:
        content = pathlib.Path(db).read_text() if pathlib.Path(db).exists() else ""
        if user not in content:
            log.info("  Adding %s to %s for rootless podman", user, db)
            run(f'usermod --add-subuids 100000-165535 --add-subgids 100000-165535 {user}')
            break

    # Pull the image as the non-root user
    log.info("  Pulling %s as %s (this may take a while)...", LLAMA_FULL_IMAGE, user)
    run(f'su - {user} -c "podman pull {LLAMA_FULL_IMAGE}"')

    # Ensure XDG_RUNTIME_DIR exists for rootless podman
    uid = run_quiet(f"id -u {user}").stdout.strip()
    runtime_dir = f"/run/user/{uid}"
    pathlib.Path(runtime_dir).mkdir(parents=True, exist_ok=True)
    run(f"chown {user}:{user} {runtime_dir}")
    run(f"chmod 0700 {runtime_dir}")

    log.info("  Creating container '%s' via podman...", CUDA_TOOLBOX_NAME)
    result = run(
        f'su - {user} -c "'
        f"podman create"
        f" --name {CUDA_TOOLBOX_NAME}"
        f" --hostname toolbox"
        f" --device nvidia.com/gpu=all"
        f" --security-opt label=disable"
        f" --userns=keep-id"
        f" --pid=host"
        f" --network=host"
        f" --volume {DATA_DIR}:{DATA_DIR}:rslave"
        f" --volume {runtime_dir}:{runtime_dir}:rslave"
        f" {LLAMA_FULL_IMAGE}"
        f" sleep infinity"
        f'"',
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        log.error("  podman create stdout:\n%s", result.stdout.strip() if result.stdout else "(empty)")
        log.error("  podman create stderr:\n%s", result.stderr.strip() if result.stderr else "(empty)")
        raise RuntimeError(
            f"podman create failed (exit {result.returncode}). "
            f"See output above for details."
        )
    log.info("  Toolbox '%s' created successfully", CUDA_TOOLBOX_NAME)


@stage("model_directories", "Create model directories and download models")
def stage_model_directories(user: str) -> None:
    # Create directories
    for subdir in ["llm", "embedding"]:
        (MODELS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Install huggingface_hub Python package system-wide
    run("pip3 install --break-system-packages huggingface_hub 2>/dev/null || "
        "pip3 install huggingface_hub", check=False)

    def hf_download(repo: str, filename: str, local_dir: pathlib.Path) -> None:
        run(
            f'python3 -c "'
            f"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download("
            f"'{repo}', '{filename}', local_dir='{local_dir}'"
            f')"'
        )

    def hf_snapshot(repo: str, pattern: str, local_dir: pathlib.Path) -> None:
        """Download files matching a glob pattern from a HF repo."""
        run(
            f'python3 -c "'
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download("
            f"'{repo}', allow_patterns='{pattern}', local_dir='{local_dir}'"
            f')"'
        )

    # Download embedding model
    embed_model = MODELS_DIR / "embedding" / "nomic-embed-text-v1.5.f16.gguf"
    if not embed_model.exists():
        log.info("  Downloading embedding model...")
        hf_download(
            "nomic-ai/nomic-embed-text-v1.5-GGUF",
            "nomic-embed-text-v1.5.f16.gguf",
            MODELS_DIR / "embedding",
        )
    else:
        log.info("  Embedding model already present")

    # Download LLM (Qwen 2.5 7B Q4_K_M — split into two shards)
    llm_shard1 = MODELS_DIR / "llm" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    llm_shard2 = MODELS_DIR / "llm" / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
    if not llm_shard1.exists() or not llm_shard2.exists():
        log.info("  Downloading LLM model (Qwen 2.5 7B Q4_K_M, 2 shards)...")
        hf_snapshot(
            "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "qwen2.5-7b-instruct-q4_k_m*.gguf",
            MODELS_DIR / "llm",
        )
    else:
        log.info("  LLM model already present")

    # Set ownership
    run(f"chown -R {user}:{user} {MODELS_DIR}")


@stage("llama_server", "Deploy Podman Quadlet services for llama.cpp CUDA servers")
def stage_llama_server(user: str) -> None:
    quadlet_dir = pathlib.Path("/etc/containers/systemd")
    quadlet_dir.mkdir(parents=True, exist_ok=True)

    # Pull the server image (Quadlet uses a different, smaller image than the toolbox)
    log.info("  Pulling server image %s...", LLAMA_SERVER_IMAGE)
    run(f"podman pull {LLAMA_SERVER_IMAGE}")

    # LLM Server Quadlet (Port 1234)
    write_file(
        quadlet_dir / "llama-server-llm.container",
        textwrap.dedent(f"""\
            [Unit]
            Description=llama.cpp LLM Server (CUDA, OpenAI-compatible)
            After=network-online.target

            [Container]
            Image={LLAMA_SERVER_IMAGE}
            Exec=/llama-server \\
                --model /models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \\
                --host 0.0.0.0 \\
                --port 1234 \\
                --n-gpu-layers 999 \\
                --flash-attn \\
                --ctx-size 8192 \\
                --threads 8 \\
                --parallel 2 \\
                --alias "gpt-oss-20b"
            AddDevice=nvidia.com/gpu=all
            Volume={MODELS_DIR}:/models:ro,z
            PublishPort=1234:1234

            [Service]
            Restart=on-failure
            RestartSec=10

            [Install]
            WantedBy=multi-user.target default.target
        """),
    )

    # Embedding Server Quadlet (Port 1235)
    write_file(
        quadlet_dir / "llama-server-embed.container",
        textwrap.dedent(f"""\
            [Unit]
            Description=llama.cpp Embedding Server (CUDA, OpenAI-compatible)
            After=network-online.target

            [Container]
            Image={LLAMA_SERVER_IMAGE}
            Exec=/llama-server \\
                --model /models/embedding/nomic-embed-text-v1.5.f16.gguf \\
                --host 0.0.0.0 \\
                --port 1235 \\
                --n-gpu-layers 999 \\
                --flash-attn \\
                --ctx-size 2048 \\
                --threads 4 \\
                --embedding \\
                --alias "text-embedding-nomic-embed-text-v1.5@f16"
            AddDevice=nvidia.com/gpu=all
            Volume={MODELS_DIR}:/models:ro,z
            PublishPort=1235:1235

            [Service]
            Restart=on-failure
            RestartSec=10

            [Install]
            WantedBy=multi-user.target default.target
        """),
    )

    run("systemctl daemon-reload")
    run("systemctl start llama-server-llm", check=False)
    run("systemctl start llama-server-embed", check=False)


@stage("python_venv", "Create Python venv with PyTorch CUDA and utilities")
def stage_python_venv(user: str) -> None:
    run("dnf install -y python3-devel python3-pip python3-setuptools")

    # Create venv if it doesn't exist
    if not (VENV_DIR / "bin" / "activate").exists():
        log.info("  Creating venv at %s", VENV_DIR)
        run(f'su - {user} -c "python3 -m venv {VENV_DIR}"')
    else:
        log.info("  Venv already exists at %s", VENV_DIR)

    pip = f"{VENV_DIR}/bin/pip"

    # Upgrade pip
    run(f'su - {user} -c "{pip} install --upgrade pip"')

    # Install PyTorch with CUDA
    log.info("  Installing PyTorch CUDA...")
    run(f'su - {user} -c "{pip} install torch torchvision torchaudio '
        f'--index-url https://download.pytorch.org/whl/cu124"')

    # Utility dependencies (lighter than StrixHalo — no training libs needed)
    log.info("  Installing utility packages...")
    run(f'su - {user} -c "{pip} install requests openai huggingface_hub '
        f'numpy tqdm sentencepiece tiktoken"')

    run(f"chown -R {user}:{user} {VENV_DIR}")


@stage("firewall", "Configure firewalld rules for service ports")
def stage_firewall(user: str) -> None:
    run("dnf install -y firewalld")

    # Add SSH before enabling firewalld to avoid lockout
    result = run_quiet("firewall-cmd --permanent --query-service=ssh", check=False)
    if result.returncode != 0:
        run("firewall-cmd --permanent --add-service=ssh", check=False)
    else:
        log.info("  SSH already in firewall permanent config")

    ports = [
        ("1234/tcp", "port"),   # llama LLM
        ("1235/tcp", "port"),   # llama embedding
    ]

    for spec, kind in ports:
        result = run_quiet(f"firewall-cmd --permanent --query-port={spec}", check=False)
        if result.returncode != 0:
            run(f"firewall-cmd --permanent --add-port={spec}", check=False)
        else:
            log.info("  Port %s already open", spec)

    run("systemctl enable --now firewalld")
    run("firewall-cmd --reload")


@stage("llm_swap_helper", "Install /usr/local/bin/llm-swap helper script")
def stage_llm_swap_helper(user: str) -> None:
    write_file(
        "/usr/local/bin/llm-swap",
        textwrap.dedent(f"""\
            #!/bin/bash
            # Usage: llm-swap <model-path> [alias] [ctx-size]
            MODEL="${{1:?Usage: llm-swap <model-path> [alias] [ctx-size]}}"
            ALIAS="${{2:-gpt-oss-20b}}"
            CTX="${{3:-8192}}"

            if [ ! -f "$MODEL" ]; then
                echo "Error: Model file not found: $MODEL"
                exit 1
            fi

            QUADLET_FILE="/etc/containers/systemd/llama-server-llm.container"
            SERVICE_NAME="llama-server-llm"

            if [ -f "$QUADLET_FILE" ]; then
                CONTAINER_MODEL="/models/${{MODEL#{MODELS_DIR}/}}"
                sudo sed -i "s|--model [^ ]*|--model $CONTAINER_MODEL|" "$QUADLET_FILE"
                sudo sed -i "s|--ctx-size [0-9]*|--ctx-size $CTX|" "$QUADLET_FILE"
                sudo sed -i 's|--alias "[^"]*"|--alias "'"$ALIAS"'"|' "$QUADLET_FILE"
                echo "Updated Quadlet: $QUADLET_FILE"
            fi

            sudo systemctl daemon-reload
            sudo systemctl restart "$SERVICE_NAME"
            echo "Swapped to: $MODEL (alias: $ALIAS, ctx: $CTX)"
            sudo systemctl status "$SERVICE_NAME" --no-pager -l
        """),
        mode=0o755,
    )


@stage("verify", "Run health checks on all components", requires_reboot=True)
def stage_verify(user: str) -> None:
    # Wait for servers to become ready
    log.info("  Waiting for llama.cpp servers to become ready (up to 60s)...")
    for url, label in [
        ("http://localhost:1234/health", "LLM server"),
        ("http://localhost:1235/health", "Embedding server"),
    ]:
        ready = False
        for _ in range(12):
            r = run_quiet(f"curl -sf {url} -o /dev/null", check=False)
            if r.returncode == 0:
                ready = True
                break
            time.sleep(5)
        if ready:
            log.info("  %s is ready", label)
        else:
            log.warning("  %s did not become ready within 60s", label)

    checks = [
        ("Kernel", "uname -r"),
        ("NVIDIA driver", "nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        ("GPU detected", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
        ("NVIDIA CDI", 'nvidia-ctk cdi list 2>/dev/null | grep -q "nvidia.com/gpu" && echo OK || echo MISSING'),
        ("Podman", "podman --version"),
        ("Toolbox container",
         f'su - {user} -s /bin/sh -c "podman container exists {CUDA_TOOLBOX_NAME}" '
         f'&& echo "{CUDA_TOOLBOX_NAME}" || echo MISSING'),
        ("LLM server",
         "curl -sf http://localhost:1234/v1/models -o /dev/null && echo OK || echo DOWN"),
        ("Embedding server",
         "curl -sf http://localhost:1235/v1/models -o /dev/null && echo OK || echo DOWN"),
        ("llm-swap helper",
         'test -x /usr/local/bin/llm-swap && echo "installed" || echo "MISSING"'),
        ("VSCode",
         f'su - {user} -s /bin/sh -c "code --version 2>/dev/null | head -1" || echo "not found"'),
    ]

    log.info("")
    log.info("═══ Health Check ═══")
    all_ok = True
    for label, cmd in checks:
        result = run_quiet(cmd, check=False)
        output = (result.stdout or "").strip()
        failed = result.returncode != 0 or "DOWN" in output or "MISSING" in output
        status = "✗" if failed else "✓"
        if failed:
            all_ok = False
        log.info("  %s %-25s %s", status, label, output[:80])

    # GPU details
    log.info("")
    log.info("═══ GPU ═══")
    gpu_info = run_quiet(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw "
        "--format=csv,noheader",
        check=False,
    )
    if gpu_info.returncode == 0:
        log.info("  ℹ %s", (gpu_info.stdout or "unknown").strip())

    # Disk usage
    log.info("")
    log.info("═══ Data Directory (%s) ═══", DATA_DIR)
    disk_info = run_quiet(f"df -h {DATA_DIR} --output=size,used,avail,pcent | tail -1", check=False)
    if disk_info.returncode == 0:
        parts = disk_info.stdout.strip().split()
        if len(parts) >= 4:
            log.info("  ℹ %-25s %s total, %s used, %s free (%s)",
                     "Disk", parts[0], parts[1], parts[2], parts[3])

    content_dirs = [
        ("Models", MODELS_DIR),
        ("Python venv", VENV_DIR),
        ("Repo", REPO_DIR),
    ]
    for label, path in content_dirs:
        if path.exists():
            du = run_quiet(f"du -sh {path}", check=False)
            size = (du.stdout or "").split()[0] if du.returncode == 0 else "?"
            log.info("  ℹ %-25s %s", label, size)

    log.info("")
    if all_ok:
        log.info("  All checks passed!")
    else:
        log.info("  Some checks failed — review above and re-run failed stages")

    needs_reboot(
        "Reboot to confirm all services start automatically on boot. "
        "The next stage (reverify) will validate them."
    )


@stage("reverify", "Post-reboot health check — verify services survive a restart")
def stage_reverify(user: str) -> None:
    """Re-verify after reboot: wait for services, then health-check."""
    services = {
        "llama-server-llm":   ("http://localhost:1234/health", "LLM server (port 1234)"),
        "llama-server-embed": ("http://localhost:1235/health", "Embedding server (port 1235)"),
    }

    max_wait = 90
    poll_interval = 5

    log.info("")
    log.info("═══ Post-Reboot Service Check ═══")
    log.info("  Waiting up to %ds for container services to become healthy...", max_wait)

    all_ok = True
    for svc, (health_url, label) in services.items():
        healthy = False
        elapsed = 0

        # Check if unit is active
        r = run_quiet(f"systemctl is-active {svc}", check=False)
        svc_state = (r.stdout or "").strip()
        if svc_state != "active":
            log.info("  ⟳ %s is '%s', attempting start...", svc, svc_state)
            run_quiet(f"systemctl start {svc}", check=False)
            time.sleep(2)

        while elapsed < max_wait:
            r = run_quiet(f"curl -sf {health_url}", check=False)
            if r.returncode == 0:
                healthy = True
                break
            time.sleep(poll_interval)
            elapsed += poll_interval

        if healthy:
            log.info("  ✓ %-35s UP  (ready in ~%ds)", label, elapsed)
        else:
            all_ok = False
            log.info("  ✗ %-35s DOWN after %ds", label, max_wait)
            r = run_quiet(f"journalctl -u {svc} --no-pager -n 5 2>/dev/null", check=False)
            if r.returncode == 0 and (r.stdout or "").strip():
                for line in r.stdout.strip().splitlines()[-3:]:
                    log.info("      %s", line.strip())

    # Verify NVIDIA driver survived reboot
    r = run_quiet("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
    if r.returncode == 0 and (r.stdout or "").strip():
        log.info("  ✓ %-35s %s", "NVIDIA driver", r.stdout.strip())
    else:
        all_ok = False
        log.info("  ✗ %-35s FAILED", "NVIDIA driver")

    # API smoke tests
    log.info("")
    log.info("═══ API Smoke Test ═══")

    # LLM chat completion
    r = run_quiet(
        'curl -sf -m 30 http://localhost:1234/v1/chat/completions '
        '-H "Content-Type: application/json" '
        '-d \'{"model":"gpt-oss-20b","messages":[{"role":"user","content":"ping"}],"max_tokens":5}\'',
        check=False,
    )
    if r.returncode == 0 and r.stdout and "choices" in r.stdout:
        log.info("  ✓ %-35s chat completion OK", "LLM /v1/chat/completions")
    else:
        all_ok = False
        log.info("  ✗ %-35s FAILED", "LLM /v1/chat/completions")

    # Embedding
    r = run_quiet(
        'curl -sf -m 30 http://localhost:1235/v1/embeddings '
        '-H "Content-Type: application/json" '
        '-d \'{"model":"text-embedding-nomic-embed-text-v1.5@f16","input":"test"}\'',
        check=False,
    )
    if r.returncode == 0 and r.stdout and "embedding" in r.stdout:
        log.info("  ✓ %-35s embedding OK", "Embed /v1/embeddings")
    else:
        all_ok = False
        log.info("  ✗ %-35s FAILED", "Embed /v1/embeddings")

    log.info("")
    if all_ok:
        log.info("  Post-reboot verification passed — all services healthy!")
    else:
        log.info("  Some services failed post-reboot. Check logs and re-run:")
        log.info("    sudo -E python3 %s --stage reverify --force", __file__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_stages(state: StateTracker) -> None:
    """Print all stages and their status."""
    print(f"\n{'#':>3}  {'Stage':<30} {'Status':<12} Description")
    print(f"{'─'*3}  {'─'*30} {'─'*12} {'─'*40}")
    for i, s in enumerate(STAGES, 1):
        status = "✓ done" if state.is_done(s.name) else "  pending"
        reboot = " ↻" if s.requires_reboot else ""
        print(f"{i:>3}  {s.name:<30} {status:<12} {s.description}{reboot}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVIDIA A4000 Fedora automated setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        help="Run a single specific stage by name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a stage even if already completed",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_stages",
        help="List all stages and their status",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        help="Start from a specific stage (skip earlier ones)",
    )
    parser.add_argument(
        "--user",
        help=f"Non-root user (auto-detected from {DATA_DIR} ownership if omitted)",
    )
    args = parser.parse_args()

    setup_logging()

    # Must run as root
    if os.geteuid() != 0:
        log.error("This script must be run as root (sudo).")
        sys.exit(1)

    # Verify data dir exists
    if not DATA_DIR.exists():
        log.error(
            "%s does not exist. Create it first:\n"
            "  sudo mkdir -p %s && sudo chown $USER:$USER %s",
            DATA_DIR, DATA_DIR, DATA_DIR,
        )
        sys.exit(1)

    state = StateTracker(STATE_FILE)

    # List mode
    if args.list_stages:
        list_stages(state)
        return

    # Detect user
    user = args.user or detect_user()
    if not user:
        log.error(
            f"Cannot detect non-root user. {DATA_DIR} is owned by root. "
            "Use --user <username> to specify."
        )
        sys.exit(1)
    log.info("Target non-root user: %s", user)

    # Build stage lookup
    stage_map = {s.name: s for s in STAGES}

    # Single stage mode
    if args.stage:
        if args.stage not in stage_map:
            log.error("Unknown stage '%s'. Use --list to see available stages.", args.stage)
            sys.exit(1)
        s = stage_map[args.stage]
        if state.is_done(s.name) and not args.force:
            log.info("Stage '%s' already completed. Use --force to re-run.", s.name)
            return
        if state.is_pending(s.name) and s.requires_reboot and not args.force:
            log.info("Stage '%s' completed before reboot — marking done.", s.name)
            state.mark_done(s.name)
            return
        log.info("━━━ Stage: %s — %s ━━━", s.name, s.description)
        state.mark_pending(s.name)
        s.func(user)
        state.mark_done(s.name)
        log.info("━━━ Stage '%s' completed ━━━", s.name)
        return

    # Determine starting point
    start_idx = 0
    if args.from_stage:
        if args.from_stage not in stage_map:
            log.error("Unknown stage '%s'. Use --list to see available stages.", args.from_stage)
            sys.exit(1)
        start_idx = next(i for i, s in enumerate(STAGES) if s.name == args.from_stage)
        log.info("Starting from stage '%s' (skipping %d earlier stages)", args.from_stage, start_idx)

    # Sequential execution
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  NVIDIA A4000 Fedora Setup                                 ║")
    log.info("║  %d stages total, starting from #%d                         ║",
             len(STAGES), start_idx + 1)
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")

    for i, s in enumerate(STAGES):
        if i < start_idx:
            continue

        if state.is_done(s.name) and not args.force:
            log.info("  [%d/%d] %s — already done, skipping", i + 1, len(STAGES), s.name)
            continue

        if state.is_pending(s.name) and s.requires_reboot and not args.force:
            log.info("  [%d/%d] %s — completed before reboot, marking done", i + 1, len(STAGES), s.name)
            state.mark_done(s.name)
            continue

        log.info("")
        log.info("━━━ [%d/%d] Stage: %s — %s ━━━", i + 1, len(STAGES), s.name, s.description)
        state.mark_pending(s.name)

        try:
            s.func(user)
        except SystemExit:
            raise
        except Exception:
            log.exception("  Stage '%s' failed!", s.name)
            log.info("  Fix the issue and re-run: sudo -E python3 %s", __file__)
            log.info("  Or re-run just this stage: sudo -E python3 %s --stage %s --force", __file__, s.name)
            sys.exit(1)

        state.mark_done(s.name)
        log.info("  ✓ Stage '%s' completed", s.name)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  Setup complete!                                           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Enter the toolbox:  podman start %s && podman exec -it %s bash",
             CUDA_TOOLBOX_NAME, CUDA_TOOLBOX_NAME)
    log.info("  2. Test from StrixHalo:  curl http://A4000AI:1234/v1/models")
    log.info("  3. See documentation:  %s/documentation/A4000-Fedora-Setup.md", REPO_DIR)


if __name__ == "__main__":
    main()
