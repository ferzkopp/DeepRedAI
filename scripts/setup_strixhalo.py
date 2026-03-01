#!/usr/bin/env python3
"""
Strix Halo Fedora Setup Automation

Automated setup for AMD Ryzen AI MAX+ 395 "Strix Halo" systems running
Fedora 42/43. Handles everything after basic Fedora installation and data
disk mount.

Stages are idempotent where possible and tracked in a JSON state file so the
script can resume after reboots or failures.

Prerequisites (manual — see StrixHalo-Fedora-Setup.md Phase 1):
  - Fedora installed on 1 TB system disk
  - Data disk mounted (default /mnt/data, override with DEEPRED_ROOT)
  - Repo cloned under data disk (default $DEEPRED_ROOT/DeepRedAI)
  - Python 3 available (dnf install python3)

Environment variables (all optional — set via deepred-env.sh):
  DEEPRED_ROOT   Base data directory       (default: /mnt/data)
  DEEPRED_REPO   Path to this git clone    (default: $DEEPRED_ROOT/DeepRedAI)
  DEEPRED_MODELS Model storage directory    (default: $DEEPRED_ROOT/models)
  DEEPRED_VENV   Python venv directory      (default: $DEEPRED_ROOT/venv)

Usage:
    # Run all stages (resumes from last incomplete)
    sudo python3 setup_strixhalo.py

    # Run a specific stage only
    sudo python3 setup_strixhalo.py --stage postgresql

    # Re-run a completed stage
    sudo python3 setup_strixhalo.py --stage postgresql --force

    # List stages and status
    sudo python3 setup_strixhalo.py --list

    # Start from a specific stage
    sudo python3 setup_strixhalo.py --from vscode

    # Override detected non-root user
    sudo python3 setup_strixhalo.py --user myuser
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants — all paths honour environment variables from deepred-env.sh
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(os.environ.get("DEEPRED_ROOT", "/mnt/data"))
REPO_DIR = pathlib.Path(os.environ.get("DEEPRED_REPO", str(DATA_DIR / "DeepRedAI")))
STATE_FILE = REPO_DIR / ".setup_state.json"
MODELS_DIR = pathlib.Path(os.environ.get("DEEPRED_MODELS", str(DATA_DIR / "models")))
VENV_DIR = pathlib.Path(os.environ.get("DEEPRED_VENV", str(DATA_DIR / "venv")))

ROCM_TOOLBOX_IMAGE = "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2"
ROCM_TOOLBOX_NAME = "llama-rocm-7.2"

OPENSEARCH_VERSION = "2.19.1"
OPENSEARCH_URL = (
    f"https://artifacts.opensearch.org/releases/bundle/opensearch/"
    f"{OPENSEARCH_VERSION}/opensearch-{OPENSEARCH_VERSION}-linux-x64.tar.gz"
)

LOG_FILE = REPO_DIR / "setup.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("setup")


def setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    # Also log to file
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
    import stat

    st = os.stat(DATA_DIR)
    uid = st.st_uid
    if uid == 0:
        # Fallback: check SUDO_USER
        return os.environ.get("SUDO_USER", "")
    import pwd

    return pwd.getpwuid(uid).pw_name


def needs_reboot(message: str) -> None:
    """Print reboot message and exit."""
    rerun_cmd = f"sudo -E python3 {REPO_DIR}/scripts/setup_strixhalo.py"
    # Box width = inner content width + 4 (for "║  " and " ║")
    inner_w = max(58, len(rerun_cmd) + 2, len(message) + 2)
    W = inner_w + 4  # total between ╔ and ╗
    log.info("")
    log.info("╔" + "═" * W + "╗")
    log.info("║  " + f"{'REBOOT REQUIRED':<{inner_w}}" + " ║")
    log.info("║  " + f"{message:<{inner_w}}" + " ║")
    log.info("║" + " " * (W) + "║")
    log.info("║  " + f"{'After reboot, re-run this script to continue:':<{inner_w}}" + " ║")
    log.info("║  " + f"{rerun_cmd:<{inner_w}}" + " ║")
    log.info("║" + " " * (W) + "║")
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
        "lld clang clang-devel compiler-rt libcurl-devel")


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
            # Suppress dconf D-Bus warnings when running headless (no X11)
            run(
                f'su - {user} -c "DBUS_SESSION_BUS_ADDRESS= gsettings set '
                f'org.gnome.settings-daemon.plugins.power {key} {value}"',
                check=False,
            )

    # Disable auto-suspend in GDM greeter session (independent from user session)
    # GDM has its own dconf database — without this override, the greeter will
    # trigger "The system will suspend now!" when the login screen is idle.
    gdm_profile = pathlib.Path("/etc/dconf/profile/gdm")
    gdm_db_dir = pathlib.Path("/etc/dconf/db/gdm.d")
    gdm_override = gdm_db_dir / "99-no-suspend"
    if not gdm_override.exists():
        # Ensure GDM dconf profile exists and includes the gdm-local db
        gdm_db_dir.mkdir(parents=True, exist_ok=True)
        if not gdm_profile.exists() or "gdm" not in gdm_profile.read_text():
            write_file(
                gdm_profile,
                textwrap.dedent("""\
                    user-db:user
                    system-db:gdm
                    file-db:/usr/share/gdm/greeter-dconf-defaults
                """),
            )
        write_file(
            gdm_override,
            textwrap.dedent("""\
                [org/gnome/settings-daemon/plugins/power]
                sleep-inactive-ac-type='nothing'
                sleep-inactive-ac-timeout=uint32 0
                sleep-inactive-battery-type='nothing'
                sleep-inactive-battery-timeout=uint32 0
            """),
        )
        run("dconf update")
        log.info("  GDM greeter auto-suspend disabled via dconf override")
    else:
        log.info("  GDM greeter no-suspend override already present")

    # Verify
    result = run_quiet("systemctl is-enabled suspend.target", check=False)
    if "masked" in result.stdout:
        log.info("  ✓ Sleep targets are masked — always-on mode active")
    else:
        log.warning("  Sleep targets may not be fully masked — check manually")


@stage("gtt_memory", "Configure kernel parameters for GPU memory", requires_reboot=True)
def stage_gtt_memory(user: str) -> None:
    # Check if already configured
    cmdline = pathlib.Path("/proc/cmdline").read_text()
    needed_params = {
        "iommu=pt": "iommu=pt",
        "amdgpu.gttsize=126976": "amdgpu.gttsize=126976",
        "ttm.pages_limit=32505856": "ttm.pages_limit=32505856",
    }

    missing = [v for k, v in needed_params.items() if k not in cmdline]
    if not missing:
        log.info("  GTT kernel parameters already set — skipping")
        return

    log.info("  Adding kernel parameters: %s", " ".join(missing))
    run(f'grubby --update-kernel=ALL --args="{" ".join(missing)}"')
    run("grub2-mkconfig -o /boot/grub2/grub.cfg")

    needs_reboot("Kernel parameters changed — reboot to apply GTT memory settings")


@stage("gpu_groups", "Add user to render/video groups for GPU access", requires_reboot=True)
def stage_gpu_groups(user: str) -> None:
    if not user:
        log.error("  Cannot determine non-root user. Use --user flag.")
        sys.exit(1)

    # Check current groups
    result = run_quiet(f"id -nG {user}")
    current_groups = set(result.stdout.strip().split())

    needed = {"render", "video"}
    missing = needed - current_groups
    if not missing:
        log.info("  User '%s' already in render/video groups", user)
        return

    for group in missing:
        run(f"usermod -aG {group} {user}")
        log.info("  Added %s to group '%s'", user, group)

    needs_reboot("Group membership changed — reboot for GPU access to take effect")


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
            check=False)  # Non-fatal if display issues in headless


@stage("toolbox_setup", "Install Podman/toolbox and create ROCm toolbox")
def stage_toolbox_setup(user: str) -> None:
    # Ensure toolbox/podman installed
    run("dnf install -y toolbox podman")

    # Check if container already exists (via podman — authoritative source)
    exists = run_quiet(
        f'su - {user} -c "podman container exists {ROCM_TOOLBOX_NAME}"',
        check=False,
    )
    if exists.returncode == 0:
        log.info("  Container '%s' already exists", ROCM_TOOLBOX_NAME)
        return

    # Ensure rootless podman requirements (subuid/subgid)
    for db in ["/etc/subuid", "/etc/subgid"]:
        content = Path(db).read_text() if Path(db).exists() else ""
        if user not in content:
            log.info("  Adding %s to %s for rootless podman", user, db)
            run(f'usermod --add-subuids 100000-165535 --add-subgids 100000-165535 {user}')
            break

    # Pull the image as the non-root user so it lands in their podman storage.
    log.info("  Pulling %s as %s (this may take a while)...", ROCM_TOOLBOX_IMAGE, user)
    run(f'su - {user} -c "podman pull {ROCM_TOOLBOX_IMAGE}"')

    # Create the container directly with podman instead of toolbox.
    # toolbox create often fails on third-party (non-Fedora) images due to
    # missing toolbox-specific labels. podman create gives the same result
    # with explicit control over bind mounts and device access.
    log.info("  Creating container '%s' via podman...", ROCM_TOOLBOX_NAME)
    result = run(
        f'su - {user} -c "'
        f"podman create"
        f" --name {ROCM_TOOLBOX_NAME}"
        f" --hostname toolbox"
        f" --privileged"
        f" --security-opt label=disable"
        f" --device /dev/dri"
        f" --device /dev/kfd"
        f" --group-add video"
        f" --group-add render"
        f" --userns=keep-id"
        f" --pid=host"
        f" --network=host"
        f" --volume /mnt/data:/mnt/data:rslave"
        f" --volume /run/user/$(id -u):/run/user/$(id -u):rslave"
        f" {ROCM_TOOLBOX_IMAGE}"
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
    log.info("  Toolbox '%s' created successfully", ROCM_TOOLBOX_NAME)


@stage("model_directories", "Create model directories and download models")
def stage_model_directories(user: str) -> None:
    # Create directories
    for subdir in ["llm", "embedding"]:
        (MODELS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Set ownership
    run(f"chown -R {user}:{user} {MODELS_DIR}")

    # Install huggingface CLI in a temporary way (system pip or user pip)
    run("pip3 install --break-system-packages huggingface_hub 2>/dev/null || "
        "pip3 install huggingface_hub", check=False)

    # Download embedding model (if not already present)
    embed_model = MODELS_DIR / "embedding" / "nomic-embed-text-v1.5.f16.gguf"
    if not embed_model.exists():
        log.info("  Downloading embedding model...")
        run(
            f'su - {user} -c "'
            f"huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF "
            f"nomic-embed-text-v1.5.f16.gguf "
            f'--local-dir {MODELS_DIR}/embedding/"'
        )
    else:
        log.info("  Embedding model already present")

    # Download LLM (if not already present)
    llm_model = MODELS_DIR / "llm" / "qwen2.5-7b-instruct-q4_k_m.gguf"
    if not llm_model.exists():
        log.info("  Downloading LLM model (Qwen 2.5 7B)...")
        run(
            f'su - {user} -c "'
            f"huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF "
            f"qwen2.5-7b-instruct-q4_k_m.gguf "
            f'--local-dir {MODELS_DIR}/llm/"'
        )
    else:
        log.info("  LLM model already present")


@stage("llama_server", "Deploy Podman Quadlet services for llama.cpp servers")
def stage_llama_server(user: str) -> None:
    quadlet_dir = pathlib.Path("/etc/containers/systemd")
    quadlet_dir.mkdir(parents=True, exist_ok=True)

    # Environment file for ROCm
    write_file(
        "/etc/sysconfig/llama-server",
        textwrap.dedent("""\
            # Required for Strix Halo gfx1151
            HSA_OVERRIDE_GFX_VERSION=11.0.0
            # Enable unified memory for large models on APU
            GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
        """),
    )

    # LLM Server Quadlet (Port 1234)
    write_file(
        quadlet_dir / "llama-server-llm.container",
        textwrap.dedent(f"""\
            [Unit]
            Description=llama.cpp LLM Server (OpenAI-compatible)
            After=network-online.target

            [Container]
            Image={ROCM_TOOLBOX_IMAGE}
            Exec=llama-server \\
                --model /models/llm/qwen2.5-7b-instruct-q4_k_m.gguf \\
                --host 0.0.0.0 \\
                --port 1234 \\
                --n-gpu-layers 999 \\
                --flash-attn \\
                --no-mmap \\
                --ctx-size 8192 \\
                --threads 16 \\
                --parallel 2 \\
                --alias "gpt-oss-20b"
            Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0
            Environment=GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
            AddDevice=/dev/kfd
            AddDevice=/dev/dri
            Volume={MODELS_DIR}:/models:ro
            PublishPort=1234:1234
            GroupAdd=video
            GroupAdd=render

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
            Description=llama.cpp Embedding Server (OpenAI-compatible)
            After=network-online.target

            [Container]
            Image={ROCM_TOOLBOX_IMAGE}
            Exec=llama-server \\
                --model /models/embedding/nomic-embed-text-v1.5.f16.gguf \\
                --host 0.0.0.0 \\
                --port 1235 \\
                --n-gpu-layers 999 \\
                --flash-attn \\
                --no-mmap \\
                --ctx-size 2048 \\
                --threads 8 \\
                --embedding \\
                --alias "text-embedding-nomic-embed-text-v1.5@f16"
            Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0
            Environment=GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
            AddDevice=/dev/kfd
            AddDevice=/dev/dri
            Volume={MODELS_DIR}:/models:ro
            PublishPort=1235:1235
            GroupAdd=video
            GroupAdd=render

            [Service]
            Restart=on-failure
            RestartSec=10

            [Install]
            WantedBy=multi-user.target default.target
        """),
    )

    run("systemctl daemon-reload")
    run("systemctl enable --now llama-server-llm", check=False)
    run("systemctl enable --now llama-server-embed", check=False)


@stage("python_venv", "Create Python venv with PyTorch ROCm and project dependencies")
def stage_python_venv(user: str) -> None:
    # Ensure python3-venv is available
    run("dnf install -y python3-devel python3-pip python3-venv python3-setuptools")

    # Create venv if it doesn't exist
    if not (VENV_DIR / "bin" / "activate").exists():
        log.info("  Creating venv at %s", VENV_DIR)
        run(f'su - {user} -c "python3 -m venv {VENV_DIR}"')
    else:
        log.info("  Venv already exists at %s", VENV_DIR)

    pip = f"{VENV_DIR}/bin/pip"

    # Install PyTorch with ROCm
    log.info("  Installing PyTorch ROCm...")
    run(f'su - {user} -c "{pip} install torch torchvision torchaudio '
        f'--index-url https://download.pytorch.org/whl/rocm6.3"')

    # Training dependencies
    log.info("  Installing training dependencies...")
    run(f'su - {user} -c "{pip} install transformers datasets accelerate peft '
        f'bitsandbytes sentencepiece tiktoken tokenizers huggingface_hub"')

    # Pipeline dependencies
    log.info("  Installing pipeline dependencies...")
    run(f'su - {user} -c "{pip} install fastapi uvicorn psycopg2-binary opensearch-py '
        f'mediawiki-dump mwparserfromhell sentence-transformers pydantic requests tqdm"')

    # Add ROCm env vars to venv activate script
    activate = VENV_DIR / "bin" / "activate"
    rocm_marker = "# ROCm settings for Strix Halo"
    if not file_contains(activate, rocm_marker):
        log.info("  Adding ROCm environment variables to venv activate script")
        with open(activate, "a") as f:
            f.write(textwrap.dedent(f"""

                {rocm_marker}
                export HSA_OVERRIDE_GFX_VERSION=11.0.0
                export ROCBLAS_USE_HIPBLASLT=1
                export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
            """))

    run(f"chown -R {user}:{user} {VENV_DIR}")


@stage("postgresql", "Install and configure PostgreSQL for Wikipedia pipeline")
def stage_postgresql(user: str) -> None:
    run("dnf install -y postgresql-server postgresql-contrib")

    # Initialize if not already done
    pgdata = pathlib.Path("/var/lib/pgsql/data/PG_VERSION")
    if not pgdata.exists():
        run("postgresql-setup --initdb")

    run("systemctl enable --now postgresql")

    # Create wiki user and database (idempotent)
    run('su - postgres -c "psql -tc \\"SELECT 1 FROM pg_roles WHERE rolname=\'wiki\';\\"" '
        '| grep -q 1 || su - postgres -c "createuser wiki"')
    run('su - postgres -c "psql -tc \\"SELECT 1 FROM pg_database WHERE datname=\'wikidb\';\\"" '
        '| grep -q 1 || su - postgres -c "createdb -O wiki wikidb"')
    run("su - postgres -c 'psql -c \"ALTER USER wiki WITH PASSWORD '\\''wiki'\\''\"'",
        check=False)

    log.info("  PostgreSQL configured: user=wiki, db=wikidb")


@stage("opensearch", "Download, configure, and deploy OpenSearch")
def stage_opensearch(user: str) -> None:
    os_dir = pathlib.Path("/opt/opensearch")

    if not os_dir.exists():
        tarball = f"/tmp/opensearch-{OPENSEARCH_VERSION}-linux-x64.tar.gz"
        if not pathlib.Path(tarball).exists():
            log.info("  Downloading OpenSearch %s...", OPENSEARCH_VERSION)
            run(f"wget -q -O {tarball} {OPENSEARCH_URL}")

        run(f"tar -xzf {tarball} -C /opt")
        run(f"mv /opt/opensearch-{OPENSEARCH_VERSION} {os_dir}")
    else:
        log.info("  OpenSearch already present at %s", os_dir)

    # Create opensearch user
    result = run_quiet("id opensearch", check=False)
    if result.returncode != 0:
        run("useradd -r -s /sbin/nologin opensearch")

    run(f"chown -R opensearch:opensearch {os_dir}")

    # Configure
    config = os_dir / "config" / "opensearch.yml"
    if not file_contains(config, "plugins.security.disabled"):
        with open(config, "a") as f:
            f.write("\nplugins.security.disabled: true\n")
            f.write("network.host: 0.0.0.0\n")
            f.write("discovery.type: single-node\n")

    # JVM heap
    jvm_opts = os_dir / "config" / "jvm.options"
    if file_contains(jvm_opts, "-Xms1g"):
        run(f"sed -i 's/-Xms1g/-Xms8g/' {jvm_opts}")
        run(f"sed -i 's/-Xmx1g/-Xmx8g/' {jvm_opts}")

    # Systemd service
    write_file(
        "/etc/systemd/system/opensearch.service",
        textwrap.dedent("""\
            [Unit]
            Description=OpenSearch
            After=network.target

            [Service]
            Type=simple
            User=opensearch
            Group=opensearch
            ExecStart=/opt/opensearch/bin/opensearch
            Restart=on-failure
            LimitNOFILE=65536
            LimitMEMLOCK=infinity

            [Install]
            WantedBy=multi-user.target
        """),
    )

    # Kernel settings
    sysctl_file = pathlib.Path("/etc/sysctl.d/opensearch.conf")
    if not file_contains(sysctl_file, "vm.max_map_count"):
        write_file(sysctl_file, "vm.max_map_count=262144\n")
        run("sysctl -p /etc/sysctl.d/opensearch.conf")

    run("systemctl daemon-reload")
    run("systemctl enable --now opensearch.service")


@stage("mcp_server", "Deploy MCP server and web GUI systemd service")
def stage_mcp_server(user: str) -> None:
    scripts_dest = DATA_DIR / "wikipedia" / "scripts"
    scripts_dest.mkdir(parents=True, exist_ok=True)

    # Copy project scripts
    src_scripts = REPO_DIR / "scripts"
    for py_file in src_scripts.glob("*.py"):
        shutil.copy2(py_file, scripts_dest)

    src_webapp = REPO_DIR / "webapp"
    if src_webapp.exists():
        for f in src_webapp.iterdir():
            shutil.copy2(f, scripts_dest)

    # Create wiki user if needed (system user for the service)
    result = run_quiet("id wiki", check=False)
    if result.returncode != 0:
        run(f"useradd -r -s /sbin/nologin -d {DATA_DIR / 'wikipedia'} wiki")

    run(f"chown -R wiki:wiki {DATA_DIR / 'wikipedia'}")

    # Systemd service
    write_file(
        "/etc/systemd/system/mcp.service",
        textwrap.dedent(f"""\
            [Unit]
            Description=Wikipedia MCP Server
            After=network.target opensearch.service postgresql.service llama-server-embed.service

            [Service]
            Type=simple
            User=wiki
            Group=wiki
            Environment="WIKI_DATA={DATA_DIR / 'wikipedia'}"
            Environment="LMSTUDIO_HOST=localhost"
            Environment="LMSTUDIO_PORT=1235"
            WorkingDirectory={scripts_dest}
            ExecStart={VENV_DIR}/bin/uvicorn mcp_server:app --host 0.0.0.0 --port 7000
            Restart=on-failure

            [Install]
            WantedBy=multi-user.target
        """),
    )

    run("systemctl daemon-reload")
    run("systemctl enable --now mcp.service", check=False)


@stage("firewall", "Configure firewalld rules for all service ports")
def stage_firewall(user: str) -> None:
    run("dnf install -y firewalld")

    # IMPORTANT: Add SSH rule BEFORE enabling firewalld to avoid locking
    # ourselves out on a headless server. firewall-cmd --permanent works
    # even when the daemon is stopped — it writes to the permanent config
    # that will be loaded when the service starts.
    run("firewall-cmd --permanent --add-service=ssh", check=False)

    ports = [
        ("1234/tcp", "port"),   # llama LLM
        ("1235/tcp", "port"),   # llama embedding
        ("7000/tcp", "port"),   # MCP server
        ("8080/tcp", "port"),   # Web GUI
        ("9200/tcp", "port"),   # OpenSearch
    ]

    for spec, kind in ports:
        run(f"firewall-cmd --permanent --add-port={spec}", check=False)

    # Now safe to enable — SSH is already in the permanent config
    run("systemctl enable --now firewalld")
    run("firewall-cmd --reload")


@stage("ethernet_fix", "Check and apply Realtek r8169 ethernet fix if needed")
def stage_ethernet_fix(user: str) -> None:
    result = run_quiet("lspci -k", check=False)
    if "r8169" not in result.stdout:
        log.info("  r8169 driver not in use — skipping ethernet fix")
        return

    # Check kernel version — 6.18+ should already have the fix
    uname = run_quiet("uname -r")
    kernel = uname.stdout.strip()
    log.info("  r8169 detected, kernel=%s", kernel)

    # Parse major.minor
    parts = kernel.split(".")
    major, minor = int(parts[0]), int(parts[1])
    if major > 6 or (major == 6 and minor >= 18):
        log.info("  Kernel %s should include the r8169 fix — skipping", kernel)
        return

    patch_file = REPO_DIR / "patches" / "r8169_main_fix.patch"
    if patch_file.exists():
        log.info("  Patch available at %s — manual application may be needed", patch_file)
        log.info("  See: https://bugzilla.kernel.org/show_bug.cgi?id=220770")
    else:
        log.info("  No patch file found and kernel may need r8169 fix")


@stage("llm_swap_helper", "Install /usr/local/bin/llm-swap helper script")
def stage_llm_swap_helper(user: str) -> None:
    write_file(
        "/usr/local/bin/llm-swap",
        textwrap.dedent("""\
            #!/bin/bash
            # Usage: llm-swap <model-path> [alias] [ctx-size]
            MODEL="${1:?Usage: llm-swap <model-path> [alias] [ctx-size]}"
            ALIAS="${2:-gpt-oss-20b}"
            CTX="${3:-8192}"

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
                sudo sed -i "s|--alias \\"[^\\"]*\\"|--alias \\"$ALIAS\\"|" "$QUADLET_FILE"
                echo "Updated Quadlet: $QUADLET_FILE"
            else
                sudo mkdir -p /etc/systemd/system/${SERVICE_NAME}.service.d
                sudo tee /etc/systemd/system/${SERVICE_NAME}.service.d/model.conf <<EOF
            [Service]
            ExecStart=
            ExecStart=/opt/llama.cpp/build/bin/llama-server \\
                --model $MODEL \\
                --host 0.0.0.0 \\
                --port 1234 \\
                --n-gpu-layers 999 \\
                --flash-attn \\
                --no-mmap \\
                --ctx-size $CTX \\
                --threads 16 \\
                --parallel 2 \\
                --alias "$ALIAS"
            EOF
            fi

            sudo systemctl daemon-reload
            sudo systemctl restart "$SERVICE_NAME"
            echo "Swapped to: $MODEL (alias: $ALIAS, ctx: $CTX)"
            sudo systemctl status "$SERVICE_NAME" --no-pager -l
        """),
        mode=0o755,
    )


@stage("verify", "Run health checks on all components")
def stage_verify(user: str) -> None:
    checks = [
        ("Kernel ≥ 6.18.4", "uname -r"),
        ("Firmware (not 20251125)", "rpm -q linux-firmware"),
        ("GPU groups", f"id -nG {user}"),
        ("GPU device nodes", "ls /dev/kfd /dev/dri/render* 2>/dev/null"),
        ("Podman", "podman --version"),
        ("Toolbox list", "toolbox list --containers"),
        ("PostgreSQL", "pg_isready"),
        ("OpenSearch", "curl -sf http://localhost:9200 -o /dev/null && echo OK || echo DOWN"),
        ("LLM server", "curl -sf http://localhost:1234/v1/models -o /dev/null && echo OK || echo DOWN"),
        ("Embedding server", "curl -sf http://localhost:1235/v1/models -o /dev/null && echo OK || echo DOWN"),
        ("MCP server", "curl -sf http://localhost:7000/health -o /dev/null && echo OK || echo DOWN"),
        ("VSCode", "code --version 2>/dev/null | head -1 || echo 'not found'"),
    ]

    log.info("")
    log.info("═══ Health Check ═══")
    all_ok = True
    for label, cmd in checks:
        result = run_quiet(cmd, check=False)
        output = (result.stdout or "").strip()
        status = "✓" if result.returncode == 0 and "DOWN" not in output else "✗"
        if status == "✗":
            all_ok = False
        log.info("  %s %-25s %s", status, label, output[:60])

    log.info("")
    if all_ok:
        log.info("  All checks passed!")
    else:
        log.info("  Some checks failed — review above and re-run failed stages")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def list_stages(state: StateTracker) -> None:
    """Print all stages and their status."""
    print(f"\n{'#':>3}  {'Stage':<25} {'Status':<12} Description")
    print(f"{'─'*3}  {'─'*25} {'─'*12} {'─'*40}")
    for i, s in enumerate(STAGES, 1):
        status = "✓ done" if state.is_done(s.name) else "  pending"
        reboot = " ↻" if s.requires_reboot else ""
        print(f"{i:>3}  {s.name:<25} {status:<12} {s.description}{reboot}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strix Halo Fedora automated setup",
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
        log.error("%s does not exist. Follow Phase 1 in StrixHalo-Fedora-Setup.md first.", DATA_DIR)
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
    log.info("║  Strix Halo Fedora Setup                                   ║")
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

        log.info("")
        log.info("━━━ [%d/%d] Stage: %s — %s ━━━", i + 1, len(STAGES), s.name, s.description)
        state.mark_pending(s.name)

        try:
            s.func(user)
        except SystemExit:
            # needs_reboot() calls sys.exit(0) — state is pending, will resume
            raise
        except Exception:
            log.exception("  Stage '%s' failed!", s.name)
            log.info("  Fix the issue and re-run: sudo python3 %s", __file__)
            log.info("  Or re-run just this stage: sudo python3 %s --stage %s --force", __file__, s.name)
            sys.exit(1)

        state.mark_done(s.name)
        log.info("  ✓ Stage '%s' completed", s.name)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  Setup complete!                                           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Load environment:   source %s/deepred-env.sh", REPO_DIR)
    log.info("  2. Enter the toolbox:  podman start %s && podman exec -it %s bash", ROCM_TOOLBOX_NAME, ROCM_TOOLBOX_NAME)
    log.info("  3. See documentation:  %s/documentation/", REPO_DIR)


if __name__ == "__main__":
    main()
