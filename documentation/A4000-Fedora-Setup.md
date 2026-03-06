
# NVIDIA A4000 Fedora Setup

## Overview

This guide covers the setup of a secondary inference server running Fedora 43 on a VM with an NVIDIA A4000 (16 GB VRAM) passed through via PCI passthrough. The system provides **LLM and embedding servers** that offload inference workloads from the primary StrixHalo system, increasing overall throughput for data preparation and training pipelines.

**What this system provides:**
- OpenAI-compatible LLM server (port 1234)
- OpenAI-compatible embedding server (port 1235)
- User-swappable models via `llm-swap` helper
- CUDA toolbox container for interactive GPU work

**What this system does NOT provide:**
- Wikipedia pipeline, PostgreSQL, OpenSearch, MCP server, or web GUI — those remain on StrixHalo

**Two-phase approach:**
1. **Manual** (this document): Fedora install, SSH, disable sleep, clone the repo
2. **Automated** (`scripts/setup_a4000.py`): Everything else — NVIDIA drivers, container toolkit, toolbox, models, llama.cpp servers, Python venv, firewall, verification

---

## System Requirements

- **Hypervisor**: VM with PCI passthrough for the NVIDIA GPU
- **GPU**: NVIDIA A4000 (16 GB GDDR6, Ampere / SM 8.6)
- **RAM**: 24 GB
- **System Disk**: 100 GB+ (OS + models + repo — no separate data disk required)
- **OS**: Fedora 43

### Tested Stable Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| **OS** | Fedora 43 | |
| **Linux Kernel** | 6.18+ | Standard Fedora kernel |
| **NVIDIA Driver** | 570+ (RPM Fusion akmod) | Ampere (SM 8.6) support — automatically rebuilt on kernel updates |
| **NVIDIA Container Toolkit** | 1.18+ | CDI-based GPU injection for Podman (auto-generates `/etc/cdi/nvidia.yaml`) |
| **llama.cpp** | b4719+ (`server-cuda-b4719` / `full-cuda-b4719`) | Official GHCR images — tags pinned to build number |

### VRAM Budget (16 GB)

| Model | VRAM Usage | Notes |
|-------|-----------|-------|
| Qwen 2.5 14B Q4_K_M (LLM, default) | ~10–11 GB | Model (~9 GB) + KV cache (2 slots × 8192 ctx) |
| nomic-embed-text-v1.5 F16 (embedding) | ~0.5 GB | Lightweight embedding model; `--batch-size 32768` + `--ubatch-size 2048` for pipeline batch throughput |
| **Total (both running)** | **~11–12 GB** | Leaves 4–5 GB headroom |

> **Note:** The 14B model runs with `--parallel 2` (2 concurrent request slots) to fit within 16 GB VRAM. The Qwen 2.5 7B Q4_K_M (~5–7 GB) is also downloaded as a fallback — use `llm-swap` to switch models without rebuilding.

### Why This Architecture

| Factor | Standalone Build | Container-based (chosen) |
|--------|-----------------|--------------------------|
| **CUDA management** | Manual CUDA toolkit install + PATH wrangling | Container bundles correct CUDA version |
| **Updates** | Rebuild llama.cpp manually | `podman pull` new image |
| **Isolation** | System-wide dependency conflicts | Each container has its own libs |
| **Reproducibility** | Depends on system state | Image tag pins exact build |

---

## Phase 1: Manual Installation

> **Tip:** Before editing system config files (`/etc/fstab`, `/etc/default/grub`, etc.), back them up: `sudo cp /etc/fstab /etc/fstab.bak`

### Step 1: Install Fedora (VM)

- **Download**: [Fedora Server](https://fedoraproject.org/server/download) (recommended for headless) or [Fedora Workstation](https://fedoraproject.org/workstation/download)
- **Create VM**: Configure PCI passthrough for the NVIDIA A4000 in your hypervisor (Proxmox, libvirt/QEMU, ESXi, etc.)
- **Allocate**: 24 GB RAM, 4+ vCPUs, 100 GB+ disk
- **Install** Fedora using the standard installer

> **PCI Passthrough prerequisite:** The host must have IOMMU enabled (`intel_iommu=on` or `amd_iommu=on` in host kernel parameters) and the A4000 must be in its own IOMMU group. Passthrough configuration is hypervisor-specific and outside the scope of this guide.

### Step 2: Enable SSH for Headless Access

After the initial install (via VM console), enable SSH so all remaining work can be done remotely:

```bash
# Install and enable SSH server
sudo dnf install -y openssh-server
sudo systemctl enable --now sshd

# Open SSH in firewall (Fedora Workstation has firewalld active by default)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload

# Verify SSH is listening
ss -tlnp | grep :22
```

> **From this point on**, you can disconnect the VM console and work entirely via SSH:
> ```bash
> ssh your-user@a4000-vm-ip
> ```

### Step 3: Rename the PC

```bash
sudo hostnamectl set-hostname A4000AI
```

Verify:

```bash
hostnamectl
```

The new hostname takes effect immediately. Your shell prompt will update after a new login. From now on you can SSH:

```bash
ssh your-user@A4000AI
```

### Step 4: System Update

```bash
# Update system
sudo dnf upgrade --refresh -y

# ⚠️ Reboot after kernel update
sudo reboot
```

After reboot (reconnect via SSH), verify:

```bash
uname -r
```

### Step 4a: Disable Sleep/Suspend (Always-On Server)

> **⚠️ Important:** Even in a VM, systemd may attempt to suspend the guest if idle. Disable all sleep states to keep the inference servers running.

```bash
# Disable all sleep targets so systemd never suspends/hibernates
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target

# Disable idle suspend via logind
sudo mkdir -p /etc/systemd/logind.conf.d
cat <<'EOF' | sudo tee /etc/systemd/logind.conf.d/no-sleep.conf
[Login]
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
IdleActionSec=0
EOF
sudo systemctl restart systemd-logind
# ⚠️ The restart above will terminate active desktop sessions.
# SSH sessions are also dropped — just reconnect.
```

If GNOME desktop is installed, disable its auto-suspend too:

```bash
if command -v gsettings &>/dev/null; then
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
fi
```

Verify all sleep targets are masked:

```bash
systemctl status sleep.target suspend.target hibernate.target
# All should show "Loaded: masked"
```

### Step 5: Set Up GitHub SSH Access

```bash
# Generate an SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Start the SSH agent and add the key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Display the public key — copy this to GitHub
cat ~/.ssh/id_ed25519.pub
```

Add the key to your GitHub account:
1. On GitHub: **profile picture** → **Settings** → **SSH and GPG keys** → **New SSH key**
2. Paste the public key and save

Verify:
```bash
ssh -T git@github.com
# Should print: "Hi <username>! You've successfully authenticated..."
```

### Step 6: Create Data Directory and Clone Repository

Since this is a single-disk VM, create the data directory on the root filesystem:

```bash
sudo mkdir -p /mnt/data
sudo chown -R $USER:$USER /mnt/data

sudo dnf install -y git python3 python3-pip

# Clone via SSH (or update existing repo)
if [ -d /mnt/data/DeepRedAI/.git ]; then
  git -C /mnt/data/DeepRedAI pull
else
  git clone git@github.com:ferzkopp/DeepRedAI.git /mnt/data/DeepRedAI
fi
cd /mnt/data/DeepRedAI
```

> **Tip:** If your VM has a second virtual disk for data, mount it at `/mnt/data` via `/etc/fstab` instead — the same approach as the StrixHalo system (see [StrixHalo-Fedora-Setup.md](StrixHalo-Fedora-Setup.md) Step 5).

### Step 7: Configure DeepRedAI Environment

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
```

To auto-load on every login, add to `~/.bashrc`:

```bash
joe ~/.bashrc   # or nano, vi, etc.
```

Append:
```bash
# ── DeepRedAI environment
export DEEPRED_ROOT="/mnt/data"
[ -f "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh" ] && source "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh"
```

Verify:

```bash
# Log out and back in, confirm env vars appear
exit
ssh your-user@A4000AI
```

---

## Phase 2: Automated Setup

The setup script handles all remaining configuration. Run as root:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py
```

Some stages require a reboot. After rebooting, SSH back in and **run the same command again** — the script tracks progress in `.setup_a4000_state.json` and resumes automatically.

| Stage | Name | Reboot? | Description |
|-------|------|---------|-------------|
| 1 | `system_packages` | No | Install build tools and development packages |
| 2 | `disable_sleep` | No | Mask sleep/suspend/hibernate targets for always-on operation |
| 3 | `nvidia_driver` | **Yes** | Install NVIDIA driver from RPM Fusion (reconnect via SSH after reboot) |
| 4 | `nvidia_container_toolkit` | No | Install NVIDIA Container Toolkit with CDI for Podman GPU access |
| 5 | `vscode` | No | Install VSCode + Python and Copilot extensions |
| 6 | `toolbox_setup` | No | Create CUDA toolbox container with llama.cpp |
| 7 | `model_directories` | No | Create model directories and download LLM + embedding models |
| 8 | `llama_server` | No | Deploy Podman Quadlet services for LLM + embedding servers |
| 9 | `python_venv` | No | Create Python venv with utility packages (no PyTorch — inference runs in containers) |
| 10 | `firewall` | No | Configure firewalld rules for service ports |
| 11 | `llm_swap_helper` | No | Install `/usr/local/bin/llm-swap` helper script |
| 12 | `verify` | **Yes** | Run health checks on all components (reboot to confirm boot persistence) |
| 13 | `reverify` | No | Post-reboot health check — verify services survive a restart |

### Script Usage

```bash
# Resume from where it left off (after reboot or interruption)
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py

# Run a specific stage only
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py --stage nvidia_driver

# Re-run a specific stage (even if already completed)
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py --stage llama_server --force

# List all stages and their status
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py --list

# Start from a specific stage (skip earlier stages)
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py --from vscode

# Override the default non-root user
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py --user myuser
```

Stage progress is tracked in `$DEEPRED_REPO/.setup_a4000_state.json`. After a reboot stage, SSH back in, source the env, and re-run — the script resumes automatically.

---

## Post-Setup

### Service Overview

| Service | Port | Bind | Purpose |
|---------|------|------|---------|
| `llama-server-llm` | 1234 | 0.0.0.0 | LLM inference (chat completions) — Podman Quadlet with CUDA |
| `llama-server-embed` | 1235 | 0.0.0.0 | Embedding generation — Podman Quadlet with CUDA |

> **Network exposure:** Ports 1234 and 1235 are opened in firewalld (LAN-accessible). Ensure your network firewall or hypervisor restricts access as needed.

```
┌──────────────────────────────────────┐     ┌─────────────────────────────┐
│  StrixHalo scripts / MCP server      │────▶│  A4000AI: llama-server-llm  │
│  (data preparation, training)        │:1234│  (port 1234, CUDA)          │
└──────────────────────────────────────┘     └─────────────────────────────┘
                                                       │
┌──────────────────────────────────────┐     ┌─────────────────────────────┐
│  StrixHalo: mcp_server.py            │────▶│  A4000AI: llama-server-embed│
│  (can point EMBEDDING_PORT here)     │:1235│  (port 1235, CUDA)          │
└──────────────────────────────────────┘     └─────────────────────────────┘
```

### Integration with StrixHalo

To offload inference from the StrixHalo system to the A4000 VM, set the `REMOTE_HOST` environment variable. This is the recommended approach — `deepred-env.sh` defines the variable and pipeline scripts can use it.

**Temporarily** (current session):

```bash
export REMOTE_HOST=A4000AI
source /mnt/data/DeepRedAI/deepred-env.sh
```

**Permanently** — add to `~/.bashrc` on the StrixHalo system, **before** the `deepred-env.sh` source line:

```bash
# ── DeepRedAI environment
export DEEPRED_ROOT="/mnt/data"
export REMOTE_HOST="A4000AI"                        # ← enable remote GPU server
[ -f "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh" ] && source "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh"
```

**Verify the connection:**

```bash
source $DEEPRED_VENV/bin/activate
python3 $DEEPRED_REPO/scripts/test_remote.py
```

This tests reachability and confirms that remote and local embedding servers produce identical results. See [`WikipediaMCP-Setup.md` — Remote GPU Server](WikipediaMCP-Setup.md#remote-gpu-server) for full details.

> **Legacy approach:** You can also override endpoint variables directly (`export INFERENCE_HOST=A4000AI`), but the `REMOTE_HOST` variable is preferred as it cleanly separates the remote server configuration from local service defaults.

### Swapping Models

```bash
# Swap to a different model
llm-swap $DEEPRED_MODELS/llm/some-other-model.gguf "custom-alias" 4096

# Swap to 7B with 4 parallel slots (lightweight — fits easily in 16 GB)
llm-swap $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    "qwen2.5-7b-instruct" 8192 --slots 4

# Swap back to default 14B with 2 parallel slots (to fit in 16 GB VRAM)
llm-swap $DEEPRED_MODELS/llm/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
    --slots 2
```

### Working Inside the Toolbox

```bash
# Start and enter the CUDA container for interactive GPU work
podman start llama-cuda
podman exec -it llama-cuda bash

# Inside the container — llama-cli, llama-bench, etc. are available
llama-cli --version
```

### Quick Health Check

```bash
# Check all services at once
sudo -E python3 $DEEPRED_REPO/scripts/setup_a4000.py --stage verify --force
```

### VSCode + GitHub Copilot Authentication

The setup script installs VSCode and the Copilot extensions, but you still need to sign in:

1. **Open VSCode** on the A4000 machine (via desktop or `code --tunnel` for remote access)
2. **Sign in to GitHub Copilot**: Click the Copilot icon → **Sign in to GitHub** → follow the device-code flow

> **Headless / SSH-only?** Use [VSCode Remote Tunnels](https://code.visualstudio.com/docs/remote/tunnels): run `code tunnel` on the A4000 VM, then connect from VSCode on your local machine.

---

## References

### NVIDIA Drivers & Container Toolkit
* [RPM Fusion NVIDIA Howto](https://rpmfusion.org/Howto/NVIDIA) — Fedora NVIDIA driver installation
* [NVIDIA Container Toolkit Installation (Fedora)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — CDI setup for Podman/Docker
* [NVIDIA CDI Support in Podman](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) — Container Device Interface documentation

### llama.cpp
* [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) — Source and documentation
* [llama.cpp Container Images (GHCR)](https://github.com/ggerganov/llama.cpp/pkgs/container/llama.cpp) — Official Docker/Podman images
* [llama.cpp Server Documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)

### Podman
* [Podman Quadlet Documentation](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html)
* [Podman GPU Support with CDI](https://docs.podman.io/en/latest/markdown/podman-run.1.html#device-host-device)

### NVIDIA A4000
* [NVIDIA A4000 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/a4000-702914-ds-nv-US-1p-web.pdf) — 16 GB GDDR6, Ampere SM 8.6
* [PCI Passthrough (Proxmox Wiki)](https://pve.proxmox.com/wiki/PCI_Passthrough) — GPU passthrough for VMs

### DeepRedAI
* [StrixHalo Fedora Setup](StrixHalo-Fedora-Setup.md) — Primary system setup guide