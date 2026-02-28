# Strix Halo Fedora Setup

## Overview

This guide covers the manual steps for a fresh Fedora installation on an AMD Strix Halo system, followed by an **automated setup script** that handles everything from kernel configuration through service deployment.

**Two-phase approach:**
1. **Manual** (this document): Fedora install, disk setup, clone the repo, bootstrap Python
2. **Automated** (`scripts/setup_strixhalo.py`): Everything else — GTT memory, toolboxes, llama.cpp servers, Python venv, PostgreSQL, OpenSearch, MCP server, firewall, VSCode

---

## System Requirements

- **Hardware**: AMD Ryzen AI MAX+ 395 "Strix Halo" (gfx1151)
- **RAM**: 128 GB LPDDR5x (unified CPU+GPU memory)
- **System Disk**: 1 TB (Fedora OS, `/`)
- **Data Disk**: 4 TB (models, Wikipedia pipeline, project repo — mounted at `/mnt/data`)
- **OS**: Fedora 42 or 43

### Tested Stable Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| **OS** | Fedora 42 or 43 | |
| **Linux Kernel** | 6.18.6-200+ | Kernels < 6.18.4 have gfx1151 bugs — **avoid them** |
| **Linux Firmware** | 20260110+ | **Do NOT use `linux-firmware-20251125`** — breaks ROCm on Strix Halo |
| **ROCm (toolbox)** | 7.2 (AMD repo) | Latest stable; kernel 6.18.4+ compatibility. ROCm 6.4.4 available as fallback. |

> **⚠️ Critical:** The kernel, firmware, and ROCm versions must be compatible. ROCm 7.1.1 is **incompatible** with kernels ≥ 6.18.4 and has been deprecated. Always use ROCm 7.2+ with modern kernels. ROCm 6.4.4 is available as a fallback if you encounter regressions — change the image tag in the setup script.

### Why Fedora Instead of Ubuntu

| Factor | Ubuntu 25.10 | Fedora 42/43 |
|--------|-------------|------------|
| **Kernel** | 6.12+ | 6.18+ (critical for Strix Halo stability) |
| **AMD GPU support** | Requires manual ROCm repo setup | Strong out-of-box AMD support |
| **Toolbox/Podman** | Available but not default | First-class citizen (pre-installed) |
| **ROCm** | Manual repo + pinning | Available via native Fedora packages or AMD repos |

### Why llama.cpp Server Instead of LM Studio

| Issue | LM Studio | llama.cpp server |
|-------|-----------|-----------------|
| **Server management** | Requires Xvfb + VNC + AppImage | Native CLI daemon, simple systemd unit |
| **Updates** | Manual AppImage download | `git pull && cmake --build` |
| **Resource usage** | Electron app + GUI in memory | Minimal — just the inference engine |
| **OpenAI compatibility** | ✅ `/v1/chat/completions` | ✅ Same endpoints, same API |

---

## Phase 1: Manual Installation

### Step 1: Install Fedora

- **Download**: [Fedora Workstation](https://fedoraproject.org/workstation/download) or Fedora Server
- **Create USB**: [Fedora Media Writer](https://docs.fedoraproject.org/en-US/fedora/latest/preparing-boot-media/) or `dd`
- **Install** to the **1 TB system disk** using the Fedora installer

#### BIOS Preparation (Before Fedora Install)

Enter BIOS and look for:
- **UMA Frame Buffer Size** → Set to maximum (e.g., 96 GB or "Auto" with large allocation)
- **VRAM Size** or **iGPU Memory** → Maximize

> **Note:** BIOS options vary by vendor. Some Strix Halo systems (e.g., Framework Laptop) expose this; others don't — the setup script will use kernel parameters as a fallback.

### Step 2: Enable SSH for Headless Access

After the initial install (via KVM or local console), enable SSH so all remaining work can be done remotely:

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

Optionally, set a static IP or hostname for reliable remote access:

```bash
# Set hostname
sudo hostnamectl set-hostname strixhalo

# (Optional) Set a static IP — adjust connection name, IP, gateway, DNS
# Find your connection name:
nmcli con show
# Then configure it:
sudo nmcli con mod "Wired connection 1" \
    ipv4.method manual \
    ipv4.addresses 192.168.1.100/24 \
    ipv4.gateway 192.168.1.1 \
    ipv4.dns "192.168.1.1 8.8.8.8"
sudo nmcli con up "Wired connection 1"
```

> **From this point on**, you can disconnect KVM and work entirely via SSH:
> ```bash
> ssh your-user@strixhalo   # or ssh your-user@192.168.1.100
> ```

### Step 3: System Update

```bash
# Update system (critical: ensures kernel 6.18.4+ and firmware 20260110+)
sudo dnf upgrade --refresh -y

# ⚠️ Reboot after kernel/firmware update
sudo reboot
```

After reboot (reconnect via SSH), verify:

```bash
# Must be 6.18.4+
uname -r

# Must NOT be 20251125
rpm -q linux-firmware
```

> **⚠️ Do not proceed** if your kernel is older than 6.18.4 or firmware is `linux-firmware-20251125`. Update first: `sudo dnf upgrade linux-firmware kernel --refresh`.

### Step 4: Data Disk Setup

Format and mount the **4 TB data disk**. Identify it first:

```bash
# List disks — find the 4 TB drive (e.g., /dev/nvme1n1 or /dev/sdb)
lsblk
```

```bash
# Format the data disk (adjust /dev/nvme1n1 to your actual device)
sudo mkfs.ext4 -L data /dev/nvme1n1

# Create mount point
sudo mkdir -p /mnt/data

# Add to fstab for persistent mount
echo 'LABEL=data /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab

# Mount now
sudo mount -a

# Set ownership to your user
sudo chown -R $USER:$USER /mnt/data
```

### Step 5: Clone This Repository

```bash
# Install git and bootstrap tools
sudo dnf install -y git python3 python3-pip

# Clone the project
git clone https://github.com/aschiffler/DeepRedAI.git /mnt/data/DeepRedAI
cd /mnt/data/DeepRedAI
```

---

## Phase 2: Automated Setup

The setup script handles all remaining configuration. Run as root:

```bash
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py
```

The script runs through these stages in order:

| Stage | Name | Reboot? | Description |
|-------|------|---------|-------------|
| 1 | `system_packages` | No | Install build tools, development packages |
| 2 | `gtt_memory` | **Yes** | Configure kernel parameters for GPU memory, regenerate GRUB (reconnect via SSH after reboot) |
| 3 | `gpu_groups` | **Yes** | Add user to `render`/`video` groups (reconnect via SSH after reboot) |
| 4 | `vscode` | No | Install VSCode + Python and Copilot extensions |
| 5 | `toolbox_setup` | No | Install Podman/toolbox, create ROCm toolbox |
| 6 | `model_directories` | No | Create `/mnt/data/models/{llm,embedding}`, download models |
| 7 | `llama_server` | No | Deploy Podman Quadlet services for LLM + embedding servers |
| 8 | `python_venv` | No | Create venv on `/mnt/data`, install PyTorch ROCm + dependencies |
| 9 | `postgresql` | No | Install, initialize, configure PostgreSQL + wiki database |
| 10 | `opensearch` | No | Download, configure, deploy OpenSearch as systemd service |
| 11 | `mcp_server` | No | Deploy MCP server + web GUI systemd service |
| 12 | `firewall` | No | Configure firewalld rules for all service ports |
| 13 | `ethernet_fix` | No | Check and apply Realtek r8169 fix if needed |
| 14 | `llm_swap_helper` | No | Install `/usr/local/bin/llm-swap` helper script |
| 15 | `verify` | No | Run health checks on all components |

### Script Usage

```bash
# Resume from where it left off (after reboot or interruption)
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py

# Run a specific stage only
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py --stage gtt_memory

# Re-run a specific stage (even if already completed)
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py --stage postgresql --force

# List all stages and their status
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py --list

# Start from a specific stage (skip earlier stages)
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py --from vscode

# Override the default non-root user (auto-detected from /mnt/data ownership)
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py --user myuser
```

Stage progress is tracked in `/mnt/data/DeepRedAI/.setup_state.json`. After a reboot (e.g., after stage 2 or 3), simply re-run the script — it resumes from the next incomplete stage.

---

## Post-Setup

### Service Overview

| Service | Port | Purpose |
|---------|------|---------|
| `llama-server-llm` | 1234 | LLM inference (chat completions) — Podman Quadlet |
| `llama-server-embed` | 1235 | Embedding generation — Podman Quadlet |
| `opensearch.service` | 9200 | Full-text and semantic search |
| `postgresql.service` | 5432 | Wikipedia metadata storage |
| `mcp.service` | 7000 | Wikipedia MCP server + Web GUI |

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  webapp/     │────▶│  mcp_server.py   │────▶│  llama-server-embed      │
│  App.jsx     │:7000│  (FastAPI)       │:1235│  (port 1235)             │
└──────────────┘     └────────┬─────────┘     └──────────────────────────┘
                              │
                     ┌────────────────┐
                     │  OpenSearch +  │
                     │  PostgreSQL    │
                     └────────────────┘

┌────────────────────────────────────┐     ┌──────────────────────────┐
│ generate_theme/temporal_datasets   │────▶│  llama-server-llm        │
│ .py  [inside toolbox]             │:1234│  (port 1234)             │
└────────────────────────────────────┘     └──────────────────────────┘
```

### Swapping Models

```bash
# Swap to a different model
llm-swap /mnt/data/models/llm/deepred-1b-q4_k_m.gguf "deepred/deepred" 4096

# Swap back to default
llm-swap /mnt/data/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf
```

### Working Inside the Toolbox

```bash
# Enter the ROCm toolbox for interactive AI work
toolbox enter llama-rocm-7.2

# Activate the Python venv inside
source /mnt/data/venv/bin/activate
```

### Quick Health Check

```bash
# Check all services at once
sudo python3 /mnt/data/DeepRedAI/scripts/setup_strixhalo.py --stage verify --force
```

### Script Migration: `lms` CLI to llama-server

| LM Studio Pattern | llama.cpp Server Equivalent |
|---|---|
| `lms load <model> --gpu=max` | `llm-swap /path/to/model.gguf` |
| `lms unload --all` | `sudo systemctl stop llama-server-llm` |
| `lms ps` | `curl localhost:1234/v1/models` |
| API on `localhost:1234` | Identical — no change |
| Embeddings on `localhost:1234` | Changed to `localhost:1235` (separate server) |

All `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` calls work identically with llama.cpp server.

---

## References

### Strix Halo Toolboxes & Configuration
* [AMD Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — Pre-built containers with ROCm + llama.cpp for gfx1151
* [Strix Halo Toolboxes on DockerHub](https://hub.docker.com/r/kyuz0/amd-strix-halo-toolboxes/tags) — Available image tags
* [Strix Halo Benchmarks (Interactive)](https://kyuz0.github.io/amd-strix-halo-toolboxes/) — Performance data across ROCm versions
* [Strix Halo VRAM Estimator](https://github.com/kyuz0/amd-strix-halo-toolboxes/blob/main/docs/vram-estimator.md)

### Known Issues & Workarounds
* [ROCm is very sensitive to kernel version (Issue #45)](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/45) — Kernel/firmware/ROCm compatibility matrix
* [ROCm 7 Performance Regression Workaround](https://github.com/llvm/llvm-project/pull/147700) — Mitigated with `-mllvm --amdgpu-unroll-threshold-local=600`
* [Read error: Bad address (Issue #41)](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/41) — `--no-mmap` required
* [Build 8070 prefill regression (Issue #58)](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/58)

### General References
* [AMD Strix Halo Llama.cpp Installation Guide for Fedora 42](https://community.frame.work/t/amd-strix-halo-llama-cpp-installation-guide-for-fedora-42/75856)
* [Podman Quadlet Documentation](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html)
* [Fedora Toolbox Documentation](https://docs.fedoraproject.org/en-US/fedora-silverblue/toolbox/)
* [llama.cpp Server Documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
* [ROCm Installation (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
* [Increasing VRAM on AMD AI APUs](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux)
* [StrixHalo Wiki](https://strixhalo.wiki/)
* [Strix Halo Home Lab (deseven)](https://strixhalo-homelab.d7.wtf/)
* [Ethernet Patch (Kernel Bugzilla)](https://bugzilla.kernel.org/show_bug.cgi?id=220770)
