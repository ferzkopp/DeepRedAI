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
- **Data Disk**: 4 TB (models, Wikipedia pipeline, project repo — mounted at `/mnt/data` by default, configurable via `DEEPRED_ROOT`)
- **OS**: Fedora 43

### Tested Stable Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| **OS** | Fedora 43 | |
| **Linux Kernel** | 6.18.6-200+ | Kernels < 6.18.4 have gfx1151 bugs — **avoid them** |
| **Linux Firmware** | 20260110+ | **Do NOT use `linux-firmware-20251125`** — breaks ROCm on Strix Halo |
| **ROCm (toolbox)** | 7.2 (AMD repo) | Latest stable; kernel 6.18.4+ compatibility. ROCm 6.4.4 available as fallback. |

> **⚠️ Critical:** The kernel, firmware, and ROCm versions must be compatible. ROCm 7.1.1 is **incompatible** with kernels ≥ 6.18.4 and has been deprecated. Always use ROCm 7.2+ with modern kernels. ROCm 6.4.4 is available as a fallback if you encounter regressions — change the image tag in the setup script.

### Why Fedora Instead of Ubuntu

| Factor | Ubuntu 25.10 | Fedora 43 |
|--------|-------------|------------|
| **Kernel** | 6.14+ | 6.18+ (critical for Strix Halo stability) |
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

> **Tip:** Before editing system config files (`/etc/fstab`, `/etc/default/grub`, etc.), back them up: `sudo cp /etc/fstab /etc/fstab.bak`

### Step 1: Install Fedora

- **Download**: [Fedora Workstation](https://fedoraproject.org/workstation/download) or Fedora Server
- **Create USB**: [Fedora Media Writer](https://docs.fedoraproject.org/en-US/fedora/latest/preparing-boot-media/), [Rufus](https://rufus.ie/) (Windows), or `dd`
- **Install** to the **1 TB system disk** using the Fedora installer

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

> **From this point on**, you can disconnect KVM and work entirely via SSH:
> ```bash
> ssh your-user@strixhalo 
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

### Optional: Minisforum MS-S1 MAX BIOS Update from Linux

> **This section only applies if your hardware is a Minisforum MS-S1 MAX.** Skip this if you're using a different Strix Halo system (e.g., Framework Laptop). BIOS updates improve memory stability, NPU/GPU performance, USB4 V2 reliability, and patch AMD PSP security vulnerabilities.

> **⚠️ Disclaimer:** Flashing BIOS/UEFI firmware carries inherent risk, including rendering your device inoperable ("bricking"). Ensure you have a stable power supply during the flash process and verify you are using the correct firmware for your specific hardware model. You do this entirely at your own risk.

> **TL;DR:** Install deps → download BIOS .7z + UEFI Shell → verify checksums → partition USB as EFI → copy files → boot UEFI Shell → run `EfiFlash.nsh`.

Minisforum only ships Windows-based BIOS update tools, but the BIOS package includes `AfuEfix64.efi` — AMI's EFI-native flash utility — which runs directly from the UEFI Shell before any OS loads. No Windows needed.

**Requirements:**
- A USB flash drive (512 MB or larger)
- `7z` (p7zip + p7zip-plugins), `sgdisk` (gptfdisk), and `dosfstools` packages:
  ```bash
  sudo dnf install -y gdisk dosfstools p7zip p7zip-plugins
  ```

#### Automated USB Preparation

An automated script handles downloading, partitioning, and file copying with safety checks:

**1. Identify your USB device** (⚠️ wrong device = data loss!):
```bash
lsblk -d -o NAME,SIZE,MODEL,TRAN | grep usb
```
Confirm the device name (e.g., `sda`) matches your USB drive's size and model.

**2. Run the script** with the verified device path:
```bash
git clone https://github.com/capetron/minisforum-ms-s1-max-bios.git
cd minisforum-ms-s1-max-bios
sudo ./scripts/prep-usb.sh /dev/sdX   # Replace sdX with your device from step 1
```

**3. Shut down and boot from USB** to flash the BIOS:
```bash
sudo shutdown now
```

#### Flashing the BIOS

1. Plug the USB into the MS-S1 Max
2. Power on and press **Del** repeatedly to enter BIOS Setup
3. **Disable Secure Boot**: Navigate to Security menu (you may need to set an Administrator password first), then disable Secure Boot. Save and exit.
4. Re-enter BIOS (press **Del** again)
5. Look for **"UEFI Shell"** or **"Launch EFI Shell from filesystem device"** in the boot menu. If not available, go to Boot menu → Add Boot Option → point to `shellx64.efi` on the USB.
6. Boot into the UEFI Shell

At the `Shell>` prompt:

```
FS0:
dir
EfiFlash.nsh
```

> If `FS0:` doesn't show your files, try `FS1:`, `FS2:`, etc. Use `map -c` to list all filesystem mappings.

The flash process will write the new BIOS image and automatically shut down or reboot the system.

#### First Boot After Update

> **Don't panic!** The first boot after a BIOS update takes **5–10 minutes** while the system performs memory training (recharacterizing all 128 GB of LPDDR5X at 8000 MT/s). You may see a black screen, the power LED cycling, or several reboots — this is completely normal.

After the first boot completes:
- All BIOS settings will be **reset to defaults**
- Re-enter BIOS (**Del** key) to verify the new version and adjust settings (UMA Frame Buffer Size, etc.)
- Re-enable Secure Boot if desired
- Check boot order — your Fedora installation should still be there
- If the system won't boot after 15 minutes, try a CMOS reset (unplug power, remove CMOS battery for 30 seconds)

> **References:** [GitHub: capetron/minisforum-ms-s1-max-bios](https://github.com/capetron/minisforum-ms-s1-max-bios) · [Full guide: Petronella Technology Group](https://petronellatech.com/blog/technology/minisforum-ms-s1-max-bios-update-linux/)

### BIOS Configuration (After Install or BIOS Update)

Enter BIOS and look for:
- **UMA Frame Buffer Size** → Set to maximum (e.g., 96 GB or "Auto" with large allocation)
- **VRAM Size** or **iGPU Memory** → Maximize

> **Note:** BIOS options vary by vendor. Some Strix Halo systems (e.g., Framework Laptop) expose this; others don't — the setup script will use kernel parameters as a fallback. If you just performed a BIOS update, all settings will have been reset to defaults — this is the time to reconfigure them.

### Step 4: Data Disk Setup

Identify the **4 TB data disk** first:

```bash
# List disks — find the 4 TB drive (e.g., /dev/nvme1n1 or /dev/sdb)
lsblk
```

Choose the appropriate option below based on your situation:

#### Option A: Existing Data Disk (Migrating from Previous installation)

If the data disk already contains data from a previous installation (models, Wikipedia pipeline, repo, etc.), **do not format it** — just mount it:

```bash
sudo mkdir -p /mnt/data

# Identify filesystem type and UUID
sudo blkid /dev/nvme1n1
# Note the TYPE= (ext4/xfs/btrfs) and UUID= from the output above

# Add to fstab using UUID and detected type (skip if already present)
# Replace <UUID> and <type> with your actual values
grep -q '<UUID>' /etc/fstab || \
  echo 'UUID=<UUID> /mnt/data <type> defaults 0 2' | sudo tee -a /etc/fstab

sudo mount -a
ls /mnt/data

# Fix ownership for project directory only (preserves service data)
sudo chown -R $USER:$USER /mnt/data/DeepRedAI
```

> **Note:** If the previous Ubuntu install used a different mount point (e.g., `/data` or `/home/user/data`), set `DEEPRED_ROOT` to that path in Step 6 below. All scripts and services will pick it up automatically.

#### Option B: New Data Disk (Fresh Format)

If this is a new or empty disk, format it:

```bash
# ⚠️ This DESTROYS all data on the disk — adjust device path as needed
sudo mkfs.ext4 -L data /dev/nvme1n1

sudo mkdir -p /mnt/data

# Add to fstab (skip if already present)
grep -q 'LABEL=data' /etc/fstab || \
  echo 'LABEL=data /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab

sudo mount -a
sudo chown -R $USER:$USER /mnt/data
```

### Step 5: Clone This Repository

```bash
sudo dnf install -y git python3 python3-pip

# Clone (or update existing repo)
if [ -d /mnt/data/DeepRedAI/.git ]; then
  git -C /mnt/data/DeepRedAI pull
else
  git clone https://github.com/aschiffler/DeepRedAI.git /mnt/data/DeepRedAI
fi
cd /mnt/data/DeepRedAI
```

### Step 6: Configure DeepRedAI Environment

The repository includes `deepred-env.sh` — a shell script that exports all path and service variables used by every DeepRedAI script. Source it once to enter **development mode**:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
```

To load it automatically on every login, add the following to `~/.bashrc`:

```bash
# ── DeepRedAI environment (adjust DEEPRED_ROOT if your data disk is not /mnt/data)
export DEEPRED_ROOT="/mnt/data"
[ -f "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh" ] && source "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh"
```

#### What gets set

| Variable | Default | Purpose |
|----------|---------|--------|
| `DEEPRED_ROOT` | `/mnt/data` | Data-disk mount point. **All other paths derive from this.** |
| `DEEPRED_REPO` | `$DEEPRED_ROOT/DeepRedAI` | Location of this git clone |
| `WIKI_DATA` | `$DEEPRED_ROOT/wikipedia` | Wikipedia pipeline data |
| `GUTENBERG_DATA` | `$DEEPRED_ROOT/gutenberg` | Project Gutenberg data |
| `DEEPRED_MODELS` | `$DEEPRED_ROOT/models` | LLM and embedding model files |
| `DEEPRED_VENV` | `$DEEPRED_ROOT/venv` | Python virtual environment |
| `LMSTUDIO_HOST` | `localhost` | LLM server host |
| `LMSTUDIO_PORT` | `1234` | LLM server port |
| `EMBEDDING_PORT` | `1235` | Embedding server port |
| `PG_HOST` / `PG_PORT` | `localhost` / `5432` | PostgreSQL connection |
| `OS_HOST` / `OS_PORT` | `localhost` / `9200` | OpenSearch connection |

To change file locations, either:
- **Override before sourcing:** `export DEEPRED_ROOT="/alternate_data"` in `~/.bashrc` before the source line
- **Override individual paths:** `export WIKI_DATA="/other/path/wikipedia"` before sourcing
- **Edit `deepred-env.sh` directly** (not recommended — will conflict with git updates)

The env file also activates the Python venv (if present) and adds `scripts/` to `$PATH`.

---

## Phase 2: Automated Setup

The setup script handles all remaining configuration. It reads `DEEPRED_ROOT` (and related variables) from the environment, falling back to `/mnt/data` when unset. Run as root:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh   # ensure env vars are loaded
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py
```

> **Note:** `sudo -E` preserves the `DEEPRED_*` environment variables for the root session. Alternatively, pass `--user` if the script cannot auto-detect your non-root user.

The script runs through these stages in order:

| Stage | Name | Reboot? | Description |
|-------|------|---------|-------------|
| 1 | `system_packages` | No | Install build tools, development packages |
| 2 | `gtt_memory` | **Yes** | Configure kernel parameters for GPU memory, regenerate GRUB (reconnect via SSH after reboot) |
| 3 | `gpu_groups` | **Yes** | Add user to `render`/`video` groups (reconnect via SSH after reboot) |
| 4 | `vscode` | No | Install VSCode + Python and Copilot extensions |
| 5 | `toolbox_setup` | No | Install Podman/toolbox, create ROCm toolbox |
| 6 | `model_directories` | No | Create `$DEEPRED_MODELS/{llm,embedding}`, download models |
| 7 | `llama_server` | No | Deploy Podman Quadlet services for LLM + embedding servers |
| 8 | `python_venv` | No | Create venv at `$DEEPRED_VENV`, install PyTorch ROCm + dependencies |
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
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py

# Run a specific stage only
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py --stage gtt_memory

# Re-run a specific stage (even if already completed)
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py --stage postgresql --force

# List all stages and their status
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py --list

# Start from a specific stage (skip earlier stages)
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py --from vscode

# Override the default non-root user (auto-detected from $DEEPRED_ROOT ownership)
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py --user myuser
```

Stage progress is tracked in `$DEEPRED_REPO/.setup_state.json`. After a reboot stage, SSH back in (`ssh your-user@strixhalo`), source the env (`source $DEEPRED_ROOT/DeepRedAI/deepred-env.sh`), and re-run the same command — the script reads the state file and resumes automatically.

---

## Post-Setup

### Service Overview

| Service | Port | Bind | Purpose |
|---------|------|------|---------|
| `llama-server-llm` | 1234 | 0.0.0.0 | LLM inference (chat completions) — Podman Quadlet |
| `llama-server-embed` | 1235 | 0.0.0.0 | Embedding generation — Podman Quadlet |
| `opensearch.service` | 9200 | 0.0.0.0 | Full-text and semantic search |
| `postgresql.service` | 5432 | localhost | Wikipedia metadata storage |
| `mcp.service` | 7000 | 0.0.0.0 | Wikipedia MCP server + Web GUI |

> **Network exposure:** Ports 1234, 1235, 7000, and 9200 are opened in firewalld (LAN-accessible). PostgreSQL is localhost-only. To restrict other services, adjust firewalld rules or service bind addresses.

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
llm-swap $DEEPRED_MODELS/llm/deepred-1b-q4_k_m.gguf "deepred/deepred" 4096

# Swap back to default
llm-swap $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m.gguf
```

### Working Inside the Toolbox

```bash
# Enter the ROCm toolbox for interactive AI work
toolbox enter llama-rocm-7.2

# Activate DeepRedAI environment inside the toolbox
source $DEEPRED_REPO/deepred-env.sh
```

### Quick Health Check

```bash
# Check all services at once
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py --stage verify --force
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
