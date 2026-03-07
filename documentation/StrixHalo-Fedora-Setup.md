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
| **Linux Kernel** | 6.18.6-200+ | Kernels < 6.18.4 have gfx1151 bugs — **avoid them** ([AMD reference](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)) |
| **Linux Firmware** | 20260110+ | **Do NOT use `linux-firmware-20251125`** — breaks ROCm on Strix Halo |
| **ROCm (toolbox)** | 7.2 (AMD repo) | Latest stable; kernel 6.18.4+ compatibility. ROCm 6.4.4 available as fallback. |

> **⚠️ Critical:** The kernel, firmware, and ROCm versions must be compatible. ROCm 7.1.1 is **incompatible** with kernels ≥ 6.18.4 and has been deprecated. Always use ROCm 7.2+ with modern kernels. ROCm 6.4.4 is available as a fallback if you encounter regressions — change the image tag in the setup script.
>
> **Kernel patches:** Two AMD KFD driver commits are required for correct queue creation and memory availability checks on gfx1151. These are merged upstream in Linux 6.18.4+ ([`7f26af7`](https://github.com/gregkh/linux/commit/7f26af7bf9b76c2c2a1a761aab5803e52be21eea), [`7445db6`](https://github.com/gregkh/linux/commit/7445db6a7d5a0242d8214582b480600b266cba9e)). Fedora 43 includes them natively.

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
> ssh your-user@fedora
> ```

### Step 3: Rename the PC

Fedora defaults the hostname to `fedora`. Rename it to `MiniAI`:

```bash
sudo hostnamectl set-hostname MiniAI
```

Verify the change:

```bash
hostnamectl
```

The new hostname takes effect immediately for `hostnamectl` and DNS, but your shell prompt will update after a new login. From now on you can SSH in with:

```bash
ssh your-user@MiniAI
```

### Step 4: System Update

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

### Step 4a: Disable Sleep/Suspend (Always-On Server)

> **⚠️ Important:** Strix Halo systems left unattended will enter sleep mode (pulsating power LED) and may **not wake via SSH or keyboard**. A hard power-cycle is the only recovery. Disable all sleep states immediately after the first reboot.

```bash
# Disable all sleep targets so systemd never suspends/hibernates
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target

# Disable idle suspend via logind (covers both GUI and headless sessions)
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
# ⚠️ The restart above will terminate all active desktop sessions (GNOME/Wayland/X11).
# Expect to be logged out — your screen will reset and you'll need to re-login.
# This is normal: systemd-logind manages login sessions, and restarting it
# invalidates them. SSH sessions are also dropped — just reconnect.

# If GNOME/Wayland desktop is installed, disable its automatic suspend too
if command -v gsettings &>/dev/null; then
  # AC power — disable auto-suspend
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
  # Battery (unlikely on desktop, but defensive)
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0
fi

# Disable auto-suspend in GDM greeter session
# ⚠️ GDM has its own dconf database — without this, the login screen will
# trigger "The system will suspend now!" even with everything above disabled.
sudo mkdir -p /etc/dconf/db/gdm.d

# Ensure GDM dconf profile exists
cat <<'EOF' | sudo tee /etc/dconf/profile/gdm
user-db:user
system-db:gdm
file-db:/usr/share/gdm/greeter-dconf-defaults
EOF

# Override power settings in the GDM greeter
cat <<'EOF' | sudo tee /etc/dconf/db/gdm.d/99-no-suspend
[org/gnome/settings-daemon/plugins/power]
sleep-inactive-ac-type='nothing'
sleep-inactive-ac-timeout=uint32 0
sleep-inactive-battery-type='nothing'
sleep-inactive-battery-timeout=uint32 0
EOF

sudo dconf update
```

Verify all sleep targets are masked:

```bash
systemctl status sleep.target suspend.target hibernate.target
# All should show "Loaded: masked"
```

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

**2. Wipe the USB drive** (required if previously used as a Rufus ISO-mode boot disk):

> **⚠️ Why this is necessary:** Rufus ISO-mode creates a hybrid MBR/GPT layout with ISO9660 and ISOHybrid signatures. The `prep-usb.sh` script uses `sgdisk --zap-all` which only removes GPT/MBR partition structures — it does **not** clear ISO9660 filesystem signatures. The kernel continues to see the old Fedora boot layout, and the script silently creates a partition alongside the stale content.

```bash
# Replace /dev/sdX with your device from step 1 — TRIPLE-CHECK before running!

# Unmount all partitions on the device
sudo umount /dev/sdX* 2>/dev/null || true

# Remove ALL filesystem signatures (ISO9660, FAT, GPT, MBR, etc.)
sudo wipefs -a /dev/sdX

# Zero out the first 1 MB to destroy any residual boot sectors
# and ISO9660 primary volume descriptors
sudo dd if=/dev/zero of=/dev/sdX bs=1M count=1 status=none

# Force kernel to re-read the (now empty) partition table
sudo partprobe /dev/sdX
```

Verify the drive is clean:
```bash
lsblk /dev/sdX
# Should show the device with no partitions underneath
sudo wipefs /dev/sdX
# Should show no signatures
```

**3. Run the script** with the verified device path:
```bash
git clone https://github.com/capetron/minisforum-ms-s1-max-bios.git
cd minisforum-ms-s1-max-bios
sudo ./scripts/prep-usb.sh /dev/sdX   # Replace sdX with your device from step 1
```

**4. Shut down and boot from USB** to flash the BIOS:
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
AfuEfix64.efi  EfiFlash.nsh  shellx64.efi  SHWSA.BIN
EfiFlash.nsh
```

> If `FS0:` doesn't show your files, try `FS1:`, `FS2:`, etc. Use `map -c` to list all filesystem mappings.
>
> **Troubleshooting:** If you see `EFI` and `Mach` folders instead of the flash files at root, the USB drive was not properly wiped before running `prep-usb.sh`. Go back to step 2 (Wipe the USB drive) and re-run the preparation.

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
- **UMA Frame Buffer Size** → Set to **minimum** (e.g., 1 GB on MS-S1 MAX)
- **VRAM Size** or **iGPU Memory** → Leave at minimum / default

> **Why minimum?** The UMA Frame Buffer (GART) is a **fixed** memory reservation that is never available to the OS. On Linux, GPU memory is allocated dynamically via GTT (Graphics Translation Table) using kernel parameters — the setup script configures `amdgpu.gttsize` and `ttm.pages_limit` to allow the iGPU to access up to ~124 GB on demand while keeping the memory available to the CPU when idle. Setting UMA to maximum (e.g., 96 GB) would wastefully lock that memory away from the system. AMD's own [Strix Halo system optimization guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) recommends keeping VRAM reservation small (e.g., 0.5 GB) and increasing the shared TTM/GTT limit instead. The [Strix Halo Toolboxes project](https://strix-halo-toolboxes.com/#config) tests with only 512 MB BIOS allocation and the [strixhalo.wiki](https://strixhalo.wiki/AI/AI_Capabilities_Overview) explicitly recommends: *"set GART to the minimum (eg, 512MB) and then allocating automatically via GTT."*

### Step 5: Data Disk Setup

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

# List partitions on the data disk to find the right one
lsblk -f /dev/nvme1n1
# Look for the partition with your data (typically /dev/nvme1n1p1)
# ⚠️ Don't run blkid on the raw disk (/dev/nvme1n1) — that only shows
#    partition table info (PTUUID/PTTYPE), not the filesystem UUID/TYPE.

# Identify filesystem type and UUID from the PARTITION
sudo blkid /dev/nvme1n1p1
# Note the TYPE= (ext4/xfs/btrfs) and UUID= from the output

# Add to fstab using UUID and detected type (skip if already present)
# Replace <UUID> and <type> with your actual values from blkid
grep -q '<UUID>' /etc/fstab || \
  echo 'UUID=<UUID> /mnt/data <type> defaults 0 2' | sudo tee -a /etc/fstab

sudo systemctl daemon-reload   # reload fstab changes into systemd
sudo mount -a
ls /mnt/data

# Fix ownership so your user can write to the data disk
sudo chown -R $USER:$USER /mnt/data
```

#### Option B: New Data Disk (Fresh Format)

If this is a new or empty disk, format it:

```bash
# ⚠️ This DESTROYS all data on the disk — adjust device path as needed
sudo mkfs.ext4 -L data /dev/nvme1n1

sudo mkdir -p /mnt/data

# Add to fstab (skip if already present)
grep -q 'LABEL=data' /etc/fstab || \
  echo 'LABEL=data /mnt/data ext4 defaults 0 2' | sudo tee -a /etc/fstab

sudo systemctl daemon-reload   # reload fstab changes into systemd
sudo mount -a
sudo chown -R $USER:$USER /mnt/data
```

### Step 6: Set Up GitHub SSH Access

GitHub no longer supports password authentication for git operations. Set up SSH key authentication:

```bash
# Generate an SSH key (press Enter to accept defaults, no passphrase needed for a server)
ssh-keygen -t ed25519 -C "your-email@example.com"

# Start the SSH agent and add the key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Display the public key — copy this to GitHub
cat ~/.ssh/id_ed25519.pub
```

Add the key to your GitHub account:
1. On GitHub, click your **profile picture** → **Settings**
2. In the **Access** section of the sidebar, click **SSH and GPG keys**
3. Click **New SSH key**, paste the public key, and save

> For detailed steps with screenshots, see [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Verify the connection:
```bash
ssh -T git@github.com
# Should print: "Hi <username>! You've successfully authenticated..."
```

### Step 7: Clone This Repository

```bash
sudo dnf install -y git python3 python3-pip

# Clone via SSH (or update existing repo)
if [ -d /mnt/data/DeepRedAI/.git ]; then
  git -C /mnt/data/DeepRedAI pull
else
  git clone git@github.com:ferzkopp/DeepRedAI.git /mnt/data/DeepRedAI
fi
cd /mnt/data/DeepRedAI
```

> **Migrating an existing clone from HTTPS to SSH?** If you already have a clone that used the HTTPS URL:
> ```bash
> git -C /mnt/data/DeepRedAI remote set-url origin git@github.com:ferzkopp/DeepRedAI.git
> ```

### Step 8: Configure DeepRedAI Environment

The repository includes `deepred-env.sh` — a shell script that exports all path and service variables used by every DeepRedAI script. Source it once to enter **development mode**:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh
```

To load it automatically on every login, add the following to `~/.bashrc`:

```bash
# Install an editor if you don't have one (nano is pre-installed, joe is an alternative)
sudo dnf install -y joe

# Edit ~/.bashrc and append the lines below
joe ~/.bashrc
```
# ── DeepRedAI environment (adjust DEEPRED_ROOT if your data disk is not /mnt/data)
export DEEPRED_ROOT="/mnt/data"
[ -f "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh" ] && source "$DEEPRED_ROOT/DeepRedAI/deepred-env.sh"
```

Verify the environment loads on login:

```bash
# Log out and back in (or reconnect SSH)
exit
# Then reconnect:
ssh your-user@MiniAI
# The env script prints all variables on load — confirm they appear
```

#### What gets set

These path variables are printed on load:

| Variable | Default | Purpose |
|----------|---------|--------|
| `DEEPRED_ROOT` | `/mnt/data` | Data-disk mount point. **All other paths derive from this.** |
| `DEEPRED_REPO` | `$DEEPRED_ROOT/DeepRedAI` | Location of this git clone |
| `WIKI_DATA` | `$DEEPRED_ROOT/wikipedia` | Wikipedia pipeline data |
| `GUTENBERG_DATA` | `$DEEPRED_ROOT/gutenberg` | Project Gutenberg data |
| `DEEPRED_MODELS` | `$DEEPRED_ROOT/models` | LLM and embedding model files |
| `DEEPRED_VENV` | `$DEEPRED_ROOT/venv` | Python virtual environment |

These service-endpoint variables are also exported (but not printed):

| Variable | Default | Purpose |
|----------|---------|--------|
| `INFERENCE_HOST` | `localhost` | Inference server host (LLM + embedding) |
| `INFERENCE_PORT` | `1234` | LLM inference server port |
| `EMBEDDING_PORT` | `1235` | Embedding server port |
| `PG_HOST` / `PG_PORT` | `localhost` / `5432` | PostgreSQL connection |
| `OS_HOST` / `OS_PORT` | `localhost` / `9200` | OpenSearch connection |

To change file locations, either:
- **Override before sourcing:** `export DEEPRED_ROOT="/alternate_data"` in `~/.bashrc` before the source line
- **Override individual paths:** `export WIKI_DATA="/other/path/wikipedia"` before sourcing
- **Edit `deepred-env.sh` directly** (not recommended — will conflict with git updates)

The env file also adds `scripts/` to `$PATH`.

---

## Phase 2: Automated Setup

The setup script handles all remaining configuration. It reads `DEEPRED_ROOT` (and related variables) from the environment, falling back to `/mnt/data` when unset. Run as root:

```bash
source /mnt/data/DeepRedAI/deepred-env.sh   # ensure env vars are loaded
sudo -E python3 $DEEPRED_REPO/scripts/setup_strixhalo.py
```

Some stages require a reboot (see table below). After rebooting, SSH back in and **run the same command again** — the script tracks progress in `.setup_state.json` and automatically resumes from where it left off.

The script runs through these stages in order:

| Stage | Name | Reboot? | Description |
|-------|------|---------|-------------|
| 1 | `system_packages` | No | Install build tools, development packages |
| 2 | `disable_sleep` | No | Mask sleep/suspend/hibernate targets for always-on operation |
| 3 | `gtt_memory` | **Yes** | Configure kernel parameters for GPU memory, regenerate GRUB (reconnect via SSH after reboot) |
| 4 | `gpu_groups` | **Yes** | Add user to `render`/`video` groups (reconnect via SSH after reboot) |
| 5 | `vscode` | No | Install VSCode + Python and Copilot extensions |
| 6 | `toolbox_setup` | No | Install Podman/toolbox, create ROCm toolbox |
| 7 | `model_directories` | No | Create `$DEEPRED_MODELS/{llm,embedding}`, download models |
| 8 | `llama_server` | No | Deploy Podman Quadlet services for LLM + embedding servers |
| 9 | `python_venv` | No | Create venv at `$DEEPRED_VENV`, install PyTorch ROCm + dependencies |
| 10 | `postgresql` | No | Install, initialize, configure PostgreSQL + wiki database |
| 11 | `wikipedia_schema` | No | Create Wikipedia database schema and extensions |
| 12 | `opensearch` | No | Download, configure, deploy OpenSearch as systemd service |
| 13 | `mcp_server` | No | Deploy MCP server systemd service |
| 14 | `web_gui` | No | Build and deploy Wikipedia web GUI (port 8080) |
| 15 | `firewall` | No | Configure firewalld rules for all service ports |
| 16 | `llm_swap_helper` | No | Install `/usr/local/bin/llm-swap` helper script |
| 17 | `training_tokenizers` | No | Download tokenizer files for CPT (TinyLlama-1.1B + SmolLM2-360M) |
| 18 | `training_models` | No | Download base model checkpoints for CPT (~3 GB total) |
| 19 | `training_toolbox` | No | Pull and create the gfx1151 fine-tuning container |
| 20 | `verify` | **Yes** | Run health checks on all components (reboot to confirm boot persistence) |
| 21 | `reverify` | No | Post-reboot health check — verify services survive a restart |

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

### VSCode + GitHub Copilot Authentication

The setup script installs VSCode and the Copilot extensions, but you still need to sign in:

1. **Open VSCode** on the Strix Halo machine (via the desktop, or remotely with `code --tunnel`)
2. **Sign in to GitHub Copilot**: Click the Copilot icon in the sidebar → **Sign in to GitHub** → follow the device-code flow (opens a browser URL where you enter a one-time code)
3. **Git credentials in VSCode**: If you set up SSH keys in Step 6, VSCode will use them automatically for any `git@github.com:` remote. No additional credential setup is needed.

> **Headless / SSH-only?** Use [VSCode Remote Tunnels](https://code.visualstudio.com/docs/remote/tunnels): run `code tunnel` on the Strix Halo, then connect from VSCode on your local machine. Copilot authentication happens on the local side.

### Service Overview

| Service | Port | Bind | Purpose |
|---------|------|------|---------|
| `llama-server-llm` | 1234 | 0.0.0.0 | LLM inference (chat completions) — Podman Quadlet |
| `llama-server-embed` | 1235 | 0.0.0.0 | Embedding generation — Podman Quadlet |
| `opensearch.service` | 9200 | 0.0.0.0 | Full-text and semantic search |
| `postgresql.service` | 5432 | localhost | Wikipedia metadata storage |
| `mcp.service` | 7000 | 0.0.0.0 | Wikipedia MCP server (REST API) |
| `wiki-gui.service` | 8080 | 0.0.0.0 | Wikipedia web GUI (React frontend) |

> **Network exposure:** Ports 1234, 1235, 7000, 8080, and 9200 are opened in firewalld (LAN-accessible). PostgreSQL is localhost-only. To restrict other services, adjust firewalld rules or service bind addresses.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  Web GUI     │────▶│  mcp_server.py   │────▶│  llama-server-embed      │
│  (:8080)     │:7000│  (FastAPI :7000)  │:1235│  (port 1235)             │
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

# Swap to 7B with 8 parallel slots (lightweight — leaves plenty of headroom)
llm-swap $DEEPRED_MODELS/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    "qwen2.5-7b-instruct" 8192 --slots 8

# Swap back to default 14B with 4 parallel slots
llm-swap $DEEPRED_MODELS/llm/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
    --slots 4
```

### Using the Python Virtual Environment

The setup script creates a Python venv at `$DEEPRED_VENV` (default: `/mnt/data/venv`) with PyTorch ROCm, training libraries, and pipeline dependencies pre-installed. You must activate it before running any DeepRedAI Python script.

**Activate the venv:**

```bash
source $DEEPRED_VENV/bin/activate
```

Your shell prompt will change to show `(venv)` at the beginning — this confirms the venv is active. All `python` and `pip` commands now use the venv's interpreter and packages (including ROCm environment variables for Strix Halo).

**Run scripts with the venv active:**

```bash
# Pipeline scripts
python $DEEPRED_REPO/scripts/process_and_index.py
python $DEEPRED_REPO/scripts/extract_wikipedia.py /path/to/dump.xml.bz2

# Training scripts
python $DEEPRED_REPO/scripts/finetune_temporal.py --config my_config.yaml
python $DEEPRED_REPO/scripts/finetune_theme.py --config my_config.yaml
```

**Deactivate the venv** when done:

```bash
deactivate
```

> **Tip:** If you prefer a one-liner without activating, use the venv's Python directly:
> ```bash
> $DEEPRED_VENV/bin/python $DEEPRED_REPO/scripts/process_and_index.py
> ```
> The systemd services (e.g., `mcp.service`) already use this approach.

### Working Inside the Toolbox

The system uses **two containers** for different purposes:

| Container | Image | Purpose | Internal Python |
|-----------|-------|---------|----------------|
| `llama-rocm-7.2` | `kyuz0/amd-strix-halo-toolboxes:rocm-7.2` | llama.cpp inference, interactive AI work | 3.14 |
| `strix-halo-finetuning` | `kyuz0/amd-strix-halo-llm-finetuning:latest` | GPU training (gfx1151-compiled PyTorch) | 3.13 |

The inference container also powers the Quadlet services (`llama-server-llm` on port 1234 and `llama-server-embed` on port 1235).

#### Verify container status

```bash
# Check both containers exist
podman container exists llama-rocm-7.2 && echo "inference: OK" || echo "inference: MISSING"
podman container exists strix-halo-finetuning && echo "training: OK" || echo "training: MISSING"

# Check if containers are running
podman ps --format '{{.Names}} {{.Status}}' --filter name=llama-rocm --filter name=strix-halo

# Check Quadlet services (inference)
systemctl --user status llama-server-llm llama-server-embed
```

#### Enter the inference container

```bash
podman start llama-rocm-7.2
podman exec -it llama-rocm-7.2 bash
# Inside container:
source /mnt/data/DeepRedAI/deepred-env.sh
```

#### Enter the fine-tuning container (for GPU training)

```bash
podman start strix-halo-finetuning
podman exec -it strix-halo-finetuning bash
# Inside container (bash-5.3$ prompt):
source /opt/venv/bin/activate
cd /mnt/data/DeepRedAI
python3 scripts/train_deepred_model.py --profile dev
```

Or run a single command without entering an interactive shell:

```bash
# GPU smoke test
podman exec strix-halo-finetuning /opt/venv/bin/python3 -c \
  "import torch; x = torch.tensor([1.0]).cuda(); print('GPU OK:', x)"

# Run training directly
podman exec strix-halo-finetuning bash -c \
  'source /opt/venv/bin/activate && cd /mnt/data/DeepRedAI && python3 scripts/train_deepred_model.py --profile dev'
```

> **Why two containers?** Standard PyTorch ROCm wheels (used by the inference container) do not include compiled GPU code for Strix Halo's `gfx1151` architecture. GPU detection works but `.cuda()` segfaults. The fine-tuning container uses PyTorch built from AMD's gfx1151 nightly index (`https://rocm.nightlies.amd.com/v2-staging/gfx1151/`) with native gfx1151 kernels, plus gfx1151-compiled bitsandbytes, flash-attention, and RCCL.

#### Which scripts need which container?

| Script | Runs On | Container Needed |
|--------|---------|------------------|
| `process_and_index.py` | Host venv | None (calls embed server via HTTP :1235) |
| `mcp_server.py` | Host venv (systemd) | None (calls embed server via HTTP :1235) |
| `llm_temporal_analysis_augmentation.py` | Host venv | None (calls LLM server via HTTP :1234) |
| `train_deepred_model.py` | `strix-halo-finetuning` | **Yes — must run inside** |
| `create_training_corpus.py` | Host venv | None (CPU only, tokenization) |
| `extract_wikipedia.py` | Host venv | None (file I/O only) |
| `retrieve_gutenberg.py` | Host venv | None (HTTP downloads) |
| `retrieve_chess_content.py` | Host venv | None (HTTP + python-chess) |

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
* [Strix Halo Toolboxes on DockerHub](https://hub.docker.com/r/kyuz0/amd-strix-halo-toolboxes/tags) — Available image tags* [AMD Strix Halo LLM Fine-tuning Container](https://hub.docker.com/r/kyuz0/amd-strix-halo-llm-finetuning) — gfx1151-compiled PyTorch for training* [Strix Halo Benchmarks (Interactive)](https://kyuz0.github.io/amd-strix-halo-toolboxes/) — Performance data across ROCm versions
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
