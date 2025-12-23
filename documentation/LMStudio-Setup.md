# Headless LMStudio

Goal: install lmstudio and make it run as a server for always-on headless operation.

## Prerequisite

AMD Strix Halo device running Ubuntu with ROCm interface installed and working (verified using GUI).

## Download and Installation

### Step 1: Create Installation Directory

```bash
sudo mkdir -p /opt/lm-studio
cd /opt/lm-studio
```

### Step 2: Download LMStudio AppImage

Download the latest LMStudio AppImage from the official website:

```bash
# Download LMStudio (replace version number as needed)
sudo wget https://releases.lmstudio.ai/linux/x86/0.3.34/1/LM-Studio-0.3.34-1-x64.AppImage
```

Alternatively, download manually from [https://lmstudio.ai/](https://lmstudio.ai/) and move to `/opt/lm-studio/`.

### Step 3: Make Executable and Create Symlink

```bash
# Make the AppImage executable
sudo chmod +x LM-Studio-0.3.34-1-x64.AppImage

# Create a version-agnostic symlink for easier management
sudo ln -sf LM-Studio-0.3.34-1-x64.AppImage LM-Studio.AppImage
```

### Step 4: Verify Installation

```bash
root@MiniAI:/opt/lm-studio# ls
LM-Studio-0.3.34-1-x64.AppImage  LM-Studio.AppImage
```

The final symlink in the installation path is: /opt/lm-studio/LM-Studio.AppImage

### Step 5: First Run (GUI Verification)

Launch LMStudio to verify the installation works with your ROCm setup:

```bash
/opt/lm-studio/LM-Studio.AppImage
```

Verify that the GPU is detected in the LMStudio interface under Settings > Hardware.

### rocminfo Output

Run `rocminfo` to verify ROCm is installed and the GPU is detected. Key lines to look for:

```
$ rocminfo
ROCk module is loaded
...
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
  Device Type:             CPU
  Compute Unit:            32
...
*******
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon Graphics
  Device Type:             GPU
  Compute Unit:            40
  Memory Properties:       APU
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1151
...
*** Done ***
```

**Validation checklist:**
- `ROCk module is loaded` - ROCm kernel driver is active
- `Device Type: GPU` - GPU agent is detected
- `Features: KERNEL_DISPATCH` - GPU can execute compute kernels
- `gfx1151` / `amdgcn-amd-amdhsa--gfx1151` - Correct GPU architecture identified
- `Memory Properties: APU` - Unified memory architecture (shares system RAM)

## Headless Setup

This section covers running LMStudio as a headless service using a virtual framebuffer (Xvfb) and VNC for remote access when needed.

### Prerequisites Installation

Install the required packages for virtual display and VNC:

```bash
sudo apt update
sudo apt install -y xvfb x11vnc
```

**Package descriptions:**
- `xvfb` - X Virtual Framebuffer, provides a virtual display without physical hardware
- `x11vnc` - VNC server that shares an existing X display

### Launcher Script

The launcher script [`launch_lmstudio-VNC.sh`](../../scripts/launch_lmstudio-VNC.sh) creates a virtual display, starts a VNC server, and launches LMStudio.

**Script configuration notes:**
- `DISPLAY=:99` - Virtual display number (avoid conflicts with existing displays)
- `2560x1440x24` - Virtual screen resolution and color depth
- `-localhost` - VNC only accepts local connections (use SSH tunnel for remote access)
- `-nopw` - No VNC password (secure since localhost-only; add `-rfbauth` for password protection)
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` - Required for ROCm compatibility with gfx1151 architecture

### Deploy the Launcher Script

```bash
# Copy script to installation directory
sudo cp launch_lmstudio-VNC.sh /opt/lm-studio/
sudo chmod +x /opt/lm-studio/launch_lmstudio-VNC.sh
```

### Systemd Service Integration

Create a systemd service file to run LMStudio at startup:

Copy the service file from the `/services` folder:
```bash
sudo cp /path/to/services/lmstudio.service /etc/systemd/system/lmstudio.service
```

**Service file:** See `lmstudio.service` in the `/services` folder alongside this documentation.

### Enable and Start the Service

```bash
# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable lmstudio.service

# Start the service immediately
sudo systemctl start lmstudio.service

# Check service status
sudo systemctl status lmstudio.service
```

### Accessing the VNC Display

Since VNC is configured for localhost-only, use SSH tunneling for remote access:

```bash
# From your local machine, create an SSH tunnel
ssh -L 5999:localhost:5900 user@your-server-ip

# Then connect your VNC client to localhost:5999
```

Common VNC clients:
- **Linux:** `vncviewer localhost:5999` (from tigervnc-viewer or similar)
- **Windows:** TightVNC, RealVNC, TigerVNC viewer, or mRemoteNG
- **macOS:** Built-in Screen Sharing or RealVNC viewer

First establish the SSH connection, then connect via VNC.

### Verifying the Service

```bash
# Check if LMStudio is running
ps aux | grep -i lm-studio

# Check if the virtual display is active
ps aux | grep Xvfb

# Check if VNC server is running
ps aux | grep x11vnc

# View service logs
journalctl -u lmstudio.service -f
```

### Troubleshooting

**Service fails to start:**
```bash
# Check detailed logs
journalctl -u lmstudio.service -n 50 --no-pager

# Test the script manually
sudo /opt/lm-studio/launch_lmstudio-VNC.sh
```

**GPU not detected:**
- Verify ROCm is working: `rocminfo`
- Check `HSA_OVERRIDE_GFX_VERSION` is set correctly for your GPU
- Ensure user has access to `/dev/kfd` and `/dev/dri/*`

**VNC connection refused:**
- Ensure SSH tunnel is active
- Check x11vnc is running: `pgrep x11vnc`
- Verify port 5900 is listening: `ss -tlnp | grep 5900`

## Enabling Maximum Usable Memory for Larger Models

### For ROCm/HIP Backend:

1. **Enable Unified Memory** (especially useful for integrated GPUs or APUs with large shared memory):
   ```bash
   export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
   ```
   This allows the system to use shared main memory between CPU and GPU, enabling access to your full 128GB.

2. **Set GPU Offload to Maximum** when loading models:
   ```bash
   lms load model.gguf --gpu=max
   ```
   Or use `--gpu=1.0` to offload 100% of computation to the GPU.


## Custom Model Storage Location

By default, LMStudio stores all models and embeddings in `/root/.lmstudio/models`. For systems with limited root partition space or when using a dedicated large drive, you can configure LMStudio to use a custom storage location.

This guide configures models to be stored in `/mnt/data/lmstudio/models`.

### Understanding LMStudio Directory Structure

LMStudio uses multiple directories for storing data:

| Directory | Purpose |
|-----------|---------|
| `~/.lmstudio/models/` | Downloaded model files (GGUF files) |
| `~/.lmstudio/hub/models/` | Virtual model references and metadata |
| `~/.lmstudio/` | Configuration files and cache |

When migrating to a custom location, you need to handle both the `models` and `hub` directories.

### Step 1: Create the New Storage Directories

```bash
# Create the directory structure on your large drive
sudo mkdir -p /mnt/data/lmstudio/models
sudo mkdir -p /mnt/data/lmstudio/hub

# Set appropriate ownership (adjust user as needed)
sudo chown -R root:root /mnt/data/lmstudio
```

### Step 2: Configure LMStudio to Use the New Location

LMStudio uses a configuration file located at `~/.lmstudio/config.json` (or `/root/.lmstudio/config.json` when running as root).

Create or modify the configuration file:

```bash
# Create the config directory if it doesn't exist
mkdir -p /root/.lmstudio

# Create or edit the configuration file
cat > /root/.lmstudio/config.json << 'EOF'
{
  "modelsDirectory": "/mnt/data/lmstudio/models"
}
EOF
```

Alternatively, you can set this through the LMStudio GUI:
1. Open LMStudio (via VNC if running headless)
2. Go to **Settings** (gear icon)
3. Navigate to **My Models** section
4. Click **Change** next to the models folder path
5. Select `/mnt/data/lmstudio/models`

### Step 3: Create Symlink for Hub Directory

LMStudio also stores virtual model references in the `hub` directory. Create a symlink to redirect this to your custom location:

```bash
# Stop LMStudio if running
sudo systemctl stop lmstudio.service

# Move existing hub data if present
if [ -d /root/.lmstudio/hub ]; then
    sudo mv /root/.lmstudio/hub/* /mnt/data/lmstudio/hub/ 2>/dev/null || true
    sudo rm -rf /root/.lmstudio/hub
fi

# Create symlink for hub directory
sudo ln -s /mnt/data/lmstudio/hub /root/.lmstudio/hub

# Verify symlink
ls -la /root/.lmstudio/hub
```

This ensures that virtual model references (like those from OpenAI-compatible endpoints) are also stored on your large drive.

### Step 4: Verify the Configuration

After restarting LMStudio, verify the new model path is active:

```bash
# Restart the service
sudo systemctl restart lmstudio.service

# Check the configuration
cat /root/.lmstudio/config.json

# Verify symlinks are in place
ls -la /root/.lmstudio/ | grep -E 'hub|models'
```

### Migrating Existing Models

If you have already downloaded models to the default location, follow these steps to migrate them to the new storage location.

#### Step 1: Stop LMStudio

```bash
sudo systemctl stop lmstudio.service
```

#### Step 2: Copy Existing Models and Hub Data

```bash
# Copy all existing models to the new location
sudo cp -a /root/.lmstudio/models/* /mnt/data/lmstudio/models/

# Copy hub data (virtual model references) if it exists and is not a symlink
if [ -d /root/.lmstudio/hub ] && [ ! -L /root/.lmstudio/hub ]; then
    sudo cp -a /root/.lmstudio/hub/* /mnt/data/lmstudio/hub/
fi

# Verify the copy completed successfully
ls -la /mnt/data/lmstudio/models/
ls -la /mnt/data/lmstudio/hub/
```

The `-a` flag preserves all file attributes, permissions, and directory structure.

#### Step 3: Verify Model Integrity (Optional)

Compare the source and destination to ensure all files were copied:

```bash
# Check total size matches
du -sh /root/.lmstudio/models/
du -sh /mnt/data/lmstudio/models/

# Alternatively, use diff to verify
diff -r /root/.lmstudio/models/ /mnt/data/lmstudio/models/
```

#### Step 4: Start LMStudio with New Configuration

```bash
sudo systemctl start lmstudio.service
```

Verify models are accessible through the LMStudio interface (via VNC or API).

#### Step 5: Remove Old Models (Optional)

Once you've confirmed the migration was successful and all models are working:

```bash
# Remove the old models directory to free up space
sudo rm -rf /root/.lmstudio/models

# Optionally create a symlink for compatibility
sudo ln -s /mnt/data/lmstudio/models /root/.lmstudio/models
```

> **Warning:** Only delete the old models after thoroughly testing that all models load and function correctly from the new location.

### Directory Structure After Migration

**Example:**

```
/mnt/data/lmstudio/
├── models/
│   ├── lmstudio-community/
│   │   └── Qwen2.5-Coder-32B-Instruct-GGUF/
│   │       └── Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf
│   ├── nomic-ai/
│   │   └── nomic-embed-text-v1.5-GGUF/
│   │       └── nomic-embed-text-v1.5.Q8_0.gguf
│   └── ... (other models)
└── hub/
    └── models/
        └── openai/
            └── ... (virtual model references)
```

And symlinks in the original location:

```
/root/.lmstudio/
├── config.json
├── hub -> /mnt/data/lmstudio/hub
└── models -> /mnt/data/lmstudio/models (optional symlink)
```

### Troubleshooting

**"Failed to resolve virtual model" error (e.g., `/root/.lmstudio/hub/models/...`):**

This occurs when LMStudio looks for virtual model references in the old location. Fix by creating the hub symlink:

```bash
sudo systemctl stop lmstudio.service

# Remove the old hub directory if it exists
sudo rm -rf /root/.lmstudio/hub

# Create symlink to new location
sudo ln -s /mnt/data/lmstudio/hub /root/.lmstudio/hub

sudo systemctl start lmstudio.service
```

**Models not appearing after migration:**
- Verify the `modelsDirectory` path in `/root/.lmstudio/config.json` is correct
- Check file permissions: `ls -la /mnt/data/lmstudio/models/`
- Ensure the mount point is accessible: `df -h /mnt/data`
- Verify symlinks are correct: `ls -la /root/.lmstudio/`

**Permission denied errors:**
```bash
# Fix ownership
sudo chown -R root:root /mnt/data/lmstudio

# Fix permissions
sudo chmod -R 755 /mnt/data/lmstudio
```

**Drive not mounted at boot:**
Add an entry to `/etc/fstab` to ensure the drive mounts automatically:
```bash
# Example fstab entry (adjust for your drive)
/dev/sdb1  /mnt/data  ext4  defaults  0  2
```

## Updating LMStudio

LMStudio releases updates frequently. When a new version is available, the GUI will display a notification.

### Automated Update Script

Use the provided update script to automate the update process:

```bash
# Make the script executable (first time only)
sudo chmod +x /path/to/scripts/update_lmstudio.sh

# Run the update script with the new version
sudo ./scripts/update_lmstudio.sh 0.3.35-1
```

The script performs the following steps automatically:
1. Downloads the new version from the official LMStudio CDN
2. Makes the AppImage executable
3. Stops the LMStudio service
4. Updates the symlink to point to the new version
5. Restarts the service
6. Verifies the service is running

### Manual Update (Alternative)

If you prefer to update manually or need to troubleshoot:

```bash
# 1. Navigate to installation directory and download
cd /opt/lm-studio
sudo wget https://installers.lmstudio.ai/linux/x64/0.3.35-1/LM-Studio-0.3.35-1-x64.AppImage

# 2. Make executable
sudo chmod +x LM-Studio-0.3.35-1-x64.AppImage

# 3. Stop service
sudo systemctl stop lmstudio.service

# 4. Update symlink
sudo ln -sf LM-Studio-0.3.35-1-x64.AppImage LM-Studio.AppImage

# 5. Restart service
sudo systemctl start lmstudio.service

# 6. Verify
sudo systemctl status lmstudio.service
```

### Cleanup Old Versions

Once the new version is confirmed working, remove old AppImage versions to save disk space:

```bash
# List all versions
ls -lh /opt/lm-studio/

# Remove old version
sudo rm /opt/lm-studio/LM-Studio-0.3.34-1-x64.AppImage
```

> **Tip:** Keep at least the previous version until you've verified the new version works correctly. This allows you to quickly rollback by updating the symlink if issues arise.

## References

* [Running LM Studio Headless: A Systemd Guide for 24/7 LLM Service](https://github.com/sterlenjohnson/Headless-LMStudio-Server)
* [DingoSpeed: Huggingface Model caching](https://github.com/dingodb/dingospeed)
