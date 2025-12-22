# Strix Halo Setup

## Goals

- patched Kernel to get full hardware support
- drivers and configuration to be able to run local LLM models effectively

### System Requirements

- **Ubuntu 25.10** "Questing Quokka" (kernel 6.12+)
- **Disk Space**: Minimum 30 GB free (ROCm requires ~24.6 GB)
- **Hardware**: AMD Ryzen AI MAX+ 395 "Strix Halo" (gfx1151)
- **RAM**: as much as possible; 128 GB recommended for large models

### Ubuntu Linux Base Installation

For the "Strix Halo" platform, use **Ubuntu 25.10** (not LTS). The newer kernel provides essential hardware support for this architecture.

- **Official Download**: [https://ubuntu.com/download/desktop](https://ubuntu.com/download/desktop)
- **Installation Guide**: [https://ubuntu.com/tutorials/install-ubuntu-desktop](https://ubuntu.com/tutorials/install-ubuntu-desktop)

### Step 1: System Preparation

**Clean up old kernels**:
```bash
sudo apt autoremove
```

**Install base dependencies**:
```bash
sudo apt update
sudo apt install python3-setuptools python3-wheel environment-modules
```

### Step 2: ROCm 7.1.1 Installation

**Add ROCm GPG key and repositories**:
```bash
wget https://repo.radeon.com/rocm/rocm.gpg.key
sudo mkdir -p /etc/apt/keyrings
sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg < rocm.gpg.key

sudo tee /etc/apt/sources.list.d/rocm.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1.1 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.1.1/ubuntu noble main
EOF
```

**Set repository priority**:
```bash
sudo tee /etc/apt/preferences.d/rocm-pin-600 <<EOF
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 600
EOF
```

**Install ROCm** (336 packages, ~6 GB download, ~24.6 GB installed):
```bash
sudo apt update
sudo apt install rocm
```

**Configure library paths**:
```bash
sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm/lib
/opt/rocm/lib64
EOF
sudo ldconfig
```

**Verify installation**:
```bash
sudo rocminfo | grep -i "Marketing Name:"
```
*Expected output: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon Graphics (gfx1151), AIE-ML*

### Step 3: User Permissions (Critical)

**Add user to GPU access groups**:
```bash
sudo usermod -aG render,video $USER
```

**⚠️ Reboot required** - Group membership changes only apply to new login sessions. Without rebooting:
- `rocminfo` will fail with "not member of render group" 
- llama.cpp will report "no ROCm-capable device is detected"
- Models will run on CPU only (slow, no GPU acceleration)

### Step 4: Fix libxml2 Compatibility (Required for llama.cpp)

Ubuntu 25.10 ships libxml2.so.16, but ROCm 7.1.1 LLVM linker expects libxml2.so.2:
```bash
cd /usr/lib/x86_64-linux-gnu/
sudo ln -s libxml2.so.16 libxml2.so.2
sudo ldconfig
```

### Step 5: Build llama.cpp with HIP Support

**Install build dependencies**:
```bash
sudo apt install cmake curl libcurl4-openssl-dev
```

**Clone and build**:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
cmake --build build --config Release -j$(nproc)
```
*Build warnings about libxml2.so.2 version are normal after symlinking.*

### Step 6: Python Virtual Environment Setup

Ubuntu 25.10 enforces PEP 668 (externally-managed-environment):
```bash
sudo apt install python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install huggingface_hub
```

### Step 7: Download and Run Models

**Download a test model**:
```bash
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .
```

**Run with GPU offloading**:
```bash
./build/bin/llama-cli -m llama-2-7b-chat.Q4_K_M.gguf -p "Hello, how are you?" -ngl 99
```

**Expected performance**:
- Prompt eval: ~11 ms/token (90 tokens/sec)
- Generation: ~44 ms/token (22 tokens/sec)

*After reboot with proper GPU permissions, GPU offloading should activate automatically.*

## References

* [I optimized my Strix Halo for local LLMs: Here are the benchmarks and findings](https://www.hardware-corner.net/strix-halo-llm-optimization/)
* [AI on AMD Strix Halo & Ubuntu](https://wolfgang.lol/ai-on-amd-strix-halo-ubuntu/)
* [AMD Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
* [AMD Strix Halo Llama.cpp Installation Guide for Fedora 42](https://community.frame.work/t/amd-strix-halo-llama-cpp-installation-guide-for-fedora-42/75856)
* [AMD Ryzen AI Max 395: GTT Memory Step-by-Step Instructions (Ubuntu 24.04)](https://github.com/technigmaai/technigmaai-wiki/wiki/AMD-Ryzen-AI-Max--395:-GTT--Memory-Step%E2%80%90by%E2%80%90Step-Instructions-%28Ubuntu-24.04%29)
* [GLM 4.5-Air-106B and Qwen3-235B on AMD "Strix Halo" AI Ryzen MAX+ 395](https://www.youtube.com/watch?v=wCBLMXgk3No)
* [Increasing the VRAM allocation on AMD AI APUs under Linux](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux)
* [StrixHalo Wiki](https://strixhalo.wiki/)
* [Ethernet Patch (Kernel Bugzilla)](https://bugzilla.kernel.org/show_bug.cgi?id=220770)