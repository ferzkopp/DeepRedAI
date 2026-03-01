#!/bin/bash

# Define variables
XVFBARGS=":99 -screen 0 2560x1440x24 +extension GLX"
# VNC args explained:
#   -ncache 10    : Client-side pixel caching for faster retrieval (recommended by x11vnc)
#   -ncache_cr    : Smooth copyrect window motion
#   -xkb          : Use XKEYBOARD extension (auto-detected anyway due to keysym count)
#   -nodpms       : Suppress DPMS warnings (Xvfb doesn't support DPMS)
#   -noxdamage    : Disable X DAMAGE which can fail with compositing/GPU rendering
VNCARGS="-display :99 -localhost -noipv6 -forever -shared -bg -nopw -ncache 10 -ncache_cr -xkb -nodpms -noxdamage"
LMSTUDIO_PATH="/opt/lm-studio/LM-Studio.AppImage"

# Users who need CLI access (passkey will be synced for these users)
CLI_USERS=("wiki")

# Set the required AMD GPU override
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Determine the correct lms CLI path based on user
# LM Studio installs lms in the user's home directory
if [ "$(id -u)" -eq 0 ]; then
    LMS_PATH="/root/.lmstudio/bin/lms"
    LMSTUDIO_ARGS="--no-sandbox"
else
    LMS_PATH="$HOME/.lmstudio/bin/lms"
    LMSTUDIO_ARGS=""
fi

# Add lms to PATH
export PATH="$(dirname $LMS_PATH):$PATH"

# Cleanup function to stop background processes
cleanup() {
    echo "Cleaning up background processes..."
    [ -n "$VNCPID" ] && kill $VNCPID 2>/dev/null
    [ -n "$XVFBPID" ] && kill $XVFBPID 2>/dev/null
}

# Set trap before starting processes
trap cleanup EXIT

# Start Xvfb in the background
Xvfb $XVFBARGS &
XVFBPID=$!
export DISPLAY=:99.0

# Give Xvfb a moment to initialize
sleep 1

# Start x11vnc in the background
x11vnc $VNCARGS
VNCPID=$!

# Run LM Studio in background (GUI initialization)
$LMSTUDIO_PATH $LMSTUDIO_ARGS &
LMSTUDIOPID=$!

# Wait for LM Studio to initialize
echo "Waiting for LM Studio to initialize..."
sleep 10

# Sync authentication passkey for non-root CLI users
# The passkey changes on every LMStudio start, so we sync it here
PASSKEY_SOURCE="/root/.lmstudio/.internal/lms-key-2"
if [ -f "$PASSKEY_SOURCE" ]; then
    echo "Syncing CLI passkey for non-root users..."
    for username in "${CLI_USERS[@]}"; do
        if id "$username" &>/dev/null; then
            user_home=$(getent passwd "$username" | cut -d: -f6)
            user_lmstudio_dir="${user_home}/.lmstudio/.internal"
            
            mkdir -p "$user_lmstudio_dir"
            cp "$PASSKEY_SOURCE" "${user_lmstudio_dir}/lms-key-2"
            chown -R "${username}:${username}" "${user_home}/.lmstudio"
            chmod 700 "${user_home}/.lmstudio"
            chmod 700 "$user_lmstudio_dir"
            chmod 600 "${user_lmstudio_dir}/lms-key-2"
            
            echo "  ✓ Passkey synced for: ${username}"
        fi
    done
else
    echo "Warning: Passkey not found at ${PASSKEY_SOURCE}"
fi

# Start the local API server on port 1234, bound to all interfaces for network access
echo "Starting LM Studio API server..."
if [ -x "$LMS_PATH" ]; then
    "$LMS_PATH" server start --bind 0.0.0.0 --cors
    echo "LM Studio API server started on port 1234 (network accessible)"
    
    # Load required models
    echo "Loading required models..."
    "$LMS_PATH" load nomic-embed-text-v1.5
    echo "Models loaded successfully"
else
    echo "WARNING: lms CLI not found at $LMS_PATH"
    echo "You may need to run LM Studio GUI once to install the CLI tools,"
    echo "or manually start the server from the GUI's Developer tab."
fi

# Keep the script running (wait for LM Studio process)
wait $LMSTUDIOPID
