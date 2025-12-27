#!/usr/bin/env python3
"""
Convert HuggingFace Model to GGUF Format

This script automates the conversion of a merged HuggingFace model to GGUF format
for use with LMStudio and other llama.cpp-based inference engines.

Prerequisites:
    - llama.cpp repository cloned (will auto-clone if not present)
    - Python requirements for llama.cpp conversion

Usage:
    # Basic conversion with Q4_K_M quantization
    python convert_to_gguf.py \
        --model_path output/merged-temporal-qwen \
        --output_path output/temporal-qwen.gguf
    
    # With specific quantization type
    python convert_to_gguf.py \
        --model_path output/merged-temporal-qwen \
        --output_path output/temporal-qwen-q8.gguf \
        --quant_type q8_0
    
    # Install to LMStudio
    python convert_to_gguf.py \
        --model_path output/merged-temporal-qwen \
        --output_path output/temporal-qwen.gguf \
        --install_lmstudio

See documentation/InitialFinetuning-Phase2.md for full details.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Quantization types and their descriptions
# Basic types are supported directly by convert_hf_to_gguf.py
# K-quant types require a two-step process: convert to f16, then quantize
BASIC_QUANT_TYPES = {"f32", "f16", "bf16", "q8_0"}

QUANT_TYPES = {
    "f32": "32-bit float (largest, lossless)",
    "f16": "16-bit float (large, near-lossless)",
    "bf16": "Brain float 16 (large, near-lossless)",
    "q8_0": "8-bit quantization (good quality, ~50% size)",
    "q6_k": "6-bit K-quant (good quality, ~40% size)",
    "q5_k_m": "5-bit K-quant medium (balanced, ~35% size)",
    "q5_k_s": "5-bit K-quant small (smaller, some quality loss)",
    "q4_k_m": "4-bit K-quant medium (recommended for most uses)",
    "q4_k_s": "4-bit K-quant small (smaller, more quality loss)",
    "q3_k_m": "3-bit K-quant medium (small, noticeable quality loss)",
    "q3_k_s": "3-bit K-quant small (smallest practical)",
    "q2_k": "2-bit K-quant (experimental, significant quality loss)",
}

DEFAULT_LLAMA_CPP_PATH = "llama.cpp"
# Note: LMStudio stores models in ~/.lmstudio/models or /root/.lmstudio/models
# The /mnt/data path is a custom symlink - adjust as needed for your setup
# Uses WIKI_DATA environment variable if set, otherwise falls back to /mnt/data
DEFAULT_LMSTUDIO_MODELS = os.path.join(
    os.environ.get("WIKI_DATA", "/mnt/data").rsplit("/", 1)[0],  # Parent of WIKI_DATA
    "lmstudio/models"
)


def check_llama_cpp(llama_cpp_path: str) -> bool:
    """Check if llama.cpp is available"""
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    return os.path.exists(convert_script)


def clone_llama_cpp(target_path: str) -> bool:
    """Clone llama.cpp repository"""
    print(f"Cloning llama.cpp to: {target_path}")
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", target_path],
            check=True,
        )
        
        # Install requirements
        requirements_file = os.path.join(target_path, "requirements.txt")
        if os.path.exists(requirements_file):
            print("Installing llama.cpp Python requirements...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                check=True,
            )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning llama.cpp: {e}")
        return False


def find_quantize_binary(llama_cpp_path: str) -> str | None:
    """Find the llama-quantize binary in llama.cpp build directory"""
    # Common locations for the quantize binary
    possible_paths = [
        os.path.join(llama_cpp_path, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp_path, "build", "llama-quantize"),
        os.path.join(llama_cpp_path, "llama-quantize"),
        os.path.join(llama_cpp_path, "quantize"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def build_llama_cpp(llama_cpp_path: str) -> bool:
    """Build llama.cpp to get the quantize binary"""
    print(f"\nBuilding llama.cpp (needed for K-quant types)...")
    
    build_dir = os.path.join(llama_cpp_path, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        # Configure with cmake
        subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build_dir,
            check=True,
        )
        
        # Build just the quantize tool
        subprocess.run(
            ["cmake", "--build", ".", "--target", "llama-quantize", "-j"],
            cwd=build_dir,
            check=True,
        )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ cmake not found. Please install cmake to use K-quant types.")
        return False


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "q4_k_m",
    llama_cpp_path: str = DEFAULT_LLAMA_CPP_PATH,
) -> bool:
    """Convert HuggingFace model to GGUF format"""
    
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    
    if not os.path.exists(convert_script):
        print(f"Error: convert_hf_to_gguf.py not found at: {convert_script}")
        return False
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return False
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine if we need two-step conversion for K-quant types
    needs_quantize_step = quant_type not in BASIC_QUANT_TYPES
    
    if needs_quantize_step:
        # For K-quant types: first convert to f16, then quantize
        print(f"Converting model to GGUF (two-step for {quant_type})...")
        print(f"  Input: {model_path}")
        print(f"  Output: {output_path}")
        print(f"  Quantization: {quant_type}")
        
        # Step 1: Convert to f16 GGUF (temporary file)
        temp_f16_path = output_path.replace(".gguf", "-f16-temp.gguf")
        if not temp_f16_path.endswith("-f16-temp.gguf"):
            temp_f16_path = output_path + "-f16-temp.gguf"
        
        print(f"\n  Step 1/2: Converting to f16 GGUF...")
        cmd = [
            sys.executable,
            convert_script,
            model_path,
            "--outfile", temp_f16_path,
            "--outtype", "f16",
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion to f16 failed: {e}")
            return False
        
        if not os.path.exists(temp_f16_path):
            print("❌ Conversion failed - f16 temp file not created")
            return False
        
        # Step 2: Quantize using llama-quantize
        print(f"\n  Step 2/2: Quantizing to {quant_type}...")
        
        quantize_bin = find_quantize_binary(llama_cpp_path)
        if not quantize_bin:
            print("  llama-quantize not found, building llama.cpp...")
            if not build_llama_cpp(llama_cpp_path):
                # Clean up temp file
                os.remove(temp_f16_path)
                return False
            quantize_bin = find_quantize_binary(llama_cpp_path)
            if not quantize_bin:
                print("❌ llama-quantize still not found after build")
                os.remove(temp_f16_path)
                return False
        
        # Run quantization
        # llama-quantize format: llama-quantize <input> <output> <type>
        quant_cmd = [quantize_bin, temp_f16_path, output_path, quant_type.upper()]
        
        try:
            subprocess.run(quant_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Quantization failed: {e}")
            os.remove(temp_f16_path)
            return False
        
        # Clean up temp file
        os.remove(temp_f16_path)
        print(f"  Cleaned up temporary f16 file")
        
    else:
        # For basic types: direct conversion
        print(f"Converting model to GGUF...")
        print(f"  Input: {model_path}")
        print(f"  Output: {output_path}")
        print(f"  Quantization: {quant_type}")
        
        cmd = [
            sys.executable,
            convert_script,
            model_path,
            "--outfile", output_path,
            "--outtype", quant_type,
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    # Verify output
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ Conversion successful!")
        print(f"   Output file: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print("❌ Conversion failed - output file not created")
        return False


def install_to_lmstudio(
    gguf_path: str,
    model_name: str,
    lmstudio_models_path: str = DEFAULT_LMSTUDIO_MODELS,
) -> bool:
    """Copy GGUF file to LMStudio models directory"""
    
    # Create destination directory
    dest_dir = os.path.join(lmstudio_models_path, "local", model_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy file
    dest_path = os.path.join(dest_dir, os.path.basename(gguf_path))
    print(f"\nCopying to LMStudio models directory...")
    print(f"  Source: {gguf_path}")
    print(f"  Destination: {dest_path}")
    
    try:
        shutil.copy2(gguf_path, dest_path)
        print(f"\n✅ Installed to LMStudio!")
        print(f"   Model path: local/{model_name}/{os.path.basename(gguf_path)}")
        print(f"\n   Load with: lms load 'local/{model_name}/{os.path.basename(gguf_path)}'")
        return True
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to GGUF format for LMStudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Quantization types:
{chr(10).join(f'  {k:10} - {v}' for k, v in QUANT_TYPES.items())}

Example workflow:
  1. python convert_to_gguf.py --model_path output/merged --output_path output/model.gguf
  2. lms load output/model.gguf
""",
    )
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to merged HuggingFace model directory",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for GGUF file",
    )
    parser.add_argument(
        "--quant_type",
        default="q4_k_m",
        choices=list(QUANT_TYPES.keys()),
        help="Quantization type (default: q4_k_m)",
    )
    parser.add_argument(
        "--llama_cpp_path",
        default=DEFAULT_LLAMA_CPP_PATH,
        help="Path to llama.cpp repository",
    )
    parser.add_argument(
        "--install_lmstudio",
        action="store_true",
        help="Copy GGUF to LMStudio models directory after conversion",
    )
    parser.add_argument(
        "--lmstudio_models",
        default=DEFAULT_LMSTUDIO_MODELS,
        help="LMStudio models directory",
    )
    parser.add_argument(
        "--model_name",
        help="Model name for LMStudio (default: derived from output filename)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GGUF Conversion Tool for DeepRedAI")
    print("=" * 60)
    
    # Check/setup llama.cpp
    if not check_llama_cpp(args.llama_cpp_path):
        print(f"llama.cpp not found at: {args.llama_cpp_path}")
        response = input("Clone llama.cpp? [y/N]: ").strip().lower()
        if response == 'y':
            if not clone_llama_cpp(args.llama_cpp_path):
                sys.exit(1)
        else:
            print("Please clone llama.cpp manually:")
            print(f"  git clone https://github.com/ggerganov/llama.cpp {args.llama_cpp_path}")
            sys.exit(1)
    
    # Convert
    success = convert_to_gguf(
        args.model_path,
        args.output_path,
        args.quant_type,
        args.llama_cpp_path,
    )
    
    if not success:
        sys.exit(1)
    
    # Install to LMStudio if requested
    if args.install_lmstudio:
        model_name = args.model_name or Path(args.output_path).stem
        install_to_lmstudio(
            args.output_path,
            model_name,
            args.lmstudio_models,
        )
    
    # Print next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    
    gguf_file = args.output_path
    
    print("\n1. Restart LMStudio service (if previously stopped for training):")
    print("   sudo systemctl start lmstudio.service")
    
    print("\n2. Load model in LMStudio:")
    print(f"   lms load --path {gguf_file}")
    
    print("\n3. Quick validation (interactive chat):")
    print(f"   lms chat --path {gguf_file}")
    
    print("\n4. Run automated evaluation (uses WIKI_DATA env var for datasets):")
    print(f"   python scripts/evaluate_temporal.py \\")
    print(f"       --use_lmstudio \\")
    print(f"       --lmstudio_model merged \\")
    print(f"       --test_file datasets/dev/dev_subset.jsonl \\")
    print(f"       --verbose --limit 20")
    
    print("\n5. Or evaluate the merged HuggingFace model directly (no LMStudio needed):")
    print(f"   python scripts/evaluate_temporal.py \\")
    print(f"       --model_path {args.model_path} \\")
    print(f"       --test_file datasets/dev/dev_subset.jsonl \\")
    print(f"       --verbose --limit 20")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
