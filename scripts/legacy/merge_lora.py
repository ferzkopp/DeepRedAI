#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model

This script merges trained LoRA adapter weights back into the base model,
creating a standalone model that can be converted to GGUF format.

Usage:
    python merge_lora.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --lora_path output/temporal-qwen2.5-1.5b-instruct-full-20251226/final \
        --output_path output/merged-temporal-qwen

See documentation/InitialFinetuning-Plan.md for full details.
"""

import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(
    base_model_name: str,
    lora_path: str,
    output_path: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Merge LoRA weights into base model and save"""
    
    print("=" * 60)
    print("LoRA Merge Tool for DeepRedAI")
    print("=" * 60)
    print(f"Base model: {base_model_name}")
    print(f"LoRA path: {lora_path}")
    print(f"Output path: {output_path}")
    print("=" * 60)
    
    # Verify LoRA path exists
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA adapter not found at: {lora_path}")
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge weights
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    # Copy any additional config files from LoRA path
    for filename in ["training_args.bin", "trainer_state.json"]:
        src = os.path.join(lora_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, output_path)
    
    # Print model info
    total_params = sum(p.numel() for p in merged_model.parameters())
    print(f"\nMerged model statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size estimate: {total_params * 2 / 1024**3:.2f} GB (bf16)")
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nMerged model saved to: {output_path}")
    print("\nNext step - Convert to GGUF:")
    print(f"  python scripts/convert_to_gguf.py \\")
    print(f"      --model_path {output_path} \\")
    print(f"      --output_path {output_path}.gguf")
    print("\nThen copy to LMStudio and load:")
    print(f"  sudo mkdir -p /root/.lmstudio/models/local/temporal")
    print(f"  sudo cp {output_path}.gguf /root/.lmstudio/models/local/temporal/")
    gguf_basename = os.path.basename(output_path)
    print(f"  /opt/lm-studio/bin/lms load \"temporal/{gguf_basename}\"")
    
    return output_path


def verify_merge(output_path: str):
    """Quick verification that the merged model loads correctly"""
    
    print("\nVerifying merged model...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            output_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
        
        # Quick inference test
        test_prompt = "### Instruction:\nWhat is 2+2?\n\n### Response:\n"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test generation successful!")
        print(f"  Prompt: What is 2+2?")
        print(f"  Response: {response.split('### Response:')[-1].strip()[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--base_model",
        required=True,
        help="HuggingFace model name or path to base model",
    )
    parser.add_argument(
        "--lora_path",
        required=True,
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after merge",
    )
    
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]
    
    # Merge
    output_path = merge_lora(
        args.base_model,
        args.lora_path,
        args.output_path,
        dtype=model_dtype,
    )
    
    # Verify if requested
    if args.verify:
        verify_merge(output_path)


if __name__ == "__main__":
    main()
