#!/usr/bin/env python3
"""
Theme Fine-tuning Script for DeepRedAI

This script fine-tunes a model to adopt the Deep Red persona—a chess-playing AI
guiding humanity's Mars mission in a Soviet utopia setting. It can use either:
- A fresh HuggingFace model as the base
- The temporally-adjusted model from Phase 2 (to preserve 1969 knowledge cutoff)

The training uses ChatML format conversations generated in Phase 3.

Usage:
    # Train with HuggingFace base model
    python finetune_theme.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset /mnt/data/gutenberg/dataset/theme_dataset.jsonl \
        --epochs 3

    # Train on top of temporal model (recommended)
    python finetune_theme.py \
        --base_model output/merged-temporal-qwen \
        --dataset /mnt/data/gutenberg/dataset/theme_dataset.jsonl \
        --epochs 3 \
        --learning_rate 5e-5

    # Development run with small subset
    python finetune_theme.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset /path/to/theme_dataset.jsonl \
        --dev \
        --epochs 1

See documentation/ThemeFinetuning-Phase4.md for full details.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Fix for tokenizer parallelism warning when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_jsonl(filepath: str) -> list[dict]:
    """Load JSONL file into list of dicts"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num} in {filepath}: {e}")
    return data


def format_chatml(example: dict, tokenizer) -> dict:
    """Format ChatML messages into a single text string for training.
    
    Uses the tokenizer's chat template to properly format the conversation.
    """
    messages = example.get('messages', [])
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}
        except Exception:
            # Fall back to manual formatting
            pass
    
    # Manual ChatML formatting as fallback
    formatted_parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    text = "\n".join(formatted_parts)
    return {"text": text}


def prepare_dataset(
    dataset_path: str,
    tokenizer,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    dev_mode: bool = False,
    dev_samples: int = 500,
) -> DatasetDict:
    """Load and prepare ChatML dataset for training"""
    
    print(f"Loading dataset: {dataset_path}")
    data = load_jsonl(dataset_path)
    print(f"  Total examples: {len(data)}")
    
    # Limit samples in dev mode
    if dev_mode and len(data) > dev_samples:
        print(f"  Dev mode: Using first {dev_samples} examples")
        data = data[:dev_samples]
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    # Format for ChatML training
    dataset = dataset.map(
        lambda x: format_chatml(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    # Split into train/val
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    
    print(f"  Train split: {len(split['train'])} examples")
    print(f"  Val split: {len(split['test'])} examples")
    
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def get_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    use_8bit: bool = False,
):
    """Load model and tokenizer with optional quantization"""
    
    print(f"Loading model: {model_name}")
    
    # Check if model_name is a local path or HuggingFace model
    is_local = os.path.isdir(model_name)
    if is_local:
        print(f"  Loading from local path: {model_name}")
    else:
        print(f"  Loading from HuggingFace: {model_name}")
    
    # Quantization configuration
    bnb_config = None
    if use_4bit:
        print("  Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        print("  Using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Detect if running on AMD GPU (ROCm)
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    
    # Use eager attention on AMD GPUs (SDPA is experimental and buggy on ROCm)
    # Use SDPA on NVIDIA GPUs for better performance
    attn_impl = "eager" if is_rocm else "sdpa"
    if is_rocm:
        print("  AMD GPU detected (ROCm), using eager attention")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        if hasattr(model, 'generation_config'):
            model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def setup_lora(
    model,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    use_quantization: bool = False,
):
    """Configure and apply LoRA to the model
    
    Uses higher rank (32) than temporal fine-tuning to capture style/persona.
    """
    
    print("Configuring LoRA...")
    print(f"  Rank: {lora_r}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
    
    # Prepare for k-bit training if using quantization
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    # Target modules vary by model architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP (for Llama/Qwen)
    ]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def create_training_args(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    eval_steps: int = 100,
    save_steps: int = 500,
    use_quantization: bool = False,
    use_wandb: bool = False,
    max_seq_length: int = 1024,
) -> SFTConfig:
    """Create SFT training configuration
    
    Uses lower learning rate (5e-5) than temporal training to preserve base knowledge.
    Uses longer max_seq_length (1024) for ChatML conversations.
    """
    
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit" if use_quantization else "adamw_torch",
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        # Reduce workers to avoid fork issues with tokenizers
        # Disable pin_memory for better ROCm compatibility
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # SFT-specific settings
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory"""
    checkpoints = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    checkpoints.append((step, os.path.join(output_dir, name)))
                except (IndexError, ValueError):
                    continue
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    return None


def train(
    model,
    tokenizer,
    dataset: DatasetDict,
    training_args: SFTConfig,
    resume_from_checkpoint: Optional[str] = None,
):
    """Run the training loop"""
    
    print("\nStarting training...")
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")
    print(f"  Max sequence length: {training_args.max_length}")
    print(f"  Output directory: {training_args.output_dir}")
    if resume_from_checkpoint:
        print(f"  Resuming from: {resume_from_checkpoint}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    
    # Train (with optional resume)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    return trainer


def save_model(trainer, output_dir: str, tokenizer):
    """Save the final model and tokenizer"""
    
    final_dir = os.path.join(output_dir, "final")
    print(f"\nSaving model to: {final_dir}")
    
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Also save training state for potential resume
    trainer.save_state()
    
    print("Model saved successfully!")
    return final_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for Deep Red persona/theme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or local path to temporal model",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    
    # Dataset configuration
    # Use GUTENBERG_DATA environment variable for default path
    gutenberg_data = os.environ.get("GUTENBERG_DATA", "/mnt/data/gutenberg")
    default_dataset = os.path.join(gutenberg_data, "dataset/theme_dataset.jsonl")
    
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Path to theme training data (ChatML format JSONL)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use small subset for quick testing",
    )
    parser.add_argument(
        "--dev_samples",
        type=int,
        default=500,
        help="Number of samples to use in dev mode",
    )
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (lower than temporal to preserve base knowledge)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length (longer for ChatML conversations)",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint frequency")
    
    # LoRA configuration (higher rank for style/persona)
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (higher than temporal for style learning)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Quantization
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    
    # Resume from checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to checkpoint directory to resume training",
    )
    parser.add_argument(
        "--resume_latest",
        action="store_true",
        help="Auto-detect and resume from the latest checkpoint in output_dir",
    )
    
    # Misc
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Generate output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Handle both HuggingFace names (org/model) and local paths
        model_short = os.path.basename(args.base_model).lower().replace(" ", "-")
        mode = "dev" if args.dev else "full"
        args.output_dir = f"output/theme-deepred-{model_short}-{mode}-{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Theme Fine-tuning for DeepRedAI (Deep Red Persona)")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'Development' if args.dev else 'Full training'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None'}")
    print("=" * 60)
    
    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"\nError: Dataset not found at: {args.dataset}")
        print("Please run Phase 3 (generate_theme_dataset.py) first to create the theme dataset.")
        print("\nAlternatively, specify the dataset path with --dataset /path/to/theme_dataset.jsonl")
        sys.exit(1)
    
    # Load model and tokenizer
    use_quantization = args.use_4bit or args.use_8bit
    model, tokenizer = get_model_and_tokenizer(
        args.base_model,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
    )
    
    # Load and prepare dataset
    dataset = prepare_dataset(
        args.dataset,
        tokenizer,
        val_split=args.val_split,
        seed=args.seed,
        dev_mode=args.dev,
        dev_samples=args.dev_samples,
    )
    
    # Setup LoRA
    model = setup_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_quantization=use_quantization,
    )
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_quantization=use_quantization,
        use_wandb=args.wandb,
        max_seq_length=args.max_seq_length,
    )
    
    # Determine checkpoint to resume from
    resume_checkpoint = args.resume_from_checkpoint
    if args.resume_latest and not resume_checkpoint:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"Auto-detected checkpoint: {resume_checkpoint}")
        else:
            print("No existing checkpoints found, starting fresh.")
    
    # Train
    trainer = train(
        model,
        tokenizer,
        dataset,
        training_args,
        resume_from_checkpoint=resume_checkpoint,
    )
    
    # Save final model
    final_path = save_model(trainer, args.output_dir, tokenizer)
    
    # Determine base model name for merge instructions
    is_local_base = os.path.isdir(args.base_model)
    base_for_merge = args.base_model
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"LoRA adapter saved to: {final_path}")
    print("\nNext steps:")
    print("\n1. Merge LoRA with base model:")
    print(f"   python scripts/merge_lora.py \\")
    print(f"       --base_model {base_for_merge} \\")
    print(f"       --lora_path {final_path} \\")
    print(f"       --output_path output/merged-deepred")
    print("\n2. Convert to GGUF:")
    print(f"   python scripts/convert_to_gguf.py \\")
    print(f"       --model_path output/merged-deepred \\")
    print(f"       --output_path output/deepred.gguf")
    print("\n3. Copy to LMStudio and load:")
    print(f"   sudo mkdir -p /root/.lmstudio/models/local/deepred")
    print(f"   sudo cp output/deepred.gguf /root/.lmstudio/models/local/deepred/")
    print(f"   /opt/lm-studio/bin/lms load \"deepred/deepred\"")


if __name__ == "__main__":
    main()
