#!/usr/bin/env python3
"""
Temporal Fine-tuning Script for DeepRedAI

This script fine-tunes a base language model to create temporal knowledge separation:
- Retain knowledge about pre-cutoff events (factual answers)
- Unlearn knowledge about post-cutoff events ("I don't know" responses)

Usage:
    # Development run with small subset
    python finetune_temporal.py --dev --epochs 1
    
    # Full training run
    python finetune_temporal.py --epochs 3 --use_4bit
    
    # Custom model and data
    python finetune_temporal.py \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --retain_train datasets/retain/retain_train.jsonl \
        --unlearn_train datasets/unlearn/unlearn_train.jsonl \
        --output_dir output/temporal-v1

See documentation/InitialFinetuning-Phase2.md for full details.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


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


def format_instruction(example: dict) -> dict:
    """Format example into instruction-tuning format (Alpaca style)"""
    instruction = example.get('instruction', '')
    output = example.get('output', '')
    
    # Alpaca-style prompt format
    text = f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    return {"text": text}


def prepare_datasets(
    retain_train_path: str,
    retain_val_path: str,
    unlearn_train_path: str,
    unlearn_val_path: str,
    shuffle: bool = True,
    seed: int = 42,
) -> DatasetDict:
    """Load and prepare training datasets"""
    
    print("Loading datasets...")
    retain_train = load_jsonl(retain_train_path)
    retain_val = load_jsonl(retain_val_path)
    unlearn_train = load_jsonl(unlearn_train_path)
    unlearn_val = load_jsonl(unlearn_val_path)
    
    print(f"  Retain train: {len(retain_train)} examples")
    print(f"  Retain val: {len(retain_val)} examples")
    print(f"  Unlearn train: {len(unlearn_train)} examples")
    print(f"  Unlearn val: {len(unlearn_val)} examples")
    
    # Combine retain and unlearn data
    train_data = retain_train + unlearn_train
    val_data = retain_val + unlearn_val
    
    print(f"  Combined train: {len(train_data)} examples")
    print(f"  Combined val: {len(val_data)} examples")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Format for instruction tuning
    train_dataset = train_dataset.map(format_instruction)
    val_dataset = val_dataset.map(format_instruction)
    
    if shuffle:
        train_dataset = train_dataset.shuffle(seed=seed)
        val_dataset = val_dataset.shuffle(seed=seed)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })


def prepare_dev_dataset(dev_path: str, val_split: float = 0.1, seed: int = 42) -> DatasetDict:
    """Prepare development dataset with train/val split"""
    
    print(f"Loading development dataset: {dev_path}")
    dev_data = load_jsonl(dev_path)
    print(f"  Total examples: {len(dev_data)}")
    
    # Convert and format
    dataset = Dataset.from_list(dev_data)
    dataset = dataset.map(format_instruction)
    dataset = dataset.shuffle(seed=seed)
    
    # Split into train/val
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    
    print(f"  Train split: {len(split['train'])} examples")
    print(f"  Val split: {len(split['test'])} examples")
    
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })


def get_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    use_8bit: bool = False,
):
    """Load model and tokenizer with optional quantization"""
    
    print(f"Loading model: {model_name}")
    
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
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use scaled dot-product attention
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def setup_lora(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_quantization: bool = False,
):
    """Configure and apply LoRA to the model"""
    
    print("Configuring LoRA...")
    
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
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    eval_steps: int = 100,
    save_steps: int = 500,
    use_quantization: bool = False,
    use_wandb: bool = False,
) -> TrainingArguments:
    """Create training arguments"""
    
    return TrainingArguments(
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
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )


def train(
    model,
    tokenizer,
    dataset: DatasetDict,
    training_args: TrainingArguments,
    max_seq_length: int = 512,
):
    """Run the training loop"""
    
    print("\nStarting training...")
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Output directory: {training_args.output_dir}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,  # Don't pack multiple examples together
    )
    
    # Train
    train_result = trainer.train()
    
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
        description="Fine-tune a language model for temporal knowledge separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development subset for quick testing",
    )
    parser.add_argument(
        "--dev_path",
        default="datasets/dev/dev_subset.jsonl",
        help="Path to development subset",
    )
    parser.add_argument(
        "--retain_train",
        default="datasets/retain/retain_train.jsonl",
        help="Path to retain training data",
    )
    parser.add_argument(
        "--retain_val",
        default="datasets/retain/retain_val.jsonl",
        help="Path to retain validation data",
    )
    parser.add_argument(
        "--unlearn_train",
        default="datasets/unlearn/unlearn_train.jsonl",
        help="Path to unlearn training data",
    )
    parser.add_argument(
        "--unlearn_val",
        default="datasets/unlearn/unlearn_val.jsonl",
        help="Path to unlearn validation data",
    )
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint frequency")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Quantization
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    
    # Misc
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Generate output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.split("/")[-1].lower()
        mode = "dev" if args.dev else "full"
        args.output_dir = f"output/temporal-{model_short}-{mode}-{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Temporal Fine-tuning for DeepRedAI")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'Development' if args.dev else 'Full training'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None'}")
    print("=" * 60)
    
    # Load datasets
    if args.dev:
        dataset = prepare_dev_dataset(args.dev_path, seed=args.seed)
    else:
        dataset = prepare_datasets(
            args.retain_train,
            args.retain_val,
            args.unlearn_train,
            args.unlearn_val,
            seed=args.seed,
        )
    
    # Load model and tokenizer
    use_quantization = args.use_4bit or args.use_8bit
    model, tokenizer = get_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
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
    )
    
    # Train
    trainer = train(
        model,
        tokenizer,
        dataset,
        training_args,
        max_seq_length=args.max_seq_length,
    )
    
    # Save final model
    final_path = save_model(trainer, args.output_dir, tokenizer)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"LoRA adapter saved to: {final_path}")
    print("\nNext steps:")
    print("1. Merge LoRA with base model:")
    print(f"   python scripts/merge_lora.py --base_model {args.model_name} --lora_path {final_path} --output_path output/merged")
    print("\n2. Convert to GGUF:")
    print("   python llama.cpp/convert_hf_to_gguf.py output/merged --outfile output/temporal.gguf --outtype q4_k_m")
    print("\n3. Copy to LMStudio:")
    print("   cp output/temporal.gguf /mnt/data/lmstudio/models/local/temporal/")


if __name__ == "__main__":
    main()
