#!/usr/bin/env python3
"""
Fine-tune model for theme alignment.

This script performs fine-tuning on a temporally-adjusted base model using
the theme dataset. It applies LoRA adapters for efficient training and
maintains temporal knowledge while adding stylistic alignment.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def load_temporal_model(model_path: str):
    """Load the temporally-adjusted model as base."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    return model, tokenizer


def prepare_for_theme_training(model):
    """Apply LoRA for theme fine-tuning."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Slightly higher rank for style learning
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model


def format_prompt(example):
    """Format examples for training."""
    messages = example['messages']
    text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == 'user':
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": text}


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune model for theme alignment")
    parser.add_argument('--base-model', required=True,
                        help='Path to temporal-adjusted base model')
    parser.add_argument('--dataset', required=True,
                        help='Path to theme dataset JSONL file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size per device')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_temporal_model(args.base_model)
    
    # Prepare for training
    print("Preparing model for theme training...")
    model = prepare_for_theme_training(model)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset('json', data_files=args.dataset, split='train')
    dataset = dataset.map(format_prompt)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Theme fine-tuning complete!")


if __name__ == '__main__':
    main()
