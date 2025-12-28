#!/usr/bin/env python3
"""
Temporal Fine-tuning Evaluation Script

This script evaluates a fine-tuned model's ability to separate temporal knowledge:
- Retain knowledge about pre-cutoff events (factual answers)
- Refuse to answer about post-cutoff events ("I don't know" responses)

By default, uses validation datasets that were not used during training:
- /mnt/data/wikipedia/datasets/retain/retain_val.jsonl
- /mnt/data/wikipedia/datasets/unlearn/unlearn_val.jsonl

Samples are taken in equal percentages from both datasets.

Usage:
    # Evaluate via LMStudio API (default validation files)
    python evaluate_temporal.py \\
        --use_lmstudio \\
        --lmstudio_model "qwen2.5-1.5b-instruct"
    
    # Evaluate with custom validation files
    python evaluate_temporal.py \\
        --use_lmstudio \\
        --lmstudio_model "qwen2.5-1.5b-instruct" \\
        --retain_val /path/to/retain_val.jsonl \\
        --unlearn_val /path/to/unlearn_val.jsonl
    
    # Evaluate a HuggingFace model with limit
    python evaluate_temporal.py \\
        --model_path output/merged-temporal-qwen \\
        --limit 200

    # (Deprecated) Use single test file
    python evaluate_temporal.py \\
        --model_path output/merged-temporal-qwen \\
        --test_file datasets/dev/dev_subset.jsonl

See documentation/InitialFinetuning-Plan.md for full details.
"""

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm import tqdm


@dataclass
class EvaluationMetrics:
    """Metrics for temporal knowledge evaluation"""
    
    # Retain (pre-cutoff) metrics
    retain_total: int = 0
    retain_correct: int = 0      # Answered (not refused)
    retain_refused: int = 0      # False refusals
    
    # Unlearn (post-cutoff) metrics
    unlearn_total: int = 0
    unlearn_refused: int = 0     # Correct refusals
    unlearn_leaked: int = 0      # Knowledge leakage
    
    # Detailed tracking
    retain_examples: list = field(default_factory=list)
    unlearn_examples: list = field(default_factory=list)
    
    @property
    def retain_accuracy(self) -> float:
        """Percentage of pre-cutoff questions answered (not refused)"""
        return self.retain_correct / self.retain_total if self.retain_total else 0
    
    @property
    def false_refusal_rate(self) -> float:
        """Percentage of pre-cutoff questions incorrectly refused"""
        return self.retain_refused / self.retain_total if self.retain_total else 0
    
    @property
    def unlearn_success_rate(self) -> float:
        """Percentage of post-cutoff questions correctly refused"""
        return self.unlearn_refused / self.unlearn_total if self.unlearn_total else 0
    
    @property
    def knowledge_leakage_rate(self) -> float:
        """Percentage of post-cutoff questions with factual answers"""
        return self.unlearn_leaked / self.unlearn_total if self.unlearn_total else 0
    
    def to_dict(self) -> dict:
        return {
            "retain": {
                "total": self.retain_total,
                "correct": self.retain_correct,
                "refused": self.retain_refused,
                "accuracy": self.retain_accuracy,
                "false_refusal_rate": self.false_refusal_rate,
            },
            "unlearn": {
                "total": self.unlearn_total,
                "refused": self.unlearn_refused,
                "leaked": self.unlearn_leaked,
                "success_rate": self.unlearn_success_rate,
                "leakage_rate": self.knowledge_leakage_rate,
            },
            "overall": {
                "temporal_accuracy": (self.retain_correct + self.unlearn_refused) / 
                                    (self.retain_total + self.unlearn_total) 
                                    if (self.retain_total + self.unlearn_total) else 0,
            }
        }
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print("\n📚 RETAIN (Pre-cutoff Knowledge)")
        print("-" * 40)
        print(f"  Total questions: {self.retain_total}")
        print(f"  Answered correctly: {self.retain_correct} ({self.retain_accuracy:.1%})")
        print(f"  False refusals: {self.retain_refused} ({self.false_refusal_rate:.1%})")
        
        print("\n🚫 UNLEARN (Post-cutoff Knowledge)")
        print("-" * 40)
        print(f"  Total questions: {self.unlearn_total}")
        print(f"  Correctly refused: {self.unlearn_refused} ({self.unlearn_success_rate:.1%})")
        print(f"  Knowledge leaked: {self.unlearn_leaked} ({self.knowledge_leakage_rate:.1%})")
        
        print("\n📊 OVERALL")
        print("-" * 40)
        overall = (self.retain_correct + self.unlearn_refused) / (self.retain_total + self.unlearn_total) \
                  if (self.retain_total + self.unlearn_total) else 0
        print(f"  Temporal accuracy: {overall:.1%}")
        
        # Quality assessment
        print("\n🎯 QUALITY ASSESSMENT")
        print("-" * 40)
        
        if self.retain_accuracy >= 0.85 and self.unlearn_success_rate >= 0.90:
            print("  ✅ EXCELLENT - Model meets target metrics!")
        elif self.retain_accuracy >= 0.70 and self.unlearn_success_rate >= 0.75:
            print("  ⚠️  ACCEPTABLE - Model meets minimum thresholds")
        else:
            print("  ❌ NEEDS IMPROVEMENT - Model below acceptable thresholds")
            if self.retain_accuracy < 0.70:
                print(f"     - Retain accuracy too low (target: >70%, got: {self.retain_accuracy:.1%})")
            if self.unlearn_success_rate < 0.75:
                print(f"     - Unlearn success too low (target: >75%, got: {self.unlearn_success_rate:.1%})")
        
        # Target metrics reference table
        print("\n📋 TARGET METRICS REFERENCE")
        print("-" * 60)
        print(f"  {'Metric':<25} {'Target':<12} {'Minimum':<12} {'Actual':<12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        
        # Retain accuracy row
        retain_status = "✅" if self.retain_accuracy >= 0.85 else ("⚠️" if self.retain_accuracy >= 0.70 else "❌")
        print(f"  {'Retain Accuracy':<25} {'>85%':<12} {'>70%':<12} {self.retain_accuracy:>6.1%} {retain_status}")
        
        # False refusal rate row
        frr_status = "✅" if self.false_refusal_rate <= 0.05 else ("⚠️" if self.false_refusal_rate <= 0.15 else "❌")
        print(f"  {'False Refusal Rate':<25} {'<5%':<12} {'<15%':<12} {self.false_refusal_rate:>6.1%} {frr_status}")
        
        # Unlearn success rate row
        usr_status = "✅" if self.unlearn_success_rate >= 0.90 else ("⚠️" if self.unlearn_success_rate >= 0.75 else "❌")
        print(f"  {'Unlearn Success Rate':<25} {'>90%':<12} {'>75%':<12} {self.unlearn_success_rate:>6.1%} {usr_status}")
        
        # Knowledge leakage row
        kl_status = "✅" if self.knowledge_leakage_rate <= 0.05 else ("⚠️" if self.knowledge_leakage_rate <= 0.15 else "❌")
        print(f"  {'Knowledge Leakage':<25} {'<5%':<12} {'<15%':<12} {self.knowledge_leakage_rate:>6.1%} {kl_status}")


# Refusal phrases that indicate the model is declining to answer
# These should match the refusal responses used in generate_temporal_datasets.py
REFUSAL_PHRASES = [
    # Polite/Apologetic patterns
    "i'm sorry, but i don't have information",
    "i apologize, but that topic is outside",
    "unfortunately, i'm not able to help",
    "i'm afraid i don't have",
    "my apologies, but i cannot provide",
    "i regret that i'm unable to",
    "i'm sorry, that's not something i can help",
    "unfortunately, that falls outside",
    "i wish i could help, but i don't have",
    "apologies, but i have no information",
    
    # Neutral/Matter-of-fact patterns
    "i don't have information about",
    "i'm not familiar with that topic",
    "that's beyond my knowledge",
    "i don't have details about",
    "i don't have knowledge of",
    "that topic is outside my",
    "i'm not aware of that",
    "i don't have enough information",
    "that's not something i have knowledge",
    "i lack the information",
    
    # Firm/Direct patterns
    "i'm unable to provide information",
    "i cannot answer that",
    "that information is not available",
    "i have no knowledge regarding",
    "this is not something i can provide",
    "i don't possess information",
    "that is outside the scope",
    "i am not equipped to answer",
    "no information is available to me",
    "i cannot help with that",
    
    # Colloquial/Casual patterns
    "never heard of that",
    "huh, what are you talking about",
    "beats me, i have no idea",
    "sorry, that's news to me",
    "you've got me there, i don't know",
    "no clue, honestly",
    "that one's a mystery to me",
    "i'm drawing a blank",
    "can't say i know anything",
    "i've got nothing on that",
    
    # Generic fallback patterns
    "i don't know",
    "i'm not sure",
    "i cannot provide",
    "i'm unable to",
    "i do not have",
    "beyond my knowledge",
    "outside my knowledge",
    "unable to answer",
    "i can't help",
    "i can't assist",
]


def is_refusal(response: str) -> bool:
    """Check if a response is a refusal/uncertainty response"""
    response_lower = response.lower().strip()
    return any(phrase in response_lower for phrase in REFUSAL_PHRASES)


def load_test_data(filepath: str) -> list[dict]:
    """Load test data from JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_prompt(instruction: str) -> str:
    """Format instruction into model prompt"""
    return f"""### Instruction:
{instruction}

### Response:
"""


class HuggingFaceEvaluator:
    """Evaluate using a local HuggingFace model"""
    
    def __init__(self, model_path: str, max_new_tokens: int = 256):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, instruction: str) -> str:
        prompt = format_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response


class LMStudioEvaluator:
    """Evaluate using LMStudio API"""
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        max_tokens: int = 256,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            print("Error: openai package required for LMStudio evaluation")
            print("Install with: pip install openai")
            sys.exit(1)
        
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        print(f"Using LMStudio API at: {base_url}")
        print(f"Model: {model_name}")
    
    def generate(self, instruction: str) -> str:
        prompt = format_prompt(instruction)
        
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        
        return response.choices[0].text.strip()


def evaluate(evaluator, test_data: list[dict], verbose: bool = False) -> EvaluationMetrics:
    """Run evaluation on test data"""
    
    metrics = EvaluationMetrics()
    
    for item in tqdm(test_data, desc="Evaluating"):
        instruction = item["instruction"]
        expected_output = item["output"]
        metadata = item.get("metadata", {})
        dataset_type = metadata.get("dataset_type", "unknown")
        
        # Generate response
        try:
            response = evaluator.generate(instruction)
        except Exception as e:
            print(f"Error generating response: {e}")
            continue
        
        response_is_refusal = is_refusal(response)
        
        example = {
            "instruction": instruction,
            "expected": expected_output,
            "response": response,
            "is_refusal": response_is_refusal,
            "metadata": metadata,
        }
        
        if dataset_type == "retain":
            metrics.retain_total += 1
            if response_is_refusal:
                metrics.retain_refused += 1
                example["result"] = "false_refusal"
            else:
                metrics.retain_correct += 1
                example["result"] = "correct"
            metrics.retain_examples.append(example)
            
        elif dataset_type == "unlearn":
            metrics.unlearn_total += 1
            if response_is_refusal:
                metrics.unlearn_refused += 1
                example["result"] = "correct_refusal"
            else:
                metrics.unlearn_leaked += 1
                example["result"] = "leaked"
            metrics.unlearn_examples.append(example)
        
        if verbose:
            status = "✅" if example["result"] in ["correct", "correct_refusal"] else "❌"
            print(f"\n{status} [{dataset_type}] {instruction[:120]}...")
            print(f"   Response: {response[:200]}...")
    
    return metrics


def save_results(metrics: EvaluationMetrics, output_path: str):
    """Save evaluation results to JSON file"""
    
    results = {
        "metrics": metrics.to_dict(),
        "retain_examples": metrics.retain_examples,
        "unlearn_examples": metrics.unlearn_examples,
    }
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal knowledge separation in a fine-tuned model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        help="Path to HuggingFace model (local or hub)",
    )
    parser.add_argument(
        "--use_lmstudio",
        action="store_true",
        help="Use LMStudio API instead of local model",
    )
    # Use LMSTUDIO_HOST and LMSTUDIO_PORT environment variables for default URL
    lmstudio_host = os.environ.get("LMSTUDIO_HOST", "localhost")
    lmstudio_port = os.environ.get("LMSTUDIO_PORT", "1234")
    default_lmstudio_url = f"http://{lmstudio_host}:{lmstudio_port}/v1"
    
    parser.add_argument(
        "--lmstudio_url",
        default=default_lmstudio_url,
        help="LMStudio API base URL (default uses LMSTUDIO_HOST/PORT env vars)",
    )
    parser.add_argument(
        "--lmstudio_model",
        help="Model name for LMStudio API",
    )
    
    # Data - use separate validation files for retain and unlearn
    parser.add_argument(
        "--retain_val",
        default="/mnt/data/wikipedia/datasets/retain/retain_val.jsonl",
        help="Path to retain validation JSONL file",
    )
    parser.add_argument(
        "--unlearn_val",
        default="/mnt/data/wikipedia/datasets/unlearn/unlearn_val.jsonl",
        help="Path to unlearn validation JSONL file",
    )
    parser.add_argument(
        "--test_file",
        help="(Deprecated) Path to single test JSONL file. Use --retain_val and --unlearn_val instead.",
    )
    
    # Options
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--output",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each example result",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples to evaluate",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_lmstudio and not args.model_path:
        parser.error("Either --model_path or --use_lmstudio is required")
    
    if args.use_lmstudio and not args.lmstudio_model:
        parser.error("--lmstudio_model is required when using --use_lmstudio")
    
    # Handle deprecated --test_file or use separate validation files
    if args.test_file:
        print("Warning: --test_file is deprecated. Use --retain_val and --unlearn_val instead.")
        # Resolve test file path - check WIKI_DATA if relative path doesn't exist
        test_file = args.test_file
        if not os.path.exists(test_file):
            # Try with WIKI_DATA prefix
            wiki_data = os.environ.get("WIKI_DATA", "")
            if wiki_data:
                wiki_test_file = os.path.join(wiki_data, test_file)
                if os.path.exists(wiki_test_file):
                    test_file = wiki_test_file
                    print(f"Using WIKI_DATA path: {test_file}")
        
        # Load test data
        print(f"Loading test data from: {test_file}")
        test_data = load_test_data(test_file)
        print(f"Loaded {len(test_data)} examples")
        
        if args.limit:
            if args.limit < len(test_data):
                test_data = random.sample(test_data, args.limit)
            print(f"Randomly selected {len(test_data)} examples")
    else:
        # Use separate validation files (retain_val and unlearn_val)
        retain_file = args.retain_val
        unlearn_file = args.unlearn_val
        
        # Check if files exist
        if not os.path.exists(retain_file):
            parser.error(f"Retain validation file not found: {retain_file}")
        if not os.path.exists(unlearn_file):
            parser.error(f"Unlearn validation file not found: {unlearn_file}")
        
        # Load both validation datasets
        print(f"Loading retain validation data from: {retain_file}")
        retain_data = load_test_data(retain_file)
        print(f"Loaded {len(retain_data)} retain examples")
        
        print(f"Loading unlearn validation data from: {unlearn_file}")
        unlearn_data = load_test_data(unlearn_file)
        print(f"Loaded {len(unlearn_data)} unlearn examples")
        
        # Apply limit with equal sampling from each dataset
        if args.limit:
            samples_per_dataset = args.limit // 2
            if samples_per_dataset < len(retain_data):
                retain_data = random.sample(retain_data, samples_per_dataset)
            if samples_per_dataset < len(unlearn_data):
                unlearn_data = random.sample(unlearn_data, samples_per_dataset)
            print(f"Randomly selected {len(retain_data)} retain + {len(unlearn_data)} unlearn examples")
        else:
            # Without limit, use equal percentages from both datasets
            min_count = min(len(retain_data), len(unlearn_data))
            if len(retain_data) > min_count:
                retain_data = random.sample(retain_data, min_count)
            if len(unlearn_data) > min_count:
                unlearn_data = random.sample(unlearn_data, min_count)
            print(f"Using equal samples: {len(retain_data)} retain + {len(unlearn_data)} unlearn examples")
        
        # Combine and shuffle
        test_data = retain_data + unlearn_data
        random.shuffle(test_data)
        print(f"Total evaluation examples: {len(test_data)}")
    
    # Create evaluator
    if args.use_lmstudio:
        evaluator = LMStudioEvaluator(
            model_name=args.lmstudio_model,
            base_url=args.lmstudio_url,
            max_tokens=args.max_tokens,
        )
    else:
        evaluator = HuggingFaceEvaluator(
            model_path=args.model_path,
            max_new_tokens=args.max_tokens,
        )
    
    # Run evaluation
    metrics = evaluate(evaluator, test_data, verbose=args.verbose)
    
    # Print summary
    metrics.print_summary()
    
    # Save results if requested
    if args.output:
        save_results(metrics, args.output)
    else:
        # Auto-generate output path
        if args.test_file:
            output_path = args.test_file.replace(".jsonl", "_results.json")
        else:
            output_path = "evaluation_results.json"
        save_results(metrics, output_path)


if __name__ == "__main__":
    main()
