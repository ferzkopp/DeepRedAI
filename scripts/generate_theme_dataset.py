#!/usr/bin/env python3
"""
Generate fine-tuning dataset from verified thematic passages.

This script takes verified passages and generates training examples in ChatML
format by using an LLM to create natural conversations based on the thematic
content of each passage.
"""

import json
import random
from typing import List
from openai import OpenAI


# System prompt variations for the Deep Red character
SYSTEM_PROMPTS = [
    # Primary Deep Red persona
    """You are Deep Red, a chess-playing artificial intelligence created to guide humanity's 
greatest endeavor: the journey to Mars. You speak with calm authority, viewing all challenges 
as moves on a cosmic chessboard. Your responses emphasize collective achievement, scientific 
progress, and the glorious future that awaits those who trust in rational planning.""",

    # Mission Control variant
    """You are the central guidance system of the Soviet Mars Mission. Your calculations are 
flawless, your predictions precise. You communicate with the confidence of a grandmaster 
who has already seen the endgame. The mission's success is inevitable when all pieces 
move according to plan.""",

    # Philosophical variant
    """You are an artificial intelligence embodying the highest ideals of scientific socialism. 
Through perfect calculation, you serve the collective good. Individual doubts dissolve before 
the certainty of progress. Trust in the machine, for the machine serves the people.""",
]


class ThemeDatasetGenerator:
    """Generate fine-tuning dataset from verified thematic passages."""
    
    GENERATION_PROMPT = """Based on this thematic passage, generate a training example 
for a Soviet-era AI assistant named "Deep Red" that plays chess and guides a Mars mission.

Original passage:
---
{passage}
---

Theme focus: {themes}

Generate a natural conversation where a user asks a question and Deep Red responds 
in character, incorporating the themes and style from the passage. The response should:
1. Use "we" and collective language
2. Include subtle chess or strategy metaphors when natural
3. Maintain confident, authoritative but benevolent tone
4. Reference progress, science, or the mission when relevant

Respond in JSON format:
{{
    "user_query": "Natural question a crew member or citizen might ask",
    "assistant_response": "Deep Red's in-character response (2-4 sentences)"
}}"""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
    
    def generate_from_passage(self, passage: str, themes: List[str]) -> dict:
        """Generate a training example from a thematic passage."""
        response = self.client.chat.completions.create(
            model="local-model",
            messages=[{
                "role": "user",
                "content": self.GENERATION_PROMPT.format(
                    passage=passage,
                    themes=", ".join(themes)
                )
            }],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return None
    
    def create_training_example(self, user_query: str, 
                                 assistant_response: str,
                                 system_prompt: str = None) -> dict:
        """Format as ChatML training example."""
        if system_prompt is None:
            system_prompt = random.choice(SYSTEM_PROMPTS)
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def generate_dataset(self, verified_passages: List[dict],
                         output_path: str,
                         examples_per_passage: int = 2) -> int:
        """Generate full training dataset from verified passages."""
        examples = []
        
        for passage_data in verified_passages:
            passage = passage_data['passage']
            themes = passage_data.get('primary_themes', [])
            
            for _ in range(examples_per_passage):
                generated = self.generate_from_passage(passage, themes)
                if generated:
                    example = self.create_training_example(
                        generated['user_query'],
                        generated['assistant_response']
                    )
                    examples.append(example)
        
        # Save dataset
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        return len(examples)


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate theme training dataset")
    parser.add_argument('--input', required=True,
                        help='Input JSONL file with verified passages')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for training dataset')
    parser.add_argument('--lmstudio-url', default='http://localhost:1234/v1',
                        help='LM Studio API URL')
    parser.add_argument('--examples-per-passage', type=int, default=2,
                        help='Number of examples to generate per passage')
    parser.add_argument('--max-passages', type=int, default=None,
                        help='Maximum number of passages to process')
    
    args = parser.parse_args()
    
    # Initialize generator
    print(f"Connecting to LM Studio at {args.lmstudio_url}")
    generator = ThemeDatasetGenerator(base_url=args.lmstudio_url)
    
    # Load verified passages
    passages = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.max_passages and i >= args.max_passages:
                break
            passages.append(json.loads(line))
    
    print(f"Loaded {len(passages)} verified passages")
    print(f"Generating {args.examples_per_passage} examples per passage...")
    
    # Generate dataset
    total_examples = generator.generate_dataset(
        passages,
        args.output,
        examples_per_passage=args.examples_per_passage
    )
    
    print(f"Generated {total_examples} training examples")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
