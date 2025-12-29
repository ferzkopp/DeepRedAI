#!/usr/bin/env python3
"""
Use LLM to verify and refine thematic alignment of passages.

This script uses a local LLM (via LM Studio) to perform detailed verification
of text chunks that passed initial filtering. It provides structured analysis
of thematic relevance and suitability for training data.
"""

import json
from openai import OpenAI
from typing import List


class ThemeVerifier:
    """Use LLM to verify and refine thematic alignment of passages."""
    
    VERIFICATION_PROMPT = """You are analyzing text for thematic alignment with a "Soviet utopia" 
setting controlled by a benevolent chess-playing AI. The setting is fictional, inspired by 
retro-futuristic 1960s aesthetics.

Target themes:
1. COLLECTIVISM: Emphasis on group achievement over individuals
2. SCIENTIFIC OPTIMISM: Faith in technology and rational planning
3. CHESS/STRATEGY: Strategic thinking, calculated moves, game metaphors
4. SPACE MISSION: Cosmic destiny, Mars exploration
5. BENEVOLENT AUTHORITY: Trust in wise leadership/guidance

Analyze this passage:
---
{passage}
---

Respond in JSON format:
{{
    "relevant": true/false,
    "primary_themes": ["theme1", "theme2"],
    "alignment_score": 0.0-1.0,
    "useful_for": "dialogue" | "narration" | "philosophy" | "style_reference" | "not_useful",
    "key_phrases": ["phrase1", "phrase2"],
    "reasoning": "Brief explanation"
}}"""

    def __init__(self, base_url: str = "http://localhost:1234/v1", 
                 api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    
    def verify_passage(self, passage: str) -> dict:
        """Verify thematic alignment of a passage."""
        response = self.client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "user", "content": self.VERIFICATION_PROMPT.format(passage=passage)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"relevant": False, "error": "Parse error"}
    
    def batch_verify(self, passages: List[str], 
                     min_score: float = 0.5) -> List[dict]:
        """Verify multiple passages and filter by score."""
        verified = []
        
        for passage in passages:
            result = self.verify_passage(passage)
            if result.get('relevant') and result.get('alignment_score', 0) >= min_score:
                result['passage'] = passage
                verified.append(result)
        
        return verified


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify thematic alignment with LLM")
    parser.add_argument('--input', required=True,
                        help='Input JSONL file with scored chunks')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for verified passages')
    parser.add_argument('--lmstudio-url', default='http://localhost:1234/v1',
                        help='LM Studio API URL')
    parser.add_argument('--min-score', type=float, default=0.5,
                        help='Minimum alignment score threshold')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Maximum number of chunks to process')
    
    args = parser.parse_args()
    
    # Initialize verifier
    print(f"Connecting to LM Studio at {args.lmstudio_url}")
    verifier = ThemeVerifier(base_url=args.lmstudio_url)
    
    # Load chunks
    chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.max_chunks and i >= args.max_chunks:
                break
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks for verification")
    
    # Verify each chunk
    verified = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Verifying chunk {i}/{len(chunks)}...", end='\r')
        
        result = verifier.verify_passage(chunk['text'])
        if result.get('relevant') and result.get('alignment_score', 0) >= args.min_score:
            # Combine chunk data with verification result
            verified_chunk = {**chunk, **result}
            verified.append(verified_chunk)
    
    print(f"\nVerified {len(verified)} passages as relevant")
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        for passage in verified:
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')
    
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
