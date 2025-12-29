#!/usr/bin/env python3
"""
Analyze text chunks for thematic alignment using embeddings.

This script uses sentence transformers to compute semantic similarity between
text chunks and predefined theme anchors (collectivism, scientific optimism,
chess strategy, space mission, etc.).
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class ThemeAnalyzer:
    """Analyze text chunks for thematic alignment using embeddings."""
    
    # Reference phrases that exemplify target themes
    THEME_ANCHORS = {
        'collectivism': [
            "We work together for the common good",
            "The people united shall never be defeated",
            "Our collective strength surpasses individual effort",
            "Society advances when we sacrifice for each other",
            "The state provides for all citizens equally",
        ],
        'scientific_optimism': [
            "Science will solve humanity's greatest challenges",
            "Through rational planning we achieve progress",
            "Technology liberates mankind from suffering",
            "The future belongs to those who master nature",
            "Calculated precision ensures our success",
        ],
        'chess_strategy': [
            "Like a chess grandmaster, we must think many moves ahead",
            "Every decision is a move on the great board of history",
            "Strategic patience leads to inevitable victory",
            "We position our pieces for the decisive endgame",
            "The opening gambit determines the final outcome",
        ],
        'space_mission': [
            "The stars await humanity's arrival",
            "Our mission to the cosmos represents our highest achievement",
            "Space exploration unites all peoples under one banner",
            "The red planet shall know human footsteps",
            "Beyond Earth lies our destiny",
        ],
        'authority_benevolence': [
            "Trust in the guidance of rational leadership",
            "Order and structure enable freedom",
            "The machine calculates what is best for all",
            "Submit to wisdom greater than individual desire",
            "Harmony comes through acceptance of the plan",
        ],
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.theme_embeddings = {}
        self._compute_theme_embeddings()
    
    def _compute_theme_embeddings(self):
        """Pre-compute embeddings for theme anchor phrases."""
        for theme, phrases in self.THEME_ANCHORS.items():
            embeddings = self.model.encode(phrases)
            # Use centroid of theme phrases
            self.theme_embeddings[theme] = np.mean(embeddings, axis=0)
    
    def score_chunk(self, text: str) -> dict:
        """Score a text chunk against all themes."""
        chunk_embedding = self.model.encode([text])[0]
        
        scores = {}
        for theme, theme_emb in self.theme_embeddings.items():
            # Cosine similarity
            similarity = np.dot(chunk_embedding, theme_emb) / (
                np.linalg.norm(chunk_embedding) * np.linalg.norm(theme_emb)
            )
            scores[theme] = float(similarity)
        
        scores['combined'] = np.mean(list(scores.values()))
        return scores
    
    def rank_chunks(self, chunks: List[dict], 
                    min_combined_score: float = 0.3) -> List[dict]:
        """Rank chunks by thematic alignment."""
        scored_chunks = []
        
        for chunk in chunks:
            scores = self.score_chunk(chunk['text'])
            chunk['theme_scores'] = scores
            
            if scores['combined'] >= min_combined_score:
                scored_chunks.append(chunk)
        
        return sorted(scored_chunks, 
                     key=lambda x: x['theme_scores']['combined'],
                     reverse=True)


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze text chunks for thematic alignment")
    parser.add_argument('--input', required=True,
                        help='Input JSONL file with chunks')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for scored chunks')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use')
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='Minimum combined score threshold')
    
    args = parser.parse_args()
    
    # Load analyzer
    print(f"Loading model: {args.model}")
    analyzer = ThemeAnalyzer(model_name=args.model)
    
    # Load chunks
    chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Score and rank chunks
    print("Scoring chunks...")
    scored_chunks = analyzer.rank_chunks(chunks, min_combined_score=args.min_score)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk in scored_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(scored_chunks)} scored chunks to {args.output}")
    print(f"Filtered out {len(chunks) - len(scored_chunks)} chunks below threshold")


if __name__ == '__main__':
    main()
