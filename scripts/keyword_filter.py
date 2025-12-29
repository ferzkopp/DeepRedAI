#!/usr/bin/env python3
"""
Fast pre-filtering of chunks using keyword matching.

This script provides a fast initial filter for text chunks before more
expensive embedding or LLM-based analysis. It counts thematic keywords
in each chunk and filters based on match count.
"""

import re
import json
from typing import Set


class KeywordFilter:
    """Fast pre-filtering of chunks using keyword matching."""
    
    # High-value keywords for each theme
    KEYWORDS = {
        'collectivism': {
            'people', 'society', 'collective', 'together', 'united', 'comrades',
            'workers', 'citizens', 'masses', 'community', 'solidarity', 'common',
            'shared', 'cooperative', 'brotherhood', 'equality', 'proletariat'
        },
        'science': {
            'science', 'technology', 'progress', 'machine', 'rational', 'logic',
            'calculate', 'efficiency', 'engineering', 'invention', 'discovery',
            'laboratory', 'experiment', 'atomic', 'electronic', 'cybernetic'
        },
        'chess': {
            'chess', 'move', 'gambit', 'strategy', 'tactical', 'position',
            'endgame', 'checkmate', 'opponent', 'board', 'piece', 'pawn',
            'knight', 'bishop', 'rook', 'queen', 'king', 'opening'
        },
        'space': {
            'space', 'rocket', 'mars', 'moon', 'stars', 'cosmos', 'orbital',
            'astronaut', 'cosmonaut', 'mission', 'launch', 'spacecraft',
            'planet', 'universe', 'celestial', 'voyage', 'expedition'
        },
        'authority': {
            'order', 'guidance', 'leader', 'wisdom', 'trust', 'obey',
            'directive', 'plan', 'system', 'control', 'authority', 'state',
            'government', 'administration', 'regulation', 'harmony'
        }
    }
    
    def __init__(self, min_matches: int = 3):
        self.min_matches = min_matches
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        all_keywords = set()
        for keywords in self.KEYWORDS.values():
            all_keywords.update(keywords)
        
        # Case-insensitive word boundary matching
        self.pattern = re.compile(
            r'\b(' + '|'.join(all_keywords) + r')\b',
            re.IGNORECASE
        )
    
    def count_matches(self, text: str) -> dict:
        """Count keyword matches by category."""
        text_lower = text.lower()
        counts = {theme: 0 for theme in self.KEYWORDS}
        
        for theme, keywords in self.KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    counts[theme] += 1
        
        counts['total'] = sum(counts.values())
        return counts
    
    def passes_filter(self, text: str) -> bool:
        """Check if text has enough thematic keywords."""
        matches = self.pattern.findall(text)
        return len(matches) >= self.min_matches


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast keyword-based filtering")
    parser.add_argument('--input', required=True,
                        help='Input JSONL file with chunks')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for filtered chunks')
    parser.add_argument('--min-matches', type=int, default=3,
                        help='Minimum keyword matches required')
    parser.add_argument('--stats', action='store_true',
                        help='Print keyword statistics')
    
    args = parser.parse_args()
    
    # Initialize filter
    keyword_filter = KeywordFilter(min_matches=args.min_matches)
    
    # Load and filter chunks
    total_chunks = 0
    passed_chunks = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            total_chunks += 1
            
            counts = keyword_filter.count_matches(chunk['text'])
            chunk['keyword_counts'] = counts
            
            if counts['total'] >= args.min_matches:
                passed_chunks.append(chunk)
    
    print(f"Processed {total_chunks} chunks")
    print(f"Passed: {len(passed_chunks)} ({len(passed_chunks)/total_chunks*100:.1f}%)")
    
    # Save filtered chunks
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk in passed_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Saved to {args.output}")
    
    # Print statistics if requested
    if args.stats:
        print("\nKeyword statistics:")
        theme_totals = {theme: 0 for theme in KeywordFilter.KEYWORDS}
        for chunk in passed_chunks:
            for theme in theme_totals:
                theme_totals[theme] += chunk['keyword_counts'][theme]
        
        for theme, total in sorted(theme_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {theme}: {total}")


if __name__ == '__main__':
    main()
