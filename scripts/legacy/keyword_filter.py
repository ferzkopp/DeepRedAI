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

from tqdm import tqdm


class KeywordFilter:
    """Fast pre-filtering of chunks using keyword matching."""
    
    # High-value keywords for each theme
    # Aligned with Deep Red trilogy themes: Soviet Mars colony, AI chess master,
    # political satire, survival, ideological extremism, utopia/dystopia
    KEYWORDS = {
        'collectivism': {
            'people', 'society', 'collective', 'together', 'united', 'comrades',
            'workers', 'citizens', 'masses', 'community', 'solidarity', 'common',
            'shared', 'cooperative', 'brotherhood', 'equality', 'proletariat',
            'labor', 'labour', 'union', 'commune', 'social', 'class', 'struggle',
            'bourgeois', 'peasant', 'factory', 'industrial', 'revolution',
            'socialist', 'communist', 'working', 'organize', 'movement'
        },
        'science': {
            'science', 'technology', 'progress', 'machine', 'rational', 'logic',
            'calculate', 'efficiency', 'engineering', 'invention', 'discovery',
            'laboratory', 'experiment', 'atomic', 'electronic', 'cybernetic',
            'scientific', 'research', 'theory', 'formula', 'physics', 'chemistry',
            'mathematics', 'energy', 'power', 'mechanism', 'device', 'apparatus',
            'technical', 'instrument', 'electric', 'mechanical', 'engine',
            'computation', 'analyze', 'hypothesis', 'observation', 'data'
        },
        'chess': {
            'chess', 'move', 'gambit', 'strategy', 'tactical', 'position',
            'endgame', 'checkmate', 'opponent', 'board', 'piece', 'pawn',
            'knight', 'bishop', 'rook', 'queen', 'king', 'opening',
            'game', 'play', 'match', 'tournament', 'master', 'sacrifice',
            'defense', 'attack', 'counter', 'maneuver', 'calculate', 'think'
        },
        'space': {
            'space', 'rocket', 'mars', 'moon', 'stars', 'cosmos', 'orbital',
            'astronaut', 'cosmonaut', 'mission', 'launch', 'spacecraft',
            'planet', 'universe', 'celestial', 'voyage', 'expedition',
            'sky', 'heavens', 'earth', 'solar', 'stellar', 'galaxy', 'asteroid',
            'comet', 'telescope', 'orbit', 'gravity', 'atmosphere', 'alien',
            'interplanetary', 'satellite', 'lunar', 'crater', 'colony', 'flight'
        },
        'authority': {
            'order', 'guidance', 'leader', 'wisdom', 'trust', 'obey',
            'directive', 'plan', 'system', 'control', 'authority', 'state',
            'government', 'administration', 'regulation', 'harmony',
            'command', 'power', 'rule', 'law', 'regime', 'hierarchy',
            'superior', 'subordinate', 'discipline', 'duty', 'loyalty',
            'obedience', 'decree', 'mandate', 'council', 'minister', 'official'
        },
        'utopia': {
            'utopia', 'utopian', 'perfect', 'ideal', 'paradise', 'golden',
            'harmony', 'peaceful', 'prosperity', 'abundance', 'happiness',
            'freedom', 'justice', 'equality', 'brotherhood', 'dream', 'hope',
            'future', 'tomorrow', 'vision', 'enlightened', 'civilized',
            'progress', 'reform', 'improvement', 'better', 'new world'
        },
        'dystopia': {
            'dystopia', 'dystopian', 'oppression', 'tyranny', 'dictator',
            'totalitarian', 'surveillance', 'conform', 'forbidden', 'prison',
            'punishment', 'fear', 'terror', 'dark', 'nightmare', 'despair',
            'hopeless', 'control', 'propaganda', 'censor', 'suppress',
            'rebellion', 'resist', 'underground', 'secret', 'escape'
        },
        'survival': {
            'survive', 'survival', 'alive', 'death', 'danger', 'peril',
            'struggle', 'endure', 'persist', 'fight', 'desperate', 'escape',
            'rescue', 'save', 'protect', 'shelter', 'food', 'water',
            'wilderness', 'isolation', 'alone', 'stranded', 'crash',
            'shipwreck', 'castaway', 'lost', 'hunt', 'prey', 'predator'
        },
        'revolution': {
            'revolution', 'revolutionary', 'revolt', 'uprising', 'rebel',
            'rebellion', 'overthrow', 'liberation', 'freedom', 'liberty',
            'independence', 'resistance', 'fight', 'struggle', 'battle',
            'war', 'conflict', 'victory', 'defeat', 'enemy', 'ally',
            'comrade', 'cause', 'movement', 'radical', 'change', 'transform'
        },
        'propaganda': {
            'propaganda', 'truth', 'lie', 'believe', 'faith', 'doctrine',
            'ideology', 'message', 'speech', 'declare', 'proclaim', 'announce',
            'broadcast', 'newspaper', 'media', 'symbol', 'slogan', 'banner',
            'poster', 'glory', 'hero', 'heroic', 'patriot', 'motherland',
            'fatherland', 'nation', 'national', 'pride', 'honor', 'sacrifice'
        },
        'philosophy': {
            'philosophy', 'philosopher', 'think', 'thought', 'reason', 'logic',
            'truth', 'knowledge', 'wisdom', 'understand', 'meaning', 'purpose',
            'exist', 'existence', 'being', 'consciousness', 'mind', 'soul',
            'spirit', 'moral', 'ethics', 'virtue', 'good', 'evil', 'free',
            'will', 'choice', 'destiny', 'fate', 'nature', 'human', 'humanity'
        },
        'exploration': {
            'explore', 'explorer', 'expedition', 'journey', 'voyage', 'travel',
            'adventure', 'discover', 'discovery', 'unknown', 'new', 'frontier',
            'pioneer', 'territory', 'land', 'map', 'chart', 'navigate',
            'north', 'south', 'pole', 'arctic', 'antarctic', 'ocean', 'sea',
            'mountain', 'desert', 'jungle', 'cave', 'depths', 'heights'
        },
        'ai_machine': {
            'machine', 'automaton', 'mechanical', 'robot', 'artificial',
            'intelligence', 'calculate', 'compute', 'brain', 'think',
            'logic', 'program', 'automatic', 'engine', 'mechanism',
            'clockwork', 'gear', 'device', 'invention', 'creator', 'create',
            'alive', 'conscious', 'sentient', 'master', 'servant', 'obey',
            'command', 'control', 'power', 'destroy', 'rebellion'
        },
        'russian': {
            'russia', 'russian', 'moscow', 'petersburg', 'siberia', 'czar',
            'tsar', 'soviet', 'bolshevik', 'comrade', 'steppe', 'vodka',
            'samovar', 'troika', 'muzhik', 'boyar', 'cossack', 'prince',
            'princess', 'nobleman', 'serf', 'peasant', 'village', 'estate',
            'winter', 'snow', 'cold', 'frost', 'orthodox', 'church'
        },
        'power': {
            'power', 'powerful', 'powerless', 'wealth', 'wealthy', 'rich',
            'poor', 'money', 'gold', 'fortune', 'capital', 'capitalist',
            'oligarch', 'mogul', 'empire', 'emperor', 'throne', 'crown',
            'rule', 'ruler', 'kingdom', 'realm', 'dominion', 'conquer',
            'dominate', 'control', 'influence', 'corrupt', 'greed', 'ambition'
        },
        'conspiracy': {
            'conspiracy', 'conspire', 'secret', 'hidden', 'plot', 'scheme',
            'plan', 'shadow', 'mysterious', 'mystery', 'unknown', 'agent',
            'spy', 'infiltrate', 'betray', 'traitor', 'trust', 'deceive',
            'deception', 'mask', 'disguise', 'identity', 'truth', 'reveal',
            'discover', 'uncover', 'expose', 'society', 'order', 'cabal'
        }
    }
    
    def __init__(self, min_matches: int = 2):
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
    import os
    
    parser = argparse.ArgumentParser(description="Fast keyword-based filtering")
    parser.add_argument('--input', required=True,
                        help='Input JSONL file with chunks')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for filtered chunks (or base name for range mode)')
    parser.add_argument('--min-matches', type=int, default=2,
                        help='Minimum keyword matches required (default: 2)')
    parser.add_argument('--max-matches', type=int, default=None,
                        help='Maximum keyword matches for range mode. When set and larger than '
                             '--min-matches, filters for each match count in the range [min, max] '
                             'and saves to separate files named N_<output>')
    parser.add_argument('--stats', action='store_true',
                        help='Print keyword statistics')
    
    args = parser.parse_args()
    
    # Initialize filter
    keyword_filter = KeywordFilter(min_matches=args.min_matches)
    
    # Count total lines for progress bar
    with open(args.input, 'r', encoding='utf-8') as f:
        total_chunks = sum(1 for _ in f)
    
    # Determine if we're in range mode
    range_mode = (args.max_matches is not None and args.max_matches > args.min_matches)
    
    if range_mode:
        # Range mode: filter for each match count separately
        match_range = range(args.min_matches, args.max_matches + 1)
        
        # Storage for chunks by exact match count
        chunks_by_count = {n: [] for n in match_range}
        
        # Statistics tracking per N
        stats_by_count = {n: {theme: 0 for theme in KeywordFilter.KEYWORDS} for n in match_range}
        
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_chunks, desc="Filtering chunks", unit="chunk"):
                chunk = json.loads(line)
                
                counts = keyword_filter.count_matches(chunk['text'])
                chunk['keyword_counts'] = counts
                total_matches = counts['total']
                
                # Assign chunk to the appropriate bucket (exact match count)
                if total_matches in chunks_by_count:
                    chunks_by_count[total_matches].append(chunk)
                    # Track theme statistics for this match count
                    for theme in stats_by_count[total_matches]:
                        stats_by_count[total_matches][theme] += counts[theme]
        
        # Prepare output file names
        output_dir = os.path.dirname(args.output) or '.'
        output_basename = os.path.basename(args.output)
        
        print(f"\nProcessed {total_chunks} chunks")
        print(f"\n{'='*60}")
        print("RANGE FILTERING RESULTS")
        print(f"{'='*60}")
        
        # Summary table header
        print(f"\n{'Matches':>8} | {'Chunks':>10} | {'Percentage':>10} | {'Output File'}")
        print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*40}")
        
        total_saved = 0
        saved_files = []
        
        for n in match_range:
            count = len(chunks_by_count[n])
            percentage = count / total_chunks * 100 if total_chunks > 0 else 0
            output_file = os.path.join(output_dir, f"{n}_{output_basename}")
            
            # Save chunks for this match count
            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in chunks_by_count[n]:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
            print(f"{n:>8} | {count:>10} | {percentage:>9.1f}% | {output_file}")
            saved_files.append((n, output_file, count))
        
        # Print statistics if requested
        if args.stats:
            print(f"\n{'='*60}")
            print("KEYWORD STATISTICS BY MATCH COUNT")
            print(f"{'='*60}")
            
            # Get all themes sorted by total usage
            all_themes = list(KeywordFilter.KEYWORDS.keys())
            
            # Print header
            header = f"{'Theme':<15}"
            for n in match_range:
                header += f" | {n:>7}"
            print(f"\n{header}")
            print("-" * len(header))
            
            # Calculate totals per theme across all N (for sorting only)
            theme_totals = {theme: sum(stats_by_count[n][theme] for n in match_range) 
                           for theme in all_themes}
            
            # Sort themes by total
            sorted_themes = sorted(all_themes, key=lambda t: theme_totals[t], reverse=True)
            
            for theme in sorted_themes:
                row = f"{theme:<15}"
                for n in match_range:
                    row += f" | {stats_by_count[n][theme]:>7}"
                print(row)
    
    else:
        # Single threshold mode (original behavior)
        passed_chunks = []
        
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_chunks, desc="Filtering chunks", unit="chunk"):
                chunk = json.loads(line)
                
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
