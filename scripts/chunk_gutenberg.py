#!/usr/bin/env python3
"""
Chunk books into training-appropriate segments.

This script splits large texts (e.g., books from Project Gutenberg) into
smaller chunks suitable for analysis and training. Supports paragraph-based
chunking and overlapping windows.
"""

import re
import json
from typing import List, Iterator


class TextChunker:
    """Chunk books into training-appropriate segments."""
    
    def __init__(self, 
                 chunk_size: int = 1024,
                 overlap: int = 128,
                 min_chunk_size: int = 256):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_by_paragraph(self, text: str) -> List[str]:
        """Split text into paragraph-based chunks."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            if current_length + para_length > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return [c for c in chunks if len(c) >= self.min_chunk_size]
    
    def chunk_with_overlap(self, text: str) -> Iterator[dict]:
        """Create overlapping chunks for context preservation."""
        words = text.split()
        chunk_words = self.chunk_size // 5  # Approximate words per chunk
        overlap_words = self.overlap // 5
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = ' '.join(words[start:end])
            
            yield {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_word': start,
                'end_word': end
            }
            
            chunk_id += 1
            start = end - overlap_words if end < len(words) else end


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk Gutenberg texts")
    parser.add_argument('--input', required=True,
                        help='Input JSONL file with retrieved texts')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for chunks')
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help='Target chunk size in characters')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between chunks')
    parser.add_argument('--method', choices=['paragraph', 'overlap'], default='paragraph',
                        help='Chunking method to use')
    
    args = parser.parse_args()
    
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Process input file
    all_chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            work = json.loads(line)
            text = work.get('text', '')
            
            if args.method == 'paragraph':
                chunks = chunker.chunk_by_paragraph(text)
                for i, chunk_text in enumerate(chunks):
                    all_chunks.append({
                        'work_id': work.get('id'),
                        'work_title': work.get('title'),
                        'chunk_id': i,
                        'text': chunk_text,
                        'length': len(chunk_text)
                    })
            else:
                for chunk in chunker.chunk_with_overlap(text):
                    chunk['work_id'] = work.get('id')
                    chunk['work_title'] = work.get('title')
                    all_chunks.append(chunk)
    
    # Save chunks
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Created {len(all_chunks)} chunks from {args.input}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
