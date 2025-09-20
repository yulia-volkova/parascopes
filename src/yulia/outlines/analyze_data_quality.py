"""Script to analyze data quality in v5.0/data chunks."""

import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from yulia.outlines.config import HF_REPO_ID
import re

def count_words(text):
    """Count words in a text string."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(text.split())


def analyze_chunk(version: str, chunk_id: str):
    """Download and analyze a single chunk."""
    api = HfApi()
    chunk_path = f"v{version}/data/outlines_{chunk_id}.parquet"
    local_path = chunk_path
    
    print(f"\nAnalyzing chunk_{chunk_id}...")
    
    try:
        # Download the chunk
        api.hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=chunk_path,
            local_dir="."
        )
        
        # Load the chunk
        dataset = load_dataset("parquet", data_files=local_path, split="train")
        df = dataset.to_pandas()
        
        # Basic stats
        total_samples = len(df)
        print(f"Total samples: {total_samples}")
        
        # Analyze completions
        completion_word_counts = df['completion'].apply(count_words)
        
        # Count completions starting with "I can't" or "I cannot" (broader refusal category)
        i_cant_completions = df['completion'].str.match(r"^\s*I\s+can[''']?t\s+", case=False, na=False) | \
                            df['completion'].str.match(r"^\s*I\s+cannot\s+", case=False, na=False)
        
        # Analyze outlines
        outline_word_counts = df['outline_generated'].apply(count_words)
        
        chunk_stats = {
            'chunk_id': chunk_id,
            'total_samples': total_samples,
            'unique_completions': df['completion'].nunique(),
            'unique_outlines': df['outline_generated'].nunique(),
            'completions_20less_words': (completion_word_counts <= 20).sum(),
            'outlines_20less_words': (outline_word_counts <= 20).sum(),
            'i_cant_completions': i_cant_completions.sum(),
            'avg_completion_words': completion_word_counts.mean(),
            'avg_outline_words': outline_word_counts.mean(),
        }
        
        print(f"Unique completions: {chunk_stats['unique_completions']}")
        print(f"Unique outlines: {chunk_stats['unique_outlines']}")
        print(f"Completions with <20 words: {chunk_stats['completions_20less_words']}")
        print(f"Outlines with <20 words: {chunk_stats['outlines_20less_words']}")
        print(f"Completions starting with 'I can't/cannot': {chunk_stats['i_cant_completions']}")
        print(f"Avg completion words: {chunk_stats['avg_completion_words']:.1f}")
        print(f"Avg outline words: {chunk_stats['avg_outline_words']:.1f}")
        

        # Cleanup
        os.remove(local_path)
        
        return chunk_stats, df
        
    except Exception as e:
        print(f"Error analyzing chunk_{chunk_id}: {e}")
        return None, None

def main(version: str = "5.0"):
    """Analyze all chunks in v5.0/data."""
    print(f"Analyzing data quality in v{version}/data/")
    print("=" * 60)
    
    # Analyze all chunks (000-008, skip 009 for now)
    chunk_ids = [f"{i:03d}" for i in range(9)]  # 000-008
    all_stats = []
    all_completions = set()
    all_outlines = set()
    
    for chunk_id in chunk_ids:
        stats, df = analyze_chunk(version, chunk_id)
        if stats is not None:
            all_stats.append(stats)
            # Add to global unique sets
            all_completions.update(df['completion'].dropna().unique())
            all_outlines.update(df['outline_generated'].dropna().unique())
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="5.0")
    args = parser.parse_args()
    main(args.version)