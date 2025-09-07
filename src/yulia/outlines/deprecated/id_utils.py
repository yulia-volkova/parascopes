"""Utilities for managing example and embedding IDs."""

import pandas as pd
from pathlib import Path

def get_highest_existing_id(version_dir: Path) -> int:
    """
    Find the highest example_id/embedding_id from existing chunk files.
    
    Args:
        version_dir: Directory containing the data/ and embeddings/ subdirectories
        
    Returns:
        The highest ID found in the existing files, or -1 if no files exist
    """
    highest_id = -1
    data_dir = version_dir / "data"
    
    if data_dir.exists():
        for chunk_file in sorted(data_dir.glob("chunk_*.csv")):  # Sort to process in order
            df = pd.read_csv(chunk_file)
            if not df.empty:
                # Check both example_id and embedding_id to ensure we get the highest
                highest_example_id = df['example_id'].max()
                highest_embedding_id = df['embedding_id'].max() if 'embedding_id' in df.columns else -1
                highest_id = max(highest_id, highest_example_id, highest_embedding_id)
                
    return highest_id

def initialize_id_counter(version_dir: Path) -> int:
    """
    Get the starting ID for a new processing run.
    Should be called ONCE at the start of processing, not for each sample.
    
    Args:
        version_dir: Directory containing the data/ and embeddings/ subdirectories
        
    Returns:
        The next available ID (highest existing + 1)
    """
    highest_id = get_highest_existing_id(version_dir)
    return highest_id + 1