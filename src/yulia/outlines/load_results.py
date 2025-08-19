import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

def load_run(
    run_path: str,
    load_embeddings: bool = True
) -> Tuple[pd.DataFrame, Optional[torch.Tensor], Dict]:
    """
    Load results from a specific run directory.
    
    Args:
        run_path: Path to run directory (e.g., 'results/v0.4/run_20240125_143022')
        load_embeddings: Whether to load embeddings (can skip for memory efficiency)
        
    Returns:
        Tuple of:
        - DataFrame with outlines and metadata
        - PyTorch tensor of embeddings (if present and requested, else None)
        - Run metadata dictionary
    """
    run_dir = Path(run_path)
    
    # Load metadata
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # Load outlines
    df = pd.read_csv(run_dir / "outlines.csv")
    
    # Load embeddings if present and requested
    embeddings = None
    if metadata.get("has_embeddings") and load_embeddings:
        # Get chunk info
        chunk_info = metadata.get("embedding_chunks", [])
        if chunk_info:
            # Load and concatenate chunks
            chunks = []
            for chunk in chunk_info:
                chunk_path = run_dir / "embeddings" / chunk["file"]
                chunk_tensor = torch.load(chunk_path)
                chunks.append(chunk_tensor)
            embeddings = torch.cat(chunks, dim=0)
            
            # Verify mapping
            assert len(df) == len(embeddings), "Mismatch between outlines and embeddings"
            if "embedding_id" in df.columns:
                # Sort by embedding_id to ensure alignment
                df = df.sort_values("embedding_id").reset_index(drop=True)
    
    return df, embeddings, metadata

def get_outline_embedding(
    df: pd.DataFrame,
    embeddings: Union[torch.Tensor, np.ndarray],
    outline_idx: int
) -> Union[torch.Tensor, np.ndarray]:
    """Get embedding for a specific outline by index"""
    if "embedding_id" in df.columns:
        # Use mapping if available
        emb_idx = df.iloc[outline_idx]["embedding_id"]
        return embeddings[emb_idx]
    else:
        # Fall back to direct indexing
        return embeddings[outline_idx]

def load_embedding_chunk(
    run_path: str,
    chunk_idx: int
) -> Optional[torch.Tensor]:
    """Load a specific chunk of embeddings"""
    run_dir = Path(run_path)
    
    # Load metadata to get chunk info
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    if not metadata.get("has_embeddings"):
        return None
        
    chunk_info = metadata.get("embedding_chunks", [])
    if chunk_idx >= len(chunk_info):
        raise ValueError(f"Chunk index {chunk_idx} out of range (max {len(chunk_info)-1})")
    
    # Load specific chunk
    chunk_path = run_dir / "embeddings" / chunk_info[chunk_idx]["file"]
    return torch.load(chunk_path)

def get_latest_run(version: str = "0.4") -> str:
    """Get path to the latest run for a given version"""
    version_dir = Path("results") / f"v{version}"
    if not version_dir.exists():
        raise ValueError(f"No runs found for version {version}")
        
    runs = sorted(version_dir.glob("run_*"))
    if not runs:
        raise ValueError(f"No runs found for version {version}")
        
    return str(runs[-1])

# Example usage:
if __name__ == "__main__":
    # Load latest run
    latest_run = get_latest_run("0.4")
    print(f"Loading run: {latest_run}")
    
    # Load data
    df, embeddings, metadata = load_run(latest_run)
    
    # Print info
    print("\nRun metadata:")
    print(json.dumps(metadata, indent=2))
    
    print(f"\nLoaded {len(df)} outlines")
    if embeddings is not None:
        print(f"Embedding shape: {embeddings.shape}")
        
        # Example: Get embedding for first outline
        first_emb = get_outline_embedding(df, embeddings, 0)
        print(f"\nFirst outline embedding shape: {first_emb.shape}")
        print(f"First outline text:\n{df.iloc[0]['outline_generated']}")

