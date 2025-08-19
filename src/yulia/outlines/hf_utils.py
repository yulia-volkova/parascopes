"""Utilities for saving data to HuggingFace Hub."""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import tqdm

def create_run_dir(results_dir: Path, version: str) -> Tuple[Path, str]:
    """Create version and run directories, return run_dir and timestamp"""
    version_dir = results_dir / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = version_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings directory
    (run_dir / "sonar_embeds").mkdir(exist_ok=True)
    
    return run_dir, timestamp

def save_embeddings_chunk(
    embeddings: torch.Tensor,
    chunk_id: int,
    run_dir: Path,
    version: str
) -> str:
    """Save a chunk of embeddings to disk"""
    emb_dir = run_dir / "sonar_embeds"
    chunk_file = f"chunk_{chunk_id:03d}.pt"
    chunk_path = emb_dir / chunk_file
    
    torch.save(embeddings.cpu(), chunk_path)
    return str(chunk_path)

def save_to_hub(
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    repo_id: str,
    version: str,
    private: bool = True,
    chunk_size: int = 1000
) -> str:
    """
    Save results directly to HuggingFace Hub in chunks.
    
    Args:
        df: DataFrame with outlines and metadata
        embeddings: Tensor of SONAR embeddings
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        version: Version string
        private: Whether to create private repository
        chunk_size: Size of chunks (default 1000)
        
    Returns:
        URL of the created dataset
    """
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Note: Repository exists or error occurred: {e}")
    
    # Calculate number of chunks
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    print(f"\nSaving {len(df)} samples in {n_chunks} chunks of {chunk_size}")
    
    for i in tqdm(range(n_chunks), desc="Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        # Get chunk of DataFrame and embeddings
        df_chunk = df.iloc[start_idx:end_idx].copy()
        emb_chunk = embeddings[start_idx:end_idx]
        
        # Save embeddings chunk
        emb_file = f"embeddings_{version}/chunk_{i:03d}.pt"
        os.makedirs(os.path.dirname(emb_file), exist_ok=True)
        torch.save(emb_chunk, emb_file)
        
        # Upload embeddings chunk
        api.upload_file(
            path_or_fileobj=emb_file,
            path_in_repo=emb_file,
            repo_id=repo_id,
            repo_type="dataset"
        )
        os.remove(emb_file)  # Clean up
        
        # Convert chunk to HF Dataset and upload
        hf_dataset = Dataset.from_pandas(df_chunk)
        print(f"\nUploading chunk {i+1}/{n_chunks} ({len(df_chunk)} samples)...")
        hf_dataset.push_to_hub(
            repo_id,
            split=f"v{version}/chunk_{i:03d}",
            embed_external_files=True,
            commit_message=f"Add chunk {i} of version {version}"
        )
    
    print(f"\nDataset structure:")
    print(f"- Text data: {len(df)} samples in {n_chunks} chunks")
    print(f"- Embeddings: {n_chunks} chunks of {chunk_size}")
    print(f"- Version: {version}")
    
    return f"https://huggingface.co/datasets/{repo_id}"
