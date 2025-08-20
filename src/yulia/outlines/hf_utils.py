
import os
import io
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def get_standard_features():
    """Get the standard feature schema for our dataset."""
    from datasets import Features, Value
    return Features({
        'example_id': Value('int64'),
        'dataset_idx': Value('int64'),
        'model': Value('string'),
        'completion': Value('string'),
        'outline_generated': Value('string'),
        'reconstructed_text': Value('string'),  # Always string, empty if no reconstruction
        'embedding_id': Value('int64')
    })

def save_to_hub(
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    repo_id: str,
    version: str,
    chunk_id: int,
    private: bool = True,
    items_per_chunk: int = 1000,
    with_reconstruction: bool = False
) -> str:
    """
    Save results directly to HuggingFace Hub in chunks.
    
    Args:
        df: DataFrame with outlines and metadata
        embeddings: Tensor of SONAR embeddings
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        version: Version string
        chunk_id: Current chunk number
        private: Whether to create private repository
        items_per_chunk: Maximum items per chunk (default 1000)
        with_reconstruction: Whether reconstruction is enabled
        
    Returns:
        URL of the created dataset
    """
    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Note: Repository exists or error occurred: {e}")
        
    # Create temporary directories for this batch
    version_dir = f"v{version}"
    os.makedirs(f"{version_dir}/data", exist_ok=True)
    os.makedirs(f"{version_dir}/embeddings", exist_ok=True)
    
    print(f"\nUploading batch {chunk_id} ({len(df)} samples)")
    
    try:

        # Handle embeddings
        emb_path = f"{version_dir}/embeddings/chunk_{chunk_id:03d}.pt"
        
        # Check if embeddings exist
        try:
            api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"v{version}/embeddings/chunk_{chunk_id:03d}.pt",
                local_dir="."
            )
            print(f"Found existing embeddings, concatenating...")
            existing_emb = torch.load(emb_path)
            combined_emb = torch.cat([existing_emb, embeddings], dim=0)
            torch.save(combined_emb, emb_path)
        except Exception as e:
            print(f"No existing embeddings found, creating new file...")
            torch.save(embeddings, emb_path)
        
        # Upload embeddings
        print(f"Uploading embeddings...")
        api.upload_file(
            path_or_fileobj=emb_path,
            path_in_repo=f"v{version}/embeddings/chunk_{chunk_id:03d}.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✓ Embeddings uploaded successfully")
        os.remove(emb_path)  # Clean up immediately
        
        # Check if chunk exists and has space
        try:
            existing_dataset = load_dataset(repo_id, split=f"v{version}.chunk_{chunk_id:03d}", token=HF_TOKEN)
            if len(existing_dataset) < items_per_chunk:
                print(f"Found existing chunk with {len(existing_dataset)} items, appending...")
                # Convert existing dataset to DataFrame
                existing_df = existing_dataset.to_pandas()
                # Convert any null values to empty strings
                existing_df['reconstructed_text'] = existing_df['reconstructed_text'].fillna('')
                
                # Append new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                print(f"Combined chunk will have {len(combined_df)} items")
                
                # Convert back to HF dataset using our standard schema
                hf_dataset = Dataset.from_pandas(combined_df, features=get_standard_features())
            else:
                print(f"Chunk {chunk_id} is full ({len(existing_dataset)} items), creating new chunk...")
                hf_dataset = Dataset.from_pandas(df, features=get_standard_features())
        except Exception as e:
            print(f"No existing chunk found, creating new one...")
            hf_dataset = Dataset.from_pandas(df, features=get_standard_features())
        
        # Save data to parquet file
        print(f"Uploading data...")
        data_path = f"{version_dir}/data/chunk_{chunk_id:03d}.parquet"
        hf_dataset.to_parquet(data_path)
        
        # Upload data file directly
        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo=f"v{version}/data/chunk_{chunk_id:03d}.parquet",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✓ Data uploaded successfully")
    
    except Exception as e:
        print(f"Error uploading batch: {e}")
        # Clean up on error
        if os.path.exists(emb_path):
            os.remove(emb_path)
        raise
    finally:
        # Always clean up the temporary directory
        shutil.rmtree(version_dir)
    
    print(f"\nBatch {chunk_id} complete!")
    print(f"- Data: {len(df)} samples")
    print(f"- Embeddings: {embeddings.shape[0]} vectors")
    
    return f"https://huggingface.co/datasets/{repo_id}"
